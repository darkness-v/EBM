import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BPL(nn.Module):
    def __init__(self, alpha=20.0):
        super().__init__()
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, x, y):
            
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        cos_sim = x @ y.transpose(0, 1)
        
        loss = self.softplus(self.alpha * (1 - cos_sim)) - 0.3
        
        loss = loss.mean()
        
        return loss

    
class OCSoftmax(nn.Module):
    def __init__(self, embedding_size, num_class=1, r_real=0.9, r_fake=0.2, alpha=20.0, class_weight=[1, 1]):
        super().__init__()
        self.embedding_size = embedding_size
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.class_weight = class_weight
        
        self.weight = nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2,1,1e-5).mul_(1e5)
        
        self.softplus = nn.Softplus()

    def forward(self, x, label=None):
            
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        scores = x @ w.transpose(0, 1)
        final_scores = None
        if label is None:
            final_scores = scores # if bonafide label is 1

        scores[label == 1] = self.r_real - scores[label == 1]
        scores[label == 0] = scores[label == 0] - self.r_fake

        loss = self.softplus(self.alpha * scores)
        # class_weight
        if x.size(0) > 4:
            loss[label == 0] = loss[label == 0] * self.class_weight[0]
            loss[label == 1] = loss[label == 1] * self.class_weight[1]
        
        loss = loss.mean() if label is not None else loss.sum()
        
        return loss, final_scores


class EMAPrototypeManager(nn.Module):
    """Maintains K L2-normalized EMA prototypes for bona-fide class.

    Buffers are broadcast by DDP as module buffers (no gradients).
    """
    def __init__(self, embedding_size: int, K: int = 1, beta: float = 0.995, assignment: str = 'hard', tau: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.embedding_size = embedding_size
        self.K = K
        self.beta = beta
        self.assignment = assignment  # 'hard' or 'soft'
        self.tau = tau
        self.eps = eps
        # Prototypes buffer: (K, D)
        self.register_buffer('mu', torch.zeros(K, embedding_size))
        # Initialization flag
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def maybe_init(self, x_bona: torch.Tensor):
        """Initialize prototypes from a batch of bona-fide embeddings if not initialized."""
        if self.initialized.item() == 1:
            return
        if x_bona is None or x_bona.numel() == 0:
            return
        x_bona = F.normalize(x_bona, p=2, dim=1)
        # pick up to K random bona-fide embeddings
        idx = torch.randperm(x_bona.size(0), device=x_bona.device)[:self.K]
        init = x_bona[idx]
        if init.size(0) < self.K:
            # pad by repeating
            reps = self.K - init.size(0)
            init = torch.cat([init, init[:reps]], dim=0)
        self.mu.copy_(F.normalize(init[:self.K], p=2, dim=1))
        self.initialized.fill_(1)

    def score(self, x: torch.Tensor):
        """Compute cosine similarity scores using current prototypes.
        Returns sims per-prototype and aggregated score s (B,), and soft weights if used.
        """
        mu = F.normalize(self.mu, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        sims = x @ mu.t()  # (B, K)
        if self.K == 1:
            return sims, sims.squeeze(1), None
        if self.assignment == 'hard':
            s, _ = sims.max(dim=1)
            return sims, s, None
        else:
            # soft
            w = F.softmax(self.tau * sims, dim=1)  # (B, K)
            s = (w * sims).sum(dim=1)
            return sims, s, w

    @torch.no_grad()
    def update(self, x_bona: torch.Tensor):
        """EMA update with bona-fide embeddings only. Call this on rank 0.
        x_bona: (B_bf, D)
        """
        if x_bona is None or x_bona.numel() == 0:
            return
        x_bona = F.normalize(x_bona, p=2, dim=1)
        self.maybe_init(x_bona)
        # compute assignments
        sims = x_bona @ F.normalize(self.mu, p=2, dim=1).t()  # (B_bf, K)
        if self.K == 1:
            weights = torch.ones(x_bona.size(0), 1, device=x_bona.device)
        else:
            if self.assignment == 'hard':
                max_idx = sims.argmax(dim=1)
                weights = torch.zeros_like(sims)
                weights.scatter_(1, max_idx.unsqueeze(1), 1.0)
            else:
                weights = F.softmax(self.tau * sims, dim=1)
        # per-prototype weighted mean
        denom = weights.sum(dim=0) + self.eps  # (K,)
        xbar = (weights.t() @ x_bona) / denom.unsqueeze(1)  # (K, D)
        # EMA update per prototype
        mu_prime = self.beta * self.mu + (1.0 - self.beta) * xbar
        self.mu.copy_(F.normalize(mu_prime, p=2, dim=1))

    def proto_pull(self, x_bona: torch.Tensor, use_soft: bool = False):
        """Compute proto-pull term for bona-fide embeddings.
        Returns mean(1 - mu_k*^T x_i). If use_soft, use soft weights aggregation.
        """
        if x_bona is None or x_bona.numel() == 0:
            return x_bona.new_zeros(())
        x_bona = F.normalize(x_bona, p=2, dim=1)
        sims = x_bona @ F.normalize(self.mu, p=2, dim=1).t()  # (B, K)
        if self.K == 1:
            s = sims.squeeze(1)
        else:
            if use_soft or self.assignment == 'soft':
                w = F.softmax(self.tau * sims, dim=1)
                s = (w * sims).sum(dim=1)
            else:
                s, _ = sims.max(dim=1)
        return (1.0 - s).mean()


class EOCS_EMA(nn.Module):
    """EOC-S style loss using EMA prototypes for bona-fide scoring.

    Loss: softplus(alpha * (m_y - s) * sign), where sign = (-1)^y.
    """
    def __init__(self, embedding_size: int, r_real: float = 0.9, r_fake: float = 0.2, alpha: float = 20.0,
                 class_weight = [1, 1], K: int = 1, beta: float = 0.995, assignment: str = 'hard', tau: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.embedding_size = embedding_size
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.class_weight = class_weight
        self.softplus = nn.Softplus()
        self.protos = EMAPrototypeManager(embedding_size, K=K, beta=beta, assignment=assignment, tau=tau, eps=eps)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        # Score with current prototypes
        _, s, _ = self.protos.score(x)  # s: (B,)
        if label is None:
            # return scores for evaluation
            return x.new_zeros(()), s.unsqueeze(1) if s.dim() == 1 else s
        # Prepare per-sample margins and signs
        y = label.to(x.dtype)
        m = torch.where(label == 1, torch.tensor(self.r_real, dtype=x.dtype, device=x.device),
                        torch.tensor(self.r_fake, dtype=x.dtype, device=x.device))
        sign = torch.where(label == 1, -torch.ones_like(y), torch.ones_like(y))  # (-1)^y
        loss = self.softplus(self.alpha * (m - s) * sign)
        # class weights
        if x.size(0) > 4:
            cw = torch.where(label == 1, torch.tensor(self.class_weight[1], dtype=x.dtype, device=x.device),
                             torch.tensor(self.class_weight[0], dtype=x.dtype, device=x.device))
            loss = loss * cw
        return loss.mean(), s

    @torch.no_grad()
    def update_prototypes(self, x_bona: torch.Tensor):
        self.protos.update(x_bona)

    def proto_pull(self, x_bona: torch.Tensor, use_soft: bool = False):
        return self.protos.proto_pull(x_bona, use_soft=use_soft)