import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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