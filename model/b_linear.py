import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CAUTION
The model includes OC-softmax loss functions.
"""
class B_Linear(nn.Module):
    def __init__(self, num_layer=25, hidden_size=1024, output_size=128, agg_size=128, loss=None, loss_bpl=None):
        super().__init__()
        ####
        # pre-processing input from wav2vec 2.0
        ####
        
        self.bridge_module = Bridge_module(num_layer, hidden_size, agg_size)
        self.AHP_ASP = SelfWeightedPooling(agg_size, num_head=1, asp=True)
        self.linear = nn.Linear(agg_size*2, output_size)
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-12)
        self.loss = loss
        self.loss_bpl = loss_bpl

    def forward(self, x, x_short=None, label=None, bona_size=None, return_embed: bool = False):
        # Full linear
        x = self.bridge_module(x)
        x = self.AHP_ASP(x)
        x = self.linear(x)
        x = self.layer_norm(x)
        
        if x_short != None:
            x_short = self.bridge_module(x_short)
            x_short = self.AHP_ASP(x_short)
            x_short = self.linear(x_short)
            x_short = self.layer_norm(x_short)

        if self.loss:
            loss, scores = self.loss(x, label)
            if x_short != None and bona_size != None and self.loss_bpl != None:
                loss_cos = self.loss_bpl(x[:bona_size, :], x_short[:bona_size, :])

                return (loss + loss_cos, scores, x) if return_embed else (loss + loss_cos, scores)
            
            return (loss, scores, x) if return_embed else (loss, scores)
        
        return (x, None, x) if return_embed else x
    
    
class Bridge_module(nn.Module):
    def __init__(self, num_layer, hidden_size, agg_size, asp=True):
        super().__init__()
        self.num_layer = num_layer

        # squeeze module
        self.squeeze_module_linears = nn.ModuleList([
            PositionwiseFeedForward(idim=hidden_size, hidden_units=agg_size, dropout_rate=0.35, change_dim=True) 
            for _ in range(num_layer)
        ])
        self.squeeze_module_norms = nn.ModuleList([
            nn.LayerNorm(agg_size, eps=1e-12) for _ in range(num_layer)
        ])

        # attentive hidden pooling (AHP)
        self.AHP_ASP = nn.ModuleList([
            SelfWeightedPooling(agg_size, num_head=1, asp=True) for _ in range(num_layer)
        ])
        if asp:
            self.AHP_W_att = nn.Parameter(
                torch.Tensor(1, agg_size * 2), requires_grad=True)
        else:
            self.AHP_W_att = nn.Parameter(
                torch.Tensor(1, agg_size), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.AHP_W_att)
    
    def forward(self, x):
        B, L, T, H = x.size()   # B, L, T, H
        output = None
        att = None
        for i in range(self.num_layer):
            x_i = x[:, i, :, :]
            x_i_ouput = self.squeeze_module_linears[i](x_i)
            x_i_ouput = self.squeeze_module_norms[i](x_i_ouput)
            x_i = self.AHP_ASP[i](x_i_ouput)  # (B, H)
            if output == None:
                output = x_i_ouput.unsqueeze(1) # (B, 1, T, H)
            else:
                output = torch.cat((output, x_i_ouput.unsqueeze(1)), dim=1)
            
            if att == None:
                att = x_i.unsqueeze(1) # (B, 1, H)
            else:
                att = torch.cat((att, x_i.unsqueeze(1)), dim=1)

        # attn_distribution
        weights = torch.bmm(att, 
                            self.AHP_W_att.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(B, 1, 1))
        attn_distribution = nn.functional.softmax(torch.tanh(weights), dim=1).unsqueeze(-1)
        attn_distribution = attn_distribution.repeat(1, 1, 1, 1) 

        weighted_x = output * attn_distribution
        weighted_x = weighted_x.sum(dim=1)

        return weighted_x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, change_dim=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        if change_dim:
            self.w_2 = nn.Linear(hidden_units, hidden_units)
        else:
            self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class SelfWeightedPooling(nn.Module):
    def __init__(self, feature_dim, num_head=1, asp=False):
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.asp = asp
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.mm_weights)
        
    def forward(self, inputs, get_w=False, tanh=True):
        batch_size = inputs.size(0)
        feat_dim = inputs.size(2)
        
        weights = torch.bmm(inputs, 
                            self.mm_weights.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        if tanh:
            attentions = nn.functional.softmax(torch.tanh(weights), dim=1)    
        else: 
            attentions = nn.functional.softmax(weights, dim=1)  
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            weighted = torch.bmm(
                inputs.view(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling       
        if self.asp:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)
        else:
            # only output the mean vector
            representations = weighted.sum(1)

        # done
        if get_w:
            return representations, attentions.squeeze(-1)
        return representations
