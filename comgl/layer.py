import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import calculate_weight

class DotPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def reset_parameter(self):
         return
    
    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x

class MLPPredictor(nn.Module):
    def __init__(self, mlp_num_layers, input_channels, mlp_hidden_channels, out_channels, dropout) -> None:
        super().__init__()
        self.lins_u, self.lins_v = nn.ModuleList(), nn.ModuleList()
        self.lins_u.append(nn.Linear(input_channels, mlp_hidden_channels))
        self.lins_v.append(nn.Linear(input_channels, mlp_hidden_channels))
        for _ in range(1, mlp_num_layers-1):
            self.lins_u.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels))
            self.lins_v.append(nn.Linear(mlp_hidden_channels, mlp_hidden_channels))
        self.lins = nn.Linear(mlp_hidden_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, u_emb, v_emb, edge_index):
        u, v = u_emb[edge_index[0]], v_emb[edge_index[1]]
        for i in range(len(self.lins_u)):
            u, v = self.lins_u[i](u).relu(), self.lins_v[i](v).relu()
            u, v = self.dropout(u), self.dropout(v)
        z = u + v
        z = self.lins(z)
        return z
    
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, model, data, view_dict):
        u_type = data.u_type
        z = model.encode_aux(data.x_dict, data.edge_index_dict)
        x = torch.dstack([z[0][u_type], z[1][u_type]]).transpose(2, 1)   # batch, num_aux_view, dim_in
        # n: auxiliary view num
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v).sum(axis=1)
        return z, att


class Uncertrainty_estimate(nn.Module):
    def __init__(self):
        super(Uncertrainty_estimate, self).__init__()
    
    def forward(self, model, data, view_dict):
        u_type = data.u_type
        with torch.no_grad():
            z = model.encode_aux(data.x_dict, data.edge_index_dict, len(view_dict) - 1)

        data.x_dict_ = data.x_dict.copy()
        for idx, view in enumerate(view_dict[1:]): 
            if u_type == 'paper':
                edge_type = view[1]
            else:
                edge_type = view[0]
            v_type = edge_type[2]
            num_nodes = [data.x_dict[u_type].size(0), data.x_dict[v_type].size(0)]

            edge_label_index_aux, edge_label_aux = data[edge_type].edge_label_index, data[edge_type].edge_label
            with torch.no_grad():
                logits = model.predictor[idx+1](z[idx][u_type], z[idx][v_type], edge_label_index_aux.to(model.device)).sigmoid()

            y_pred = F.gumbel_softmax(logits, tau=0.01, hard=True)[:, 1]
            acc = (y_pred.cpu() == edge_label_aux).type(torch.float)  # 边的预测准确率
            weight = torch.ones(num_nodes[1], 1, dtype=torch.float) 
            weight = calculate_weight(weight, acc, edge_label_index_aux.cpu(), loc=1)
            
            data.x_dict_[v_type] *= weight.repeat(1, data.x_dict_[v_type].size(-1)).to(model.device)   # 与不确定性系数相乘

        z = model.encode_aux(data.x_dict, data.edge_index_dict, len(view_dict) - 1)
        paper_emb = z[0][u_type] + z[1][u_type]
        return z, paper_emb

class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, model, data, view_dict):
        u_type = data.u_type
        paper_fea = data.x_dict[u_type]
        z = model.encode_aux(data.x_dict, data.edge_index_dict)
        paper_emb = z[0][u_type] + z[1][u_type] + paper_fea
        return z, paper_emb
