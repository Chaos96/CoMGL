import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pos_neg_edges(edge_index, node_set):
    paper_set, author_set = node_set[0], node_set[1]
    row = np.random.choice(paper_set, size=edge_index.size(-1)).reshape(1, -1)
    col = np.random.choice(author_set, size=edge_index.size(-1)).reshape(1, -1)
    neg_edge_index = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0).to(edge_index)
    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(edge_index.size(-1)), torch.zeros(neg_edge_index.size(-1))], dim=0)
    return edge_label_index, edge_label

def batch_get_neg_edges(edge_index, num_nodes):
    paper_set, author_set = np.arange(num_nodes[0]), np.arange(num_nodes[1])
    row = np.random.choice(paper_set, size=edge_index.size(-1)).reshape(1, -1)
    col = np.random.choice(author_set, size=edge_index.size(-1)).reshape(1, -1)
    neg_edge_index = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0).to(edge_index)
    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(edge_index.size(-1)), torch.zeros(neg_edge_index.size(-1))], dim=0)
    return edge_label_index, edge_label

def calculate_weight(weight, y_pred, edge_index, loc=1):
    from torch_scatter import scatter_mean
    idx = edge_index[loc, :].max() + 1
    weight[:idx, 0] = scatter_mean(y_pred, edge_index[loc, :])
    return weight
