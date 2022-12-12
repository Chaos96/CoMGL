import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_candidate_edges(num_nodes, num_edges):
    paper_set, author_set = np.arange(num_nodes[0]), np.arange(num_nodes[1])
    row = np.random.choice(paper_set, size=num_edges).reshape(1, -1)
    col = np.random.choice(author_set, size=num_edges).reshape(1, -1)
    candidate_edges = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0)
    return candidate_edges

def calculate_weight(weight, y_pred, edge_index, loc=1):
    from torch_scatter import scatter_mean
    idx = edge_index[loc, :].max() + 1
    weight[:idx, 0] = scatter_mean(y_pred, edge_index[loc, :])
    return weight

def evaluate_hits(evaluator, pos_pred, neg_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits

    return results

def evaluate_mrr(evaluator, pos_pred, neg_pred):
    neg_pred = neg_pred.view(neg_pred.shape[0], -1)

    results = {}
    mrr = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = mrr

    return results

def evaluate_rocauc(evaluator, pos_pred, neg_pred):
    pos_pred = pos_pred.view(-1)
    neg_pred = neg_pred.view(-1)

    results = {}
    rocauc = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['rocauc'].mean().item()

    results['rocauc'] = rocauc

    return results

def evaluate_recall_my(evaluator, pos_pred, neg_pred):   # topk=None):
    results = {}
    recall_train = cal_recall(pos_train_pred, neg_train_pred, topk=topk)
    recall_valid = cal_recall(pos_val_pred, neg_val_pred, topk=topk)
    recall_test = cal_recall(pos_test_pred, neg_test_pred, topk=topk)
    results['recall@100%'] = (recall_train, recall_valid, recall_test)

    return results