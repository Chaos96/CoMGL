import os
import time
import subprocess

import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP, OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import (AddMetaPaths, RandomLinkSplit,
                                        RandomNodeSplit)
from torch_geometric.utils import bipartite_subgraph
from ogb.nodeproppred import Evaluator
import torch


def assign_free_gpus(threshold_vram_usage=10000, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = np.array([
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ])  # Remove garbage
        rank = np.argsort(gpu_info)
        # Keep gpus under threshold only
        free_gpus = [i for i in rank if gpu_info[i] < threshold_vram_usage]

        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join([str(i) for i in free_gpus])
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
    return gpus_to_use

def load_dataset(args):
    if args.dataset == 'mag':
        dataset = OGB_MAG(root=args.data_path, preprocess='metapath2vec')
        data = dataset[0]
        del data[('paper', 'cites', 'paper')]

        args.node_class_num = data['paper'].y.unique().max() + 1  # 349

        data['author'].nid = torch.arange(data['author'].x.size(0)).reshape(-1, 1)

        view_1 = [('paper', 'writes_by', 'author'), ('author', 'writes', 'paper')]  # ('paper', 'cites', 'paper')
        view_2_ = [('institution', 'affiliated_with_by', 'author'), ('author', 'affiliated_with', 'institution')]  # Initial graph's edge type
        view_3 = [('field_of_study', 'has_topic_by', 'paper'), ('paper', 'has_topic', 'field_of_study')]
        view_dict = [view_1, view_2_, view_3]
        for view in view_dict:
            data[view[0]].edge_index = torch.flipud(data.edge_index_dict[view[1]]) # directed -> undirected
        
        if os.path.exists('./data/mag_processed.pt'):
            print('load data successfully !')
            data = torch.load('../data/mag_processed.pt')
        else:
            metapaths = [[view_1[0], view_2_[1]], [view_2_[0], view_1[1]]]
            data = AddMetaPaths(metapaths)(data)
            del data[view_2_[0]]
            del data[view_2_[1]]
            torch.save(data, './data/mag_processed.pt')

        view_2 = [('institution', 'metapath_1', 'paper'), ('paper', 'metapath_0', 'institution')]  # Generate new edge type around paper
        args.view_dict = view_dict = [view_1, view_2, view_3]
        args.u_type = data.u_type = 'paper'
        args.v_type = data.v_type = 'author'   # main view 的两种节点类型
        v_emb = data['author'].x.clone()    # record authors'embedding during training for inductive settings

    elif args.dataset == 'dblp':
        dataset = DBLP(root=args.data_path)
        data = dataset[0]

        args.u_type = data.u_type = 'paper'
        args.v_type = data.v_type = 'author'   # main view 的两种节点类型
        args.node_class_num = data['author'].y.unique().max() + 1  # 4

        data["conference"].x = torch.ones(data["conference"].num_nodes, 1)

        view_1 = [('paper', 'to', 'author'), ('author', 'to', 'paper')]
        view_2 = [('paper', 'to', 'conference'), ('conference', 'to', 'paper')]
        view_3 = [('paper', 'to', 'term'), ('term', 'to', 'paper')]
        args.view_dict = view_dict = [view_1, view_2, view_3]
    
    else:
        data = HeteroData()

        path_1 = 'data/openid/openid2projectid.csv'
        path_2 = 'data/openid/openid2q36.csv'
        path_3 = 'data/openid/openid2uin.csv'

        data_1 = pd.read_csv(path_1)
        data_2 = pd.read_csv(path_2)
        data_3 = pd.read_csv(path_3)
        openid_index, openid_map = pd.factorize(data_1['open_id'])
        num_openid_1 = len(openid_map)
        project_index, project_map = pd.factorize(data_1['proj_id'])
        num_project = len(project_map)
        org_index, org_map = pd.factorize(data_1['org_id'])
        num_institution = len(org_map)
        openid2project_edge_index = np.vstack([openid_index, project_index])
        openid2project_edge_index = torch.tensor(openid2project_edge_index, dtype=torch.long)
        project2org_edge_index = np.vstack([project_index, org_index])
        project2org_edge_index = torch.tensor(project2org_edge_index, dtype=torch.long)

        openid_index, openid_map = pd.factorize(data_2['open_id'])
        num_openid_2 = len(openid_map)
        qimei36_index, qimei36_map = pd.factorize(data_2['qimei36'])
        num_qimei36 = len(qimei36_map)
        openid2qimei36_edge_index = np.vstack([openid_index, qimei36_index])
        openid2qimei36_edge_index = torch.tensor(openid2qimei36_edge_index, dtype=torch.long)

        openid_index, openid_map = pd.factorize(data_3['open_id'])
        num_openid_3 = len(openid_map)
        uin_index, uin_map = pd.factorize(data_3['uin'])
        num_uin = len(uin_map)
        openid2uin_edge_index = np.vstack([openid_index, uin_index])
        openid2uin_edge_index = torch.tensor(openid2uin_edge_index, dtype=torch.long)

        num_openid = max([num_openid_1, num_openid_2, num_openid_3])

        num_openid_features = args.embed_size
        num_project_features = args.embed_size
        num_institution_features = args.embed_size
        num_qimei36_features = args.embed_size
        num_uin_features = args.embed_size
        data['openid'].x = torch.randn(num_openid, num_openid_features)
        data['openid'].y = torch.randint(0, 2, (num_openid, ))
        data['project'].x = torch.randn(num_project, num_project_features)
        data['institution'].x = torch.rand(num_institution, num_institution_features)
        data['qimei36'].x = torch.randn(num_qimei36, num_qimei36_features)
        data['uin'].x = torch.randn(num_uin, num_uin_features)

        # Create an edge type "(author, writes, paper)" and building the
        # graph connectivity:

        data['openid', 'to', 'project'].edge_index = openid2project_edge_index  # [2, num_edges]
        data['project', 'to', 'institution'].edge_index = project2org_edge_index
        data['openid', 'to', 'qimei36'].edge_index = openid2qimei36_edge_index
        data['openid', 'to', 'uin'].edge_index = openid2uin_edge_index 

        # data['openid', 'to', 'project'].edge_index = torch.vstack([torch.randint()])  # [2, num_edges]
        # data['project', 'to', 'institution'].edge_index = project2org_edge_index
        # data['openid', 'to', 'qimei36'].edge_index = openid2qimei36_edge_index
        # data['openid', 'to', 'uin'].edge_index = openid2uin_edge_index 

        # view_1 = [('openid', 'to', 'project'), ('project', 'to', 'openid'), ('project', 'to', 'institution'), ('institution', 'to', 'project')]
        view_1 = [('openid', 'to', 'project'), ('project', 'to', 'openid')]
        view_2 = [('openid', 'to', 'qimei36'), ('qimei36', 'to', 'openid')]
        view_3 = [('openid', 'to', 'uin'), ('uin', 'to', 'openid')]
        args.view_dict = view_dict = [view_1, view_2, view_3]

        for view in view_dict:
            data[view[1]].edge_index = torch.flipud(data.edge_index_dict[view[0]]) # directed -> undirected

        args.u_type = data.u_type = 'openid'
        args.v_type = data.v_type = 'project'   # main view 的两种节点类型
        args.node_class_num = 2   # main view 节点预测类别数

        # 保持原始特征维度与后续auxiliary views的表征维度一致，对预测结果影响待定...
        # with torch.no_grad():
        #     for item in data.metadata()[0]:
        #         if data.x_dict[item].size(1) != args.embed_size:
        #             rand_weight = torch.Tensor(data.x_dict[item].size(1), args.embed_size).uniform_(-0.5, 0.5)
        #             data[item].x = data.x_dict[item] @ rand_weight
        
        data[args.v_type].nid = torch.arange(data[args.v_type].x.size(0)).reshape(-1, 1)
        v_emb = data[args.v_type].x.clone()  # record authors'embedding during training for inductive settings

    # main view's edge_type
    args.edge_type = edge_type = view_1[0]
    args.rev_edge_type = rev_edge_type = view_1[1]   
    args.metadata = data.metadata()

    print(data)
    train_data, val_data, test_data = get_data_split(data, edge_type, rev_edge_type, view_dict)

    train_data = add_delete_edges(train_data, view_dict=view_dict, noise_ratio=args.noise_ratio)

    if args.train_on_subgraph:
        train_loader = NeighborLoader(
            train_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [15] * 2 for key in train_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=args.batch_size,
            input_nodes=(args.u_type, train_data[args.u_type].train_mask),
        )

        mask = torch.ones(val_data[args.u_type].x.size(0), dtype=torch.bool)
        val_loader = NeighborLoader(
            val_data,
            num_neighbors={key: [15] * 2 for key in val_data.edge_types},
            batch_size=args.batch_size,
            input_nodes=(args.u_type, mask),
        )

        mask = torch.ones(test_data[args.u_type].x.size(0), dtype=torch.bool)
        test_loader = NeighborLoader(
            test_data,
            num_neighbors={key: [15] * 2 for key in test_data.edge_types},
            batch_size=args.batch_size,
            input_nodes=(args.u_type, mask),
        )
        return train_loader, val_loader, test_loader, v_emb

    return train_data, val_data, test_data, v_emb

def filter_edges(edge_index, node_mask, loc=0):
    node_set = node_mask.nonzero()
    mask = torch.isin(edge_index[loc], node_set)
    return edge_index[:, mask]

def negative_sampling(edge_index, u_set, v_set, ratio=1):
    size = int(edge_index.size(-1) * ratio)
    row = np.random.choice(u_set, size=size).reshape(1, -1)
    col = np.random.choice(v_set, size=size).reshape(1, -1)
    neg_edge_index = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0).to(edge_index)
    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(edge_index.size(-1)), torch.zeros(neg_edge_index.size(-1))], dim=0)
    return edge_label_index, edge_label

def add_delete_edges(data, view_dict, noise_ratio):
    for view in view_dict:
        for edge_type in view:
            edge_index = data.edge_index_dict[edge_type]
            num_edges = edge_index.size(-1)
            if noise_ratio <= 0:   # delete edges
                noise_ratio = np.abs(noise_ratio)
                perm = torch.randperm(num_edges)[:int(num_edges * noise_ratio)]
                data.edge_index_dict[edge_type] = edge_index[:, perm]
            
            else:
                loc = 0 if edge_type[0]==data.u_type else 1
                paper_set = torch.unique(edge_index[loc]).numpy()
                v_type = edge_type[2] if loc==0 else edge_type[0]
                v_set = np.arange(data.x_dict[v_type].size(0))
                edge_index, _ = negative_sampling(edge_index, paper_set, v_set, ratio=noise_ratio)
                data.edge_index_dict[edge_type] = edge_index

    return data

def get_subgraph(data, mask, view_dict):
    u_type = data.u_type  
    
    for view in view_dict:
        for edge_type in view:
            if edge_type[0] == u_type:
                v_set = torch.arange(data.x_dict[edge_type[2]].size(0))  # 只对paper/user 节点进行分割，其他节点全部保留
                subset = (mask.nonzero(), v_set)
            else:
                u_set = torch.arange(data.x_dict[edge_type[0]].size(0))
                subset = (u_set, mask.nonzero())
            edge_index, _ = bipartite_subgraph(subset, data.edge_index_dict[edge_type], relabel_nodes=True)
            data[edge_type].edge_index = edge_index

    data[u_type].x = data.x_dict[u_type][mask]
    return data

def get_data_split(data, edge_type, rev_edge_type, view_dict, num_train=0.8, num_val=0.1, num_test=0.1):
    u_type, v_type = data.u_type, data.v_type
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
 
    num_u = data[u_type].x.size(0)
    num_v = data[v_type].x.size(0)
    
    if not hasattr(data[u_type], 'train_mask'):
        print('Manual division ')
        num_train = int(num_train * num_u)
        num_val = int(num_val * num_u)
        
        perm = torch.randperm(num_u)
        train_mask = torch.zeros(num_u, dtype=torch.bool)
        val_mask = torch.zeros(num_u, dtype=torch.bool)
        test_mask = torch.zeros(num_u, dtype=torch.bool)
        train_mask[perm[:num_train]] = True       
        val_mask[perm[num_train:num_train+num_val]] = True
        test_mask[perm[num_train+num_val:]] = True

        train_data[u_type].train_mask = train_mask
        val_data[u_type].val_mask = val_mask
        test_data[u_type].test_mask = test_mask

    # Eliminate redundant edges to satisfy "inductive" and "strict cold start" scenarios, only for view 1.
    print(f'the num of train data edges: {train_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    train_data[edge_type].edge_index = filter_edges(train_data[edge_type].edge_index, train_data[u_type].train_mask, 0)
    train_data[rev_edge_type].edge_index = filter_edges(train_data[rev_edge_type].edge_index, train_data[u_type].train_mask, 1)
    # train_data = get_subgraph(train_data, train_data['paper'].train_mask, view_dict[1:])
    print(f'{train_data[edge_type].edge_index.size(-1)}')


    val_data = get_subgraph(val_data, val_data[u_type].val_mask, view_dict)

    test_data = get_subgraph(test_data, test_data[u_type].test_mask, view_dict)

    return train_data, val_data, test_data