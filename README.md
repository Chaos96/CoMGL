# CoMGL: Collaborative Multi-view Graph Learning for Strict Cold-start Tasks

- Main view: ("paper", "to", "author")
- Auxiliary views: ("paper", "to", "Conference"), ("paper", "to", "term")

## Data 

### Data Preparation

数据集按照`OGB Dataset`格式，放置在`./data/`目录下.

Template class for the `dataset`
```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Create node types "paper" and "author" holding a feature matrix:
data['paper'].x = torch.randn(num_papers, num_paper_features)
data['author'].x = torch.randn(num_authors, num_authors_features)

# Create an edge type "(author, writes, paper)" and building the
# graph connectivity:
data[("paper", "to", "author")].edge_index = ...  # [2, num_edges]

```

### Data Process

- 满足Inductive 的场景设置，划分train/val/test 数据集，去除各部分数据集Main view中多余的边，但Auxiliary view 的边全部保留；
- 去除val/test 数据集中Main view的所有边，满足"Strict Cold Start" 场景；

![](https://cdn.jsdelivr.net/gh/Zaiyun27/Imgur/img/202211132039830.png)

## Model

- Heterogeneous Grpah
    - Encoder
        - Graph SAGE: 每个view 单独建立一张异构图及对应的Encoder
    - Aggregator
        - Uncertainty Estimate / Attention: 聚合Auxiliary view 的paper 表征，用于Main view 的预测任务
    - Predictor
        - Main view Predictor: 链路预测 / 节点分类
        - Auxiliary view Predictor: 辅助任务的链路预测


## Experiments

### 1. Node Prediction

1. Auxiliary views encode
2. Auxiliary views’ paper embedding aggregate
3. Main view 边生成
4. Main view encode
5. Main view’s node prediction task

```bash
python main.py --exp_mode node_prediction --data_path ./data
```

### 2. Link Prediction

1. Auxiliary views encode
2. Auxiliary views’ paper embedding aggregate
3. Main view’s link prediction task

```bash
python main.py --exp_mode link_prediction --data_path ./data
```

