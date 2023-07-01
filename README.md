# CoMGL: Collaborative Multi-view Graph Learning for Strict Cold-start Tasks



## Data 

### Data Preparation for [OGB-MAG](http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip) dataset.

- Main view: ("paper", "to", "author")
- Auxiliary views: ("paper", "to", "Conference"), ("paper", "to", "term")

### Data Process

- Divide the train/val/test dataset, remove the redundant edges of each part of the dataset, but keep all the edges of the auxiliary view to satisfy the inductive setting;
- Remove all edges of the main view in the val/test dataset to meet the "Strict Cold Start" scenario.
```bash
python preprocess.py --data_path data/
```
## Experiments
