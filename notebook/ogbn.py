import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import pickle

def download():
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
    
    with open(dataset_name + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    download()
