# -*- Coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import minmax_scale, scale #minmax_scale 归一化

def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def RWR(A, K=10, alpha=0.96):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0] #行长
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def load_networks(path):

    networks, symbols = [], []
    for file in path:
        if file.split('.')[0][-3:] == 'gcn':
            network = np.load(file)['features']
            networks.append(network)
            symbols.append(np.load(file, allow_pickle=True)['symbol'])
        else:
            network = np.load(file)['corr']
            network = minmax_scale(network)
            networks.append(network)
            symbols.append(np.load(file, allow_pickle=True)['symbol'])


    return networks, symbols 


class netsDataset(Dataset):
    def __init__(self, net):

        super(netsDataset, self).__init__()
        self.net = net

    def __len__(self):
        #形状不是正方形，取列
        return self.net.shape[1]

    def __getitem__(self, item):
        x = self.net[item]
        y = self.net[item]
        return x, y, item
