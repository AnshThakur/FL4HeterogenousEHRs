import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import dgl
import torch.nn as nn


import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_self_loops
from sklearn.neighbors import NearestNeighbors
import numpy as np





def Makegraph(matrix):
    num_patients, num_features = matrix.shape
    # print( num_patients, num_features)
    src_nodes, dst_nodes = [], []
    for i in range(0,num_patients):
        for j in range(0, num_features):
            src_nodes.append(i)
            dst_nodes.append(j)



    g = dgl.heterograph({
        ('patient', 'feature', 'patient'): (src_nodes, dst_nodes)
    })

    edge_features = torch.tensor([matrix[i, j] for i, j in zip(src_nodes, dst_nodes)], dtype=torch.float)
    g.edges[('patient', 'feature', 'patient')].data['weight'] = edge_features
    # print(g)
    # g = dgl.to_homogeneous(g)
    # g = dgl.add_self_loop(g)
    return g



def MakegraphH(matrix):
  g = dgl.DGLGraph()
  g.add_nodes(matrix.shape[0])
  # g.ndata['feat'] = features


  # Find K nearest neighbors for each row
  knn = NearestNeighbors(n_neighbors=1)
  knn.fit(matrix.numpy())
  _, indices = knn.kneighbors(matrix.numpy())

  # Add edges to the graph based on KNN
  for i in range(matrix.shape[0]):
      g.add_edges(i, indices[i])
  g = dgl.add_self_loop(g)

  return g


def MakegraphHT(matrix):
    # Find K nearest neighbors
    edge_index = knn_graph(matrix, k=1, loop=True)
    graph = Data(x=matrix, edge_index=edge_index)
    return graph





