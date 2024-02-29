from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn.parameter import Parameter
import torch
from torch_geometric.nn import GCNConv,GATConv
from dgl.nn import GraphConv
import dgl.function as fn
import dgl.nn as dglnn


class DNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim,hidden_dim)            # fully connected layer: maps last hidden vector to model prediction
        self.activation1 = nn.ReLU()                # coz binary classification
        self.drop=nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim,1)
        self.activation2 = nn.Sigmoid()  
        self.device=device



    def forward(self, x):
        e = self.drop(self.activation1(self.fc1(x)))
        out=self.activation2(self.fc2(e))
        return out







class GATT(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(GATT, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)  # using one attention head for simplicity
        self.activation1 = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        # self.fc = nn.Linear(hidden_dim, 1) # Linear can also be used to map embedding from GATConv to preds
        self.fc=GATConv(hidden_dim,1, heads=1)
        self.activation2 = nn.Sigmoid()
        self.device = device
        
        # Move the entire model to the specified device
        self.to(self.device)

    def forward(self, edge_index, x):
        # x, edge_index = data.x, data.edge_index
      
        h = self.conv1(x, edge_index)
        h = self.activation1(h)
        
        h = self.drop(h)
        
        out = self.fc(h,edge_index)
        
        # Activation function
        return self.activation2(out.squeeze(-1))
    




