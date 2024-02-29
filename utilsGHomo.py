import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from dataclasses import dataclass, asdict
import json

@dataclass
class Params:
    num_sites: int = None
    num_rounds: int = None
    inner_epochs: int = None
    batch_size: int = None
    outer_lr: float = None
    weight_decay: float = None
    inner_lr: float = None

PARAMS_FILE = "params.json"

def save_params(run_path, params):
    with open(f'{run_path}/{PARAMS_FILE}', "w") as f:
        json.dump(asdict(params), f, indent=2)

def get_device():
    if torch.cuda.is_available():
       device=torch.device('cuda:0')
    else:
       device=torch.device('cpu')   
    return device

def combine_grads(G):

    '''Given a list of client gradients, combine them for meta gradient'''
    
    nodes = len(G)
    keys = G[0].keys() # Because the keys should be the same for all models
    
    Meta_grad = deepcopy(G[0])

    for k in keys:
        for i in range(1, nodes):
            Meta_grad[k] += G[i][k]

        Meta_grad[k] = Meta_grad[k]/nodes    

    return Meta_grad   

from MakeGraph import *

# def prediction_binary(model,loader,loss_fn,device):
#     P=[]
#     L=[]
#     model.eval()
#     val_loss=0
#     for i,batch in enumerate(loader):
#       if i<len(loader)-1:
#         data,labels=batch
#         num_patients, num_features = data.shape
#         g=MakegraphH(data).to(device)
#         # print(g)
#         g.ndata['feat'] = data.to(device)
#         data=data.to(torch.float32).to(device)
#         labels=labels.to(torch.float32).to(device)
        
#         # pred=model(data)[:,0]
#         pred = model(g, g.ndata['feat'].float())
#         # print(pred)
#         # pred=pred.squeeze(1)
#         loss=loss_fn(pred,labels)
#         val_loss=val_loss+loss.item()

#         P.append(pred.cpu().detach().numpy())
#         L.append(labels.cpu().detach().numpy())
        
#     val_loss=val_loss/len(loader)
#     P=np.concatenate(P)  
#     L=np.concatenate(L)
#     auc=roc_auc_score(L,P)
#     return val_loss,auc

def prediction_binary(model,loader,loss_fn,device):
    P=[]
    L=[]
    model.eval()
    val_loss=0
    for i,batch in enumerate(loader):
      if i<len(loader)-1:
        data,labels=batch
        num_patients, num_features = data.shape
        g = MakegraphHT(data)
        g = g.to(device)
        # print(g)
        # g.nodes['patient'].data['feat'] = data.to(device)
        # data=data.to(torch.float32).to(device)
        labels=labels.to(torch.float32).to(device)
        
        # pred=model(data)[:,0]
        pred = model(g.edge_index, g.x.float())
        # print(pred)
        # pred=pred.squeeze(1)
        loss=loss_fn(pred,labels)
        val_loss=val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss=val_loss/len(loader)
    P=np.concatenate(P)  
    L=np.concatenate(L)
    auc=roc_auc_score(L,P)
    return val_loss,auc

from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np


# def prediction_binaryT(model, loader, loss_fn, device):
#     P = []
#     L = []
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for data, labels in loader:
#             g = MakegraphHT(data).to(device)
#             features = data.to(torch.float32).to(device)
#             labels = labels.to(torch.float32).to(device)
            
#             pred = model(g.edge_index, features)
#             loss = loss_fn(pred, labels)
#             val_loss += loss.item()

#             P.append(pred.cpu().numpy())
#             L.append(labels.cpu().numpy())
            
#     val_loss /= len(loader)
#     P = np.concatenate(P)  
#     L = np.concatenate(L)
#     auc = roc_auc_score(L, P)
#     return val_loss, auc


def prediction_binaryT(model, loader, loss_fn, device):
    P = []
    L = []
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in loader:
            #************************************************************************
            mean_original = data.mean()
            std_dev_original = data.std()
            normalized_data = (data - mean_original) / std_dev_original

            # Resize the dataset to 64x64 by random sampling along the columns
            indices = torch.randint(0, normalized_data.size(1), (100-data.shape[1],))
            resized_data = normalized_data[:, indices]

            resized_data=torch.cat((data, resized_data),axis=1)

            # # Transform back to original scale (optional)
            # resized_transformed_data = (resized_data * std_dev_original) + mean_original
            #************************************************************************

            g = MakegraphHT(resized_data).to(device)
            features = resized_data.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            
            pred = model(g.edge_index, features)
            loss = loss_fn(pred, labels)
            val_loss += loss.item()

            P.append(pred.cpu().numpy())
            L.append(labels.cpu().numpy())
            
    val_loss /= len(loader)
    P = np.concatenate(P)  
    L = np.concatenate(L)
    auc = roc_auc_score(L, P)
    return val_loss, auc
             
def evaluate_modelsT(client_id, Loaders, net, TL, loss_fn, device, df, B, model_path, ae=False, ae_fl=None):
    ''' Given site i, and model net, evaluate the model peformance on the site's val set'''
    
    tl1 = TL[client_id]
    val_loss, val_auc = prediction_binaryT(net, Loaders[client_id][1], loss_fn, device)
    
    if val_auc > B[client_id]:
       B[client_id] = val_auc
       torch.save(net, f'./trained_models/{model_path}/node{client_id}') 
       if ae_fl:
           torch.save(ae_fl, f'./trained_models/AE_Unstructured/node{client_id}') 

    df = df.append({'Train_Loss': tl1, 'Val_Loss': val_loss, 'Val_AUC': val_auc}, ignore_index=True)
    return df, B[client_id] 

def evaluate_models(client_id, Loaders, net, TL, loss_fn, device, df, B, model_path, ae=False, ae_fl=None):
    ''' Given site i, and model net, evaluate the model peformance on the site's val set'''
    
    tl1 = TL[client_id]
    val_loss, val_auc = prediction_binary(net, Loaders[client_id][1], loss_fn, device)
    
    if val_auc > B[client_id]:
       B[client_id] = val_auc
       torch.save(net, f'./trained_models/{model_path}/node{client_id}') 
       if ae_fl:
           torch.save(ae_fl, f'./trained_models/AE_Unstructured/node{client_id}') 

    df = df.append({'Train_Loss': tl1, 'Val_Loss': val_loss, 'Val_AUC': val_auc}, ignore_index=True)
    return df, B[client_id] 
      