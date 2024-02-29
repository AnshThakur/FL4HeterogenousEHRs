import torch
from torch.utils.data import TensorDataset, DataLoader
import _pickle as cPickle
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(100)


torch.manual_seed(100)
np.random.seed(100)





def get_loaders(cols,path='./final_data/OUH.csv',id=0,batch=64,sampler=True):
    train=pd.read_csv(path)
    label=train['Covid-19 Positive'].fillna(0)
    train=train[cols]
    print(train.shape)

    T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
    T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)

    T=T.fillna(T.median())
    

    # print(T.median())
    print('---------------------------------------')
    Val_T=Val_T.fillna(T.median())
    Test_T=Test_T.fillna(T.median())


    train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
    val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
    val_loader = DataLoader(val_dataset, batch_size=batch)  

    test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
    test_loader = DataLoader(test_dataset, batch_size=batch)  

    Loaders=[]
    Loaders.append(train_loader)
    Loaders.append(val_loader)
    Loaders.append(test_loader)
    return Loaders


def get_loaders_same(cols,path='./final_data/OUH.csv',batch=64,sampler=True):
    train=pd.read_csv(path)
    label=train['Covid-19 Positive'].fillna(0)
    train=train[cols]

    ind=np.where(label==1)[0]
    ind=ind[0:500]
    s1=train.iloc[ind]


    ind=np.where(label==0)[0]
    ind=ind[0:500]
    s0=train.iloc[ind]

    s=pd.concat([s1,s0])
    print(s.shape)
    sim=train.dot(s.transpose())

    print(sim.shape)
    train=sim

    T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
    T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)

    T=T.fillna(T.median())
    

    # print(T.median())
    print('---------------------------------------')
    Val_T=Val_T.fillna(T.median())
    Test_T=Test_T.fillna(T.median())


    train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
    val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
    val_loader = DataLoader(val_dataset, batch_size=batch)  

    test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
    test_loader = DataLoader(test_dataset, batch_size=batch)  

    Loaders=[]
    Loaders.append(train_loader)
    Loaders.append(val_loader)
    Loaders.append(test_loader)
    return Loaders




# def get_loaders_unstructured(path='./final_data/OUH.csv',id=0,batch=64,sampler=True):
#     train=pd.read_csv(path)
#     columns1 = [col for col in train.columns if 'Vital_Sign' in col]
#     columns2 = [col for col in train.columns if 'Blood_Test' in col]
#     columns3 = [col for col in train.columns if 'Blood_Gas' in col]
    
#     if id==0:
#        cols=columns1+columns3
#     elif id==1:
#        cols=columns1+columns2+columns3
#     elif id==2:
#        cols=columns1+columns2
#     else:
#        cols=columns2+columns3            


#    #  cols=columns1+columns2

#     label=train['Covid-19 Positive'].fillna(0)
#     train=train[cols]
#     print(train.shape)
#     # train=pd.read_csv('./final_data/BH-CURIAL Processed Data.csv')
#     T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
#     T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)

#     T=T.fillna(T.median())
    

#     # print(T.median())
#     print('---------------------------------------')
#     Val_T=Val_T.fillna(T.median())
#     Test_T=Test_T.fillna(T.median())


#     train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
#     train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
#     val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
#     val_loader = DataLoader(val_dataset, batch_size=batch)  

#     test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
#     test_loader = DataLoader(test_dataset, batch_size=batch)  

#     Loaders=[]
#     Loaders.append(train_loader)
#     Loaders.append(val_loader)
#     Loaders.append(test_loader)
#     return Loaders


def get_loaders_scaled(cols,files,path='./final_data/',batch=64):
    Loaders=[]
    
    l=0
    scaling=['Blood_Test HAEMOGLOBIN','Blood_Test WHITE CELLS','Blood_Test BILIRUBIN','Blood_Test CREATININE','Blood_Test UREA','Blood_Test ALK.PHOSPHATASE','Blood_Test MONOCYTES']
    metric=[10,100,10,1000,100,1000,10009]
    
    for f in files:
      
        train=pd.read_csv(path+f+'.csv')
        label=train['Covid-19 Positive'].fillna(0)
        train=train[cols]      
        
           
        
        
        T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
        T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)
        T=T.fillna(T.median())
        
        Val_T=Val_T.fillna(T.median())
        Test_T=Test_T.fillna(T.median())
        
        if ((f == 'OUH') or (f=='BH')):
           print(f)
           for k in range(len(scaling)):
               s=scaling[k]
               T[s]=T[s]/metric[k] 
               Val_T[s]=Val_T[s]/metric[k]
               Test_T[s]=Test_T[s]/metric[k] 
               print(T[s].head()) 
        
        train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
        val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
        val_loader = DataLoader(val_dataset, batch_size=batch)  

        test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
        test_loader = DataLoader(test_dataset, batch_size=batch)  
      

        Loaders.append([train_loader,val_loader,test_loader]) 

    return Loaders   



def get_loaders_jubmled(C,files,path='./final_data/',batch=64):
    Loaders=[]
    
    l=0
    for f in files:
        temp=C[l]
        l=l+1
        train=pd.read_csv(path+f+'.csv')
        label=train['Covid-19 Positive'].fillna(0)
        train=train[temp]      
        print(train.head)
        print('------------') 
        T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
        T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)
        T=T.fillna(T.median())

        Val_T=Val_T.fillna(T.median())
        Test_T=Test_T.fillna(T.median())

        train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
        val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
        val_loader = DataLoader(val_dataset, batch_size=batch)  

        test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
        test_loader = DataLoader(test_dataset, batch_size=batch)  
      

        Loaders.append([train_loader,val_loader,test_loader]) 

    return Loaders    



def get_loaders_structured(cols1,files,path='./final_data/',batch=64,unstruct=0):
    Loaders=[]
    temp=cols1
    for f in files:
        cols=temp
        train=pd.read_csv(path+f+'.csv')
        label=train['Covid-19 Positive'].fillna(0)

        if unstruct==1:
           if (f=='OUH' or f=='UHB'):
              columns1 = [col for col in train.columns if 'Vital_Sign' in col]
              cols=cols+columns1
           
        train=train[cols]      

        print(train.shape)
        print('------------') 
        T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
        T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)
        T=T.fillna(T.median())

        Val_T=Val_T.fillna(T.median())
        Test_T=Test_T.fillna(T.median())

        train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
        val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
        val_loader = DataLoader(val_dataset, batch_size=batch)  

        test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
        test_loader = DataLoader(test_dataset, batch_size=batch)  
      

        Loaders.append([train_loader,val_loader,test_loader]) 

    return Loaders    



def get_loaders_consistant(cols,files,path='./final_data/',batch=64):
    Loaders=[]
    
    l=0

    
    for f in files:
      
        train=pd.read_csv(path+f+'.csv')
        label=train['Covid-19 Positive'].fillna(0)
        train=train[cols]      
        
           
        
        
        T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
        T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)
        T=T.fillna(T.median())
        
        Val_T=Val_T.fillna(T.median())
        Test_T=Test_T.fillna(T.median())
        
        
        train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
        val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
        val_loader = DataLoader(val_dataset, batch_size=batch)  

        test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
        test_loader = DataLoader(test_dataset, batch_size=batch)  
      

        Loaders.append([train_loader,val_loader,test_loader]) 

    return Loaders