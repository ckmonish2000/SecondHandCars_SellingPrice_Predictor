#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader



ds=pd.read_csv("car data.csv")
ds=ds.dropna()


label=ds.pop("Selling_Price")
label=pd.DataFrame(data=label)
ds=ds.drop(["Car_Name","Seller_Type","Fuel_Type","Transmission"],axis=1)
X=ds
y=label[:-1]
X.loc["Present_Price"]=X["Present_Price"].astype("int64")
X=X.dropna()
y=y.dropna()



X=np.array(X,dtype="float64")
y=np.array(y,dtype="float64")
X=torch.from_numpy(X).float()
y=torch.from_numpy(y).float()





model=nn.Linear(4,1)
loss=nn.MSELoss()
optim=torch.optim.SGD(model.parameters(),lr=0.000000000001)





# pred=model(X)
# ls=loss(pred,y)
# ls.backward()
# ls
# optim.step()
# pred=model(X)
# ls=loss(pred,y)
# ls.backward()
# print(ls)
# optim.step()
# optim.zero_grad()

### God's Grace




ds=TensorDataset(X,y)
dl=DataLoader(ds,batch_size=2,shuffle=True)

for z in range(900):
    for i,j in dl:
        pred=model(X)
        ls=loss(pred,y)
        ls.backward()
        print(ls)
        optim.step()
        optim.zero_grad()







