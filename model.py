import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

class DQN(nn.Module):
    def __init__(self,in_size,hid_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hid_size).cuda()
        self.layer2=nn.Linear(hid_size,out_size).cuda()

    def forward(self,x):
        x=fun.relu(self.layer1(x))
        x=self.layer2(x)
        return x

    def save(self,file_name='model.pth'):
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.model=model
        self.lr=lr
        self.gamma=gamma
        self.optimer=optim.Adam(model.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss()
        for i in self.model.parameters():
            print(i.is_cuda)

    def train_step(self, stan, akcja, nagroda, nast_stan, is_Done):
        stan=torch.tensor(stan,dtype=torch.float).cuda()
        nast_stan=torch.tensor(nast_stan,dtype=torch.float).cuda()
        nagroda=torch.tensor(nagroda,dtype=torch.float).cuda()
        akcja=torch.tensor(akcja, dtype=torch.long).cuda()

        if(len(stan.shape)==1):
            stan=torch.unsqueeze(stan,0).cuda()
            nast_stan=torch.unsqueeze(nast_stan,0).cuda()
            nagroda=torch.unsqueeze(nagroda,0).cuda()
            akcja=torch.unsqueeze(akcja,0).cuda()
            is_Done=(is_Done,)

        pred=self.model(stan).cuda()
        cel=pred.clone().cuda()
        for i in range(len(is_Done)):
            nowe_Q=nagroda[i]
            if not is_Done[i]:
                nowe_Q=nagroda[i]+self.gamma*torch.max(self.model(nast_stan[i])).cuda()
            cel[i][torch.argmax(akcja).item()]=nowe_Q

        self.optimer.zero_grad()
        strata=self.criterion(cel,pred)
        strata.backward()
        self.optimer.step()