import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

#Klasa sieci neuronowej złożona z warstwy wejściowej, jednej ukrytej i wejściowej
class DQN(nn.Module):
    #inicjalizaja, dane wejściowe do warstw są podawane liniowo
    def __init__(self,in_size,hid_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hid_size).cuda()
        self.layer2=nn.Linear(hid_size,out_size).cuda()

    #Funkcja bierze wejście i przeprowadza je przez sieć, funkcja aktywacji relu
    def forward(self,x):
        x=fun.relu(self.layer1(x))
        x=self.layer2(x)
        return x
    #zapisywanie modelu do pliku
    def save(self,file_name='model.pth'):
        torch.save(self.state_dict(),file_name)

#klasa trenera sieci według Q-learning i równania Bellmana
#wykorzystanie cuda() pozwala na szybsze uczenie z użyciem karty graficznej
class QTrainer:
    #inicjalizacja parametrami
    def __init__(self,model,lr,gamma):
        self.model=model
        self.lr=lr
        self.gamma=gamma
        #optimer odpowiada za aktualizaję wag i bias
        self.optimer=optim.Adam(model.parameters(),lr=self.lr)
        #Mean Squared Error
        self.criterion=nn.MSELoss()
        for i in self.model.parameters():
            print(i.is_cuda)
    #krok w trenowaniu
    def train_step(self, stan, akcja, nagroda, nast_stan, is_Done):
        #macierze na parametry
        stan=torch.tensor(stan,dtype=torch.float).cuda()
        nast_stan=torch.tensor(nast_stan,dtype=torch.float).cuda()
        nagroda=torch.tensor(nagroda,dtype=torch.float).cuda()
        akcja=torch.tensor(akcja, dtype=torch.long).cuda()
        #jeśli jest tylko jeden parametr to zmieniamy typ danych(jedna funkcja do trenowania pamięci długiej i krótkiej)
        if(len(stan.shape)==1):
            stan=torch.unsqueeze(stan,0).cuda()
            nast_stan=torch.unsqueeze(nast_stan,0).cuda()
            nagroda=torch.unsqueeze(nagroda,0).cuda()
            akcja=torch.unsqueeze(akcja,0).cuda()
            is_Done=(is_Done,)
        #przewidujemy wartość Q na podstawie modelu
        #ustawimy wartości następnych stanów i maksymlanych możliwych do osiągnięcia wartości
        pred=self.model(stan).cuda()
        cel=pred.clone().cuda()
        for i in range(len(is_Done)):
            nowe_Q=nagroda[i]
            if not is_Done[i]:
                nowe_Q=nagroda[i]+self.gamma*torch.max(self.model(nast_stan[i])).cuda()
            cel[i][torch.argmax(akcja).item()]=nowe_Q

        self.optimer.zero_grad()
        #Wsteczna propagacja błędu
        strata=self.criterion(cel,pred)
        strata.backward()
        self.optimer.step()