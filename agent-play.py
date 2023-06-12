import torch
import numpy as np
import random
from collections import  deque
from Plot import plot
from game_with_ai import Gra_AI,Kierunki,Punkt,WIELKOSC_KRATKI
from model import DQN,QTrainer
class Agent:
    #Inicjalizacja
    def __init__(self):
        self.model=DQN(11,256,3)
        self.model.load_state_dict(torch.load('./Modele/model_base.pth'))
        #self.model.load_state_dict(torch.load('./Modele/model_test1.pth'))
        #self.model.load_state_dict(torch.load('./Modele/model_test2.pth'))
        #self.model.load_state_dict(torch.load('./Modele/model_test3.pth'))
        #self.model.load_state_dict(torch.load('./Modele/model_test4.pth'))
        #self.model.load_state_dict(torch.load('./Modele/model_test5.pth'))
        # self.model.load_state_dict(torch.load('./Modele/model_test6.pth'))
        # self.model.load_state_dict(torch.load('./Modele/model_test7.pth'))
        # self.model.load_state_dict(torch.load('./Modele/model_test8.pth'))
    #funkcja zwraca obecny stan - 11 wartości 0 lub 1 na podstawie których będzie podejmować decyzję model
    def get_stan(self, gra):
        glowa=gra.wonsz[0]
        # zagrożenie-oznacza, że nastąpi kolizja
        # stan:
        # [zagrożenie po lewej, zagrożenie z przodu, zagrożenie z prawej,
        # kierunek lewo, kierunek prawo, kierunek gora, kierunek dol
        # jedzenie lewo, jedzenie prawo, jedzenie gora, jedzenie dol]
        punkt_lewo=Punkt(glowa.x-WIELKOSC_KRATKI,glowa.y)
        punkt_prawo=Punkt(glowa.x+WIELKOSC_KRATKI,glowa.y)
        punkt_gora=Punkt(glowa.x,glowa.y-WIELKOSC_KRATKI)
        punkt_dol=Punkt(glowa.x,glowa.y+WIELKOSC_KRATKI)

        kier_lewo= gra.kierunek==Kierunki.LEWO
        kier_prawo= gra.kierunek==Kierunki.PRAWO
        kier_gora= gra.kierunek==Kierunki.GORA
        kier_dol= gra.kierunek==Kierunki.DOL

        stan=[#Zagrożenie prosto
            (kier_lewo and gra._kolizja(punkt_lewo))or
            (kier_prawo and gra._kolizja(punkt_prawo))or
            (kier_gora and gra._kolizja(punkt_gora))or
            (kier_dol and gra._kolizja(punkt_dol)),
            #Zagrożenie w prawo
            (kier_gora and gra._kolizja(punkt_prawo)) or
            (kier_dol and gra._kolizja(punkt_lewo)) or
            (kier_prawo and gra._kolizja(punkt_dol)) or
            (kier_lewo and gra._kolizja(punkt_gora)),
            #Zagrożenie w lewo
            (kier_lewo and gra._kolizja(punkt_dol)) or
            (kier_prawo and gra._kolizja(punkt_gora)) or
            (kier_gora and gra._kolizja(punkt_lewo)) or
            (kier_dol and gra._kolizja(punkt_prawo)),
            kier_lewo,kier_prawo,kier_gora,kier_dol,
            gra.jedzenie.x<glowa.x,#jedzenie po lewej
            gra.jedzenie.x>glowa.x,#jedzenie po prawej
            gra.jedzenie.y<glowa.y,#jedzenie wyżej
            gra.jedzenie.y>glowa.y#jedzenie niżej
        ]
        #zwracamy stan
        return np.array(stan,dtype=int)
    #Zwracanie akcji do wykonania na podstawie modelu
    #[1,0,0] - ruch prosto
    #[0,1,0] -ruch w prawo
    #[0,0,1] - ruch w lewo
    def get_akcja(self,stan):
        koncowy_ruch=[0,0,0]
        #Przewidujemy następpny krok na podstawie sieci neuronowej
        stan0=torch.tensor(stan,dtype=torch.float).cuda()
        pred=self.model(stan0).cuda()
        #Ustawiamy odpowiedni bit rozwiązania, wybrany przez model
        ruch=torch.argmax(pred).item()
        koncowy_ruch[ruch]=1
        return koncowy_ruch
#Główna funkcja, odpowiada za trenowanie modelu
def graj():
    agent = Agent()
    gra = Gra_AI()
    #nieskończona pętla, która odpowiada za trenowanie
    while True:
        #Zapisuejmy obecny stan
        stan_stary=agent.get_stan(gra)
        #Bierzemy nastepną akcje/ruch
        ruch=agent.get_akcja(stan_stary)
        #Wykonujemy ruch w środowisku
        nagroda,is_Done,wynik=gra.krok(ruch)
        #Zapisujemy nowy stan
        stan_nowy=agent.get_stan(gra)
        #trenujemy pamięć krótkotrwałą
        if is_Done:
            print('Wynik:',wynik)
            gra.reset()

if(__name__=="__main__"):
    graj()