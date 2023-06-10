import torch
import numpy as np
import random
from collections import  deque

import game
from Plot import plot
from game_with_ai import Gra_AI,Kierunki,Punkt,WIELKOSC_KRATKI
from model import DQN,QTrainer
MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
        self.numer_gry=0
        self.epsilon=0
        self.gamma=0.9
        self.pamiec=deque(maxlen=MAX_MEMORY)
        self.model=DQN(11,256,3)
        self.trener=QTrainer(self.model,lr=LR,gamma=self.gamma)

    # zagrożenie-oznacza, że nastąpi kolizja
    # stan:
    # [zagrożenie po lewej, zagrożenie z przodu, zagrożenie z prawej,
    # kierunek lewo, kierunek prawo, kierunek gora, kierunek dol
    # jedzenie lewo, jedzenie prawo, jedzenie gora, jedzenie dol]
    def get_stan(self, gra):
        glowa=gra.wonsz[0]
        punkt_lewo=Punkt(glowa.x-WIELKOSC_KRATKI,glowa.y)
        punkt_prawo=Punkt(glowa.x+WIELKOSC_KRATKI,glowa.y)
        punkt_gora=Punkt(glowa.x,glowa.y-WIELKOSC_KRATKI)
        punkt_dol=Punkt(glowa.x,glowa.y+WIELKOSC_KRATKI)

        kier_lewo= gra.kierunek==Kierunki.LEWO
        kier_prawo= gra.kierunek==Kierunki.PRAWO
        kier_gora= gra.kierunek==Kierunki.GORA
        kier_dol= gra.kierunek==Kierunki.DOL

        stan=[
            (kier_lewo and gra._kolizja(punkt_lewo))or
            (kier_prawo and gra._kolizja(punkt_prawo))or
            (kier_gora and gra._kolizja(punkt_gora))or
            (kier_dol and gra._kolizja(punkt_dol)),
            #prawo
            (kier_gora and gra._kolizja(punkt_prawo)) or
            (kier_dol and gra._kolizja(punkt_lewo)) or
            (kier_prawo and gra._kolizja(punkt_dol)) or
            (kier_lewo and gra._kolizja(punkt_gora)),
            #lewo
            (kier_lewo and gra._kolizja(punkt_dol)) or
            (kier_prawo and gra._kolizja(punkt_gora)) or
            (kier_gora and gra._kolizja(punkt_lewo)) or
            (kier_dol and gra._kolizja(punkt_prawo)),
            kier_lewo,kier_prawo,kier_gora,kier_dol,
            gra.jedzenie.x<glowa.x,
            gra.jedzenie.x>glowa.x,
            gra.jedzenie.y<glowa.y,
            gra.jedzenie.y>glowa.y
        ]
        return np.array(stan,dtype=int)

    def zapamietaj(self,stan,akcja,nagroda,nast_stan,is_Done):
        self.pamiec.append((stan,akcja,nagroda,nast_stan,is_Done))

    def trenuj_dlugo_pamiec(self):
        if(len(self.pamiec)>BATCH_SIZE):
            sample=random.sample(self.pamiec,BATCH_SIZE)
        else:
            sample=self.pamiec
        stany,akcje,nagrody,nast_stany,is_Dones=zip(*sample)
        self.trener.train_step(stany,akcje,nagrody,nast_stany,is_Dones)

    def trenuj_krotka_pamiec(self,stan,akcja,nagroda,nast_stan,is_Done):
        self.trener.train_step(stan,akcja,nagroda,nast_stan,is_Done)

    def get_akcja(self,stan):
        self.epsilon=80-self.numer_gry
        koncowy_ruch=[0,0,0]
        if(random.randint(0,200)<self.epsilon):
            ruch=random.randint(0,2)
            koncowy_ruch[ruch]=1
        else:
            stan0=torch.tensor(stan,dtype=torch.float).cuda()
            pred=self.model(stan0).cuda()
            ruch=torch.argmax(pred).item()
            koncowy_ruch[ruch]=1
        return koncowy_ruch

def trenuj():
    plot_wyniki = []
    plot_avg_wynik = []
    laczny_wynik = 0
    rekord = 0
    agent = Agent()
    gra = Gra_AI()
    while True:
        stan_stary=agent.get_stan(gra)
        koncowy_ruch=agent.get_akcja(stan_stary)

        nagroda,is_Done,wynik=gra.krok(koncowy_ruch)
        stan_nowy=agent.get_stan(gra)
        agent.trenuj_krotka_pamiec(stan_stary,koncowy_ruch,nagroda,stan_nowy,is_Done)
        agent.zapamietaj(stan_stary,koncowy_ruch,nagroda,stan_nowy,is_Done)
        if is_Done:
            gra.reset()
            agent.numer_gry+=1
            agent.trenuj_dlugo_pamiec()
            if(wynik>rekord):
                rekord=wynik
                agent.model.save()
            print('Gra:',agent.numer_gry,'Wynik:',wynik,'Rekord:',rekord)

            plot_wyniki.append(wynik)
            laczny_wynik+=wynik
            avg_wynik=laczny_wynik/agent.numer_gry
            plot_avg_wynik.append(avg_wynik)
            plot(plot_wyniki,plot_avg_wynik)

if(__name__=="__main__"):
    trenuj()