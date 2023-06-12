import torch
import numpy as np
import random
from collections import  deque
from Plot import plot
from game_with_ai import Gra_AI,Kierunki,Punkt,WIELKOSC_KRATKI
from model import DQN,QTrainer
#Definujemy maksymalną pamięć do przechowywania poprzednich kroków uczenia
MAX_MEMORY=100_000
#Definujemy maksymalny fragment do wykorzstania w trakcie uczenia pamięci długotrwałej
BATCH_SIZE=1000
#Learning Rate
LR=0.001

#Klasa pośrednicząca pomiędzy modelem(siecią neuronową), a środowiskiem(grą)
class Agent:
    #Inicjalizacja
    def __init__(self):
        self.numer_gry=0#Numer obecnej gry, którą wykonuje AI
        self.epsilon=0#Losowość, odpowiada za szansę z jaką następny krok będzie losowy lub na podstawie sieci, ustawiane później
        self.gamma=0.9#discunt factor w Q-learning, wpływa na to jak ceniony jest przyszły stan
        self.pamiec=deque(maxlen=MAX_MEMORY)#pamięć na stany
        self.model=DQN(11,256,3)#Tworzymy sieć neuronową z trzema wejściami, 256 neuronami w jednej warstwie ukrytej i jednym wyjściu
        self.trener=QTrainer(self.model,lr=LR,gamma=self.gamma)#Ustawiamy trenera modelu
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
    #zapamiętywanie, czyli zapisywanie danych w pamięci
    def zapamietaj(self,stan,akcja,nagroda,nast_stan,is_Done):
        self.pamiec.append((stan,akcja,nagroda,nast_stan,is_Done))
    #Trenowanie pamięci długotrwałej polega na wybraniu fragmentu pamięci i trenowaniu modelu na podstawie wielu poprzednich kroków
    def trenuj_dlugo_pamiec(self):
        if(len(self.pamiec)>BATCH_SIZE):
            sample=random.sample(self.pamiec,BATCH_SIZE)
        else:
            sample=self.pamiec
        stany,akcje,nagrody,nast_stany,is_Dones=zip(*sample)
        self.trener.train_step(stany,akcje,nagrody,nast_stany,is_Dones)
    #Trenowanie pamięci krótkotrwałej polega na trenowaniu w oparciu o stan obecny, przyszły, wartość nagrody i czy gra się zakończyła
    def trenuj_krotka_pamiec(self,stan,akcja,nagroda,nast_stan,is_Done):
        self.trener.train_step(stan,akcja,nagroda,nast_stan,is_Done)
    #Zwracanie akcji do wykonania na podstawie modelu
    #[1,0,0] - ruch prosto
    #[0,1,0] -ruch w prawo
    #[0,0,1] - ruch w lewo
    def get_akcja(self,stan):
        #ustawiamy wartość, którą ma przekroczyć wylosowana liczba, aby ruch był generowany przez model
        self.epsilon=80-self.numer_gry
        koncowy_ruch=[0,0,0]
        #Jeśli wylosowana liczba jest większa niż epsilon to następny krok jest losowy
        #Zwiększa to szanse na to, że sieć nauczy się różnych akcji, a nie "utknie"
        if(random.randint(0,200)<self.epsilon):
            ruch=random.randint(0,2)
            koncowy_ruch[ruch]=1
        else:
            #Przewidujemy następpny krok na podstawie sieci neuronowej
            stan0=torch.tensor(stan,dtype=torch.float).cuda()
            pred=self.model(stan0).cuda()
            #Ustawiamy odpowiedni bit rozwiązania, wybrany przez model
            ruch=torch.argmax(pred).item()
            koncowy_ruch[ruch]=1
        return koncowy_ruch
#Główna funkcja, odpowiada za trenowanie modelu
def trenuj():
    #ustawiamy parametry początkowe i inicjalizujemy obiekty
    plot_wyniki = []
    plot_avg_wynik = []
    laczny_wynik = 0
    rekord = 0
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
        agent.trenuj_krotka_pamiec(stan_stary,ruch,nagroda,stan_nowy,is_Done)
        #zapisujemy dane kroku
        agent.zapamietaj(stan_stary,ruch,nagroda,stan_nowy,is_Done)
        #Jeśli dana gra się zakończyła
        if is_Done:
            #resetujemy stan środowiska
            gra.reset()
            #zwiększamy numer gry
            agent.numer_gry+=1
            #trenujemy pamięć długotrwałą
            agent.trenuj_dlugo_pamiec()
            #jeśli uzyskano nowy najwyższy wynik to zapisujemy stan sieci neuronowej
            if(wynik>rekord):
                rekord=wynik
                agent.model.save()
            print('Gra:',agent.numer_gry,'Wynik:',wynik,'Rekord:',rekord)
            #zapisujemy dane do wykresu
            plot_wyniki.append(wynik)
            laczny_wynik+=wynik
            avg_wynik=laczny_wynik/agent.numer_gry
            plot_avg_wynik.append(avg_wynik)
            plot(plot_wyniki,plot_avg_wynik)

if(__name__=="__main__"):
    trenuj()