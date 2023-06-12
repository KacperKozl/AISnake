import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import numpy as np
#inicjalizujemy pygame
pygame.init()
#ustawiamy czcionkę
font=pygame.font.Font('CALIST.ttf',24)
#Klasa zwierająca wartości liczbowe odpowiadające kierunkom
class Kierunki(Enum):
    PRAWO=1
    LEWO=2
    GORA=3
    DOL=4
#Tworzymy definicję punktu
Punkt=namedtuple('Punkt','x,y')
#Ustawiamy standardowe parametry gry, takie jak wielkość kratki, prędkość gry i kolory obiektów
WIELKOSC_KRATKI=20
PREDKOSC=40
BIALY=(255,255,255)
CZARNY=(0,0,0)
CZERWONY=(225,0,0)
NIEBIESKI=(0,0,255)
ZIELONY=(0,255,100)
#Główna klasa gry
class Gra_AI:
    #definiujemy inicjalizację z prametrami będącymi wielkością okna
    def __init__(self,w=640,h=480):
        #ustawiamy paramtetry
        self.h=h
        self.w=w
        #ustawiamy wyświetlanie
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Wonsz')
        #ustawiamy zegar
        self.zegar=pygame.time.Clock()
        #Ustawiamy pozycję startową
        self.reset()
    #definujemy funkcję resetującą do pozycji startowej
    def reset(self):
        #ustawiamy kierunek początkowy węża na prawo
        self.kierunek=Kierunki.PRAWO
        #ustawiamy punkt odpowiadający głowie
        self.glowa=Punkt(self.w/2,self.h/2)
        #definiujemy węża jako głowe i dwa dodatkowe segmenty
        self.wonsz=[self.glowa,Punkt(self.glowa.x-WIELKOSC_KRATKI,self.glowa.y),Punkt(self.glowa.x-(2*WIELKOSC_KRATKI),self.glowa.y)]
        self.jedzenie=None
        #ustawiamy parametry i generejumey pierwsze jedzenie
        self.wynik=0
        self._generuj__jedzenie()
        self.kroki=0

    #Funkcja generuje jedzenie, wybiera losowe koordynaty, i jeśli miałoby się pojawić w wężu to generuje ponownie
    def _generuj__jedzenie(self):
        x=random.randint(0,(self.w-WIELKOSC_KRATKI)//WIELKOSC_KRATKI)*WIELKOSC_KRATKI
        y = random.randint(0, (self.h - WIELKOSC_KRATKI) // WIELKOSC_KRATKI) * WIELKOSC_KRATKI
        self.jedzenie=Punkt(x,y)
        if(self.jedzenie in self.wonsz):
            self._generuj__jedzenie()

    #Funkcja odpowiadająca za ruch węża
    def _ruszaj(self,akcja):
        #tablica z kierunkami, wymagana do sterowania wężem za pomocą poleceń: prosto, w lewo, w prawo względem obecnego kierunku
        clock_wise = [Kierunki.PRAWO, Kierunki.DOL, Kierunki.LEWO, Kierunki.GORA]
        i=clock_wise.index(self.kierunek)
        #Jeśli akcja odpowiada kodowi ruchu prosto
        if np.array_equal(akcja, [1, 0, 0]):
            nowy_kierunek=clock_wise[i]
        # Jeśli akcja odpowiada kodowi skrętu w prawo
        elif np.array_equal(akcja, [0, 1, 0]):
            next_i=(i+1)%4
            nowy_kierunek=clock_wise[next_i]
        # Jeśli akcja odpowiada kodowi skrętu w lewo
        else:
            next_i = (i - 1) % 4
            nowy_kierunek = clock_wise[next_i]
        #ustawiamy nowy kierunek i przesuwamy głowę węża
        self.kierunek=nowy_kierunek
        y=self.glowa.y
        x=self.glowa.x
        if(self.kierunek ==Kierunki.LEWO):
            x-=WIELKOSC_KRATKI
        elif (self.kierunek == Kierunki.PRAWO):
            x += WIELKOSC_KRATKI
        elif (self.kierunek == Kierunki.GORA):
            y -= WIELKOSC_KRATKI
        elif (self.kierunek == Kierunki.DOL):
            y += WIELKOSC_KRATKI
        self.glowa=Punkt(x,y)
    #Prosta detekcja kolizji podanego punktu(lub gdy nie podano to głowy) z granicą okna lub segmentem węża
    def _kolizja(self,punkt=None):
        if(punkt is None):
            punkt=self.glowa
        if(punkt.x>self.w-WIELKOSC_KRATKI or punkt.x<0 or punkt.y<0 or punkt.y>self.h-WIELKOSC_KRATKI ):
            return True
        if(punkt in self.wonsz[1:]):
            return True
        return False
    #Aktualizacja UI
    def _ui(self):
        self.display.fill(CZARNY)
        #Rysujemy wszystkie segmenty węża, jedzenie i wypisujemy wynik
        for punkt in self.wonsz:
            pygame.draw.rect(self.display,NIEBIESKI,pygame.Rect(punkt.x,punkt.y,WIELKOSC_KRATKI,WIELKOSC_KRATKI))
            pygame.draw.rect(self.display, ZIELONY, pygame.Rect(punkt.x+4, punkt.y+4, 12, 12))
        pygame.draw.rect(self.display,CZERWONY,pygame.Rect(self.jedzenie.x,self.jedzenie.y,WIELKOSC_KRATKI,WIELKOSC_KRATKI))
        text=font.render("wynik: "+str(self.wynik),True,BIALY)
        self.display.blit(text,[0,0])
        pygame.display.flip()
    #Definujemy funkcję, która obsługuje jeden krok/ruch węża
    def krok(self,akcja):
        self.kroki+=1
        for event in pygame.event.get():
            if(event.type==pygame.QUIT):
                pygame.quit()
                quit()

        self._ruszaj(akcja)
        #Dodajemy głowę ze zmienionymi współrzędnymi
        #Jeśli nie jest ona w tym samym miejscu co jedzenie, to ostatni segment zostanie usunięty
        #takie działanie sprawia, że wąż się porusza
        self.wonsz.insert(0,self.glowa)
        game_over=False
        nagroda=0
        # W przypadku kolizji lub gdy liczba kroków przekroczy 100*długość węża(sieć sprawia, że wąż kręci się w kółko)
        # kończymy grę i ustawiamy nagrodę dla uczenia maszynowego na -10
        if(self._kolizja() or self.kroki>100*len(self.wonsz)):
            game_over=True
            nagroda=-10
            return nagroda,game_over,self.wynik
        #Jeśli zjadł jedzenie to ustawiamy nagrodę na 10 i dodajemy jeden do wyniku
        if(self.glowa==self.jedzenie):
            nagroda=10
            self.wynik+=1
            self._generuj__jedzenie()
        else:
            #W przeciwnym wypadku usuwamy nadmiarowy segment
            self.wonsz.pop()
        #Aktualizujemy UI i zwracamy stan gry - nagroda, czy gra się zakończyła i obecny wynik
        self._ui()
        self.zegar.tick(PREDKOSC)
        return nagroda,game_over,self.wynik