import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import numpy as np
pygame.init()
font=pygame.font.Font('CALIST.ttf',24)

class Kierunki(Enum):
    PRAWO=1
    LEWO=2
    GORA=3
    DOL=4

Punkt =namedtuple('Punkt','x,y')

WIELKOSC_KRATKI=20
PREDKOSC=40
BIALY=(255,255,255)
CZARNY=(0,0,0)
CZERWONY=(225,0,0)
NIEBIESKI=(0,0,255)
ZIELONY=(0,255,100)

class Gra_AI:
    def __init__(self,w=640,h=480):
        self.h=h
        self.w=w
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Wonsz')
        self.zegar=pygame.time.Clock()
        self.reset()

    def reset(self):
        self.kierunek=Kierunki.PRAWO
        self.glowa=Punkt(self.w/2,self.h/2)
        self.wonsz=[self.glowa,Punkt(self.glowa.x-WIELKOSC_KRATKI,self.glowa.y),Punkt(self.glowa.x-(2*WIELKOSC_KRATKI),self.glowa.y)]
        self.jedzenie=None
        self.wynik=0
        self._generuj__jedzenie()
        self.kroki=0

    def _generuj__jedzenie(self):
        x=random.randint(0,(self.w-WIELKOSC_KRATKI)//WIELKOSC_KRATKI)*WIELKOSC_KRATKI
        y = random.randint(0, (self.h - WIELKOSC_KRATKI) // WIELKOSC_KRATKI) * WIELKOSC_KRATKI
        self.jedzenie=Punkt(x,y)
        if(self.jedzenie in self.wonsz):
            self._generuj__jedzenie()

    def _ruszaj(self,akcja):
        clock_wise = [Kierunki.PRAWO, Kierunki.DOL, Kierunki.LEWO, Kierunki.GORA]
        i=clock_wise.index(self.kierunek)
        if np.array_equal(akcja, [1, 0, 0]):
            nowy_kierunek=clock_wise[i]
        elif np.array_equal(akcja, [0, 1, 0]):
            next_i=(i+1)%4
            nowy_kierunek=clock_wise[next_i]
        else:
            next_i = (i - 1) % 4
            nowy_kierunek = clock_wise[next_i]
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

    def _kolizja(self,punkt=None):
        if(punkt is None):
            punkt=self.glowa
        if(punkt.x>self.w-WIELKOSC_KRATKI or punkt.x<0 or punkt.y<0 or punkt.y>self.h-WIELKOSC_KRATKI ):
            return True
        if(punkt in self.wonsz[1:]):
            return True
        return False

    def _ui(self):
        self.display.fill(CZARNY)
        for punkt in self.wonsz:
            pygame.draw.rect(self.display,NIEBIESKI,pygame.Rect(punkt.x,punkt.y,WIELKOSC_KRATKI,WIELKOSC_KRATKI))
            pygame.draw.rect(self.display, ZIELONY, pygame.Rect(punkt.x+4, punkt.y+4, 12, 12))
        pygame.draw.rect(self.display,CZERWONY,pygame.Rect(self.jedzenie.x,self.jedzenie.y,WIELKOSC_KRATKI,WIELKOSC_KRATKI))
        text=font.render("wynik: "+str(self.wynik),True,BIALY)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def krok(self,akcja):
        self.kroki+=1
        for event in pygame.event.get():
            if(event.type==pygame.QUIT):
                pygame.quit()
                quit()

        self._ruszaj(akcja)
        self.wonsz.insert(0,self.glowa)
        game_over=False
        nagroda=0
        if(self._kolizja() or self.kroki>100*len(self.wonsz)):
            game_over=True
            nagroda=-10
            return nagroda,game_over,self.wynik

        if(self.glowa==self.jedzenie):
            nagroda=10
            self.wynik+=1
            self._generuj__jedzenie()
        else:
            self.wonsz.pop()
        self._ui()
        self.zegar.tick(PREDKOSC)
        return nagroda,game_over,self.wynik