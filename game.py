import pygame
import random
from enum import Enum
from collections import namedtuple
pygame.init()
font=pygame.font.Font('CALIST.ttf',24)

class Kierunki(Enum):
    PRAWO=1
    LEWO=2
    GORA=3
    DOL=4

Punkt =namedtuple('Punkt','x,y')

WIELKOSC_KRATKI=20
PREDKOSC=20
BIALY=(255,255,255)
CZERWONY=(225,0,0)
NIEBIESKI=(0,0,255)
ZIELONY=(0,255,100)
CZARNY=(0,0,0)

class Gra:
    def __init__(self,w=640,h=480):
        self.h=h
        self.w=w
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Wonsz')
        self.zegar=pygame.time.Clock()

        self.kierunek=Kierunki.PRAWO
        self.glowa=Punkt(self.w/2,self.h/2)
        self.wonsz=[self.glowa,Punkt(self.glowa.x-WIELKOSC_KRATKI,self.glowa.y),Punkt(self.glowa.x-(2*WIELKOSC_KRATKI),self.glowa.y)]
        self.jedzenie=None
        self.wynik=0
        self._generuj__jedzenie()
    def _generuj__jedzenie(self):
        x=random.randint(0,(self.w-WIELKOSC_KRATKI)//WIELKOSC_KRATKI)*WIELKOSC_KRATKI
        y = random.randint(0, (self.h - WIELKOSC_KRATKI) // WIELKOSC_KRATKI) * WIELKOSC_KRATKI
        self.jedzenie=Punkt(x,y)
        if(self.jedzenie in self.wonsz):
            self._generuj__jedzenie()

    def _ruszaj(self,kierunek):
        y=self.glowa.y
        x=self.glowa.x
        if(kierunek==Kierunki.LEWO):
            x-=WIELKOSC_KRATKI
        elif (kierunek == Kierunki.PRAWO):
            x += WIELKOSC_KRATKI
        elif (kierunek == Kierunki.GORA):
            y -= WIELKOSC_KRATKI
        elif (kierunek == Kierunki.DOL):
            y += WIELKOSC_KRATKI
        self.glowa=Punkt(x,y)

    def _kolizja(self):
        if(self.glowa.x>self.w-WIELKOSC_KRATKI or self.glowa.x<0 or self.glowa.y<0 or self.glowa.y>self.h-WIELKOSC_KRATKI ):
            return True
        if(self.glowa in self.wonsz[1:]):
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

    def krok(self):
        for event in pygame.event.get():
            if(event.type==pygame.QUIT):
                pygame.quit()
                quit()
            if(event.type==pygame.KEYDOWN):
                if(event.key==pygame.K_RIGHT):
                    self.kierunek=Kierunki.PRAWO
                elif(event.key==pygame.K_LEFT):
                    self.kierunek=Kierunki.LEWO
                elif (event.key == pygame.K_DOWN):
                    self.kierunek = Kierunki.DOL
                elif (event.key == pygame.K_UP):
                    self.kierunek = Kierunki.GORA
        self._ruszaj(self.kierunek)
        self.wonsz.insert(0,self.glowa)
        game_over=False
        if(self._kolizja()):
            game_over=True
            return game_over,self.wynik
        if(self.glowa==self.jedzenie):
            self.wynik+=1
            self._generuj__jedzenie()
        else:
            self.wonsz.pop()
        self._ui()
        self.zegar.tick(PREDKOSC)
        return game_over,self.wynik

if __name__=="__main__":
    game=Gra()

    while True:
        game_over,wynik=game.krok()
        if(game_over==True):
            break

    print('Wynik koncowy',wynik)
    pygame.quit()