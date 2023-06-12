# AISnake
W celu uruchomienia programu uruchamiamy plik agent.py
Plik agent-play.py służy do załadowania wybranego modelu w celu pokazania jak on steruje wężem w środowisku
game.py to plik zawierający grę, którą można grać na klawiaturze
Aby wczytać model, po jego inicjalizacji dodajemy linijkę "self.model.load_state_dict(torch.load('[ścieżka do pliku modelu]'))"
