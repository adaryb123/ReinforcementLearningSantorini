Hra sa spusta skriptom main.py.
Pouzivatel zadava input do konzoly.
Na zaciatku pouzivatel vyberie kto bude hrat. Ak ma hrat RL agent, pouzivatel musit zadat presne
meno modelu, tak ako je ulozeny v train/models.

Adresar engine obsahuje logiku hry.
Adrsar ai obsahuje botov proti ktorym sa da hrat.
Adresar train obsahuje vsetko potrebne na trenovanie RL agenta.

Trenovanie agenta sa spusta skriptom train/train.py. V skripte treba importnut konfiguracny subor s trenovacimi parametrami
ulozeny v train/configs. Priklad konfiguracneho suboru je v train/configs/example_config.py.

Natrenovany model sa ulozi do train/models.
Grafy a logy z trenovania sa ulozia do train/plots a train/logs.

Skript train/validate.py sluzi na otestovanie uspesnosti natrenovaneho modelu proti zakladnym botom.