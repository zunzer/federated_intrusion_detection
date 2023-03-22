#!/bin/bash

python Cohda_Simulator.py 1 64434 & 
python Cohda_Simulator.py 2 64436 & 
python Cohda_Simulator.py 3 64438 & 
python Cohda_Simulator.py 4 64440 & 
python Cohda_Simulator.py 5 64442  &
python Cohda_Simulator.py 6 64444  &
python Cohda_Simulator.py 7 64446  &
python Cohda_Simulator.py 8 64448  &
python Cohda_Simulator.py 9 64450  &
python Cohda_Simulator.py 0 64452 &

python Client_threading.py 1 64433 64434 & 
python Client_threading.py 2 64435 64436 & 
python Client_threading.py 3 64437 64438 & 
python Client_threading.py 4 64439 64440 & 
python Client_threading.py 5 64441 64442 & 
python Client_threading.py 6 64443 64444 & 
python Client_threading.py 7 64445 64446 & 
python Client_threading.py 8 64447 64448 & 
python Client_threading.py 9 64449 64450 & 
python Client_threading.py 0 64451 64452 


sleep 20