#!/bin/bash
#BSUB -J python
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
### -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process"
### BSUB -R "select[gpua100]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -o job_%J.out
#BSUB -e job_%J.err

source env/bin/activate

python3 ddpm.py train --data mnist --model model.pt --device cuda --batch-size 64 --epochs 100 --lr 1e-3 --net unet
python3 ddpm.py sample --data mnist --model model.pt --device cuda --batch-size 64 --net unet