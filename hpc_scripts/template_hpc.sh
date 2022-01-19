#!/bin/sh
module load conda
eval "$(conda shell.bash hook)"
source activate conda_k40
export LD_LIBRARY_PATH=~/.conda/envs/conda_k40/lib:$LD_LIBRARY_PATH
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

ulimit -n 500000

cd /mnt/home/evaldsu/Documents/emotion_recognition_from_IBI