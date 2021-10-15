#!/bin/sh -v
#PBS -e /mnt/home/abstrac01/aiga_andrijanova/logs/
#PBS -o /mnt/home/abstrac01/aiga_andrijanova/logs/
#PBS -q batch
#PBS -N grid_run_3
#PBS -p 1000
#PBS -l nodes=1:ppn=4:gpus=1:shared,feature=v100
#PBS -l mem=40gb
#PBS -l walltime=96:00:00

module load conda
eval "$(conda shell.bash hook)"
conda activate bt

cd C:\Users\aigaa\Documents\GitHub\BachelorsThesis


python taskgen.py \
-sequence_name Conv2d_Fourier \
-run_name run \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 1 \
-model Conv2d_Fourier \
-dataset_path \
./data/DREAMER_IBI_30sec_byperson.json \
./data/AMIGOS_IBI_30sec_byperson.json \
./data/DREAMER_IBI_30sec_byseq.json \
./data/AMIGOS_IBI_30sec_byseq.json \
-epoch_count 100 \
-learning_rate 1e-3 3e-3 1e-4 3e-4 1e-5 3e-4 \
-batch_size 32 64 128