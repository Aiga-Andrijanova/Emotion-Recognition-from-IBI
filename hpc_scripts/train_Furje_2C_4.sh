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
conda activate aiga_env

cd /mnt/home/abstrac01/aiga_andrijanova/


python taskgen.py \
-sequence_name best_runs_DREAMER_Fourier_2C_byseq \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 8 \
-model Conv2d_Fourier \
-dataset_path ./data/DREAMER_IBI_30sec_byseq.json \
-epoch_count 1000 \
-learning_rate 1e-3 1e-4 1e-5 \
-batch_size 16 32 64 128