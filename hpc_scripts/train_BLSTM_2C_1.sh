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
-sequence_name best_runs_AMIGOS_BLSTM_2C_byperson\
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 12 \
-model BLSTM_Conv1 \
-dataset_path ./data/AMIGOS_IBI_30sec_byseq_small_2C.json \
-epoch_count 1000 \
-learning_rate 1e-4 \
-batch_size 32 64 \
-rnn_layers 2 3 \
-hidden_size 64