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
-sequence_name grid_search_DREAMER_LSTM_2C_byseq \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 6 \
-dataset_path ./data/DREAMER_IBI_30sec_byseq.json \
-epoch_count 100 \
-learning_rate 1e-4 1e-5 \
-batch_size 32 64 128 \
-rnn_layers 1 2 3 \
-hidden_size 16 32 64