#!/bin/sh -v
#PBS -e /mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/logs
#PBS -o /mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/logs
#PBS -q batch
#PBS -p 1000
#PBS -l nodes=1:ppn=12:gpus=1:shared,feature=k40
#PBS -l mem=40gb
#PBS -l walltime=96:00:00
#PBS -N aiga_andrijanova

module load conda
eval "$(conda shell.bash hook)"
conda activate conda_k40
export LD_LIBRARY_PATH=~/.conda/envs/conda_k40/lib:$LD_LIBRARY_PATH

cd /mnt/home/evaldsu/Documents/emotion_recognition_from_IBI

python taskgen.py \
-sequence_name BLSTM_Conv1_grid \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 8 \
-model BLSTM_Conv1 \
-dataset_path \
./data/DREAMER_IBI_30sec_byseq.json \
./data/AMIGOS_IBI_30sec_byseq.json \
./data/DREAMER_IBI_30sec_byperson.json \
./data/AMIGOS_IBI_30sec_byperson.json \
-epoch_count 250 \
-learning_rate 1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 \
-batch_size 64 128 256 \
-rnn_layers 1 2 3 4 5 \
-hidden_size 64 128 256