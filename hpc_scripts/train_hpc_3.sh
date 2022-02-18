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
-sequence_name LSTM_V2_grid \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 8 \
-model LSTM_V2 \
-dataset_path \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_2C_standard_score_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_2C_standard_score_byperson.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_2C_minmax_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_2C_minmax_byperson.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_2C_AllVideos_standard_score_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_2C_AllVideos_standard_score_byperson.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_2C_AllVideos_minmax_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_2C_AllVideos_minmax_byperson.json \
-epoch_count 100 \
-learning_rate 3e-4 \
-batch_size 256 \
-rnn_layers 1 2 3 4 5 \
-hidden_size 64 128 256