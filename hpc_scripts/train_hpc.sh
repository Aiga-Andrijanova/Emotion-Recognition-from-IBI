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
-sequence_name Conv2d_Fourier_grid \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 8 \
-model Conv2d_Fourier \
-dataset_path \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_byseq.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/DREAMER_IBI_30sec_byperson.json \
/mnt/home/evaldsu/Documents/emotion_recognition_from_IBI/data/AMIGOS_IBI_30sec_byperson.json \
-epoch_count 250 \
-learning_rate 1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 \
-batch_size 64 128 256