#!/bin/sh -v
conda activate bt
cd C:/Users/aigaa/Documents/GitHub/BachelorsThesis
export HOME=C:/Users/aigaa/Documents


python taskgen.py \
-sequence_name Fourier_2C_grid \
-template template_hpc.sh \
-script main.py \
-is_force_start True \
-num_repeat 1 \
-num_cuda_devices_per_task 1 \
-num_tasks_in_parallel 1 \
-model Conv2d_Fourier \
-dataset_path \
./data/DREAMER_IBI_30sec_byseq.json \
./data/AMIGOS_IBI_30sec_byseq.json \
-epoch_count 100 \
-learning_rate 1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 \
-batch_size 16 32 64 128