#!/bin/bash

echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"

# OpenCVXray w/ Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation True \
 --model_name 'OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'

# OpenCVXray
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation False \
 --model_name 'OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'

# ChestXRayNN w/ Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation True \
 --model_name 'ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'

# ChestXRayNN
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation False \
 --model_name 'ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'

echo "Job finished!"