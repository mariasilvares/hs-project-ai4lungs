#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb
#SBATCH --job-name=mrs_train
#SBATCH --output=mrs_train.out
#SBATCH --error=mrs_train.err

echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"

# # OpenCVXray w/ Data Augmentation
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
#  --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
#  --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
#  --data_augmentation \
#  --model_name 'OpenCVXRayNN' \
#  --dataset_name 'OpenCVXray' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/nas-ctm01/datasets/public/MEDICAL/DatasetOpenCVXray'

# # OpenCVXray
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
#  --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
#  --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
#  --model_name 'OpenCVXRayNN' \
#  --dataset_name 'OpenCVXray' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/nas-ctm01/datasets/public/MEDICAL/DatasetOpenCVXray'

# # ChestXRayNN w/ Data Augmentation
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
#  --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
#  --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
#  --data_augmentation \
#  --model_name 'ChestXRayNN' \
#  --dataset_name 'ChestXRayAbnormalities' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/nas-ctm01/datasets/public/MEDICAL/PulmonaryChestXRaAbnormalities'

#  # ChestXRayNN
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
#  --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
#  --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
#  --model_name 'ChestXRayNN' \
#  --dataset_name 'ChestXRayAbnormalities' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/nas-ctm01/datasets/public/MEDICAL/PulmonaryChestXRaAbnormalities'


#DenseNet121OpenCVXRayNN com Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
 --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
 --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
 --data_augmentation \
 --model_name 'DenseNet121OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/nas-ctm01/datasets/public/MEDICAL/DatasetOpenCVXray'

#DenseNet121OpenCVXRayNN sem Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
 --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
 --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
 --model_name 'DenseNet121OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/nas-ctm01/datasets/public/MEDICAL/DatasetOpenCVXray'

# DenseNet121ChestXRayNN com Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
 --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
 --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
 --data_augmentation \
 --model_name 'DenseNet121ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/nas-ctm01/datasets/public/MEDICAL/PulmonaryChestXRaAbnormalities'

#DenseNet121ChestXRayNN sem Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
 --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
 --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
 --model_name 'DenseNet121ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 300 \
 --batch_size 32 \
 --base_data_path '/nas-ctm01/datasets/public/MEDICAL/PulmonaryChestXRaAbnormalities'

echo "Job finished!"
