#!/bin/bash

echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"

# # OpenCVXray w/ Data Augmentation
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
#  --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
#  --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
#  --data_augmentation \
#  --model_name 'OpenCVXRayNN' \
#  --dataset_name 'OpenCVXray' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray' 

# # OpenCVXray
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
#  --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
#  --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
#  --model_name 'OpenCVXRayNN' \
#  --dataset_name 'OpenCVXray' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'

# # ChestXRayNN w/ Data Augmentation
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
#  --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
#  --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
#  --data_augmentation \
#  --model_name 'ChestXRayNN' \
#  --dataset_name 'ChestXRayAbnormalities' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'

# # ChestXRayNN
# python src/model_trainval.py \
#  --gpu_id 0 \
#  --seed 42 \
#  --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
#  --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
#  --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
#  --model_name 'ChestXRayNN' \
#  --dataset_name 'ChestXRayAbnormalities' \
#  --channels 3 \
#  --height 64 \
#  --width 64 \
#  --nr_classes 3 \
#  --epochs 300 \
#  --batch_size 32 \
#  --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'



# DenseNet121OpenCVXRayNN com Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation \
 --model_name 'DenseNet121OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 1 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'  # Caminho base dos dados


# DenseNet121OpenCVXRayNN sem Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --model_name 'DenseNet121OpenCVXRayNN' \
 --dataset_name 'OpenCVXray' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 1 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'


# DenseNet121ChestXRayNN com Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --data_augmentation \
 --model_name 'DenseNet121ChestXRayNN' \
  --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 3 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'


# DenseNet121ChestXRayNN sem Data Augmentation
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results' \
 --weights_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights' \
 --history_dir '/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history' \
 --model_name 'DenseNet121ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 3 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'


echo "Job finished!"