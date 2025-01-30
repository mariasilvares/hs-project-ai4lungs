#!/bin/bash
#SBATCH --partition=TODO
#SBATCH --qos=TODO
#SBATCH --job-name=TODO
#SBATCH --output=TODO.out
#SBATCH --error=TODO.err



echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"


# OpenCVXray
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
 --epochs 1 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray'



# TODO: ChestXRayNN
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
 --epochs 1 \
 --batch_size 32 \
 --base_data_path '/home/mariareissilvares/Documents/hs-project-ai4lungs/data/PulmonaryChestXRaAbnormalities'
echo "Job finished!"
