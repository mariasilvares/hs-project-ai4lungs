#!/bin/bash
#SBATCH --partition=TODO
#SBATCH --qos=TODO
#SBATCH --job-name=TODO
#SBATCH --output=TODO.out
#SBATCH --error=TODO.err



echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"
python src/model_trainval.py \
 --gpu_id 0 \
 --seed 42 \
 --results_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results' \
 --weights_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/weights' \
 --history_dir '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results/history' \
 --data_augmentation True \
 --model_name 'ChestXRayNN' \
 --dataset_name 'ChestXRayAbnormalities' \
 --channels 3 \
 --height 64 \
 --width 64 \
 --nr_classes 3 \
 --epochs 1 \
 --batch_size 32 \
 --base_data_path '/nas-ctm01/datasets/public/MEDICAL/mrsilvares/data' \
 --dataset_opencv_xray "/nas-ctm01/datasets/public/MEDICAL/DatasetOpenCVXray" \
 --dataset_pulmonary_chest_xray_abnormalities "/nas-ctm01/datasets/public/MEDICAL/PulmonaryChestXRaAbnormalities"



# TODO: ChestXRayNN


echo "Job finished!"
