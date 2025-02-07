#!/bin/bash

echo "HS-Project-AI4Lungs (Maria Silvares)"
echo "Job started!"

python src/model_test.py \
  --results_dir "/home/mariareissilvares/Documents/hs-project-ai4lungs/results" \
  --weights_dir "/home/mariareissilvares/Documents/hs-project-ai4lungs/results/weights" \
  --history_dir "/home/mariareissilvares/Documents/hs-project-ai4lungs/results/history" \
  --model_name "OpenCVXRayNN" \
  --dataset_name "OpenCVXray" \
  --base_data_path "/home/mariareissilvares/Documents/hs-project-ai4lungs/data/DatasetOpenCVXray" \
  --gpu_id 0 \
  --seed 42 \
  --batch_size 32

echo "Job finished!"
