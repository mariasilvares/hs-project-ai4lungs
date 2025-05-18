#!/bin/bash



# Sync data with SLURM and local workstation
echo "Downloading data..."
rsync -azP tgoncalv@ctm-upload:/nas-ctm01/datasets/public/MEDICAL/mrsilvares/results /home/mariareissilvares/Documents/hs-project-ai4lungs
echo "Finished"