#!/bin/bash
#SBATCH --job-name=single       # Specify job name
#SBATCH --cpus-per-task=32  # Specify number of CPUs per task
#SBATCH --mem-per-cpu=1G  # Request memory in MB for the gpu
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH --time 2:00:00        # Run time (hh:mm:ss) 
#SBATCH --export=ALL         # All your environmental variables setting will passed to the compute nodes.


# Load needed modules. Example: 
# module load cuda/12.1.1


# Copy the needed files
# $PATH_TO_CODE is the path to the directory containing the code
# $PATH_TO_DATA is the path to the directory containing the datasets.

cp $PATH_TO_CODE/*py ./
cp -r $PATH_TO_CODE/crystal_builder ./
cp -r $PATH_TO_CODE/models ./


srun python main.py \
    --calculation_type single_calc \
    --epochs 250 \
    --len_dataset all \
    --name_dataset mp_gap \
    --path_dataset $PATH_TO_DATA \
    --task regression \
    --model_name GATom \
    --num_workers 4 \
    --use_matf32 \
    > output.out 2> output.err

