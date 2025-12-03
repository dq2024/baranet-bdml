#!/bin/bash

#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_3033-2025fa
#SBATCH --partition=n1s8-t4-1
#SBATCH --open-mode=append
#SBATCH --output=.test.out
#SBATCH --error=.test.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1


singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash "
source /ext3/env.sh
conda activate py311

python benchmark_latency.py --batch-size 1
"