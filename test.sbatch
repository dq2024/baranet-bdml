#!/bin/bash

#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_3033-2025fa
#SBATCH --partition=n1s8-t4-1
#SBATCH --open-mode=append
#SBATCH --output=test.out
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/dq2024/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate py311

python benchmark_latency.py --batch-size 1
"