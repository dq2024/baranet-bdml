#!/bin/bash

#SBATCH --job-name=job_wgpu
#SBATCH --output=out/test2.out
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB


singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh;
conda activate py311;
cd /scratch/dq2024/baranet-bdml/;
chmod +x run.sh;

./run.sh;
python benchmark_latency.py --batch-size 1
"