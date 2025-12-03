#!/bin/bash

#SBATCH --job-name=job_wgpu
#SBATCH --open-mode=append
#SBATCH --output=out/test.out
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1


singularity exec --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh;
conda activate py311;
cd /scratch/dq2024/baranet-bdml/;
chmod +x run.sh;
export PYBIND11_INC="$(python -c 'import pybind11; print(pybind11.get_include())')";
export PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)";
export CMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR:${CMAKE_PREFIX_PATH}";
export CPLUS_INCLUDE_PATH="$PYBIND11_INC:${CPLUS_INCLUDE_PATH}";
export CPATH="$PYBIND11_INC:${CPATH}";
./run.sh;
python benchmark_latency.py --batch-size 1
"