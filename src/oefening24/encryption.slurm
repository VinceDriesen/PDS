#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=lp_h_pds_iiw
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load GCC/10.3.0
module load CUDA/11.7.0

nvcc -O3 -DNDEBUG -o encrypt encrypt_cuda.cu encrypt.cpp encrypt_cuda.cpp main.cpp -gencode arch=compute_60,code=sm_60

./encrypt encrypted.txt decrypted.txt PDS_PDS_PDS_PDS decrypt

