#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:00:10
#SBATCH --error="%x.e_oefening16"
#SBATCH --output="%x.o_oefening16"

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load GCC/13.3.0

g++ -g -Wall -fopenmp -o oefening16 oefening16.cpp

srun ./oefening16 2
