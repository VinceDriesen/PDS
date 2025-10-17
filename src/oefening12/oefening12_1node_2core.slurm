#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:00:10
#SBATCH --error="%x.e_oefening12_1node_2cores"
#SBATCH --output="%x.o_oefening12_1node_2cores"

# Own code, remove if it causes problems
source /data/leuven/303/vsc30380/slurmhooks

module purge

module load OpenMPI/5.0.3-GCC-13.3.0
module load GCC/13.3.0

mpic++ -o oefening12 oefening12.cpp
mpirun -np 2 ./oefening12
