#!/bin/bash
module purge
module load gcc/12.5.0 fftw/3.3.10-para cuda/12.8 nvhpc/nvhpc-hpcx-2.20-cuda12.8/25.3
export PATH=/data/home/scvj672/run/VASP/vasp.6.3.2/bin:$PATH
export OMP_NUM_THREADS=1
mpirun --bind-to none -np 8 vasp_std
