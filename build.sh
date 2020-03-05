#!/bin/sh
g++ --std=c++11 -O3 -msse3 --param max-inline-insns-auto=4000 fpde_hpc_1d.cpp -lfftpack -o fpde_hpc_seq
g++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 fpde_hpc_1d.cpp -lfftpack -o fpde_hpc_par
mpic++ --std=c++11 -g3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DUSE_MPI fpde_hpc_1d.cpp -lfftpack -o fpde_hpc_mpi
mpic++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DUSE_MPI -DUSE_REPARTITIONING fpde_hpc_1d.cpp -lfftpack -o fpde_hpc_mpi_rp
mpic++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DOCL opencl_cuda_base.cpp fpde_hpc_1d.cpp -lfftpack -ldl -lOpenCL -o fpde_hpc_ocl
mpic++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DOCL -DUSE_MPI opencl_cuda_base.cpp fpde_hpc_1d.cpp -lfftpack -ldl -lOpenCL -o fpde_hpc_ocl_mpi
mpic++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DOCL -DUSE_MPI -DUSE_REPARTITIONING opencl_cuda_base.cpp fpde_hpc_1d.cpp -lfftpack -ldl -lOpenCL -o fpde_hpc_ocl_mpi_rp
#mpic++ --std=c++11 -O3 -fopenmp -msse3 --param max-inline-insns-auto=4000 -DUSE_OPENCL -DOCL -DCUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64  opencl_cuda_base.cpp fpde_hpc_1d.cpp libfftpack/libfftpack.a -ldl -lcuda -lcudart -lnvrtc -o fpde_hpc_cuda
