#!/bin/bash
ulimit -c unlimited
export MALLOC_CHECK_=3
export MALLOC_PERTURB_=30
function do_test
{
S=$1
S2=$2
S3=$3
i3=$4
echo $S $S2 $S3 "impl3d=" $i3
./fpde_hpc_seq aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
./fpde_hpc_par aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
./fpde_hpc_ocl use_ocl 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 1 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 1 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 2 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 2 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_mpi aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_ocl_mpi use_ocl 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 1 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 1 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 2 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 2 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 1 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_mpi_rp aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
mpirun --hostfile hostfile -np 3 ./fpde_hpc_ocl_mpi_rp use_ocl 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 BS 1 NB $S M $S2 K $S3 varT $vt varZ $vz varXY $vxy FK $fk FP $fp debug 0 split_bc $split_bc impl3d $i3 split3d 0 impl 2 irs 1 TMA 2 sum_alg 2 sum_param 10 mpi_add_thr 10 | grep "last errs"
}

aa=0.8
bb=0.8
A=5
B=0
S=2
Tm=1.0
Sm=1.0
Om=1.0
tau=0.0025
vt=0
vz=0
vxy=0
fk=3
fp=0.9
split_bc=0

Tm=0.01
do_test 2 12 11 0
do_test 12 2 11 0
do_test 11 12 13 0
do_test 2 12 11 1
do_test 12 2 11 1
do_test 11 12 13 1

