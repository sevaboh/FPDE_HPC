#!/bin/bash
ulimit -c unlimited
export MALLOC_CHECK_=3
export MALLOC_PERTURB_=30
function all_tests
{
time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 8 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 11 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 12 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 13 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 8 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 11 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 12 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 13 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_cuda use_ocl 1 ocl_vector 4 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_ocl use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 2 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 5 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 6 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 8 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 11 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 12 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_ocl use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 1 sum_param 0.01 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_ocl use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 2 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 5 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 6 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 8 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 11 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 12 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
}
function error_tests
{
bb=1.0
A=2
B=1
fp=0.8
time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 4 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
}
function n_tests
{
for S in 4 6 8 10; do
time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 0 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7

time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 15 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 13 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 4 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param 16 oclBS 16 ieps 1e-15 EPS2 1e-7
done
}

function bs_tests
{
for BS in 16 32 48 64; do
time ./fpde_hpc_cuda use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_cuda use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
done
}

function ocl_vec_tests
{
for BS in 16 32 48 64; do
time ./fpde_hpc_ocl use_ocl 0 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 0 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 14 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 1 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 2 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 5 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 6 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 8 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 9 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 11 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
time ./fpde_hpc_ocl use_ocl 1 ocl_vector 12 aa $aa bb $bb A $A B $B Tm $Tm Sm $Sm Om $Om Tau $tau mode 3 NB $S M $S K $S varT $vt varZ 0 varXY 0 FK $fk FP $fp debug 1 split_bc $split_bc impl3d 0 split3d 0 impl 0 irs 1 TMA 2 sum_alg 2 sum_param $BS oclBS $BS ieps 1e-15 EPS2 1e-7
done
}

aa=0.8
bb=0.8
A=10
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

Tm=0.1
n_tests
S=4
error_tests
ocl_vec_tests
bs_tests
all_tests