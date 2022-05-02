#!/bin/sh
#compile
ulimit -c unlimited
MALLOC_CHECK_=3
export MALLOC_CHECK_
#g++ -O3 -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -DOCL -DCUDA -DTi=0 -DT=float opencl_cuda_base.cpp fr_do.cpp -lcuda -lcudart -lnvrtc -o fr_do_cuda_f
#g++ -O3 -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -DOCL -DCUDA -DTi=1 -DT=double opencl_cuda_base.cpp fr_do.cpp -lcuda -lcudart -lnvrtc -o fr_do_cuda_d
# test w/o gpu
#g++ -O3 -DTi=0 -DT=float -fopenmp fr_do.cpp -o fr_do_float
#g++ -O3 -DTi=1 -DT=double -fopenmp fr_do.cpp -o fr_do_double
#time ./fr_do_float 3000 64 > f.txt
#time ./fr_do_double 3000 64 > d.txt
#time ./fr_do_float -1 128 64 32 1280 > ddc.txt
#time ./fr_do_double -1 128 64 32 1280 > dfc.txt
# test equation solution (analytic through rows and integrals)
echo > e1dc.txt
echo > e1dg.txt
echo > e1fc.txt
echo > e1fg.txt
rm s_n_t.txt
for i in 128 256 512 1024
do
time ./fr_do_cuda_d -1 128 64 32 $i 0 1 1.5 1e-17 13  >> e1dc.txt
time ./fr_do_cuda_d -2 128 64 32 $i 0 1 1.5 1e-17 13  >> e1dg.txt
time ./fr_do_cuda_f -1 128 64 32 $i 0 1 1.5 1e-17 13  >> e1fc.txt
time ./fr_do_cuda_f -2 128 64 32 $i 0 1 1.5 1e-17 13  >> e1fg.txt
done
for i in 0.5 1.0 1.5 2.0
do
rm s_n_t.txt
time ./fr_do_cuda_d -1 128 64 32 1024 1 1 $i 1e-17 13 >> e1dc.txt
time ./fr_do_cuda_d -2 128 64 32 1024 1 1 $i 1e-17 13  >> e1dg.txt
time ./fr_do_cuda_f -1 128 64 32 1024 1 1 $i 1e-17 13  >> e1fc.txt
time ./fr_do_cuda_f -2 128 64 32 1024 1 1 $i 1e-17 13  >> e1fg.txt
done
rm s_n_t.txt
for i in 32 64 96 128
do
time ./fr_do_cuda_d -1 $i 64 32 1024 0 1 1.5 1e-17 13  >> e1dc.txt
time ./fr_do_cuda_d -2 $i 64 32 1024 0 1 1.5 1e-17 13  >> e1dg.txt
time ./fr_do_cuda_f -1 $i 64 32 1024 0 1 1.5 1e-17 13  >> e1fc.txt
time ./fr_do_cuda_f -2 $i 64 32 1024 0 1 1.5 1e-17 13 >> e1fg.txt
done
for i in 32 64 96 128
do
time ./fr_do_cuda_d -1 128 $i 32 1024 0 1 1.5 1e-17 13  >> e1dc.txt
time ./fr_do_cuda_d -2 128 $i 32 1024 0 1 1.5 1e-17 13  >> e1dg.txt
time ./fr_do_cuda_f -1 128 $i 32 1024 0 1 1.5 1e-17 13  >> e1fc.txt
time ./fr_do_cuda_f -2 128 $i 32 1024 0 1 1.5 1e-17 13  >> e1fg.txt
done
for i in 32 64 128
do
time ./fr_do_cuda_d -1 128 64 $i 1024 0 1 1.5 1e-17 13 >> e1dc.txt
time ./fr_do_cuda_d -2 128 64 $i 1024 0 1 1.5 1e-17 13 >> e1dg.txt
time ./fr_do_cuda_f -1 128 64 $i 1024 0 1 1.5 1e-17 13 >> e1fc.txt
time ./fr_do_cuda_f -2 128 64 $i 1024 0 1 1.5 1e-17 13 >> e1fg.txt
done
# test derivative computation
echo > cf0.txt
echo > cf1.txt
echo > cd0.txt
echo > cd1.txt
for i in 800 2080 4160 6240 8320
do
for j in 32 64 96 128
do
time ./fr_do_cuda_f $i $j 0 0 >> cf0.txt
time ./fr_do_cuda_f $i $j 0 1 >> cf1.txt
time ./fr_do_cuda_d $i $j 0 0 >> cd0.txt
time ./fr_do_cuda_d $i $j 0 1 >> cd1.txt
done
done
# test equation solution
echo > edc.txt
echo > edg.txt
echo > efc.txt
echo > efg.txt
for i in 32 64 96 128
do
time ./fr_do_cuda_d -1 $i 64 32 1280 0 >> edc.txt
time ./fr_do_cuda_d -2 $i 64 32 1280 0 >> edg.txt
time ./fr_do_cuda_f -1 $i 64 32 1280 0 >> efc.txt
time ./fr_do_cuda_f -2 $i 64 32 1280 0 >> efg.txt
done
for i in 32 64 96 128
do
time ./fr_do_cuda_d -1 128 $i 32 1280 0 >> edc.txt
time ./fr_do_cuda_d -2 128 $i 32 1280 0 >> edg.txt
time ./fr_do_cuda_f -1 128 $i 32 1280 0 >> efc.txt
time ./fr_do_cuda_f -2 128 $i 32 1280 0 >> efg.txt
done
for i in 32 64 96 128
do
time ./fr_do_cuda_d -1 128 64 $i 1280 0 >> edc.txt
time ./fr_do_cuda_d -2 128 64 $i 1280 0 >> edg.txt
time ./fr_do_cuda_f -1 128 64 $i 1280 0 >> efc.txt
time ./fr_do_cuda_f -2 128 64 $i 1280 0 >> efg.txt
done
for i in 640 1280 1920 2560 3200
do
time ./fr_do_cuda_d -1 128 64 32 $i 0 >> edc.txt
time ./fr_do_cuda_d -2 128 64 32 $i 0 >> edg.txt
time ./fr_do_cuda_f -1 128 64 32 $i 0 >> efc.txt
time ./fr_do_cuda_f -2 128 64 32 $i 0 >> efg.txt
done
