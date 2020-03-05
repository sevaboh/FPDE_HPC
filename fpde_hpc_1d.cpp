/* Author: Vsevolod Bohaienko */
/* 1d fractional PDE for water transport/diffusion modelling with OpenMP,MPI,GPU parallelization */
//#define USE_MPI
//#define OCL
// for cuda code OCL should also be enabled
// #define CUDA
//#define USE_REPARTITIONING
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <map>
#include <condition_variable>
#include <thread>
#include <chrono>
#ifdef _OPENMP // openmp - don't work for varT, automated algorithm selection, Toeplitz matrices multiplication through FFT
#include <omp.h> 
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <unistd.h>
#include <sys/times.h>
#include <unistd.h>
#include <sys/times.h>
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}
void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}
#ifdef OCL
#include "opencl_cuda_class.h"
#endif
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
std::map<double,double> gamma_cache;
// Note that the functions Gamma and LogGamma are mutually dependent.
double Gamma
(
    double x    // We require x > 0
);

double LogGamma
(
    double x    // x must be positive
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    if (x < 12.0)
    {
        return log(fabs(Gamma(x)));
    }

	// Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    static const double c[8] =
    {
		 1.0/12.0,
		-1.0/360.0,
		1.0/1260.0,
		-1.0/1680.0,
		1.0/1188.0,
		-691.0/360360.0,
		1.0/156.0,
		-3617.0/122400.0
    };
    double z = 1.0/(x*x);
    double sum = c[7];
    for (int i=6; i >= 0; i--)
    {
        sum *= z;
        sum += c[i];
    }
    double series = sum/x;

    static const double halfLogTwoPi = 0.91893853320467274178032973640562;
    double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;    
	return logGamma;
}
double Gamma
(
    double x    // We require x > 0
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}
	std::map<double,double>::iterator it;
	if ((it=gamma_cache.find(x))!=gamma_cache.end())
		return it->second;
    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
	//
	// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
	// The relative error over this interval is less than 6e-7.

	const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

    if (x < 0.001)
	{
		double res=1.0/(x*(1.0 + gamma*x));
#pragma omp critical
		gamma_cache[x]=res;
        return res;
	}

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)
    
	if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
		
		double y = x;
        int n = 0;
        bool arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if (arg_was_less_than_one)
        {
            y += 1.0;
        }
        else
        {
            n = static_cast<int> (floor(y)) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        static const double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        static const double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;
        for (i = 0; i < 8; i++)
        {
            num = (num + p[i])*z;
            den = den*z + q[i];
        }
        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
                result *= y++;
        }
#pragma omp critical
		gamma_cache[x]=result;
		return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
 		// Correct answer too large to display. Force +infinity.
		double temp = DBL_MAX;
#pragma omp critical
		gamma_cache[x]=temp*2.0;

		return temp*2.0;
    }
	double res=exp(LogGamma(x));
#pragma omp critical
	gamma_cache[x]=res;
    return res;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
#include <fftpack.h>
#include <ls_fft.h>
#include <bluestein.h>
void fft(double *v, int n)
{
    static real_plan plan;
	static int old_n = -1;
	if (old_n != n)
	{
		if (old_n!=-1)
			kill_real_plan(plan);
		plan = make_real_plan(n);
		old_n = n;
	}
    real_plan_forward_c(plan, v);
    
}
void ifft(double *v, int n)
{
	static real_plan plan;
	static int old_n = -1;
	if (old_n != n)
	{
		if (old_n != -1)
			kill_real_plan(plan);
		plan = make_real_plan(n);
		old_n = n;
	}
	real_plan_backward_c(plan, v);
}
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
FILE *log_file=NULL;
double XYvar_coef = 100.0;
double Zvar_coef = 100.0;
int full_test = 0;
int linear_avg = 0;
int implicit = 0; //  0 - explicit, 1 - full matrix formation, 2 - tridiagonal + intmatrix
int implicit3d=0; // 3d problem - 0 - locally-onedimensional, 1 - implicit
int toeplitz_mult_alg = 0; // 0 - full, 1 - through FFT, 2 - through series
int space_der = 0; // 0 - onesided, 1 - twosided (Dl+Dr), Dr=-int(x,b), Dl=int(0,x)
int implicit_row_scaling=0;
int func_in_kernel = 0; // 0 -x, 1 - x^1/2, 2 - x^2, 3 - x^k
int integr_max_niter=100000;
double func_power = 0.6;
int BS = 10; // block size
int NB = 2, sNB; // number of block
int MB, KB;
int N, sN;  // number of nodes
int _M, K; // x,y nodes for 3D mode
int varT = 1, varZ = 1, varXY = 0;
int varZ_niter = 1000;
double A = 0.1; // 1-A for calc_V
double global_eps=1e-5;
double global_eps2=1e-5; // used to control kb_row2
int debug_level = 0;
int impl_niter = 5000;
double impl_err = 1e-9;
int rp_split_3d=0; // right part added once or distributed
int no_t=0,no_x=0; // no time or space variables in analytic test
int sum_alg=0; // 0 - full summing, 1 - restricted summing, 2 - series expantion, 3 - full summing and calculation of series sums (flag for automated algorithm selection)
int dont_clear_sum_alg1=0;
double sum_param=1e-5; // eps for restricted summing, number of terms for series expantion
double st_vmu,st_rss;
int tma2_all_diags=0;
// for automated algorithm selection (maximal admissable difference - sum of squeares of differences to sum of squares)
double difference_factor=0.5; // maximal admissable difference between implicit3d and other algs
double tma22_factor=1.1; // maximal admissable difference between base implicit alg and series expansion space fraction approximation
double tma22_minstep=1.1; // minimal step factor for optimal series expansion eps search
int time_min=50; // minimal time in ms to search for the best time derivative approximation algorithm
int timegr_min=50; // minimal iteration
double timegr_factor=1.2; // factor for time growth to search for the best time derivative approximation algorithm
double a5coef=1.0;
#ifndef _OPENMP
	double *F; // function values
	double *BVK; // values of F
#endif	
// GPGPU variables
int double_ext=0;
int use_ocl=0;
int device=0;
// MPI variables
int mpi_rank=0,mpi_size=1;
int mpi_mult=0;
int mpi_mp_add_threads=10;
// MPI functions wrappers
#ifdef USE_MPI
int ncalls[5]={0,0,0,0};
int mpi_times[5]={0,0,0,0};
// mpi recv or send request
typedef struct
{
	const void *buf;
	int count;
	MPI_Datatype datatype;
	int tag;
	int done; // received and notified
	int notified;
	std::condition_variable *cv;
	std::mutex *cv_m;
	int dt_size; // datatype size
	int thread;
} mpi_req;
// message received from another process - to be matched with recv request
typedef struct
{
	char *buf;
	int size;
	int tag;
	int processed;
} mpi_msg;
std::map<int,int> thread_state; // number of receives
std::map<int,std::pair<int,int> > thread_wait_id;
std::vector<mpi_req> *mpi_sends,*mpi_recvs; 
std::vector<mpi_msg> *mpi_received_msgs;
omp_lock_t mpi_lock,*recv_locks,*send_locks;
omp_lock_t *proc_locks;
omp_lock_t *proc_send_locks;
int *last_sends,*last_recvs,*last_msgs;
char **bufs=NULL;
int *sizes=NULL;
int *ns=NULL;
char **bs=NULL;
MPI_Request *rqs=NULL;
int *sizes2=NULL;
char **bufs2=NULL;
void mpi_set_start_thread()
{
omp_set_lock(&mpi_lock);
	thread_state.insert(std::pair<int,int>(omp_get_thread_num(),0));
	thread_wait_id.insert(std::pair<int,std::pair<int,int> >(omp_get_thread_num(),std::pair<int,int>(-1,-1)));
omp_unset_lock(&mpi_lock);
}
// match received messages and recv requests
int check_received_messages(int rank)
{
	int m=0;
omp_set_lock(&recv_locks[rank]);
	for (int i=last_msgs[rank];i<mpi_received_msgs[rank].size();i++)
	if (mpi_received_msgs[rank][i].processed==0)
		for (int j=last_recvs[rank];j<mpi_recvs[rank].size();j++)
			if (mpi_recvs[rank][j].done==0)
			    if (mpi_recvs[rank][j].tag==mpi_received_msgs[rank][i].tag)
			    {
				int s;
				MPI_Type_size(mpi_recvs[rank][j].datatype, &s);
				if ((mpi_recvs[rank][j].count*s)!=mpi_received_msgs[rank][i].size)
				{
					printf("%d %d size mismatch %d %d %d\n",mpi_rank,mpi_recvs[rank][j].tag,rank,(mpi_recvs[rank][j].count*s),mpi_received_msgs[rank][i].size);
					while(1);
					abort();
				}
				memcpy((char *)mpi_recvs[rank][j].buf,mpi_received_msgs[rank][i].buf,mpi_received_msgs[rank][i].size);
				delete [] mpi_received_msgs[rank][i].buf;
				mpi_recvs[rank][j].done=1;
				mpi_received_msgs[rank][i].processed=1;
				m++;
				break;
			    }
	// move pointers
	for (int i=last_msgs[rank];i<mpi_received_msgs[rank].size();i++)
	    if (mpi_received_msgs[rank][i].processed)
			last_msgs[rank]=i;
	    else
			break;
	for (int j=last_recvs[rank];j<mpi_recvs[rank].size();j++)
	    if (mpi_recvs[rank][j].done==1)
			last_recvs[rank]=j;
	    else
			break;
omp_unset_lock(&recv_locks[rank]);
	return m;
}
int make_sends(int rank)
{
	unsigned int t1=GetTickCount();
	// make sends
	int s;
omp_set_lock(&proc_send_locks[rank]);
	// wait for previous sends
	if (bufs[rank])
	{
		MPI_Status st;
		MPI_Wait(&rqs[2*rank+0],&st);
		if (ns[rank])
			MPI_Wait(&rqs[2*rank+1],&st);
	}
	// collect data into bufs
	sizes[rank]=0;
	ns[rank]=0;
	if (bufs[rank]) delete [] bufs[rank];
	bufs[rank]=NULL;
omp_set_lock(&send_locks[rank]);
	for (int i=last_sends[rank];i<mpi_sends[rank].size();i++)
	if (mpi_sends[rank][i].done==0)
	{
		MPI_Type_size(mpi_sends[rank][i].datatype, &s);
		mpi_sends[rank][i].dt_size=s;
		sizes[rank]+=s*mpi_sends[rank][i].count;
		ns[rank]++;
	}
	if (ns[rank])
	{
		bs[rank]=bufs[rank]=new char[sizes[rank]+8*ns[rank]];
		for (int j=last_sends[rank];j<mpi_sends[rank].size();j++)
		if (mpi_sends[rank][j].done==0)
		{
			s=mpi_sends[rank][j].dt_size;
			((int *)bs[rank])[0]=mpi_sends[rank][j].tag;
			bs[rank]+=4;
			((int *)bs[rank])[0]=mpi_sends[rank][j].count*s;
			bs[rank]+=4;
			memcpy(bs[rank],mpi_sends[rank][j].buf,mpi_sends[rank][j].count*s);
			bs[rank]+=mpi_sends[rank][j].count*s;
			mpi_sends[rank][j].done=1;
		}
	}
	last_sends[rank]=mpi_sends[rank].size()-1;
	if (last_sends[rank]<0) last_sends[rank]=0;
omp_unset_lock(&send_locks[rank]);
	// do send
	if (ns[rank])
	{
		sizes[rank]+=8*ns[rank];
		MPI_Isend(&sizes[rank],1,MPI_INTEGER,rank,mpi_rank*mpi_size+rank,MPI_COMM_WORLD,&rqs[2*rank+0]);
#pragma omp atomic
		ncalls[0]++;
		if (sizes[rank])
		{
			MPI_Isend(bufs[rank],sizes[rank],MPI_BYTE,rank,mpi_rank*mpi_size+rank,MPI_COMM_WORLD,&rqs[2*rank+1]);
#pragma omp atomic
			ncalls[0]++;
#pragma omp atomic
			ncalls[2]+=sizes[rank];
		}
	}
#pragma omp atomic
	mpi_times[0]+=GetTickCount()-t1;
omp_unset_lock(&proc_send_locks[rank]);
	return ns[rank];
}
int make_recvs(int rank)
{
	unsigned int t1=GetTickCount();
	// make sends
	int sum_recv=0;
	// receive data
	MPI_Status st;
	MPI_Recv(&sizes2[rank],1,MPI_INTEGER,rank,rank*mpi_size+mpi_rank,MPI_COMM_WORLD,&st);
#pragma omp atomic
	ncalls[1]++;
	if (sizes2[rank])
	{
		if (bufs2[rank]) delete [] bufs2[rank];
		bufs2[rank]=new char[sizes2[rank]];
		MPI_Recv(bufs2[rank],sizes2[rank],MPI_BYTE,rank,rank*mpi_size+mpi_rank,MPI_COMM_WORLD,&st);
#pragma omp atomic
		ncalls[1]++;
#pragma omp atomic
		ncalls[3]+=sizes2[rank];
		sum_recv+=sizes2[rank];
	}
	// parse data into mpi_received_msgs
omp_set_lock(&recv_locks[rank]);
	char *b=bufs2[rank];
	while ((b-bufs2[rank])<sizes2[rank])
	{
		int t=((int*)b)[0];
		int s=((int*)b)[1];
		b+=8;
		mpi_msg m={NULL,s,t,0};
		m.buf=new char[m.size];
		memcpy(m.buf,b,s);
		b+=s;
		mpi_received_msgs[rank].push_back(m);
	}	
omp_unset_lock(&recv_locks[rank]);
#pragma omp atomic
	mpi_times[1]+=GetTickCount()-t1;
	return sum_recv;
}
void do_check_and_send(int rank)
{
	int do_send=1;
	int id=omp_get_thread_num();
omp_set_lock(&send_locks[rank]);
	if (last_sends[rank]==(mpi_sends[rank].size()-1))
	{
		omp_unset_lock(&send_locks[rank]);
		return;
	}
omp_unset_lock(&send_locks[rank]);
	// make sends if all threads entered at least one receive after last send
omp_set_lock(&mpi_lock);
	do_send=1;
	int unprocessed=0;
	for (std::map<int,std::pair<int,int> >::iterator i=thread_wait_id.begin();i!=thread_wait_id.end();i++)
	{
		int ii=(*i).second.first;
		if (ii>=0)
		{
omp_set_lock(&recv_locks[rank]);
			if ((*i).second.second==rank)
			if (mpi_recvs[rank][ii].done==1)
				unprocessed=1;
omp_unset_lock(&recv_locks[rank]);
			if (unprocessed==1)
				break;
		}
		else
		{
			unprocessed=1;
			break;
		}
	}
	if (unprocessed==1)
		do_send=0;
omp_unset_lock(&mpi_lock);
	if (do_send)
		make_sends(rank);		
}
void mpi_set_end_thread()
{
	int do_s=0;
omp_set_lock(&mpi_lock);
	{
		thread_state.erase(omp_get_thread_num());
		thread_wait_id.erase(omp_get_thread_num());
		if (thread_state.size())
			do_s=1;
	}
omp_unset_lock(&mpi_lock);
	if (do_s && (mpi_size>1))
	for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
			do_check_and_send(i);
}
int _MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{
#ifndef USE_REPARTITIONING
	if (implicit3d==1)
	{
#endif
		ncalls[0]++;
		unsigned int t1=GetTickCount();
		int ret=MPI_Recv(buf,count,datatype,source,tag,comm,status);
		mpi_times[0]+=GetTickCount()-t1;
		int s;
		MPI_Type_size(datatype, &s);
		ncalls[2]+=count*s;
		return ret;
#ifndef USE_REPARTITIONING
	}
	else
	{
		unsigned int t1=GetTickCount();
		int id;
		// add request to array
		{
			mpi_req r={buf,count,datatype,tag,0,0};
			r.cv=new std::condition_variable();
			r.cv_m=new std::mutex();
omp_set_lock(&recv_locks[source]);
			mpi_recvs[source].push_back(r);
			id=mpi_recvs[source].size()-1;
omp_unset_lock(&recv_locks[source]);
		}
		// check received
		check_received_messages(source);
omp_set_lock(&recv_locks[source]);
		if (mpi_recvs[source][id].done)
		{
			omp_unset_lock(&recv_locks[source]);
			goto end;
		}
omp_unset_lock(&recv_locks[source]);
omp_set_lock(&mpi_lock);
		if ((*(thread_wait_id.find(omp_get_thread_num()))).second.first==-1) // two consequetive recvs
			(*(thread_state.find(omp_get_thread_num()))).second++;
		(*(thread_wait_id.find(omp_get_thread_num()))).second=std::pair<int,int>(id,source);
omp_unset_lock(&mpi_lock);
		// send if needed
		for (int i=0;i<mpi_size;i++)
			if (i!=mpi_rank)
				do_check_and_send(i);
// wait for receiving 
		while (1)
		{
omp_set_lock(&recv_locks[source]);
			if (mpi_recvs[source][id].done)
			{
				omp_unset_lock(&recv_locks[source]);
				break;
			}
omp_unset_lock(&recv_locks[source]);
			
if (omp_test_lock(&proc_locks[source])==false)
{		
			std::cv_status st;
			std::unique_lock<std::mutex> lk(mpi_recvs[source][id].cv_m[0]);
			st=mpi_recvs[source][id].cv->wait_for(lk,std::chrono::milliseconds(1));
			if (st==std::cv_status::timeout) 
				mpi_times[3]++;
			continue;
}
			// release threads waiting for received messages
			int found=0;
omp_set_lock(&mpi_lock);
			for (std::map<int,std::pair<int,int> >::iterator i=thread_wait_id.begin();i!=thread_wait_id.end();i++)
			{
				int ii=(*i).second.first;
				if (ii>=0)
				{
					if ((*i).second.second==source)
					{
omp_set_lock(&recv_locks[source]);
						if (mpi_recvs[source][ii].done==1)
						if (mpi_recvs[source][ii].notified==0)
						{
							found=1;
							mpi_recvs[source][ii].notified=1;
							mpi_recvs[source][ii].cv->notify_all();
						}
omp_unset_lock(&recv_locks[source]);
					}
				}
			}
omp_unset_lock(&mpi_lock);
			if (found==1)
			{
				omp_unset_lock(&proc_locks[source]);
				continue;
			}
			// send and wait for new messages
			for (int i=0;i<mpi_size;i++)
				if (i!=mpi_rank)
					do_check_and_send(i);
			if (mpi_recvs[source][id].done)
			{
				omp_unset_lock(&proc_locks[source]);
				break;
			}
			make_recvs(source);
			// check received messages
			check_received_messages(source);
omp_unset_lock(&proc_locks[source]);
		}
end:		
omp_set_lock(&mpi_lock);
		(*(thread_wait_id.find(omp_get_thread_num()))).second=std::pair<int,int>(-1,-1);
omp_unset_lock(&mpi_lock);
	}
	return 0;
#endif
}
int _MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
#ifndef USE_REPARTITIONING
	if (implicit3d==1)
	{
#endif
		ncalls[1]++;
		unsigned int t1=GetTickCount();
		int ret=MPI_Isend(buf,count,datatype,dest,tag,comm,request);
		mpi_times[2]+=GetTickCount()-t1;
		int s;
		MPI_Type_size(datatype, &s);
		ncalls[3]+=count*s;
		return ret;
#ifndef USE_REPARTITIONING
	}
	else
	{
		// add request to array
		{
			mpi_req s={buf,count,datatype,tag,0};
			int ss;
			MPI_Type_size(datatype, &ss);
			s.buf=(const void *)(new char[count*ss]);
			s.thread=omp_get_thread_num();
			memcpy((void *)s.buf,buf,count*ss);
omp_set_lock(&send_locks[dest]);
			mpi_sends[dest].push_back(s);
omp_unset_lock(&send_locks[dest]);
omp_set_lock(&mpi_lock);
			(*(thread_state.find(omp_get_thread_num()))).second++;
omp_unset_lock(&mpi_lock);
		}
omp_set_lock(&mpi_lock);
		(*(thread_wait_id.find(omp_get_thread_num()))).second=std::pair<int,int>(-2,-2); // mark that there was a send operation on this thread
omp_unset_lock(&mpi_lock);
	}
	return 0;
#endif
}
int _MPI_Wait(MPI_Request *request, MPI_Status *status)
{
#ifndef USE_REPARTITIONING
	if (implicit3d==1)
#endif
		return MPI_Wait(request,status);
	return 0;
}
int _MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
#ifndef USE_REPARTITIONING
	if (implicit3d==1)
	{
#endif
		ncalls[1]++;
		unsigned int t1=GetTickCount();
		int ret=MPI_Send(buf,count,datatype,dest,tag,comm);
		mpi_times[1]+=GetTickCount()-t1;
		int s;
		MPI_Type_size(datatype, &s);
		ncalls[3]+=count*s;
		return ret;
#ifndef USE_REPARTITIONING
	}
	else
	{
		MPI_Request r;
		MPI_Status s;
		_MPI_Isend(buf,count,datatype,dest,tag,comm,&r);
		return _MPI_Wait(&r,&s);
	}
	return 0;
#endif	
}
#endif
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
// simple matrix-vector mult
void vmul(double **K,double *x,double *r,int N)
{
#pragma omp parallel for
	for (int i=0;i<N;i++)
	{
		double v=0.0;
		double *row=K[i];
		double *xx = x;
		for (int j=0;j<N;j++,xx++,row++)
			v+=row[0]*xx[0];
		r[i]=v;
	}			
}
// matrix-vector mult for tridiag(Mat)+sum(T*Ci,i=0,..,3)
void vmul_T(double **K, double *x, double *r, int N);
// matrix-vector mult for 7-diag(Mat)+sum(T*Ci,i=0,..,3) (for 3d problems)
void vmul_T_3d(double **K, double *x, double *r, int N);
#ifdef USE_MPI
double parallel_sum(double v);
#endif
// MLN solver
template <int n> void solve_mln(double **K,double *B,double *ret,int N,double solve_thr,int maxiter, void (*vmul)(double **,double *, double *, int),int rb,int re)
{
#define qn(k) ((int)floor((float)((k)-1)/n))
#define rn(k) ((k)-n*qn(k))
	  // g,w,d arrays and c - from [iter-n to iter-1] => a[i]=a_base_iter[iter-i]
	  // arryas must be shifted right
	  double *p_q[n+1],*p_r,*p_g[n+1],*p_w[n+1],*p_u,*p_Au,*p_d[n+1],*p_zd,*p_zw,*p_g_precond[n+1],*p_u_precond,err;
	  int i,j;	  
	  int m=N;
	  double c[n+1],al,ro1,beta,norm;
	  int iter=1,sumiter=1; // iteration number from 1
	  // alloc aux arrays
		for (i=0;i<=n;i++) p_q[i]=new double [m];
		p_r=new double [m];
		for (i=0;i<=n;i++) p_g[i]=new double [m];
		for (i=0;i<=n;i++) p_g_precond[i]=p_g[i];
		for (i=0;i<=n;i++) p_w[i]=new double [m];
		p_u=new double [m];
		p_u_precond=p_u;
		p_Au=new double [m];
		for (i=0;i<=n;i++) p_d[i]=new double [m];
		p_zd=new double [m];
		p_zw=new double [m];
	 // initial values
	 // p_r = B-A*ret - residual
restart:
		vmul(K,ret,p_r,N);
#pragma omp parallel for
	 for (i=rb;i<re;i++)
		p_r[i]=B[i]-p_r[i];
      // p_g[0]=p_r
	  memcpy(p_g[1],p_r,m*sizeof(double));
	  // q[0]=p_r, q[1:n-1]=random(0:1)
	  memcpy(p_q[0],p_r,m*sizeof(double));
	  for (i=1;i<n;i++) p_q[i][i]=rand()/(double)RAND_MAX;
	  // p_w[0]=A*p_g~[0]
      vmul(K,p_g_precond[1],p_w[1],N);
	  // c[0]=q1*p_w[0]
	  c[1]=0.0;
	   for (i=rb;i<re;i++)
		  c[1]+=p_q[0][i]*p_w[1][i];
#ifdef USE_MPI
	c[1]=parallel_sum(c[1]);
#endif			
	// Compute norm - the dot product (p_r,p_r)
	  norm=0.0;
	 for (i=rb;i<re;i++)
			norm+=(p_r[i])*(p_r[i]);
#ifdef USE_MPI
	norm=parallel_sum(norm);
#endif			
	  if (fabs(norm/N)<solve_thr) goto denorm; // convergence - z is a solution
	  // iterations
	  do
	  {
		  if (rn(iter)==1)
		  {
			  //al=(q(rn(i))*r(i-1))/c(i-1)
			  al=0.0;
			  for (i=rb;i<re;i++)
				 al+=(p_q[rn(iter)-1][i])*(p_r[i]);
#ifdef USE_MPI
			al=parallel_sum(al);
#endif			
			  al/=c[1];
			  if (c[1]!=0.0)
			  {
			// u=r-al*w
#pragma omp parallel for
			for (i=rb;i<re;i++)
				p_u[i]=((p_r[i])-al*(p_w[1][i]));
			// ret=ret+al*g~
#pragma omp parallel for
			for (i=rb;i<re;i++)
				ret[i]=((ret[i])+al*(p_g_precond[1][i]));
			 // ro=-(Au~,u)/||Au~||2
		      vmul(K,p_u_precond,p_Au,N);
			  norm=0.0;
		 	  for (i=rb;i<re;i++)
				norm+=(p_Au[i])*(p_Au[i]);
#ifdef USE_MPI
			  norm=parallel_sum(norm);
#endif			
			  ro1=0.0;
		 	  for (i=rb;i<re;i++)
				ro1+=(p_Au[i])*(p_u[i]);
#ifdef USE_MPI
			  ro1=parallel_sum(ro1);
#endif			
			  ro1=-ro1/norm;
			  if (norm == 0.0)
				  goto restart;
			  // ret=ret-ro*u
#pragma omp parallel for
			 for (i=rb;i<re;i++)
				ret[i]=((ret[i])-ro1*(p_u_precond[i]));
			  }
			  else
				  goto restart;
		  }
		  else
		  {
			  //al=(q(rn(i))*u(i-1))/c(i-1)*
			  al=0.0;
			  for (i=rb;i<re;i++)
				 al+=(p_q[rn(iter)-1][i])*(p_u[i]);
#ifdef USE_MPI
			  al=parallel_sum(al);
#endif			
			  al/=c[1];
			  if (c[1]!=0.0)
			  {
			  if (rn(iter)<n)
			  {
				  // u=u-al*d[k-1]
#pragma omp parallel for
				 for (i=rb;i<re;i++)
					p_u[i]=((p_u[i])-al*(p_d[1][i]));
			  }
			  // ret=ret+ro[rn(k-1)+1]*al*g~
#pragma omp parallel for
			 for (i=rb;i<re;i++)
				ret[i]=((ret[i])+ro1*al*(p_g_precond[1][i]));
			  }
		  }

		  // recalc p_r
		  vmul(K,ret,p_r,N);
#pragma omp parallel for
		  for (i=rb;i<re;i++)
			 p_r[i]=(((B[i])-(p_r[i])));
		  if (rn(iter)<n)
		  {
			  // zd=u
			  memcpy(p_zd,p_u,m*sizeof(double));
			  // g=0
#pragma omp parallel for
			  for (i=rb;i<re;i++) p_g[0][i]=0.0;
			  // zw=0
#pragma omp parallel for
			  for (i=rb;i<re;i++) p_zw[i]=0.0;
			  
			  if (qn(iter)>=1)
			  for (j=iter-n;j<(qn(iter)*n-1);j++)
			  {
				  //beta=-q[rn(j+1)]*zd/c[s]
				  beta=0.0;
				  for (i=rb;i<re;i++)
					 beta-=(p_q[rn(j+1)-1][i])*(p_zd[i]);
#ifdef USE_MPI
				  beta=parallel_sum(beta);
#endif			
				  beta/=c[iter-j];
				  if (c[iter-j]!=0.0)
				  {
				  // zd+=beta*d[s]
#pragma omp parallel for
				  for (i=rb;i<re;i++)
				      p_zd[i]=((p_zd[i])+beta*(p_d[iter-j][i]));
				  //g[k]+=beta*g[s]
#pragma omp parallel for
				  for (i=rb;i<re;i++)
				      p_g[0][i]=((p_g[0][i])+beta*(p_g[iter-j][i]));
				  //zw+=beta*w[s]
#pragma omp parallel for
				  for (i=rb;i<re;i++)
				      p_zw[i]=((p_zw[i])+beta*(p_w[iter-j][i]));
				  }
			  }
		  	  // zw=r+ro[gn(iter)+1]*zw
#pragma omp parallel for
			  for (i=rb;i<re;i++)
			      p_zw[i]=((p_r[i])+ro1*(p_zw[i]));
			  // beta=-(q[1]*zw/c[gn(k)*n])
			  beta=0.0;
			  for (i=rb;i<re;i++)
				 beta-=(p_q[0][i])*(p_zw[i]);
#ifdef USE_MPI
			  beta=parallel_sum(beta);
#endif			
			  beta/=c[iter-qn(iter)*n];
			  if (c[iter - qn(iter)*n] == 0)
				  goto restart;
			  //zw+=beta*w[gn(k)*n]
#pragma omp parallel for
			 for (i=rb;i<re;i++)
			      p_zw[i]=((p_zw[i])+beta*(p_w[iter-(qn(iter)*n)][i]));
			  //g[k]=g[k]+zw+(beta/ro(gn(k)+1])*g[gn(k)*n]
#pragma omp parallel for
			  for (i=rb;i<re;i++)
			      p_g[0][i]=((p_g[0][i])+(p_zw[i])+(beta/ro1)*(p_g[iter-(qn(iter)*n)][i]));
			  
			  for (j=(qn(iter)*n+1);j<=iter-1;j++)
			  {
				  //beta=-q[rn(j+1)]*zw/c[s]
				  beta=0.0;
				  for (i=rb;i<re;i++)
					 beta-=(p_q[rn(j+1)-1][i])*(p_zw[i]);
#ifdef USE_MPI
				  beta=parallel_sum(beta);
#endif			
				  beta/=c[iter-j];
				  if (c[iter-j]!=0.0)
				  {
				  //g[k]+=beta*g[s]
#pragma omp parallel for
				  for (i=rb;i<re;i++)
				      p_g[0][i]=((p_g[0][i])+beta*(p_g[iter-j][i]));
				  //zw+=beta*w[s]
#pragma omp parallel for
				  for (i=rb;i<re;i++)
				      p_zw[i]=((p_zw[i])+beta*(p_d[iter-j][i]));
				  }
			  }
			  // d[k]=zw-u
#pragma omp parallel for
			  for (i=rb;i<re;i++)
			      p_d[0][i]=((p_zw[i])-(p_u[i]));
			  // c[k]=q[rn(k+1)]*d[k]
			  c[0]=0.0;
			  for (i=rb;i<re;i++)
				c[0]+=(p_q[rn(iter+1)-1][i])*(p_d[0][i]);
#ifdef USE_MPI
			 c[0]=parallel_sum(c[0]);
#endif			
			  // w[k]=Ag~[k]
		      vmul(K,p_g_precond[0],p_w[0],N);
		  }
		  else
		  {
			  // beta=-(q[1]*r/c[qn(k)*n])
			  beta=0.0;
			  for (i=rb;i<re;i++)
				 beta-=(p_q[0][i])*(p_r[i]);
#ifdef USE_MPI
			  beta=parallel_sum(beta);
#endif			
			  beta/=c[iter-qn(iter)*n];
			  if (c[iter-qn(iter)*n]==0.0)
				  beta=0.0;
			  //zw=r+betas[qn(k)*n]*w[qn(k)*n]
#pragma omp parallel for 
			  for (i=rb;i<re;i++)
			      p_zw[i]=((p_r[i])+beta*(p_w[iter-(qn(iter)*n)][i]));
			  //g[k]=zw+(betas[qn(k)*n]/ro(gn(k)+1])*g[qn(k)*n]
#pragma omp parallel for 
			  for (i=rb;i<re;i++)
			      p_g[0][i]=((p_zw[i])+(beta/ro1)*(p_g[iter-(qn(iter)*n)][i]));

			  for (j=(qn(iter)*n+1);j<=iter-1;j++)
			  {
				  //beta=-q[rn(j+1)]*zw/c[s]
				  beta=0.0;
				  for (i=rb;i<re;i++)
					 beta-=(p_q[rn(j+1)-1][i])*(p_zw[i]);
#ifdef USE_MPI
			 	 beta=parallel_sum(beta);
#endif			
				  beta/=c[iter-j];
				  if (c[iter-j]!=0.0)
				  {
				  //g[k]+=beta*g[s]
#pragma omp parallel for 
				  for (i=rb;i<re;i++)
				      p_g[0][i]=((p_g[0][i])+beta*(p_g[iter-j][i]));
				  //zw+=beta*w[s]
#pragma omp parallel for 
				  for (i=rb;i<re;i++)
				      p_zw[i]=((p_zw[i])+beta*(p_d[iter-j][i]));
				  }
			  }
			  // w[k]=Ag~[k]
		      vmul(K,p_g_precond[0],p_w[0],N);
			  // c[k]=q[rn(k+1)]*w[k]
			  c[0]=0.0;
			  for (i=rb;i<re;i++)
				c[0]+=(p_q[rn(iter+1)-1][i])*(p_w[0][i]);
#ifdef USE_MPI
			  c[0]=parallel_sum(c[0]);
#endif			
		  }
		  // right shift p_g,p_d,p_w and arrays
		  {
			  double *sgn=p_g[n],*sdn=p_d[n],*swn=p_w[n],*sgpn=p_g_precond[n], scn=c[n]; 
			  for (i=n;i>0;i--)
			  {
				  p_g[i]=p_g[i-1];
				  p_g_precond[i]=p_g_precond[i-1];
				  p_d[i]=p_d[i-1];
				  p_w[i]=p_w[i-1];
				  c[i]=c[i-1];
			  }
			  p_g[0]=sgn;
			  p_g_precond[0]=sgpn;
			  p_d[0]=sdn;
			  p_w[0]=swn;
			  c[0]=scn;
		  }
		 // check p_r norm
		 norm=0.0;
		for (i=rb;i<re;i++)
				norm+=(p_r[i])*(p_r[i]);
#ifdef USE_MPI
		norm=parallel_sum(norm);
#endif			
		 if (fabs(norm/N)<solve_thr) goto denorm; // convergence - z is a solution
		 iter++;
		 sumiter++;
#ifdef USE_MPI		  
#pragma omp atomic
		 mpi_times[4]++;
#endif		  
		 if (debug_level==2)
			if (log_file) fprintf(log_file,"iter %d norm %g\n",iter,norm);
		 if (iter==maxiter)
			goto denorm;
	  }
	  while (1);
denorm:
	  // check
	  err=0;
  	  vmul(K,ret,p_r,N);
	  for (i=rb;i<re;i++)
	  {
 		    p_r[i]-=B[i];
			err+=p_r[i]*p_r[i];
	  }
#ifdef USE_MPI
	 err=parallel_sum(err);
#endif			
	  if (debug_level==1)
		if (log_file) fprintf(log_file,"error: %g, niter: %d\n",err,sumiter);
	  // cleanup
	  delete [] p_r;
	  delete [] p_u;
	  delete [] p_Au;
	  delete [] p_zd;
	  delete [] p_zw;
	  for (i=0;i<n+1;i++)
	  {
		  delete [] p_q[i];
		  delete [] p_g[i];
		  delete [] p_w[i];
		  delete [] p_d[i];
	  }
}
// calc row with coefs - inner integrals for t0,t1 
double _kb_row2_fixed_coefs(double gtj,double gtjma, double *_coefs,int n, double alpha,int right=0)
{
	double sum = 0.0, v1;
	double *c=_coefs;
	double i = 0;
	int ii = 0;
	if (right==0)
		v1 = gtjma;
	else
		v1=1.0;
	do
	{
		sum += v1*(c++)[0];
		i+=1.0;
		ii++;
		if (right==0)
			v1 *= -(-alpha - i + 1.0) / (i*gtj);
		else
			v1 *= -(-alpha - i + 1.0)*gtj / i;
	} while (ii!=n);
	return sum;
}
#define idx(i,j,k) ((i)+(j)*(sN+2)+(k)*(sN+2)*(_M+2))
class row {
public:
	double *A;
	int x,y,c;
	row()
	{
		A=NULL;
		x=y=0;
		c=-1;
	}
	void set(double *_A,int _x,int _y,int _c)
	{
		A=_A;
		x=_x;
		y=_y;
		c=_c;
	}
	inline double& operator [] (int i)
	{
		if (c == 0)
			return A[idx(i, x, y)];
		if (c == 1)
			return A[idx(x, i, y)];
		if (c == 2)
			return A[idx(x, y, i)];
		return A[i];
	}
};
///////////// OpenCL code for time derivative summing //////////////////////
int ocl_vector=0; // (1,2,3) - <type - 1: double, 2:float, 3:half>16 vectorization (sum_alg0 do not work in vectorized mode),
				  // 4 - tensor vectorization (only sum_alg 2) - datatype - half
int oclBS=1; // BS%16 must = 0 for vector mode
#ifdef OCL
#define ocl_sum_text_vector(T,S,VT) "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#pragma OPENCL EXTENSION cl_khr_fp16 : enable \n\
#define sum_alg %d\n\
#define OS %d\n\
#define BS %d\n\
double _kb_row2_fixed_coefs(double gtj, __global "#T" *_coefs,int n,double alpha_,int ll,__local "#T" *v1s)\n\
{\n\
	"#T" sum = 0.0, v1,mult;\n\
	__local "#VT" *v1vs=(__local "#VT" *)v1s;\n\
	__global "#VT" *cs=(__global "#VT" *)_coefs;\n\
	union { "#T" s["#S"]; "#VT" v; } v;\n\
	mult = powr(gtj, -alpha_);\n\
	for (int ii=1;ii<=n;ii+=BS)\n\
	{\n\
    	    barrier(CLK_LOCAL_MEM_FENCE);\n\
	    v1s[ll]=mult;\n\
	    for (int iii=ii;iii<ii+ll;iii++)\n\
			v1s[ll]*=-(-alpha_ - iii + 1.0) / (iii*gtj);\n\
		if (!isfinite(v1s[ll])) v1s[ll]=0.0;\n\
	    barrier(CLK_LOCAL_MEM_FENCE);\n\
	    mult=v1s[BS-1]*(-(-alpha_ - (ii+BS) + 1.0) / ((ii+BS)*gtj));\n\
	    for (int iii=0;iii<BS;iii+="#S")\n\
	    {\n\
			v.v=cs[(ii+iii-1)/"#S"]*v1vs[iii/"#S"];\n\
			for (int jj=0;jj<"#S";jj++) sum+=v.s[jj];\n\
    	}\n\
    }\n\
	return sum;\n\
}\n\
__kernel void Sum(__global double *S,__global double *old,int old_size,__global double *tau,__global double *Htmp,__global double *row2_precalc,__global double *kb_3_cache,double sum_param,double gtj,double gt,double alpha,int tstep)\n\
{\n\
	int id=get_global_id(0);\n\
	int ll=get_local_id(0);\n\
	__local "#T" cc[BS],taus[BS];\n\
	__local "#T" v1s[BS]; \n\
	__local "#T" diffs[BS];\n\
	"#T" diffs1[BS];\n\
	double time_sum = 0;\n\
	__local "#VT" *kv=(__local "#VT" *)&cc[0], *tts=(__local "#VT" *)&taus[0]; \n\
	__local "#VT" *diff=(__local "#VT" *)&diffs[0];\n\
	"#VT" *diff1=("#VT" *)&diffs1[0];\n\
	if (old_size>=2)\n\
	{\n\
		if (sum_alg!=2)\n\
		for (int t = old_size - 2;t >=0 ;t-=BS)\n\
		{\n\
			int br=0;\n\
			barrier(CLK_LOCAL_MEM_FENCE);\n\
			if ((t-ll)>=0) \n\
			{\n\
				cc[ll]=kb_3_cache[t-ll]; \n\
				taus[ll]=tau[t-ll];\n\
			}\n\
			barrier(CLK_LOCAL_MEM_FENCE);\n\
			for (int tt=0;tt<BS;tt++) if ((t-tt)>=0) diffs1[tt]=old[((t-tt) + 1)*OS+id] - old[(t-tt)*OS+id]; \n\
			union { "#T" s["#S"]; "#VT" v; } v;\n\
			for (int tt=0;tt<BS;tt+="#S")\n\
			{\n\
				v.v=kv[tt/"#S"]*diff1[tt/"#S"]/tts[tt/"#S"];\n\
				for (int jj=0;jj<"#S";jj++) if ((t-tt-jj)>=0) time_sum+=v.s[jj];\n\
#if "#S"==1 \n\
				if (sum_alg==1) // restricted summing\n\
				    if (fabs(kv[tt])<sum_param)\n\
					br++;\n\
#endif\n\
			}\n\
			if (br==BS)\n\
			    break;\n\
		}\n\
		if (sum_alg==2) \n\
		{\n\
			"#T" mult=gt;\n\
			"#T" cur_diff = old[OS+id] - old[id];\n\
			for (int j=0;j<(int)sum_param;j+=BS)\n\
			{\n\
				barrier(CLK_LOCAL_MEM_FENCE);\n\
				cc[ll]=row2_precalc[j+ll];\n\
				if (!isfinite(cc[ll])) cc[ll]=0.0;\n\
				diffs[ll]=mult;\n\
				for (int jj=0;jj<j+ll;jj++) diffs[ll]*=gt;\n\
				if (!isfinite(diffs[ll])) diffs[ll]=0.0;\n\
				barrier(CLK_LOCAL_MEM_FENCE);\n\
				for (int jj=0;jj<BS;jj+="#S")\n\
					((__global "#VT" *)&Htmp[id*(int)sum_param])[(j+jj)/"#S"]+=diff[jj/"#S"]*kv[jj/"#S"]*cur_diff; \n\
			}\n\
			time_sum=_kb_row2_fixed_coefs(gtj,(__global "#T" *)&Htmp[id*(int)sum_param],(int)sum_param,alpha,ll,v1s)/tau[tstep]; \n\
		} \n\
	}\n\
	S[id]=time_sum;\n\
}"
const char *ocl_sum_text_double=ocl_sum_text_vector(double,1,double);
const char *ocl_sum_text_float=ocl_sum_text_vector(float,1,float);
const char *ocl_sum_text_half=ocl_sum_text_vector(half,1,half);
const char *ocl_sum_text_vector_double=ocl_sum_text_vector(double,16,double16);
const char *ocl_sum_text_vector_float=ocl_sum_text_vector(float,16,float16);
const char *ocl_sum_text_vector_half=ocl_sum_text_vector(half,16,half16);
const char *ocl_sum_text_vector_double8=ocl_sum_text_vector(double,8,double8);
const char *ocl_sum_text_vector_float8=ocl_sum_text_vector(float,8,float8);
const char *ocl_sum_text_vector_half8=ocl_sum_text_vector(half,8,half8);
const char *ocl_sum_text_vector_double4=ocl_sum_text_vector(double,4,double4);
const char *ocl_sum_text_vector_float4=ocl_sum_text_vector(float,4,float4);
const char *ocl_sum_text_vector_half4=ocl_sum_text_vector(half,4,half4);
const char *ocl_sum_text_vector_double2=ocl_sum_text_vector(double,2,double2);
const char *ocl_sum_text_vector_float2=ocl_sum_text_vector(float,2,float2);
const char *ocl_sum_text_vector_half2=ocl_sum_text_vector(half,2,half2);
/////////////////////////////// CUDA code ///////////////////////////////////////////
#ifdef CUDA
#define cuda_sum_text_vector(T,S,H,VT) "\n\
#include \"cuda_fp16.h\"\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define sum_alg %d\n\
#define OS %d\n\
#define BS %d\n\
#if "#H"==1 \n\
#if "#S"==2 \n\
inline __device__ "#T"2 operator*("#T"2 b, "#T" a)\n\
{\n\
    return make_"#T"2(a * b.x, a * b.y);\n\
}\n\
#endif\n\
#endif\n\
#if "#H"==0 \n\
#if "#S"==2 \n\
inline __device__ "#T"2 operator*("#T"2 b, "#T"2 a)\n\
{\n\
    return make_"#T"2(a.x * b.x, a.y * b.y);\n\
}\n\
inline __device__ void operator+=("#T"2 &b, "#T"2 a)\n\
{\n\
    b.x+=a.x;b.y+=a.y;\n\
}\n\
inline __device__ "#T"2 operator*("#T"2 b, "#T" a)\n\
{\n\
    return make_"#T"2(a * b.x, a * b.y);\n\
}\n\
inline __device__ "#T"2 operator/("#T"2 b, "#T"2 a)\n\
{\n\
    return make_"#T"2(b.x / a.x, b.y / a.y);\n\
}\n\
#endif\n\
#if "#S"==4 \n\
inline __device__ "#T"4 operator*("#T"4 b, "#T"4 a)\n\
{\n\
    return make_"#T"4(a.x * b.x, a.y * b.y,a.z*b.z,a.w*b.w);\n\
}\n\
inline __device__ void operator+=("#T"4 &b, "#T"4 a)\n\
{\n\
    b.x+=a.x;b.y+=a.y;b.z+=a.z;b.w+=a.w;\n\
}\n\
inline __device__ "#T"4 operator*("#T"4 b, "#T" a)\n\
{\n\
    return make_"#T"4(a * b.x, a * b.y,a*b.z,a*b.w);\n\
}\n\
inline __device__ "#T"4 operator/("#T"4 b, "#T"4 a)\n\
{\n\
    return make_"#T"4(b.x / a.x, b.y / a.y,b.z/a.z,b.w/a.w);\n\
}\n\
#endif\n\
#endif\n\
__device__ double _kb_row2_fixed_coefs(double gtj,  "#T" *_coefs,int n,double alpha_,int ll,"#T" *v1s)\n\
{\n\
	double sum = 0.0;\n\
	"#T" mult;\n\
	"#VT" *v1vs=("#VT" *)v1s;\n\
	 "#VT" *cs=( "#VT" *)_coefs;\n\
	mult = pow((double)gtj, (double)-alpha_);\n\
	for (int ii=1;ii<=n;ii+=BS)\n\
	{\n\
	    __syncthreads();\n\
	    v1s[ll]=mult;\n\
	    for (int iii=ii;iii<ii+ll;iii++)\n\
		v1s[ll]=v1s[ll]*("#T")(-(-alpha_ - iii + 1.0) / (iii*gtj));\n\
	    if (!isfinite((double)v1s[ll])) v1s[ll]=0.0;\n\
	    __syncthreads();\n\
	    mult=v1s[BS-1]*("#T")(-(-alpha_ - (ii+BS) + 1.0) / ((ii+BS)*gtj));\n\
	    for (int iii=0;iii<BS;iii+="#S")\n\
	    {\n\
			"#VT" v=cs[(ii+iii-1)/"#S"]*v1vs[iii/"#S"];\n\
#if "#S"==1\n\
			sum+=(double)v;\n\
#endif\n\
#if "#S"==2\n\
			sum+=(double)(v.x+v.y);\n\
#endif\n\
#if "#S"==4\n\
			sum+=(double)(v.x+v.y+v.z+v.w);\n\
#endif\n\
    	    }\n\
        }\n\
        return sum;\n\
}\n\
extern \"C\" __global__ void Sum( double *S, double *old,int old_size, double *tau, double *Htmp, double *row2_precalc, double *kb_3_cache,double sum_param,double gtj,double gt,double alpha,int tstep)\n\
{\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x; \n\
	int ll=threadIdx.x; \n\
	__shared__ __align__(8) "#T" cc[BS],taus[BS];\n\
	__shared__ __align__(8) "#T" v1s[BS]; \n\
	__shared__ __align__(8) "#T" diffs[BS];\n\
	"#T" diffs1[BS];\n\
	double time_sum = 0.0;\n\
	"#VT" *kv=("#VT" *)&cc[0], *tts=("#VT" *)&taus[0]; \n\
	"#VT" *diff=("#VT" *)&diffs[0];\n\
	"#VT" *diff1=("#VT" *)&diffs1[0];\n\
	if (old_size>=2)\n\
	{\n\
		if (sum_alg!=2)\n\
		for (int t = old_size - 2;t >=0 ;t-=BS)\n\
		{\n\
			int br=0;\n\
			__syncthreads();\n\
			if ((t-ll)>=0) \n\
			{\n\
				cc[ll]=kb_3_cache[t-ll]; \n\
				taus[ll]=tau[t-ll];\n\
			}\n\
			else\n\
			{\n\
				cc[ll]=0.0;\n\
				taus[ll]=1.0;\n\
			}\n\
			__syncthreads();\n\
			for (int tt=0;tt<BS;tt++) if ((t-tt)>=0) diffs1[tt]=old[((t-tt) + 1)*OS+id] - old[(t-tt)*OS+id]; else diffs1[tt]=0.0;\n\
			for (int tt=0;tt<BS;tt+="#S")\n\
			{\n\
				"#VT" v=kv[tt/"#S"]*diff1[tt/"#S"]/tts[tt/"#S"];\n\
#if "#S"==1\n\
				time_sum+=(double)v;\n\
				if (sum_alg==1) // restricted summing\n\
				    if (fabs(kv[tt])<sum_param)\n\
					br++;\n\
#endif\n\
#if "#S"==2\n\
				time_sum+=(double)(v.x+v.y);\n\
#endif\n\
#if "#S"==4\n\
				time_sum+=(double)(v.x+v.y+v.z+v.w);\n\
#endif\n\
			}\n\
			if (br==BS)\n\
			    break;\n\
		}\n\
		if (sum_alg==2) \n\
		{\n\
			"#T" mult=gt;\n\
			"#T" cur_diff = old[OS+id] - old[id];\n\
			if (!isfinite((double)cur_diff)) cur_diff=0.0;\n\
			for (int j=0;j<(int)sum_param;j+=BS)\n\
			{\n\
				__syncthreads();\n\
				cc[ll]=row2_precalc[j+ll];\n\
				if (!isfinite((double)cc[ll])) cc[ll]=0.0;\n\
				diffs[ll]=mult;\n\
				for (int jj=0;jj<j+ll;jj++) diffs[ll]*=mult;\n\
				if (!isfinite((double)diffs[ll])) diffs[ll]=0.0;\n\
				__syncthreads();\n\
				for (int jj=0;jj<BS;jj+="#S")\n\
					(( "#VT" *)&Htmp[id*(int)sum_param])[(j+jj)/"#S"]+=diff[jj/"#S"]*kv[jj/"#S"]*cur_diff; \n\
			}\n\
			time_sum=_kb_row2_fixed_coefs(gtj,( "#T" *)&Htmp[id*(int)sum_param],(int)sum_param,alpha,ll,v1s)/tau[tstep]; \n\
		} \n\
	}\n\
	S[id]=time_sum;\n\
}"
const char *cuda_sum_text_double=cuda_sum_text_vector(double,1,0,double);
const char *cuda_sum_text_float=cuda_sum_text_vector(float,1,0,float);
const char *cuda_sum_text_half=cuda_sum_text_vector(half,1,1,half);
const char *cuda_sum_text_vector_double4=cuda_sum_text_vector(double,4,0,double4);
const char *cuda_sum_text_vector_float4=cuda_sum_text_vector(float,4,0,float4);
const char *cuda_sum_text_vector_double2=cuda_sum_text_vector(double,2,0,double2);
const char *cuda_sum_text_vector_float2=cuda_sum_text_vector(float,2,0,float2);
const char *cuda_sum_text_vector_half2=cuda_sum_text_vector(half,2,1,half2);
// use of tensor arithmetics - only sum_alg 2, BS should be =n=16
const char *cuda_sum_text_tensor = "\n\
#pragma OPENCL EXTENSION %s : enable \n\
#define sum_alg %d\n\
#define OS %d\n\
#define BS %d\n\
#include \"cuda_fp16.h\" \n\
#include \"mma.h\" \n\
using namespace nvcuda;\n\
__device__ double _kb_row2_fixed_coefs(double gtj,  half *_coefs,double alpha_,int ll,half *v1s,int id)\n\
{\n\
	__shared__ __align__(8) float hres[BS*BS];\n\
	__shared__ __align__(8) half A[BS*BS];\n\
	// v1s -> 16x16 matrices in form ((a1,a1*a2,a1*a2*a3,...),(0,...,0),...)\n\
	__syncthreads();\n\
	v1s[ll]=pow(gtj, -alpha_);\n\
	for (int iii=1;iii<1+ll;iii++)\n\
		v1s[ll]=v1s[ll]*(half)(-(-alpha_ - iii + 1.0) / (iii*gtj));\n\
	if (!isfinite((double)v1s[ll])) v1s[ll]=0.0;\n\
	// A - _coefs\n\
	for (int i=0;i<BS;i++) A[ll*BS+i]=_coefs[i];\n\
	__syncthreads();\n\
       // Declare the fragments\n\
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;\n\
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;\n\
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;\n\
       // Initialize the output to zero\n\
       wmma::fill_fragment(c_frag, 0.0f);\n\
       // Load the inputs\n\
       wmma::load_matrix_sync(a_frag, A, 16);\n\
       wmma::load_matrix_sync(b_frag, v1s, 16);\n\
       // Perform the matrix multiplication\n\
       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);\n\
       // Store the output\n\
       wmma::store_matrix_sync(hres, c_frag, 16, wmma::mem_row_major);\n\
       return hres[ll*BS];\n\
}\n\
extern \"C\" __global__ void Sum( double *S, double *old,int old_size, double *tau, double *Htmp, double *row2_precalc, double *kb_3_cache,double sum_param,double gtj,double gt,double alpha,int tstep)\n\
{\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x; \n\
	int ll=threadIdx.x; \n\
	__shared__ __align__(8) half cc[BS];\n\
	__shared__ __align__(8) half v1s[BS*BS]; \n\
	__shared__ __align__(8) half diffs[BS];\n\
	double time_sum = 0.0;\n\
	if (old_size>=2)\n\
	{\n\
		half mult=gt;\n\
		half cur_diff = old[OS+id] - old[id];\n\
		if (!isfinite((double)cur_diff)) cur_diff=0.0;\n\
		for (int j=0;j<(int)sum_param;j+=BS)\n\
		{\n\
			__syncthreads();\n\
			cc[ll]=row2_precalc[j+ll];\n\
			if (!isfinite((double)cc[ll])) cc[ll]=0.0;\n\
			diffs[ll]=mult;\n\
			for (int jj=0;jj<j+ll;jj++) diffs[ll]*=mult;\n\
			if (!isfinite((double)diffs[ll])) diffs[ll]=0.0;\n\
			__syncthreads();\n\
			for (int jj=0;jj<BS;jj++)\n\
				(( half *)&Htmp[id*(int)sum_param])[j+jj]+=diffs[jj]*cc[jj]*cur_diff; \n\
		}\n\
		time_sum=_kb_row2_fixed_coefs(gtj,( half *)&Htmp[id*(int)sum_param],alpha,ll,v1s,id)/tau[tstep]; \n\
	}\n\
	S[id]=time_sum;\n\
}";
#endif
#endif
////////////////////////////////////////////
// time-space-fractional solver 1d /////////////
// Da_tH=div(CvDb_zH) - div((v*cv/k)Db_zC) - div((kt*cv/k)Db_zT)-S/C(H)
// sigma Da_tC=div(D Db_zC)-vdiv(CDb_zC)-ktdiv(CDb_zT)
// Ct Da_tT=div(lambda Db_zT)-v*po*Cp div(TDb_zC)-kt*po*Cp div (T Db_zT)
////////////////////////////////////////////
class H_solver {
public:
	double *b_U; // pressure head
	double *b_C; // concentration
	double *b_T; // temperature
	double *b_Utimesum;
	double *b_Ctimesum;
	double *b_Ttimesum;
	std::vector<double> *Htmp;
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
		row U,C,T;
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double *Al,*sAl; // alpha coefficients
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double *Bt,*sBt; // beta coefficients
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double *Om,*sOm; // right part
#if defined(_OPENMP)||defined(USE_MPI)
	static double *F; // function values
#endif	
#if defined(_OPENMP)||defined(USE_MPI)
	static double *BVK; // values of F
#endif	
#pragma omp threadprivate(U,C,T,Al,Bt,Om,F,BVK)	
	std::vector<double*> Als,Bts,Oms,Fs,BVKs; // thread local variables

#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double **Mat,**sMat;
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double **Mat1d; //full matrix for implicit space fractional solver	
#pragma omp threadprivate(Mat, Mat1d)	
	std::vector<double **>Mats,Mat1ds;// thread local variables
	double **Mat_3d;
	// matrices for implicit with decomposition on toeplitz matrices (for fixed steps)
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double **Mcoefs,**Mcoefs2; // coefficient matrices (3xN)
	std::vector<double **>Mcoefss,Mcoefs2s;// thread local variables

	double **Mcoefs_3d[3],**Mcoefs2_3d[3]; // coefficient matrices (3xN)
	double *Tm[3]; // Toeplitz matrix (one row per dimension)
	int vart_main;
	// row scaling coefficients
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	double *RS,*sRS;
#pragma omp threadprivate(Mcoefs,Mcoefs2,RS)	
	std::vector<double *>RSs;// thread local variables
	// integral matrices coefficients for left-sided derivative
	std::vector<double *> Tm_diagonals[3];
	std::vector< std::vector<double> > Tm_coefs[3];
	int Tm_ndiags[3];
	// integral matrices coefficients for right-sided derivative
	std::vector<double *> r_Tm_diagonals[3];
	std::vector< std::vector<double> > r_Tm_coefs[3];
	int r_Tm_ndiags[3];

	std::vector<double*> oldH,oldC,oldT; // for time-fractional derivatives
	std::vector<double> tau,Time; // time steps and time moments
	// for 3d
	double **rp_mult;
	double **spr;
	// steps
	int tstep; // current time step
	double L,sL; // domain length 
	double *dL,*sdL,*Z,*sZ; //space variable step lengths and Z values
	double *gZ,*gZg; // to precalc g(Z[i]) and pow(g(Z[i],-gamma)
	double minDL,maxDL; // min/max for z step
	// H equation and common
	double alpha, gamma; // alpha - time derivative power, gamma - space derivative power
	double H0; // initial condition for H
	double Da,Dg; 
	double v; // root layer depth
	double Tr,Ev; // transpiration and evaporation rates
	double k; // filtration coefficient in saturated soil
	double nu; // chemical osmosis coefficient
	double kt; // thermal osmosis coefficient
	// C equation
	double sigma; // average soil porosity
	double D; // diffusion coefficient
	double C0; // initial condition for C
	double irr_C; // irrigation water salt concentration for upper boundary condition
	// T equation
	double Ct; // volumetric heat capacity
	double lambda; // coefficient of thermal conductivity
	double Cp; // specific heat capacity
	double aT; // daily average air temperature
	double T0; // initial condition for T
	// irrigation scheduling 
	double min_wetness; // min wetness to apply irrigation
	double max_wetness; // wetness to be after irrigation
	double irr_volume; // volume of irrigation water
	double irr_time; // time range to apply irrigation water
	double p_soil,p_water; // densities of soil and pore water
	int irr_start_step; 
	int bottom_cond; 
	int analytic_test;
	int rp_form,cv_form,k_form,no_irr;
	int equation; // equation now solved
	int mode; // 1 - 1D, 3 - 3D
	int is_r_C; // roots water uptake dependent on salt concentration
	double rC_thr;
	double rC_50;
	double rC_exp;
	// for saving
	double kkk0;
	// for 3D
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	int m3d_x,m3d_y,m3d_c; // current row
#if defined(_OPENMP)||defined(USE_MPI)
	static 
#endif
	int rb,re; // start-end cells in row to process
#pragma omp threadprivate(m3d_x,m3d_y,m3d_c,rb,re)	
	double Lx,Ly; // lengths in (x,y)
	double *dLx,*dLy,*X,*Y; // steps in (x,y) and values
	double *gX,*gXg,*gY,*gYg; // to precalc g(X/Y[i]) and pow(g(X/Y[i],-gamma)
	double XYforceexp; // exp for force of attraction for XY variable steps
	int Nrows,Nplants,Nsprinklers,Nsprinklers_rows;
	double Poffs,Spr_offs;
	double Ploffs, Spr_spr_offs;
	double rootR, sprinklerR,betw_rows, betw_spr_rows, betw_pl, betw_spr;
	
	int second_step;
	double irr_am;
	int first_rp,first_spr;
	int s_is_c,s_is_t;
#ifdef OCL
	// OpenCL works only for single equation model
	OpenCL_program *prg;
	OpenCL_commandqueue *queue;
	OpenCL_prg *prog;
	OpenCL_kernel *kSum;
	OpenCL_buffer *boldU,*bS,*bHtmp,*bK,*bCoefs,*btau;
	double *Sums;
	int old_alloced;
	int ocl_sol_size;
	// initialize OpenCL
	void init_opencl()
	{
		int iv;
		ocl_sol_size=(sN+2)*(_M+2)*(K+2);
		if ((sum_alg==2)&&(oclBS!=1))
		{
		    if ((((int)sum_param)%oclBS)!=0)
		    {
			printf("for sum_alg=2 sum_param%%oclBS must be =0\n");
			exit(0);
		    }
		    if ((ocl_vector!=0)&&((oclBS%16)!=0))
		    {
			printf("for sum_alg=2 and vectorized algorithm oclBS%%16 must be =0\n");
			exit(0);
		    }
		}
		if (ocl_vector==4)
		{
		    oclBS=16;
		    sum_param=16;
		}
		if ((ocl_sol_size%oclBS)!=0)
			ocl_sol_size=((ocl_sol_size/oclBS)+1)*oclBS;
		const char *tt;
#ifndef CUDA
		tt=ocl_sum_text_double;
		if (ocl_vector==1) tt=ocl_sum_text_vector_double;
		if (ocl_vector==2) tt=ocl_sum_text_vector_float;
		if (ocl_vector==3) tt=ocl_sum_text_vector_half;
		if (ocl_vector==5) tt=ocl_sum_text_vector_double8;
		if (ocl_vector==6) tt=ocl_sum_text_vector_float8;
		if (ocl_vector==7) tt=ocl_sum_text_vector_half8;
		if (ocl_vector==8) tt=ocl_sum_text_vector_double4;
		if (ocl_vector==9) tt=ocl_sum_text_vector_float4;
		if (ocl_vector==10) tt=ocl_sum_text_vector_half4;
		if (ocl_vector==11) tt=ocl_sum_text_vector_double2;
		if (ocl_vector==12) tt=ocl_sum_text_vector_float2;
		if (ocl_vector==13) tt=ocl_sum_text_vector_half2;
		if (ocl_vector==14) tt=ocl_sum_text_float;
		if (ocl_vector==15) tt=ocl_sum_text_half;
#else
		tt=cuda_sum_text_double;
		if (ocl_vector==4) tt=cuda_sum_text_tensor;
		if (ocl_vector==8) tt=cuda_sum_text_vector_double4;
		if (ocl_vector==9) tt=cuda_sum_text_vector_float4;
		if (ocl_vector==11) tt=cuda_sum_text_vector_double2;
		if (ocl_vector==12) tt=cuda_sum_text_vector_float2;
		if (ocl_vector==13) tt=cuda_sum_text_vector_half2;
		if (ocl_vector==14) tt=cuda_sum_text_float;
		if (ocl_vector==15) tt=cuda_sum_text_half;
#endif
		prg = new OpenCL_program(1);
		queue = prg->create_queue(device, 0);
		{
			char *text = new char[strlen(tt) * 2];
			sprintf(text, tt, ((double_ext == 0) ? "cl_amd_fp64" : "cl_khr_fp64")
				,sum_alg
				,ocl_sol_size
				,oclBS
				);
			prog = prg->create_program(text);
			delete[] text;
		}
		kSum = prg->create_kernel(prog, "Sum");
		bS = prg->create_buffer(CL_MEM_READ_WRITE, ocl_sol_size*sizeof(double), NULL);
		old_alloced=40;
		int sp=sum_param;
		if (sum_alg!=2) sp=1;
		bHtmp = prg->create_buffer(CL_MEM_READ_WRITE, 3*sp*ocl_sol_size*sizeof(double), NULL);
		bCoefs = prg->create_buffer(CL_MEM_READ_WRITE, sp*sizeof(double), NULL);
		boldU = prg->create_buffer(CL_MEM_READ_WRITE, old_alloced*ocl_sol_size*sizeof(double), NULL);
		queue->EnqueueWriteBuffer(boldU, oldH[0], 0, (sN + 2)*(_M + 2)*(K+2)*sizeof(double));
		bK = prg->create_buffer(CL_MEM_READ_WRITE, old_alloced*sizeof(double), NULL);
		btau = prg->create_buffer(CL_MEM_READ_WRITE, old_alloced*sizeof(double), NULL);
		Sums=new double [ocl_sol_size];
	}
	void fr_ocl_check_and_resize()
	{
		if (tstep>=(old_alloced-2))
		{
			if (sum_alg!=2)
			{
				double *k=new double[old_alloced*2];
				double *oU=new double [ 2*old_alloced*ocl_sol_size];
				queue->EnqueueBuffer(bK, k, 0,  old_alloced*sizeof(double));
				queue->EnqueueBuffer(boldU, oU, 0, old_alloced*ocl_sol_size*sizeof(double));
				delete bK;
				delete boldU;
				bK = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 2*old_alloced*sizeof(double), k);
				boldU = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 2*old_alloced*ocl_sol_size*sizeof(double), oU);
				queue->Finish();
				delete [] k;
				delete [] oU;
			}
			double *t=new double[old_alloced*2];
			queue->EnqueueBuffer(btau, t, 0,  old_alloced*sizeof(double));
			delete btau;
			btau = prg->create_buffer(CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, 2*old_alloced*sizeof(double), t);
			queue->Finish();
			old_alloced*=2;		
			delete [] t;
		}
	}
	void fr_ocl_call_sum()
	{
		static int first=1;
		struct timespec i1,i2,i3,i4;
		clock_gettime(CLOCK_REALTIME,&i1);
		size_t nth=(sN+2)*(_M+2)*(K+2), lsize=1;
		int err=0;
		int iv;
		double dv;
		if (first)
		{
			init_opencl();
			first=0;
		}
		fr_ocl_check_and_resize();
		// put data
		if (kb_3_cache.size()>0)
			queue->EnqueueWriteBuffer(bK, &kb_3_cache[0], 0,  ((kb_3_cache.size()<old_alloced)?kb_3_cache.size():old_alloced)*sizeof(double));
		if (sum_alg==2)
		{
			if (row2_precalc.size()>time_i1) if (row2_precalc[time_i1].size()) row2_precalc[time_i1][0].clear();
			kb(alpha, tstep-1, tstep, tstep+1, &Time[0], (void *)&Time,time_i1,0);
			queue->EnqueueWriteBuffer(bCoefs, &row2_precalc[time_i1][0][0], 0,  ((row2_precalc[time_i1][0].size()>sum_param)?sum_param:row2_precalc[time_i1][0].size())*sizeof(double));
			if (oldH.size()>0)
				queue->EnqueueWriteBuffer(boldU, &oldH[0][0], 0*ocl_sol_size*sizeof(double),  nth*sizeof(double));
			if (oldH.size()>1)
				queue->EnqueueWriteBuffer(boldU, &oldH[1][0], 1*ocl_sol_size*sizeof(double),  nth*sizeof(double));
		}
		else
			queue->EnqueueWriteBuffer(boldU, &oldH[oldH.size()-1][0], (oldH.size()-1)*ocl_sol_size*sizeof(double),  nth*sizeof(double));
		queue->EnqueueWriteBuffer(btau, &tau[tstep], tstep*sizeof(double), sizeof(double));
		// set args
		err |= kSum->SetBufferArg(bS, 0);
		err |= kSum->SetBufferArg(boldU, 1);
		iv=oldH.size();
		err |= kSum->SetArg(2, sizeof(int), &iv);
		err |= kSum->SetBufferArg(btau, 3);
		err |= kSum->SetBufferArg(bHtmp, 4);
		err |= kSum->SetBufferArg(bCoefs, 5);
		err |= kSum->SetBufferArg(bK, 6);
		dv=sum_param;
		err |= kSum->SetArg(7, sizeof(double), &dv);
		dv=g(Time[tstep+1]);
		err |= kSum->SetArg(8, sizeof(double), &dv);
		dv=g(Time[tstep]);
		err |= kSum->SetArg(9, sizeof(double), &dv);
		dv=alpha;
		err |= kSum->SetArg(10, sizeof(double), &dv);
		iv=tstep;
		err |= kSum->SetArg(11, sizeof(int), &iv);
		if (err) { printf("Error: Failed to set kernels args"); exit(0); }
		// call
		lsize=oclBS;
		if ((nth%lsize)!=0)
			nth=((nth/lsize)+1)*lsize;
		queue->Finish();
		clock_gettime(CLOCK_REALTIME,&i2);
		queue->ExecuteKernel(kSum, 1, &nth, &lsize);
		queue->Finish();
		clock_gettime(CLOCK_REALTIME,&i3);
		// get data
		queue->EnqueueBuffer(bS, Sums, 0,  nth*sizeof(double));
		clock_gettime(CLOCK_REALTIME,&i4);
		printf("Summing times (%d,%d)",(int)(i4.tv_nsec-i1.tv_nsec),(int)(i3.tv_nsec-i2.tv_nsec));
	}
#endif
	//////////////////////////////////////
	//////////////////////////////////////
	//////////////////////////////////////
	// generalized Caputo derivative
	double g(double t)
	{
		if (func_in_kernel == 1)
			return sqrt(t);
		if (func_in_kernel == 2)
			return t*t;
		if (func_in_kernel == 3)
			return pow(t,func_power);
		return t;
	}
	// should return kf(n)(t) given kf(n-1)(t)=der_nm1
	// for n=0 return f(t)
	double inv_g_der(double t, int n, double der_nm1)
	{
		if (func_in_kernel == 1) // sqrt(t) -> f=t^2 -> df/dt= 2t, d2f/dt=2, d(n>=3)f/dt=0
		{
			if (n == 0)
				return t*t;
			if (n == 1)
				return 2 * t*(der_nm1 / (t*t));
			if (n == 2)
				return 2 * (der_nm1 / (2 * t));
			return 0.0;
		}
		if (func_in_kernel == 2) // t^2 -> f=sqrt(t) -> df/dt=0.5*t(-0.5), d(n+1)f/dt=(d(n)f/dt)*(0.5-n)/t
		{
			if (n == 0)
				return sqrt(t);
			if (n == 1)
				return (0.5 / sqrt(t))*(der_nm1 / sqrt(t));
			return der_nm1*(0.5 - n + 1) / t;
		}
		if (func_in_kernel == 3) // t^k -> f=t^(1/k) -> df/dt=(1/k)*t((1/k)-1), d(n+1)f/dt=(d(n)f/dt)*((1/k)-n)/t
		{
			if (n == 0)
				return pow(t,1.0/func_power);
			if (n == 1)
				return ((1.0/func_power) * pow(t,(1.0/func_power)-1))*(der_nm1 / pow(t,1.0/func_power));
			return der_nm1*((1.0/func_power) - n + 1) / t;
		}
		return ((n == 0) ? t : ((n == 1) ? (der_nm1 / t) : 0));
	}
	// using taylor series on gti-g(x) in g(ti)
	// for right-part - series on g(x)-gti in g(t1)
	double _kb_row(double t0, double t1, double gtj,double alpha,int &niter)
	{
		double sum = 0.0, v = 0.0, v2, v3, v4;
		double i = 0;
		niter = 0;
		if ((alpha == 1.0)&&(g(t1)==gtj))
			return 1.0;
		if ((alpha==1.0)&&(g(t1)!=gtj))
			return 0.0;
		if (g(t1) <= gtj)
		{
			v = inv_g_der(gtj, 1, inv_g_der(gtj, 0, 1));
			v *= pow(gtj, 1 - alpha);
			// ((a-g(d))^(n-b+1)-(a-g(c))^(n-b+1))/(n-b+1)
			v3 = pow(fabs(1.0 - g(t1) / gtj), 1 - alpha);
			v4 = pow(fabs(1.0 - g(t0) / gtj), 1 - alpha);
			v2 = -(v3 - v4) / (1 - alpha);
			sum += v2*v;
			while (fabs(v*v2) > global_eps)
			{
				i += 1.0;
				//(((-1)^n/n!)*f(n+1)(a))*a^(n-b+1)
				v = inv_g_der(gtj, i + 1.0, v);
				v *= -gtj / i;
				v3 *= fabs(1.0 - g(t1) / gtj);
				v4 *= fabs(1.0 - g(t0) / gtj);
				v2 = -(v3 - v4) / (i - alpha + 1.0);
				sum += v2*v;
				niter++;
				if (niter > integr_max_niter)
					break;
			}
		}
		else
		{
			double gh = g(t1) - gtj;
			double gl = g(t0) - gtj;
			double g1 = g(t1);
			// f'(g(t1))/gl^(1-a)
			v = inv_g_der(g1, 1, inv_g_der(g1, 0, 1));
			v *= pow(gl, 1 - alpha);
			// I0=1/(1-a) * (gh^(1-a) - gl^(1-a))
			v2 =( (pow(gh, 1 - alpha) / pow(gl, 1 - alpha)) - 1.0) / (1 - alpha); // I0/gl^(1-a)
			sum += v2*v;
			while (fabs(v*v2) > global_eps)
			{
				i += 1.0;
				// gh/(n+1-a) * f(n+1)(g(t1))/gl^(1-a)
				v = inv_g_der(g1, i + 1.0, v);
				v /= (1 - alpha + i) / gh;
				// v2 - Bi, v4 - Ci
				if (i == 1.0)
					v4 = (gl - gh)/gh;
				else
					v4 *= (gl - gh)*((i - 1.0) / i)*((-alpha + i) / (gh*(i - 1.0)));
				v2 = -(v2 + v4);
				sum += v2*v;
				niter++;
				if (niter > integr_max_niter)
					break;
			}
		}
		return sum;
	}
	// using newton binomial on gtj and taylor series of f'(x) in g(t1)
	int time_i1;
	std::vector< std::vector< std::vector<double> > > row2_precalc;
	double _kb_row2(double t0, double t1, double gtj, int idx,int idx2,double alpha,int &niter,int *outer_niter=NULL)
	{
		double sum = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4, v5, v6, v7;
		double gt0 = g(t0), gt1 = g(t1);
		std::vector<double> *cache;
		double i = 0, m,eeps=global_eps2;
		int mode=0; 
		int found;
		niter = 0;
		if (outer_niter)
			outer_niter[0] = 0;
		if ((alpha == 1.0)&&(g(t1)==gtj))
			return 1.0;
		if ((alpha==1.0)&&(g(t1)!=gtj))
			return 0.0;
		if (idx==time_i1) eeps=global_eps;
		if (idx!=-1)
		{
			while (row2_precalc.size()<(idx + 1))
				row2_precalc.push_back(std::vector< std::vector<double> >());
			while (row2_precalc[idx].size()<(idx2 + 1))
				row2_precalc[idx].push_back(std::vector<double>());
			cache = &row2_precalc[idx][idx2];
		}
		if (g(t1) > gtj)
			mode = 1;
		// (a,n)*(-1)^n
		v1 = 1.0;
		if (mode == 0)
		{
			v2 = pow(gtj, -alpha)*gt1;
			v6 = gt0 / gt1;
		}
		else
		{
			v2 = pow(gt0, 1-alpha);
			v6 = pow(gt1,1-alpha) / pow(gt0,1-alpha);
		}
		do
		{
			// (a,n)*(-1)^n
			if (i != 0.0)
			{
				v1 *= -(-alpha - i + 1.0) / i;
				// (a^(-b-n)*g(t1)^(n+1)
				if (mode == 0)
				{
					v2 *= gt1 / gtj;
					v6 *= gt0 / gt1;
				}
				else
				{
					v2 *= gtj / gt0;
					v6 *= gt0 / gt1;
				}
			}
			// integrate(f'(x)x^n,x,g(c),g(d))/g(t1)^(n+1)
			found = 0;
			if (idx!=-1)
			if (cache->size()>(int)i)
			{
				v3 = (*cache)[(int)i];
				found = 1;
			}
			if (found == 0)
			{
				if ((i == 0.0)&&(mode==0))
					v3 = (t1 - t0) / gt1;
				else
				{
					v3 = 0.0;
					m = 0.0;
					// f(m+1)(gd)/m!
					v4 = inv_g_der(gt1, 1, inv_g_der(gt1, 0, 0));
					// I(n,m)=integrate(x^n*(x-g(t1))^m,x,g(t0),g(t1));
					// I(n,0)=(1/n+1)*(g(t1)^(n+1)-g(t0)^n+1)
					if (mode==0)
						v5 = (1.0 / (i + 1.0))*(1.0 - v6);
					else
						v5 = -(1.0 / (-alpha-i + 1.0))*(1.0 - v6);
					v3 += v4*v5;
					v7 = 1;
					while (fabs(v4*v5) > (fabs(v3)*global_eps))
					{
						m+=1.0;
						// (f(m+1)(gd)/m!)
						v4 = inv_g_der(gt1, m + 1.0, v4);
						v4 /= m;
						if (mode==0)
							v4 /= ((i + m + 1.0) / (gt1*m));
						else
							v4 /= ((-alpha-i + m + 1.0) / (gt1*m));

						if (m == 1.0)
						{
							if (mode == 0)
								v7 = (1.0 / (gt1))*(v6*(gt0 - gt1));
							else
								v7 = (1.0 / (gt1))*(gt0 - gt1);
						}
						else
						{
							if (mode==0)
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((i + m) / (gt1*(m - 1.0)));
							else
								v7 *= (gt0 - gt1)*((m - 1.0) / m)*((-alpha-i + m) / (gt1*(m - 1.0)));
						}
						// I(n,m)=-(I(n,m-1)+(1/(g(t1)*m))*(g(t0)^(n+1)*(g(t0)-g(t1))^m))/((n+m+1)/(g(t1)*m)
						v5 = -(v5 + v7);
						v3 += v4*v5;
						niter++;
						if (niter>integr_max_niter)
							break;
					}
				}
				if (idx!=-1)
				{
					while (cache->size()<(i + 1.0))
						cache->push_back(0);
					(*cache)[(int)i] = v3;
				}
			}
			sum += v1*v2*v3;
			i+=1.0;
			niter++;
			if (outer_niter)
				outer_niter[0]++;
			if (niter>integr_max_niter)
				break;
		} while (fabs(v1*v2*v3) > eeps);
		return sum;
	}
	//b(t0,t1,a) = int((g(x_a)-g(tau))^(-alpha),tau=x_t0,x_t1)
	// use newton binomial near t0==0
	// use taylor series near t1==a
	// move usage boundary to ensure minimal number of iteration done by both algorithms
	std::vector<double *> kb_cache_pts;
	std::vector< std::vector< std::vector<double> > > kb_cache;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	std::vector< std::vector<double> > *last_vi1;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	int a3k;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	double *last_points;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	int last_i1;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	int last_a;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	double* last_vi1a;
#if defined(_OPENMP)||defined(USE_MPI)
	static
#endif		
	int last_vi1a_size;
#pragma omp threadprivate(a3k,last_points,last_i1,last_a,last_vi1a,last_vi1a_size,last_vi1)	
	double kb(double alpha, double t0,double t1,double a,double *points,void *time,int ii1=-1,int ii2=-1)
	{
		if (alpha == 1.0)
		{
			if (t1==a)
				return 1.0;
			else
				return 0.0;
		}
		int i1 = -1;
		std::vector< std::vector<double> > *vi1=NULL;
		double* vi1a=NULL;
		int vi1a_size;
		int niter, niter2;
		int ret=0;
		double v;
		// integer and float part of steps
		int tt0=(int)t0;
		double k0=t0-tt0;
		int tt1=(int)t1;
		double k1=t1-tt1;
		int tta=(int)a;
		double ka=a-tta;
		double p0=0.0,p1=0.0,pa=0.0;
		if (tt0>=0) p0=points[tt0];
		if (tt1>=0) p1=points[tt1];
		if (tta>=0) pa=points[tta];
		if (k0!=0) p0+=k0*(points[tt0+1]-p0);
		if (k1!=0) p1+=k1*(points[tt1+1]-p1);
		if (ka!=0) pa+=ka*(points[tta+1]-pa);
		// try to find in cache
		if ((k0==0)&&(k1==0)&&(ka==0))
		{
			if (kb_cache.size())
			if ((last_vi1-&kb_cache[0])!=last_i1)
			    last_points=NULL;
			if (last_points==points)
			{
				i1=last_i1;
				vi1=last_vi1;
				if (last_a==tta)
				{
					vi1a=last_vi1a;
					vi1a_size=last_vi1a_size;
				}
			}
			else
			{
				for (int i = 0;i < kb_cache_pts.size();i++)
				if ((kb_cache_pts[i]==points)||((time!=NULL)&&(kb_cache_pts[i]==time)))
				{
					i1 = i;
					break;
				}
				if (i1 == -1)
				{
#pragma omp critical
					{
						kb_cache_pts.push_back((time==NULL)?points:(double *)time);
						kb_cache.push_back(std::vector< std::vector<double> >());
						i1 = kb_cache_pts.size() - 1;
					}
					if (time)
						time_i1=i1;
				}
				last_a = -1;
				vi1a = NULL;
				last_i1=i1;
				last_points=points;
#pragma omp critical
				vi1=last_vi1=&kb_cache[i1];
			}
			if (vi1a==NULL)
			{
				if (vi1)
				if (vi1->size()>tta)
				{
					last_vi1a=vi1a=&((*vi1)[tta][0]);
					last_vi1a_size=vi1a_size=(*vi1)[tta].size();
					last_a=tta;
				}
			}
			if (vi1a)
				if (vi1a_size>tt0)
				{
					v=vi1a[tt0];
					if (v!=-1)
						return v;
				}
		}
#pragma omp critical
		{
		// calculate
		if (ret==0)
		{
			if (ii1==-1) ii1=i1;
			if ((time==NULL)||(sum_alg!=2))
			{
				if ((t0 <= a3k)&&(t1!=a)&&(t0!=a))
				{
					v = _kb_row2(p0,p1,g(pa),ii1, t0,alpha,niter);
					if (t0 == a3k)
					{
						double v2;
						v2 = _kb_row(p0, p1, g(pa),alpha,niter2);
						if (niter2 > niter)
							a3k++;
						else
							if (a3k != 1)
								a3k--;
					}
				}
				else
					v = _kb_row(p0, p1, g(pa), alpha, niter2);
			}
			else
			{
				double tt0=t0;
				if (ii2!=-1) tt0=ii2;
				if (t1!=a)
					v = _kb_row2(p0,p1,g(pa),ii1, tt0,alpha,niter);
				else
					v = _kb_row(p0, p1, g(pa), alpha, niter2);
			}
			// put in cache
			if ((k0==0)&&(k1==0)&&(ka==0)&(time==NULL))
			{
				while (kb_cache[i1].size() <= a) kb_cache[i1].push_back(std::vector<double>());
				while (kb_cache[i1][a].size() <= t0) kb_cache[i1][a].push_back(-1);
				kb_cache[i1][a][t0] = v;
			}
		}
		}
		return v;
	}
	// calculate values of F for block
	void calc_v(int block,double *F,double *V,double *points,int with_i=0)
	{
		for (int i = 0;i<BS;i++)
			V[i] = 0.0;
		if (A == 0.0)
		{
			if (with_i)
				for (int i=block*BS+1;i<(block+1)*BS+1;i++)
					V[i-(block*BS+1)]=F[i];
			return;
		}
		for (int i=block*BS+1;i<(block+1)*BS+1;i++)
		{
			double v=0.0;
			int end=i-1;
			if (with_i) 
				end=i;
			for (int j = 1;j <= end;j++)
			{
				double kv = kb(1 - A, j - 1, j, i, points,NULL);
				v += kv*F[j];
			}
			if (space_der == 1)
			{
				for (int j = i + 1;j < N;j++)
				{
					double kv = -kb(1 - A, j , j+1, i, points, NULL);
					v += kv*F[j];
				}
			}
			V[i-(block*BS+1)]=v;
		}
	}
	// set space variable steps
	// set steps to be close to exponential distribution from minDL to maxDL	
	void var_xyz(double minDL, double maxDL, double *Z,double *sZ,double *dL, double *sdL)
	{
		double a = minDL;
		double b = log(maxDL / minDL);
		for (int k = 0;k < varZ_niter;k++)
			for (int i = 1;i < N;i++)
			{
				// move for 0.1 of the distance to the needed step length
				double d0 = a*exp(b*(1.0 - (double)i / N));
				d0 = minDL + maxDL - d0;
				double d1 = Z[i] - Z[i - 1];
				double m = (d1 - d0) / 10.0;
				if (m > 0)
				{
					while (Z[i - 1]>(Z[i] - m))
						m /= 2;
					Z[i] -= m;
				}
				if (i == (N - 1))
				{
					d1 = Z[i + 1] - Z[i];
					m = (d1 - d0) / 10.0;
					if (m > 0)
					{
						while (Z[i + 1] < (Z[i] + m))
							m /= 2;
						Z[i] += m;
					}
				}
			}
		for (int i = 0;i <= N;i++)
			dL[i] = Z[i + 1] - Z[i];
		dL[N] = dL[N - 1];
		if (sZ)
			for (int i = 0;i <= N;i++)
				sZ[i] = Z[i];
		if (sdL)
			for (int i = 0;i <= N;i++)
				sdL[i] = dL[i];
		if (debug_level==1)
		for (int i = 0;i <= N;i++)
		    if (log_file) fprintf(log_file,"%d - Z %g dZ %g\n",i,Z[i],dL[i]);
	}
	void set_xyz_steps()
	{
		// uniform case
		for (int i=0;i<=N;i++)
		{
			sdL[i]=dL[i]=L/(double)N;
			sZ[i]=Z[i]=i*L/(double)N;
		}
		// exponential
		if (varZ)
			var_xyz(minDL, maxDL, Z, sZ, dL, sdL);
		if (mode==3)
		{
			for (int i=0;i<=_M;i++)
			{
				dLx[i]=Lx/(double)_M;
				X[i]=i*Lx/(double)_M;
			}
			for (int i=0;i<=K;i++)
			{
				dLy[i]=Ly/(double)K;
				Y[i]=i*Ly/(double)K;
			}
			// variable -  plants and sprinklers as the sources of attraction force
			if (varXY)
			{
			    for (int k=0;k<100;k++)
			    {
				for (int i=1;i<_M;i++)
				{
				    double f=0.0,x,ff;
				    for (int ii = 0;ii < Nsprinklers_rows;ii++)
				    {	
				        x = Spr_offs + ii*betw_spr_rows;				        
				        ff=exp(-(x-X[i])*(x-X[i])*XYforceexp);
				        if (x>X[i]) f+=ff; else f-=ff;
				    }
				    for (int ii = 0;ii < Nrows;ii++)				
				    {
    				        x = Poffs + ii*betw_rows;
				        ff=exp(-(x-X[i])*(x-X[i])*XYforceexp);
				        if (x>X[i]) f+=ff; else f-=ff;
    				    }
    				    X[i]+=f/((Nsprinklers_rows+Nrows)*XYvar_coef);
				}				
				for (int i=1;i<_M;i++)
				    X[i]=0.5*(X[i-1]+X[i+1]);
				for (int i=1;i<K;i++)
				{
				    double f=0.0,ff,y;
				    for (int j = 0;j < Nsprinklers;j++)
				    {
			    	        y = Spr_spr_offs + j*betw_spr;
				        ff=exp(-(y-Y[i])*(y-Y[i])*XYforceexp);
				        if (y>Y[i]) f+=ff; else f-=ff;
			    	    }
				    for (int j = 0;j < Nplants;j++)
				    {
    					y = Ploffs + j*betw_pl;
				        ff=exp(-(y-Y[i])*(y-Y[i])*XYforceexp);
				        if (y>Y[i]) f+=ff; else f-=ff;				        
    				    }
    				    Y[i]+=f/((Nsprinklers+Nplants)*XYvar_coef);
				}
				for (int i=1;i<K;i++)
				    Y[i]=0.5*(Y[i-1]+Y[i+1]);
			    }
			    for (int i = 0;i <= _M;i++)
				dLx[i] = X[i + 1] - X[i];
			    dLx[_M] = dLx[_M - 1];
			    for (int i = 0;i <= K;i++)
				dLy[i] = Y[i + 1] - Y[i];
			    dLy[K] = dLy[_M - 1];
				if (debug_level == 1)
				{
					for (int i = 0;i <= _M;i++)
						if (log_file) fprintf(log_file,"%d - X %g dX %g\n", i, X[i], dLx[i]);
					for (int i = 0;i <= K;i++)
						if (log_file) fprintf(log_file,"%d - Y %g dY %g\n", i, Y[i], dLy[i]);
				}
			}
			for (int i = 0;i <= N;i++)
			{
				sZ[i] = Z[i];
				dL[i]=sdL[i];
			}
		}
		// precalc g(Z),pow(g(Z),-gamma)
		for (int i=0;i<=N;i++)
		{
			gZ[i]=g(Z[i]);
			gZg[i]=pow(gZ[i],-gamma);
		}
		if (mode==3)
		{
			for (int i=0;i<=_M;i++)
			{
				gX[i]=g(X[i]);
				gXg[i]=pow(gX[i],-gamma);
			}
			for (int i=0;i<=K;i++)
			{
				gY[i]=g(Y[i]);
				gYg[i]=pow(gY[i],-gamma);
			}
		}
	}
	// minimum (0.5) ET at 0, maximum (1.5) at 12
	double day_night_coef()
	{
		return 1+0.5*sin(M_PI*((Time[tstep]/3600.0)-6.0)/12.0);
	}
	double wetness(int i, double u = -1.0, int from_u = 0)
	{
		double P;
		double Ch = 0.5;
		if (mode==1) P = -(U[i]-Z[i]);
		if (from_u)
			P = -(u-Z[i]);
		if (analytic_test)
			return 1.0;
		if (mode == 3)
			if (from_u == 0)
			{
				if (m3d_c == 0)
					P = -(b_U[idx(i, m3d_x, m3d_y)] - i*sZ[i]);
				if (m3d_c == 1)
					P = -(b_U[idx(m3d_x, i, m3d_y)] - m3d_x*sZ[i]);
				if (m3d_c == 2)
					P = -(b_U[idx(m3d_x, m3d_y, i)] - m3d_x*sZ[i]);
			}
		/////////////// van Genuchten - Mualem /////////////////////
		if (cv_form == 4)
		{
			double n = 1.25;
			double s0 = 0.0736;
			double s1 = 0.55;
			double a = 0.066;
			P *= 100.0;
			if (P <= 0.0)
				Ch = s1;
			else
				Ch = s0+((s1 - s0) / pow(1 + pow(a*P, n), (1 - 1 / n)));
		}
		return Ch;
	}
	double inv_dw_dh(int i)
	{
		double P;
		if (mode==1) P= -(U[i]-Z[i]);
		double Ch = 0.0;
		if ((analytic_test == 1)||(analytic_test==6))
			return 1.0;
		if ((analytic_test == 2) || (analytic_test == 3) ||(analytic_test == 4) || (analytic_test == 5))
		{
			if (mode==1)
				return 1.5 + sin(sZ[i] / (4.0*sL));
			if (mode == 3)
			{
				if (m3d_c == 0)
					return 1.5 + sin(sZ[i] / (4.0*sL));
				if (m3d_c == 1)
					return 1.5 + sin(sZ[m3d_x] / (4.0*sL));
				if (m3d_c == 2)
					return 1.5 + sin(sZ[m3d_x] / (4.0*sL));
			}
		}
		if (mode == 3)
		{
			if (m3d_c == 0)
				P = -(b_U[idx(i, m3d_x, m3d_y)] - i*sZ[i]);
			if (m3d_c == 1)
				P = -(b_U[idx(m3d_x, i, m3d_y)] - m3d_x*sZ[i]);
			if (m3d_c == 2)
				P = -(b_U[idx(m3d_x, m3d_y, i)] - m3d_x*sZ[i]);
		}
		/////////////// van Genuchten - Mualem /////////////////////
		if (cv_form == 4)
		{
			double n = 1.25;
			double s0 = 0.0736;
			double s1 = 0.55;
			double a = 0.066;
			if (P <= 0.0)
				Ch = 0.0;
			else
			{
				double hn = pow(100.0, n), h2n = pow(100.0, 2.0*n);
				Ch = ((((1 - n)*hn*s0 + (n - 1)*hn*s1)*pow(a*P, n)*pow(1 + hn*pow(a*P, n), (1 / n))) / (h2n*P*pow(a*P, 2 * n) + 2 *hn* P*pow(a*P, n) + P));
			}
		}
		if (Ch!=0.0)
			return 1.0/Ch;
		return 1e20;
	}
	double avg_root_layer_wetness()
	{
		double sum = 0.0;
		double n = 0;
		if (mode == 1)
			for (int i = 0;i<N + 1;i++)
				if (Z[i] <= v)
				{
					if (linear_avg)
					    sum += 0.5*(wetness(i)+wetness(i+1))*sdL[i];
					else
					    sum += wetness(i)*sdL[i];
					n+=sdL[i];
				}
				else
					break;
		if (mode == 3)
			for (int i = 0;i<N + 1;i++)
				for (int j = 1;j<_M;j++)
					for (int k = 1;k<K;k++)
						if (sZ[i] <= v)
						{
							if (linear_avg)
							    sum += 0.125*(wetness(i, b_U[idx(i, j, k)], 1)+wetness(i, b_U[idx(i+1, j, k)], 1)+
							    wetness(i, b_U[idx(i, j+1, k)], 1)+wetness(i, b_U[idx(i, j, k+1)], 1)+
							    wetness(i, b_U[idx(i+1, j+1, k)], 1)+wetness(i, b_U[idx(i+1, j, k+1)], 1)+
							    wetness(i, b_U[idx(i, j+1, k+1)], 1)+wetness(i, b_U[idx(i+1, j+1, k+1)], 1))
							    *sdL[i] * dLx[j] * dLy[k];
							else
							    sum += wetness(i, b_U[idx(i, j, k)], 1)	*sdL[i] * dLx[j] * dLy[k];
							n+= sdL[i] * dLx[j] * dLy[k];
						}
		if (n!=0.0) sum /= n;
		return sum;
	}
	double total_water_content()
	{
		double sum = 0.0;
		if (mode == 1)
			for (int i = 0;i < N + 1;i++)
				sum += wetness(i)*dL[i];
		if (mode == 3)
			for (int i = 0;i<N + 1;i++)
				for (int j = 1;j<_M;j++)
					for (int k = 1;k<K;k++)
							sum += wetness(i, b_U[idx(i, j, k)], 1)*sdL[i]*dLx[j]*dLy[k];
		return sum*p_soil/p_water;
	}
	// non-linear filtration coefficient 
	double KK(int i)
	{
		double n = 1.25;
		double s0 = 0.0736;
		double s1 = 0.55;
		double Kr=1.0;
		if (analytic_test == 0)
		{
			// by VGM
			if (k_form==1)
				Kr = pow(((wetness(i) - s0) / (s1 - s0)), 0.5)*pow(1.0 - pow(1.0 - pow(((wetness(i) - s0) / (s1 - s0)), (1.0 / (1.0 - 1.0 / n))), (1.0 - 1.0 / n)), 2.0);
			// by Averianov
			if (k_form==2)
				Kr = pow(((wetness(i) - s0) / (s1 - s0)), 3.5);
		}
		if ((analytic_test == 1)||(analytic_test == 6))
			return 1.0;
		if ((analytic_test == 2) || (analytic_test == 3) || (analytic_test == 4) || (analytic_test == 5))
		{
			if (mode==1)
				return 0.5 + sZ[i];
			if (mode == 3)
			{
				if (m3d_c == 0)
					return 0.5 + sZ[i];
				if (m3d_c == 1)
					return 0.5 + sZ[m3d_x];
				if (m3d_c == 2)
					return 0.5 + sZ[m3d_x];
			}
		}
		return Kr*k;
	}
	double Rp3d_mult()
	{
		if (mode==1) 
			return 1.0;
		if (first_rp == 1)
		{
			for (int ix = 0;ix < _M + 2;ix++)
				for (int iy = 0;iy < K + 2;iy++)
				{
					double x0, y0, r;
					double max_mult = 0.0;
					x0 = X[ix];
					y0 = Y[iy];
		    			double cr = dLx[ix];
					if (dLy[iy] > cr) cr = dLy[iy];
					cr*=0.5;
					for (int i = 0;i < Nrows;i++)
						for (int j = 0;j < Nplants;j++)
						{
							double x, y, mult;
							x = Poffs + i*betw_rows;
							y = Ploffs + j*betw_pl;
							r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
							if (r <= (rootR+cr))
							{
								if ((r - cr) < 0)
									mult = 1.0;
								else
									mult = (rootR-(r-cr)) / rootR;
								if (mult > max_mult)
									max_mult = mult;
							}
						}
					rp_mult[ix][iy] = max_mult;
				}
			first_rp = 0;
		}
		return rp_mult[m3d_x][m3d_y];
	}
	double g2,g3,g4,g5;
	double g4a,g3a,g2a;
	double g5g,g4g,g3g,g2g;
	double g1fp,g1fpg;
	double g2fp,g2fpg;
	double g2fpa;
	void calc_gammas()
	{
		g2=Gamma(2.0);
		g3=Gamma(3.0);
		g4=Gamma(4.0);
		g5=Gamma(5.0);
		g4a=Gamma(4.0-alpha);
		g3a=Gamma(3.0-alpha);
		g2a=Gamma(2.0-alpha);
		g5g=Gamma(5.0 - (1.0 + gamma)+1);
		g4g=Gamma(4.0 - (1.0 + gamma)+1);
		g3g=Gamma(3.0 - (1.0 + gamma)+1);
		g2g=Gamma(2.0 - (1.0 + gamma)+1);
		g1fp=Gamma((1.0/func_power)+1.0);
		g1fpg=Gamma((1.0/func_power) - gamma+1);
		g2fp=Gamma((2.0/func_power)+1.0);
		g2fpg=Gamma((2.0/func_power) - gamma+1);
		g2fpa=Gamma((2.0/func_power) - alpha+1);
	}

	// sink term (right part) - roots water uptake
	double Rp(int i)
	{
		double ret=0.0;
		if (analytic_test!=3)
			if (equation!=0) 
				return 0.0;
		if (!(rp_split_3d&1))
		if (mode == 3) // in 3d splitted mode all right-part goes to z-pass
			if (m3d_c != 2)
				return 0.0;
		if (analytic_test)
		{
			if (analytic_test == 1)
			{
				if (mode == 1)
				{
					double p1 = 0.0, p2 = 0.0;
					p1 = ((pow(Z[i], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(Z[i], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(Ly, 3.0);
					p2 = ((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
					return p1 - p2;
				}
				if (mode == 3)
				{
					double p1 = 0.0, p2 = 0.0;
					if ((rp_split_3d&1)==0)
					{
					    p1 = ((pow(sZ[i], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(sZ[i], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(sL, 3.0);
				    	p1 += ((pow(X[m3d_x], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(X[m3d_x], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(Lx, 3.0);
					    p1 += ((pow(Y[m3d_y], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(Y[m3d_y], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(Ly, 3.0);
					    p2 = (((pow(Time[tstep+1], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep+1], 2.0 - alpha)*g3 / g3a)) );					
					}    
					if ((rp_split_3d&1)==1)
					{
					    if (m3d_c==0) p1 = ((pow(sZ[i], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(sZ[i], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(sL, 3.0);
					    if (m3d_c==1) p1 = ((pow(X[i], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(X[i], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(Lx, 3.0);
					    if (m3d_c==2) p1 = ((pow(Y[i], 3.0 - (1.0 + gamma))*g4 / g3g) - 2.0*(pow(Y[i], 2.0 - (1.0 + gamma))*g3 / g2g)) / pow(Ly, 3.0);
					    p2 = (((pow(Time[tstep+1], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep+1], 2.0 - alpha)*g3 / g3a)) );					
					    p2/=3.0;
					}
					return p1 - p2;
				}
			}
			if (analytic_test == 2)
			{
				if (mode == 1)
				{
					double p1 = 0.0, p2 = 0.0;
					p1 = (2.0 / pow(sL, 3.0))*(0.5*(pow(Z[i], 1.0 - gamma)*g2 / g2g) + (pow(Z[i], 2.0 - gamma)*g3 / g3g));
					p2 = ((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
					return inv_dw_dh(i)*p1 - p2;
				}
				if (mode==3)
				{
				    double ret=0.0;
				    if ((rp_split_3d&1)==0)
				    {
						double p1 = 0.0, p2 = 0.0,v1,v2,v3;
				    			v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[m3d_y], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[i], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma)*g2 / g2g) + (pow(sZ[m3d_x], 2.0 - gamma)*g3 / g3g)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[m3d_y], 1.0 - gamma)*g2 / g2g));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[i], 1.0 - gamma)*g2 / g2g));
						p2 =  - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a);
						ret=inv_dw_dh(i)*p1 - p2;
				    }	    
				    if ((rp_split_3d&1)==1)
				    {
						double p1 = 0.0, p2 = 0.0,v1=1.0,v2=1.0,v3=1.0;
						if (m3d_c==0)
						{
				    			v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[m3d_x], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[m3d_y], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[i], 1.0 - gamma)*g2 / g2g) + (pow(sZ[i], 2.0 - gamma)*g3 / g3g)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[i])*(pow(X[m3d_x], 1.0 - gamma)*g2 / g2g));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[i])*(pow(Y[m3d_y], 1.0 - gamma)*g2 / g2g));
						}
						if (m3d_c==1)
						{
				    			v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[i], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[m3d_y], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma)*g2 / g2g) + (pow(sZ[m3d_x], 2.0 - gamma)*g3 / g3g)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[i], 1.0 - gamma)*g2 / g2g));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[m3d_y], 1.0 - gamma)*g2 / g2g));
						}
						if (m3d_c==2)
						{
				    			v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[m3d_y], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[i], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma)*g2 / g2g) + (pow(sZ[m3d_x], 2.0 - gamma)*g3 / g3g)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[m3d_y], 1.0 - gamma)*g2 / g2g));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[i], 1.0 - gamma)*g2 / g2g));
						}
						p2 =  - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a);
						ret=inv_dw_dh(i)*p1 - p2;
						ret/=3.0;
				    }
				    return ret;
				}
			}
			if (analytic_test == 5)
			{
				if (mode==3)
				{
				    double ret=0.0;
					double t=Time[tstep];
				    if ((rp_split_3d&1)==0)
				    {
						double p1 = 0.0, p2 = 0.0,v1,v2,v3;
						double  vt= 1.0 - a5coef*2.0*pow(t, 2.0);
						v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
						v2=(pow(X[m3d_y], 2.0) / pow(Lx, 2.0));
					v3=(pow(Y[i], 2.0) / pow(Ly, 2.0));
					p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma*func_power)*g1fp / g1fpg) + (pow(sZ[m3d_x], 2.0 - gamma*func_power)*g2fp / g2fpg)));
					p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[m3d_y], 1.0 - gamma*func_power)*g1fp / g1fpg));
					p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[i], 1.0 - gamma*func_power)*g1fp / g1fpg));
						p2 =  - a5coef*2.0*(pow(t, 2.0 - alpha*func_power)*g2fp / g2fpa);
						if ((no_t==0)&&(no_x==0))
							ret= inv_dw_dh(i)*vt*p1- p2*v1*v2*v3;
						if (no_x)
							ret= -p2;
						if (no_t)
							ret= inv_dw_dh(i)*p1;
				    }	    
				    if ((rp_split_3d&1)==1)
				    {
						double p1 = 0.0, p2 = 0.0,v1=1.0,v2=1.0,v3=1.0;
						double  vt= 1.0 - a5coef*2.0*pow(t, 2.0);
						if (m3d_c==0)
						{
				    			v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[m3d_x], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[m3d_y], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[i], 1.0 - gamma*func_power)*g1fp / g1fpg) + (pow(sZ[i], 2.0 - gamma*func_power)*g2fp / g2fpg)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[i])*(pow(X[m3d_x], 1.0 - gamma*func_power)*g1fp / g1fpg));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[i])*(pow(Y[m3d_y], 1.0 - gamma*func_power)*g1fp / g1fpg));
						}
						if (m3d_c==1)
						{
				    			v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[i], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[m3d_y], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma*func_power)*g1fp / g1fpg) + (pow(sZ[m3d_x], 2.0 - gamma*func_power)*g2fp / g2fpg)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[i], 1.0 - gamma*func_power)*g1fp / g1fpg));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[m3d_y], 1.0 - gamma*func_power)*g1fp / g1fpg));
						}
						if (m3d_c==2)
						{
				    			v1=(pow(sZ[m3d_x], 2.0) / pow(sL, 2.0));
				    			v2=(pow(X[m3d_y], 2.0) / pow(Lx, 2.0));
							v3=(pow(Y[i], 2.0) / pow(Ly, 2.0));
							p1 = v2*v3*((2.0 / pow(sL, 2.0))*(0.5*(pow(sZ[m3d_x], 1.0 - gamma*func_power)*g1fp / g1fpg) + (pow(sZ[m3d_x], 2.0 - gamma*func_power)*g2fp / g2fpg)));
							p1 += v1*v3*((2.0 / pow(Lx, 2.0))*(0.5+sZ[m3d_x])*(pow(X[m3d_y], 1.0 - gamma*func_power)*g1fp / g1fpg));
							p1 += v1*v2*((2.0 / pow(Ly, 2.0))*(0.5+sZ[m3d_x])*(pow(Y[i], 1.0 - gamma*func_power)*g1fp / g1fpg));
						}
						p2 =  - a5coef*2.0*(pow(t, 2.0 - alpha*func_power)*g2fp / g2fpa);
						if ((no_t==0)&&(no_x==0))
							ret= inv_dw_dh(i)*vt*p1- p2*v1*v2*v3;
						if (no_x)
							ret= -p2;
						if (no_t)
							ret= inv_dw_dh(i)*p1;
						ret/=3.0;
				    }
				    return ret;
				}
			}
			if (analytic_test == 3)
				if (mode == 1)
				{
					double p1 = 0.0, p2 = 0.0;
					if (equation==0)
					{
						p1 = (2.0 / pow(sL, 3.0))*(0.5*(pow(Z[i], 1.0 - gamma)*g2 / g2g) + (pow(Z[i], 2.0 - gamma)*g3 / g3g));
						p1-=(1.0 / pow(sL, 3.0))*((g3/g2g)*pow(Z[i],1.0-gamma))*(2.0*nu+0.5*kt);
						p2 = ((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
						return inv_dw_dh(i)*p1 - p2;
					}
					if (equation==1)
					{
						p1=2.0*(D / pow(sL, 3.0))*((g3/g2g)*pow(Z[i],1.0-gamma));						
						p1+=(2.0 / pow(sL, 3.0))*(
							((g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							0.5*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									(0.5*((g2/g2g)*pow(Z[i],1.0-gamma))+
									 ((g3/g3g)*pow(Z[i],2.0-gamma)))+
							2.0*((g5/g5g)*pow(Z[i],4.0-gamma)/pow(sL,3.0)));
						p1-=nu*(4.0/pow(sL, 3.0))*(
							2.0*((g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							0.5*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									((g2/g2g)*pow(Z[i],1.0-gamma)));
						p1-=kt*(1.0/pow(sL, 3.0))*(
							2.0*((g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							0.5*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									((g2/g2g)*pow(Z[i],1.0-gamma)));
						p2 = 0.5*((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
						return (p1 - p2)/sigma;
					}
					if (equation==2)
					{
						p1=0.5*(lambda / pow(sL, 3.0))*((g3/g2g)*pow(Z[i],1.0-gamma));
						p1+=p_water*(2.0 / pow(sL, 3.0))*(
							(0.25*(g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							2.0*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									(0.5*((g2/g2g)*pow(Z[i],1.0-gamma))+
									 ((g3/g3g)*pow(Z[i],2.0-gamma)))+
							0.5*((g5/g5g)*pow(Z[i],4.0-gamma)/pow(sL,3.0)));
						p1-=nu*p_water*(4.0/pow(sL, 3.0))*(
							0.5*((g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							2.0*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									((g2/g2g)*pow(Z[i],1.0-gamma)));
						p1-=kt*p_water*(1.0/pow(sL, 3.0))*(
							0.5*((g4/g4g)*pow(Z[i],3.0-gamma)/pow(sL,3.0))+
							2.0*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)))*
									((g2/g2g)*pow(Z[i],1.0-gamma)));					
						p2 = 2.0*((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
						return (p1 - p2)/Ct;
					}
				}
			if (analytic_test == 4)
			{
				if (mode == 1)
				{
					double p1 = 0.0, p2 = 0.0;
					p1 = (2.0 / pow(sL, 3.0))*(0.5*(pow(Z[i], 1.0 - gamma)*g2 / g2g) + (pow(Z[i], 2.0 - gamma)*g3 / g3g));
					p1 *= (Time[tstep] / (10.0*24.0*3600.0));
					p2 = ((pow(Time[tstep], 3.0 - alpha)*g4 / g4a) - 2.0*(pow(Time[tstep], 2.0 - alpha)*g3 / g3a)) / (pow(10.0*24.0*3600.0, 3.0));
					p2 += (((pow(Time[tstep], 1.0 - alpha)*g2 / g2a)) / (10.0*24.0*3600.0))*(pow(sZ[i], 2.0) / pow(sL, 3.0));
					return inv_dw_dh(i)*p1 - p2;
				}
			}	
			if (analytic_test==6)
				return 0;			
		}
		else
		{
			if (Z[i]<=v)
			{
				if (rp_form==1)
					ret=(2.21-3.72*(Z[i]/v)+3.46*(Z[i]*Z[i]/(v*v))-1.87*(Z[i]*Z[i]*Z[i]/(v*v*v)))*Tr*day_night_coef();
				if (rp_form==2)
					ret=Tr*day_night_coef();
			}
		}
		// reduction = 1/(1+((C-thr)/(C50-thr))^rC_exp))
		if (is_r_C)
		if (C[i]>rC_thr)
			ret*=1.0/(1.0+pow((C[i]-rC_thr)/(rC_50-rC_thr),rC_exp));
		if (rp_split_3d&1)
		    ret/=3.0; // split right part into 3 passes
		return ret*Rp3d_mult()*inv_dw_dh(i);
	}
	double is_sprinkler()
	{
		if (first_spr)
		{
			for (int ix = 0;ix < _M + 2;ix++)
				for (int iy = 0;iy < K + 2;iy++)
					spr[ix][iy]=0;
			for (int i = 0;i < Nsprinklers_rows;i++)
				for (int j = 0;j < Nsprinklers;j++)
				{
					double x, y;
					double A = 0;
					x = Spr_offs + i*betw_spr_rows;
					y = Spr_spr_offs + j*betw_spr;
					for (int ix = 0;ix < _M + 2;ix++)
						for (int iy = 0;iy < K + 2;iy++)
						{
							double x0, y0, r;
							double max_mult = 0.0;
							x0 = X[ix];
							y0 = Y[iy];
							double cr = dLx[ix];
							if (dLy[iy] > cr) cr = dLy[iy];
							cr*=0.5;
							r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
							if (r <= (sprinklerR + cr))
								A+=dLx[ix]*dLy[iy];
						}
					if (A!=0)
					for (int ix = 0;ix < _M + 2;ix++)
						for (int iy = 0;iy < K + 2;iy++)
						{
							double x0, y0, r;
							double max_mult = 0.0;
							x0 = X[ix];
							y0 = Y[iy];
							double cr = dLx[ix];
							if (dLy[iy] > cr) cr = dLy[iy];
							cr*=0.5;
							r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
							if (r <= (sprinklerR + cr))
								spr[ix][iy] += 1.0/A;
						}
				}
			for (int ix = 0;ix < _M + 2;ix++)
			{
				spr[ix][1] += spr[ix][0];
				spr[ix][K - 1] += spr[ix][K + 1] + spr[ix][K];
			}
			for (int iy = 0;iy < K + 2;iy++)
			{
				spr[1][iy] += spr[0][iy];
				spr[_M - 1][iy] += spr[_M + 1][iy] + spr[_M][iy];
			}
			first_spr = 0;
			if (analytic_test==0)
			if (debug_level == 2)				
			for (int ix=0;ix<_M;ix++)
			    for (int iy=0;iy<K;iy++)
				if (log_file) fprintf(log_file,"spr %d %d %g\n",ix,iy,spr[ix][iy]);
		}
		return spr[m3d_x][m3d_y];
	}
	// upper boundary condition (DbU=Uc)
	double Uc()
	{
		double E=Ev*day_night_coef(); // evaporation
		double I=0.0; // irrigation
		if (rp_form == 3)
			E += Tr*day_night_coef();
		if (no_irr==0)
		{
			double aw=avg_root_layer_wetness();
			if (second_step==0)
			if (irr_start_step==-1)
				if (aw<min_wetness) // start irrigation
					irr_start_step=tstep;
			if (irr_start_step!=-1) // irrigation is being applied
			{
				I=irr_volume/irr_time;
				if (second_step==0)
				if (((Time[tstep] - Time[irr_start_step]) > irr_time) ||
					(aw >= max_wetness))
					{
						if (mode == 1)
							if (log_file) fprintf(log_file,"irrigation amount - %g mm \n", irr_am*1000.0);
						if (mode == 3)
							if (log_file) fprintf(log_file,"irrigation amount - %g m^3 \n", irr_am);
						irr_am = 0.0;
						I=0.0;
						irr_start_step = -1; // stop irrigation				
					}
			}
		}
		if (mode==3)
			I *= is_sprinkler();
		if (mode == 1)
			irr_am += I*tau[tstep];
		if (mode == 3)
			irr_am += I*tau[tstep]*dLx[m3d_x]*dLy[m3d_y];
		return dL[0]*(E-I)/ KK(1);
	}	
	// upper boundary condition for C
	double Cu()
	{
		if (no_irr==0)
			if (irr_start_step!=-1) // irrigation is being applied
				return irr_C;
		return 0.0;
	}
	// upper boundary condition for T
	double Tu()
	{
		return aT*day_night_coef();
	}
	// AHi-1 - RHi + BHi+1 = Psi
	double A_(int i)
	{
		if (equation==0) // H
			return Dg*KK(i-1)*inv_dw_dh(i)*kb(gamma, i - 1, i, i, Z,NULL)/(dL[i]*dL[i-1]);
		if (equation==1) // C
			return Dg*D*(kb(gamma, i - 1, i, i, Z,NULL) / (dL[i] * dL[i-1]) )/sigma;
		if (equation==2) // T
			return Dg*lambda*(kb(gamma, i - 1, i, i, Z,NULL) / (dL[i] * dL[i-1] ))/Ct;
		return 0.0;
	}
	double B(int i)
	{
		if (equation==0) // H
			return Dg*KK(i)*inv_dw_dh(i)*kb(gamma, i - 1, i, i, Z,NULL)/(dL[i]* dL[i]);
		if (equation==1) // C
			return Dg*D*(kb(gamma, i - 1, i, i, Z,NULL) / (dL[i] * dL[i]) )/sigma;
		if (equation==2) // T
			return Dg*lambda*(kb(gamma, i - 1, i, i, Z,NULL) / (dL[i] * dL[i]) )/Ct;
		return 0.0;
	}
	double R(int i)
	{
		if (equation == 0) // H
		{
			double a1=inv_dw_dh(i)*((KK(i) / (dL[i]* dL[i])) + (KK(i - 1)/(dL[i] *dL[i-1])))*Dg*kb(gamma, i - 1, i, i, Z,NULL);
			double a2=Da*kkk0 / tau[tstep];
			if ((mode==3)&&(implicit3d)&&(m3d_c!=2))
				a2=0.0;
			return a1 + a2;			
		}
		if (equation==1) // C
		{
			double a1=(D*Dg*kb(gamma, i - 1, i, i, Z,NULL) *((1/ (dL[i] * dL[i]) )+ (1 / (dL[i] * dL[i-1])))/sigma);
			double a2=Da*kkk0/tau[tstep];
			if ((mode==3)&&(implicit3d)&&(m3d_c!=2))
				a2=0.0;
			return a1+a2;
		}
		if (equation==2) // T
		{
			double a1=(lambda*Dg*kb(gamma, i - 1, i, i, Z,NULL)  *((1 / (dL[i] * dL[i])) + (1 / (dL[i] * dL[i - 1])))/Ct);
			double a2=Da*kkk0 / tau[tstep];
			if ((mode==3)&&(implicit3d)&&(m3d_c!=2))
				a2=0.0;
			return a1+a2;
		}
		return 0.0;
	}
	double testF(int i, int tstep,int eq=0)
	{
		if (analytic_test==1)
			return ((pow(sZ[i], 3.0) - 2.0*pow(sZ[i], 2.0)) / pow(sL, 3.0))+((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
		if (analytic_test == 2)
			return (pow(sZ[i], 2.0) / pow(sL, 3.0)) + ((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
		if (analytic_test ==3)
		{
			if (eq==0)
				return (pow(sZ[i], 2.0) / pow(sL, 3.0)) + ((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
			if (eq==1)
				return 2.0*(pow(sZ[i], 2.0) / pow(sL, 3.0))+ 0.5*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
			if (eq==2)
				return 0.5*(pow(sZ[i], 2.0) / pow(sL, 3.0))+ 2.0*((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
		}
		if (analytic_test == 4)
			return (Time[tstep]/ (10.0*24.0*3600.0))*(pow(sZ[i], 2.0) / pow(sL, 3.0)) + ((pow(Time[tstep], 3.0) - 2.0*pow(Time[tstep], 2.0)) / (pow(10.0*24.0*3600.0, 3.0)));
		return 0.0;
	}
	double DxtestF(int i, int tstep, int eq = 0)
	{
		if (analytic_test == 1)
			return ((((g4 / g3)*pow(Z[i], 3.0 - 1.0)) - 2.0*((g3 / g2)*pow(Z[i], 2.0 - 1.0))) / pow(sL, 3.0));
		if (analytic_test == 2)
			return ((g3 / g2)*pow(Z[i], 2.0 - 1.0)) / pow(sL, 3.0);
		if (analytic_test == 3)
		{
			if (eq == 0)
				return ((g3 / g2)*pow(Z[i], 2.0 - 1.0)) / pow(sL, 3.0);
			if (eq == 1)
				return 2.0*(((g3 / g2)*pow(Z[i], 2.0 - 1.0)) / pow(sL, 3.0));
			if (eq == 2)
				return 0.5*(((g3 / g2)*pow(Z[i], 2.0 - 1.0)) / pow(sL, 3.0));
		}
		if (analytic_test == 4)
			return ( Time[tstep] / (10.0*24.0*3600.0))*((g3 / g2)*pow(Z[i], 2.0 - 1.0)) / pow(sL, 3.0);
		return 0.0;
	}
	double testF3d(int i, int j,int k,double ts)
	{
		double v1 = 0.0, v2 = 0.0, v3 = 0.0,vt=0.0;
		int tstep=(int)ts;
		double kk=ts-tstep;
		double t=Time[tstep];
		if (kk!=0)
			t+=kk*tau[tstep];
		if (analytic_test == 1)
		{
			vt = pow(t, 3.0) - 2.0*pow(t, 2.0);
			v1 = ((pow(sZ[i], 3.0) - 2.0*pow(sZ[i], 2.0)) / pow(sL, 3.0));
			v2 = ((pow(X[j], 3.0) - 2.0*pow(X[j], 2.0)) / pow(Lx, 3.0));
			v3 = ((pow(Y[k], 3.0) - 2.0*pow(Y[k], 2.0)) / pow(Ly, 3.0));
		}
		if (analytic_test == 2)
		{
		    vt= 1.0 - 2.0*pow(t, 2.0);
		    v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
		    v2=(pow(X[j], 2.0) / pow(Lx, 2.0));
		    v3=(pow(Y[k], 2.0) / pow(Ly, 2.0));
		    return v1*v2*v3+vt;
		}
		if (analytic_test == 5)
		{
			if (no_t==0)
		    	vt= 1.0 - a5coef*2.0*pow(t, 2.0);
			else
				vt=1.0;
			if (no_x==0)
			{
				v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
				v2=(pow(X[j], 2.0) / pow(Lx, 2.0));
				v3=(pow(Y[k], 2.0) / pow(Ly, 2.0));
			}
			else
			{
				v1=v2=v3=1.0;
			}
		    return v1*v2*v3*vt;
		}
		if (analytic_test==6)
		{
			if ((i==0)||(j==0)||(k==0)) 
				return 1.0;
			else 
				return 0.0;
			
		}
		return v1 + v2 + v3 + vt;
	}
	double DxtestF3d(int i, int j,int k,double ts)
	{
		int tstep=(int)ts;
		double kk=ts-tstep;
		double t=Time[tstep];
		if (kk!=0)
			t+=kk*tau[tstep];
		if (analytic_test == 1)
		{
		    if (m3d_c==0)
			return ((((g4 / g3)*pow(sZ[i], 3.0 - 1.0)) - 2.0*((g3 / g2)*pow(sZ[i], 2.0 - 1.0))) / pow(sL, 3.0));
		    if (m3d_c==1)
			return ((((g4 / g3)*pow(X[j], 3.0 - 1.0)) - 2.0*((g3 / g2)*pow(X[j], 2.0 - 1.0))) / pow(Lx, 3.0));
		    if (m3d_c==2)
			return ((((g4 / g3)*pow(Y[k], 3.0 - 1.0)) - 2.0*((g3 / g2)*pow(Y[k], 2.0 - 1.0))) / pow(Ly, 3.0));
		}
		if (analytic_test == 2)
		{
		    double v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
		    double v2=(pow(X[j], 2.0) / pow(Lx, 2.0));
		    double v3=(pow(Y[k], 2.0) / pow(Ly, 2.0));
		    if (m3d_c==0)
			return v2*v3*(((g3 / g2)*pow(sZ[i], 2.0 - 1.0)) / pow(sL, 2.0));
		    if (m3d_c==1)
			return v1*v3*(((g3 / g2)*pow(X[j], 2.0 - 1.0)) / pow(Lx, 2.0));
		    if (m3d_c==2)
			return v1*v2*(((g3 / g2)*pow(Y[k], 2.0 - 1.0)) / pow(Ly, 2.0));
		}
		if (analytic_test == 5)
		{
		    double vt= 1.0 - a5coef*2.0*pow(t, 2.0);
		    double v1=(pow(sZ[i], 2.0) / pow(sL, 2.0));
		    double v2=(pow(X[j], 2.0) / pow(Lx, 2.0));
		    double v3=(pow(Y[k], 2.0) / pow(Ly, 2.0));
			if (no_t) vt=1;
			if (no_x) return 0.0;
		    if (m3d_c==0)
			return vt*v2*v3*(((g3 / g2)*pow(sZ[i], 2.0 - 1.0)) / pow(sL, 2.0));
		    if (m3d_c==1)
			return vt*v1*v3*(((g3 / g2)*pow(X[j], 2.0 - 1.0)) / pow(Lx, 2.0));
		    if (m3d_c==2)
			return vt*v1*v2*(((g3 / g2)*pow(Y[k], 2.0 - 1.0)) / pow(Ly, 2.0));
		}
		return 0.0;
	}
	double analytic_compare(FILE *fi)
	{
		double err=0.0;
		double avg_err=0.0,max_err=0.0;
		if (mode==1)
		for (int i=0;i<=N;i++)
			if (testF(i,tstep+1))
			{
				double ee=(U[i]-testF(i,tstep+1))*(U[i]-testF(i,tstep+1));
				if (analytic_test==3)
				{
					ee+=(C[i]-testF(i,tstep+1,1))*(C[i]-testF(i,tstep+1,1));
					ee+=(T[i]-testF(i,tstep+1,2))*(T[i]-testF(i,tstep+1,2));
				}
				err+=ee;
				if (ee>max_err) max_err=ee;
			}
		if (mode==3)
			for (int i = 1;i < sN;i++)
			for (int j = 1;j < _M;j++)
			for (int k = 1;k < K;k++)
				{
				    double v = b_U[idx(i, j, k)];
				    double a = testF3d(i, j, k, tstep+1);
				    double ee = (v - a)*(v - a);
				    if (ee>max_err) max_err=ee;
				    err += ee;			
				}
		if (mode==1)
		    avg_err=err/(N+1);
		if (mode==3)
		    avg_err=err/((sN-1)*(_M-1)*(K-1));
		if (fi)
		    fprintf(fi," %g(%g,%g)",err,max_err,avg_err);
		return err;
	}
	void alloc(int is_c,int is_t)
	{
		dL = new double[N + 2];
		sdL = new double[N + 2];
		Z = new double[N + 2];
		gZ = new double[N + 2];
		gZg = new double[N + 2];
		sZ = new double[N + 2];
		if (mode==1)
		{
			b_U=new double[N+2];
			b_Utimesum=new double[N+2];
			if (is_c)
			{
				b_C=new double[N+2];
				b_Ctimesum=new double[N+2];
			}
			else
				b_C=b_Ctimesum=NULL;
			if (is_t)
			{
				b_T=new double[N+2];
				b_Ttimesum=new double[N+2];
			}
			else
				b_T=b_Ttimesum=NULL;				
			U.set(b_U,0,0,3);
			C.set(b_C,0,0,3);
			T.set(b_T,0,0,3);
			if ((sum_alg==2)||(sum_alg==3))
			{
				Htmp=new std::vector<double>[N+2];
				for (int i=0;i<N+2;i++)
					for (int j=0;j<sum_param;j++)
						Htmp[i].push_back(0.0);
			}
		}
		if (mode==3)
		{
			b_U=new double[(sN+2)*(_M+2)*(K+2)+1];
			b_Utimesum=new double[(sN+2)*(_M+2)*(K+2)+1];
			if (is_c)
			{
				b_C=new double[(sN+2)*(_M+2)*(K+2)+1];
				b_Ctimesum=new double[(sN+2)*(_M+2)*(K+2)+1];
			}
			else
				b_C=b_Ctimesum=NULL;				
			if (is_t)
			{
				b_T=new double[(sN+2)*(_M+2)*(K+2)+1];
				b_Ttimesum=new double[(sN+2)*(_M+2)*(K+2)+1];
			}
			else
				b_T=b_Ttimesum=NULL;
			dLx=new double[_M+2];
			dLy=new double[K+2];
			X=new double[_M+2];
			Y=new double[K+2];
			gX=new double[_M+2];
			gY=new double[K+2];
			gXg=new double[_M+2];
			gYg=new double[K+2];
			rp_mult = new double *[_M + 2];
			spr = new double *[_M + 2];
			for (int i = 0;i < _M + 2;i++)
			{
				rp_mult[i] = new double[K + 2];
				spr[i] = new double[K + 2];
			}
			if ((sum_alg==2)||(sum_alg==3))
			{
				Htmp=new std::vector<double>[(sN+2)*(_M+2)*(K+2)];
				for (int i=0;i<(sN+2)*(_M+2)*(K+2);i++)
					for (int j=0;j<sum_param;j++)
						Htmp[i].push_back(0.0);
			}
		}
		sAl=Al=new double[sN+_M+K+2];
		sBt=Bt=new double[sN+ _M + K + 2];
		sOm=Om=new double[sN+ _M + K + 2];
		sRS=RS=new double[sN+_M+K+2];
		if (implicit3d == 1)
		{
			delete[] Om;
			delete[] Al;
			delete[] Bt;
			sOm=Om = new double[(sN + 2)*(_M + 2)*(K + 2) + 1];
			sAl=Al = new double[(sN + 2)*(_M + 2)*(K + 2) + 1];
			sBt=Bt = new double[(sN + 2)*(_M + 2)*(K + 2) + 1];
		}
		if (implicit==1) // full matrix
		{
			if (implicit3d==1)
			{
				sMat=Mat=new double*[(sN+2)*(_M+2)*(K + 2)+1];
				for (int i=0;i<=(sN+2)*(_M+2)*(K + 2);i++)
					Mat[i]=new double[(sN+2)*(_M+2)*(K + 2)+1];
				Mat1d=new double*[sN+_M+K + 2];
				for (int i=0;i<sN+_M+K + 2;i++)
					Mat1d[i]=new double[sN+_M+K + 2];
			}
			if (implicit3d==0)
			{
				sMat=Mat=new double*[sN+_M+K + 2];
				for (int i=0;i<sN+_M+K + 2;i++)
					Mat[i]=new double[sN+_M+K + 2];
			}
		}
		if (implicit == 2) // matrix split on tridiagonal and toeplitz for uniform grid
		{
			sMat=Mat = new double*[3];
			for (int i = 0;i<3;i++)
				Mat[i] = new double[sN + _M + K + 2];
			Mcoefs = new double*[3];
			for (int i = 0;i<3;i++)
				Mcoefs[i] = new double[sN + _M + K + 2];
			Mcoefs2 = new double*[3];
			for (int i = 0;i<3;i++)
				Mcoefs2[i] = new double[sN + _M + K + 2];
			if (mode==3)
			{
				Mat_3d = new double*[7];
				for (int i = 0;i<7;i++)
					Mat_3d[i] = new double[(sN+2)*(_M+2)*(K + 2)+1];
				for (int j=0;j<3;j++)
				{
					Mcoefs_3d[j] = new double*[3];
					for (int i = 0;i<3;i++)
						Mcoefs_3d[j][i] = new double[(sN+2)*(_M+2)*(K + 2)+1];
					Mcoefs2_3d[j] = new double*[3];
					for (int i = 0;i<3;i++)
						Mcoefs2_3d[j][i] = new double[(sN+2)*(_M+2)*(K + 2)+1];
				}
				delete [] RS;
				sRS=RS=new double[(sN+2)*(_M+2)*(K + 2)+1];
			}
			for (int j = 0;j < 3;j++)
				Tm[j] = new double[sN + _M + K + 2];
		}
	}
	H_solver(int atest,int rpf,int cvf,int irr,int bc,double a,double b,int is_c,int is_t,int is_rc,int m,FILE *log,int kpf)
	{
		time_i1=-1;
		a3k = 1;
		last_points=NULL;
		int ssa=sum_alg;
		if (sum_alg==3)
			sum_alg=2;
		mode=m;
		s_is_c=is_c;
		s_is_t=is_t;
		if (atest==3)
			is_c=is_t=s_is_c=s_is_t=1;
		vart_main=0;
#ifdef _OPENMP
		if (toeplitz_mult_alg == 1) 
		{ 
			printf("FFT do not work with openmp\n"); 
			toeplitz_mult_alg=0;
		}
#endif
#ifdef USE_MPI
		implicit=2;
		toeplitz_mult_alg=2;
		printf("only series-based implicit schemes for MPI mode\n");
#endif
		// implicit check - 2 only for uniform grid
		if (implicit == 2)
		{
			if ((toeplitz_mult_alg == 0) || (toeplitz_mult_alg == 1))
				if (varZ || varXY || (func_in_kernel !=0))
				{
					if (log_file) fprintf(log_file,"Integral matrix is not Toeplitz for variable time steps or non-x function in integral kernel (%d %d %d)\n",varZ,varXY,func_in_kernel);
					toeplitz_mult_alg=0;
					implicit=1;
				}
			if (b==1)
			{
				if (log_file) fprintf(log_file,"no implicit=2 for gamma=1\n");
				implicit=1;
			}
		}
		if (mode==3)
			if (implicit3d==1)
				if (implicit==0)
					implicit=1;

		alloc(is_c,is_t);

		L = 3.0; // total depth
		H0 = -800.0 / 1000.0; // initial H	
		v = 0.5; // root layer depth
		Tr = 3.0*0.001 / (24 * 3600); // transp. 3.0 mm per day
		Ev = 4.0*0.001 / (24 * 3600); // evapor. 4.0 mm per day
		k = 0.0002 / 100.0; // saturated filtration coefficient
		alpha=a;
		gamma=b;
		calc_gammas();
		minDL = L / ((double)N*Zvar_coef);
		maxDL = L / N;

		if (is_c) nu=0.00132/(24.0*3600.0); else nu=0.0;
		if (is_t) kt=0.432*1e-4/(24.0*3600.0); else kt=0.0;
		sigma=0.655; 
		D=0.000184/(24.0*3600.0);
		C0=0;
		irr_C=2.5;
		Ct=2000000;
		lambda=69120.0/(24.0*3600.0);
		Cp=800;
		aT=15;
		T0=15;		
		
		bottom_cond=bc; // 0 -dU/dn=0, 1 - U=H0
		analytic_test=atest;
		k_form=kpf;
		rp_form=rpf;
		cv_form=cvf;
		no_irr=irr;

		is_r_C=is_rc; // reduction = 1/(1+((C-thr)/(C50-thr))^rC_exp))
		rC_thr=1.862; // 720 cm osmotic pressure - ~1862 mg/L
		rC_50=7.212; // 2650 cm - ~7212 mg/L
		rC_exp=1.35;
		
		min_wetness=0.385 ; // minimal wetness
		max_wetness=0.54; // needed wetness
		p_soil = 2540;
		p_water=1000;

		irr_volume = (max_wetness - min_wetness)*p_soil*v / p_water; // in m
		irr_time = 3600 * 5; // time range to apply irrigation
		irr_start_step = -1;

		// 3D
		Lx=4; // lengths of xy-domain 
		Ly=4;
		XYforceexp=1.0;

		Nrows = 3; // number of plants rows (along X)
		Nplants=3; //  number of plants in a row (along Y)
		Nsprinklers=2; //  number of sprinklers in sprinkler row (along Y)
		Nsprinklers_rows=Nrows-1; // number of sprinklers rows (along X)
		Poffs=1; // offset of first plants row from x=0 and x=Lx
		Ploffs = 1; // offset of first plant from y=0 and y=Ly
		Spr_spr_offs=1; // offset of first sprinkler from y=0 and y=Ly
		betw_rows=(Lx-Poffs*2)/(Nrows-1); // distance between plant rows
		Spr_offs=Poffs+betw_rows*0.5; // offset of first sprinkler row from x=0 and x=Lx
		betw_spr_rows=(Lx-Spr_offs*2)/(Nsprinklers_rows-1); // distance between sprinkler rows
		betw_pl = (Ly - Ploffs * 2) / (Nplants - 1); // distance between plants
		betw_spr = (Ly - Spr_spr_offs * 2) / (Nsprinklers - 1); // distance between sprinklers
		rootR=1.0; // plants root system width
		sprinklerR = 0.5;// varxy test - 0.1, 0.5; // effective radius of sprinklers

		set_xyz_steps();
		sL = L;
		tstep = -1;
		Time.push_back(0.0);
		equation = 0; // 0 -H, 1-C,2-T
		second_step=0;
		irr_am=0.0;
		first_rp=first_spr=1;
		m3d_c = 0;
		if (mode==3)
		{
		    m3d_x=m3d_y=0;
    		Rp3d_mult();
		    is_sprinkler();
		}

		if (mode == 3) // irr.volume in m^3 per sprinkler for 3d
			irr_volume *= (Lx*Ly)/((double)Nsprinklers*Nsprinklers_rows);
		if (analytic_test==3)
		{
			sigma=1.5;
			D=1.0;
			Ct=2.0;
			lambda=1.0;
			Cp=1.0;
			nu=3.0;
			kt=0.25;
			p_water=1.0;
			is_c=is_t=1;
		}
		if (alpha != 1.0)
			Da = 1.0 / Gamma(1.0 - alpha);
		else
			Da = 1.0;
		if (gamma != 1.0)
			Dg = 1.0 / Gamma(1.0 - gamma);
		else
			Dg = 1.0;
		
		// initial conditions
		if (mode==1)
		for (int i = 0;i < N + 1;i++)
			if (analytic_test==0)
			{
				U[i] = H0-Z[i];
				if ((nu!=0.0)||(is_r_C)) C[i] = C0; else{if (b_C) C[i]=0.0;}
				if (kt!=0.0) T[i] = T0; else {if (b_T) T[i]=0.0;}
			}
			else
			{
				U[i] = testF(i, 0);
				if (analytic_test==3)
				{
					C[i] = testF(i, 0,1);
					T[i] = testF(i, 0,2);
				}
			}
		if (mode == 3)
		{
			memset(b_U, 0, (sN + 2)*(_M + 2)*(K + 2));
			if (s_is_c)
				memset(b_C, 0, (sN + 2)*(_M + 2)*(K + 2));
			if (s_is_t)
				memset(b_T, 0, (sN + 2)*(_M + 2)*(K + 2));
			for (int i = 0;i <= sN + 1;i++)
				for (int j = 0;j <= _M + 1;j++)
					for (int k = 0;k <= K + 1;k++)
					{
						if (analytic_test == 0)
						{
							b_U[idx(i, j, k)] = H0 - Z[i];
							if ((nu != 0.0) || (is_r_C)) b_C[idx(i, j, k)] = C0; 
							if (kt != 0.0) T[idx(i, j, k)] = T0; 
						}
						else
							b_U[idx(i, j, k)] = testF3d(i, j, k, 0);
						if (i==(sN+1))
							b_U[idx(i, j, k)]=0.0;
						if (j==(_M+1))
							b_U[idx(i, j, k)]=0.0;
						if (k==(K+1))
							b_U[idx(i, j, k)]=0.0;
						if (s_is_c)
						{
							if (i==(sN+1))
								b_C[idx(i, j, k)]=0.0;
							if (j==(_M+1))
								b_C[idx(i, j, k)]=0.0;
							if (k==(K+1))
								b_C[idx(i, j, k)]=0.0;
						}
						if (s_is_t)
						{
							if (i==(sN+1))
								b_T[idx(i, j, k)]=0.0;
							if (j==(_M+1))
								b_T[idx(i, j, k)]=0.0;
							if (k==(K+1))
								b_T[idx(i, j, k)]=0.0;
						}						
					}
		}
		// fill integral matrices coefficients
		if (implicit == 2)
		{
			if ((toeplitz_mult_alg == 0) || (toeplitz_mult_alg == 1))
			{
				double Dg2=1.0/g2g;
				A = 1 - gamma;
				for (int i = 1;i <= sN;i++)
					if (A != 0.0)
						Tm[0][i] = Dg2*fabs(pow(fabs(Z[1] - Z[i]), A) - pow(fabs(Z[1] - Z[i+1]), A));
					else
						Tm[0][i] = ((i == 1) ? 1 : 0);
				if (mode == 3)
				{
					for (int i = 1;i <= _M;i++)
						if (A != 0.0)
							Tm[1][i] = Dg2*fabs(pow(fabs(X[1] - X[i]), A) - pow(fabs(X[1] - X[i+1]), A));
						else
							Tm[1][i] = ((i == 1) ? 1 : 0);
					for (int i = 1;i <= K;i++)
						if (A != 0.0)
							Tm[2][i] = Dg2*fabs(pow(fabs(Y[1] - Y[i]), A) - pow(fabs(Y[1] - Y[i+1]), A));
						else
							Tm[2][i] = ((i == 1) ? 1 : 0);
				}
			}
			if (toeplitz_mult_alg == 2)
			{
				int nit, niter, total, old_total[3];
				int NN;
				double *ZZ;
				int max_niter;
				// calculate main diagonal integrals using taylor series
				Tm_diagonals[0].push_back(new double[sN + 2]);
				Tm_ndiags[0]=1;
				if (mode == 3)
				{
					Tm_diagonals[1].push_back(new double[_M + 2]);
					Tm_diagonals[2].push_back(new double[K + 2]);
					Tm_ndiags[1]=Tm_ndiags[2]=1;
				}
				for (int i = 1;i < sN;i++)
				{
					Tm_diagonals[0].push_back(new double[sN + 2]);
					Tm_diagonals[0][i][0]= kb(gamma,i-1, i, i,Z,NULL);
				}
				if (mode == 3)
				{
					for (int i = 1;i < _M;i++)
					{
						Tm_diagonals[1].push_back(new double[_M + 2]);
						Tm_diagonals[1][i][0] = kb(gamma, i - 1, i, i, X, NULL);
					}
					for (int i = 1;i < K;i++)
					{
						Tm_diagonals[2].push_back(new double[K + 2]);
						Tm_diagonals[2][i][0] = kb(gamma, i - 1, i, i, Y, NULL);
					}
				}
				for (int dim = 0;dim < mode;dim++)
				{
					if (dim == 0) { NN = sN; ZZ = Z; }
					if (dim == 1) { NN = _M; ZZ = X; }
					if (dim == 2) { NN = K; ZZ = Y; }
					if (NN<=2) continue;
					// set initial total
					total = 1;
					max_niter = 0;
					for (int i = 2;i < NN;i++)
					{
						_kb_row2(ZZ[i - 2], ZZ[i -  1], g(ZZ[i]), dim, i - 2, gamma, nit, &niter);
						if (niter > max_niter)
							max_niter = niter;
					}
					total += max_niter;
					old_total[dim] = total;
					while (true)
					{
						// calculate total number of coefficients needed to calc the complete matrix
						total = Tm_ndiags[dim] + 1;
						if (total==(NN-1))
						{
							if (tma2_all_diags)
							{
								// calculate next diagonal
								Tm_ndiags[dim]++;
								int idx = Tm_ndiags[dim] - 1;
								for (int i = idx + 1;i < NN;i++)
									Tm_diagonals[dim][i][idx] = kb(gamma, i - idx - 1, i - idx, i, ZZ, NULL);
								old_total[dim] = total;
							}
							break;
						}
						_kb_row2(ZZ[NN - Tm_ndiags[dim] - 3], ZZ[NN - Tm_ndiags[dim] - 2], g(ZZ[NN-1]), dim, NN - Tm_ndiags[dim] - 3, gamma, nit, &max_niter);
						total += max_niter;
						// try to add next diagonal to the list of computed ones
						if ((total < old_total[dim])||tma2_all_diags)
						{
							// calculate next diagonal
							Tm_ndiags[dim]++;
							int idx = Tm_ndiags[dim] - 1;
							for (int i = idx + 1;i < NN;i++)
								Tm_diagonals[dim][i][idx] = kb(gamma, i - idx - 1, i - idx, i, ZZ, NULL);
							old_total[dim] = total;
						}
						else
							break;
					}
					// calculate series coefficients
					for (int i = Tm_ndiags[dim] + 2;i < NN;i++)
						_kb_row2(ZZ[i - Tm_ndiags[dim] - 2], ZZ[i - Tm_ndiags[dim] - 1], g(ZZ[i]), dim, i - Tm_ndiags[dim] - 2, gamma, nit, &niter);
					for (int i =0;i < NN-2;i++)
					{
						int size = total - Tm_ndiags[dim];
						double mult = g(ZZ[i + 1]);
						if (row2_precalc.size()<=dim) 
							size=0;
						else
						{
							if (row2_precalc[dim].size()<=i)
								size=0;
							else
								if (row2_precalc[dim][i].size() < size)
									size = row2_precalc[dim][i].size();
						}
						Tm_coefs[dim].push_back(std::vector< double>());
						for (int j = 0;j < size;j++)
						{
							Tm_coefs[dim][i].push_back(row2_precalc[dim][i][j]*mult);
							mult *= g(ZZ[i + 1]);
						}
					}
					// test
					if (debug_level)
					{
						if (log_file) fprintf(log_file,"N %d fixed %d total %d\n",NN,Tm_ndiags[dim],total);
						row2_precalc[dim].clear();
						double err = 0.0;
						for (int i = 1;i < NN;i++)
							for (int j = 1;j <= i;j++)
							{
								double v0, v1 = kb(gamma, j - 1, j, i, ZZ, NULL);
								if ((i - j) < Tm_ndiags[dim])
									v0 = Tm_diagonals[dim][i][i - j];
								else
									v0 = _kb_row2_fixed_coefs(g(ZZ[i]),pow(g(ZZ[i]),-gamma), &Tm_coefs[dim][j - 1][0],Tm_coefs[dim][j - 1].size(), gamma);
								err += (v1 - v0)*(v1 - v0);
							}
						if (log_file) fprintf(log_file,"impl matrix interpol err %g\n", err);
					}
				}
				// process for right-side derivative
				if (space_der == 1)
				{
					row2_precalc[0].clear();
					r_Tm_diagonals[0].push_back(new double[sN + 2]);
					r_Tm_ndiags[0] = 1;
					if (mode == 3)
					{
						row2_precalc[1].clear();
						row2_precalc[2].clear();
						r_Tm_diagonals[1].push_back(new double[_M + 2]);
						r_Tm_diagonals[2].push_back(new double[K + 2]);
						r_Tm_ndiags[1] = Tm_ndiags[2] = 1;
					}
					for (int i = 1;i < sN;i++)
					{
						r_Tm_diagonals[0].push_back(new double[sN + 2]);
						r_Tm_diagonals[0][i][0] = kb(gamma, i+1, i+2, i, Z, NULL);
					}
					if (mode == 3)
					{
						for (int i = 1;i < _M;i++)
						{
							r_Tm_diagonals[1].push_back(new double[_M + 2]);
							r_Tm_diagonals[1][i][0] = kb(gamma, i + 1, i + 2, i, X, NULL);
						}
						for (int i = 1;i < K;i++)
						{
							r_Tm_diagonals[2].push_back(new double[K + 2]);
							r_Tm_diagonals[2][i][0] = kb(gamma, i + 1, i + 2, i, Y, NULL);
						}
					}
					for (int dim = 0;dim < mode;dim++)
					{
						if (dim == 0) { NN = sN; ZZ = Z; }
						if (dim == 1) { NN = _M; ZZ = X; }
						if (dim == 2) { NN = K; ZZ = Y; }
						// set initial total
						total = 1;
						max_niter = 0;
						for (int i = 1;i < NN-4;i++)
						{
							_kb_row2(ZZ[i + 2], ZZ[i + 3], g(ZZ[i]), dim, i + 2, gamma, nit, &niter);
							if (niter > max_niter)
								max_niter = niter;
						}
						total += max_niter;
						old_total[dim] = total;
						while (true)
						{
							// calculate total number of coefficients needed to calc the complete matrix
							total = r_Tm_ndiags[dim] + 1;
							_kb_row2(ZZ[NN - 2], ZZ[NN - 1], g(ZZ[NN - r_Tm_ndiags[dim]-4]), dim, NN - 2, gamma, nit, &max_niter);
							total += max_niter;
							// try to add next diagonal to the list of computed ones
							if ((total < old_total[dim])||tma2_all_diags)
							{
								// calculate next diagonal
								r_Tm_ndiags[dim]++;
								int idx = r_Tm_ndiags[dim] - 1;
								for (int i = 1;i < NN- idx - 1;i++)
									r_Tm_diagonals[dim][i][idx] = kb(gamma, i + idx, i + idx+1, i, ZZ, NULL);
								old_total[dim] = total;
							}
							else
								break;
						}
						// calculate series coefficients
						for (int i = 1;i < NN - (r_Tm_ndiags[dim] + 3);i++)
							_kb_row2(ZZ[i + (r_Tm_ndiags[dim] + 1)], ZZ[i + (r_Tm_ndiags[dim] + 2)], g(ZZ[i]), dim, i + (r_Tm_ndiags[dim] + 1), gamma, nit, &niter);
						for (int i = 0;i < NN - 2;i++)
						{
							int size = total - r_Tm_ndiags[dim];
							double mult = pow(g(ZZ[i]),1-gamma);
							if (row2_precalc.size() <= dim)
								size = 0;
							else
							{
								if (row2_precalc[dim].size() <= i)
									size = 0;
								else
									if (row2_precalc[dim][i].size() < size)
										size = row2_precalc[dim][i].size();
							}
							r_Tm_coefs[dim].push_back(std::vector< double>());
							for (int j = 0;j < size;j++)
							{
								r_Tm_coefs[dim][i].push_back(row2_precalc[dim][i][j] * mult);
								mult /= g(ZZ[i]);
							}
						}
						// test
						if (debug_level)
						{
							if (log_file) fprintf(log_file, "N %d fixed %d total %d\n", NN, r_Tm_ndiags[dim], total);
							row2_precalc[dim].clear();
							double err = 0.0;
							for (int i = 1;i < NN;i++)
								for (int j = i+1;j <NN-2;j++)
								{
									double v0, v1 = kb(gamma, j, j+1, i, ZZ, NULL);
									if ((j - i) < r_Tm_ndiags[dim])
										v0 = r_Tm_diagonals[dim][i][j - i];
									else
										v0 = _kb_row2_fixed_coefs(g(ZZ[i]), pow(g(ZZ[i]),-gamma),&r_Tm_coefs[dim][j][0], r_Tm_coefs[dim][j].size(), gamma,1);
									err += (v1 - v0)*(v1 - v0);
								}
							if (log_file) fprintf(log_file, "impl matrix interpol err %g\n", err);
						}
					}
				}
			}
		}
		if (full_test==0)
		{
			fprintf(log,"H_solver%d(analytic %d,right part %d, C(H) %d,no_irr %d,boundary cond %d,a %g,b %g,is_C %d,is_T %d,is_R(C) %d)\n",mode,atest,rpf,cvf,irr,bc,a,b,is_c,is_t,is_r_C);
			printf("H_solver%d(analytic %d,right part %d, C(H) %d,no_irr %d,boundary cond %d,a %g,b %g,is_C %d,is_T %d,is_R(C) %d)\n",mode,atest,rpf,cvf,irr,bc,a,b,is_c,is_t,is_r_C);
			fflush(stdout);
		}
		sum_alg=ssa;
	}
	void init_time_step(double t)
	{
		Time.push_back(Time[Time.size()-1]+t);
		tau.push_back(t);
		tstep++;
		kkk0=kb(alpha, tstep, tstep + 1, tstep + 1, &Time[0],(void *)&Time);
		precalc_kb_3();
	}
	// set working row and its saved values vector
	void get_F_oldF(row *U_,std::vector<double*> **old,double **points=NULL)
	{
		if (mode==1)
		{
			if (equation==0)
			{
				U_[0]=U;
				old[0]=&oldH;
			}
			if (equation==1)
			{
				U_[0]=C;
				old[0]=&oldC;
			}
			if (equation==2)
			{
				U_[0]=T;		
				old[0]=&oldT;
			}
			if (points)
				points[0]=Z;
		}
		if (mode==3)
		{
			if (equation==0)
			{
				U_[0].set(b_U,m3d_x,m3d_y,m3d_c);
				old[0]=&oldH;
			}
			if (equation==1)
			{
				U_[0].set(b_C,m3d_x,m3d_y,m3d_c);
				old[0]=&oldC;
			}
			if (equation==2)
			{
				U_[0].set(b_T,m3d_x,m3d_y,m3d_c);		
				old[0]=&oldT;
			}
			if (points)
			{
				if (m3d_c==0)
					points[0]=sZ;
				if (m3d_c==1)
					points[0]=X;
				if (m3d_c==2)
					points[0]=Y;
			}
		}
	}
	// alpha coefficients
	void al1()
	{
		if (analytic_test==0)
		{
			if ((mode==1)||((mode==3)&&(m3d_c==0)))
			{
				if (equation==0)
				{
				    if (inv_dw_dh(0)!=1e20) // first order condition in saturated zone
						Al[1] = 1;
				    else
						Al[1] = 0;
				}
				if (equation==1)
					Al[1] = 0;
				if (equation==2)
					Al[1] = 0;
			}
			else
				Al[1] = 1;
		}
		else
			Al[1] = 0;
		for (int i = 1;i < N;i++)
			Al[i + 1] = B(i) / (R(i) - A_(i)*Al[i]);
	}
	// clear saved solution for alg2 to alg0 transition when doing automated algorithm selection
	void time_summing_alg0_to_alg2()
	{
		if (row2_precalc.size()>time_i1) if (row2_precalc[time_i1].size()) row2_precalc[time_i1][0].clear();
		kb(alpha, tstep-1, tstep, tstep+1, &Time[0], (void *)&Time,time_i1,0);
		for (int id=0;id<(sN+2)*(_M+2)*(K+2);id++)
		{
			Htmp[id].resize(sum_param);
			for (int j=0;j<sum_param;j++)
				Htmp[id][j]=0.0;
		}
		for (int i=2;i<oldH.size();i++)
		{
				double mult;
				double gtj=g(Time[i]);	
				if (row2_precalc.size()>time_i1) if (row2_precalc[time_i1].size()) row2_precalc[time_i1][0].clear();
				kb(alpha, i-2, i-1, i, &Time[0], (void *)&Time,time_i1,0);
				for (int id=0;id<(sN+2)*(_M+2)*(K+2);id++)
				{
					double diff = (oldH[i-1][id] - oldH[i-2][id]);
					// update Sn
					mult = g(Time[i-1]);
					for (int j=0;j<sum_param;j++)
					if (row2_precalc[time_i1][0].size()>j)
					{
						Htmp[id][j]+=mult*row2_precalc[time_i1][0][j]*diff;
						mult *= g(Time[i-1]);
					}
				}
		}
		if (oldH.size()>2)
		{
			oldH.erase(oldH.begin(),oldH.end()-2);
			if (s_is_c)
				oldC.erase(oldC.begin(),oldC.end()-2);
			if (s_is_t)
				oldT.erase(oldT.begin(),oldT.end()-2);
		}
	}
	// right part for explicit scheme
	std::vector<double> kb_3_cache;
	void precalc_kb_3()
	{
		row U_;
		std::vector<double*> *old;
		equation=0;
		get_F_oldF(&U_,&old);
		kb_3_cache.clear();
		if (sum_alg==2)
		{
			if (row2_precalc.size()>time_i1) if (row2_precalc[time_i1].size()) row2_precalc[time_i1][0].clear();
			kb(alpha, tstep-1, tstep, tstep+1, &Time[0], (void *)&Time,time_i1,0);
			return;			
		}
		if (old->size()>=2)
			for (int t = 0;t < old->size() - 1;t++)
				kb_3_cache.push_back(kb(alpha, t, t+1, tstep+1, &Time[0], (void *)&Time));
	}
	double time_summing(int i,int j,std::vector<double*> *old,row &U_,int iid=-1)
	{
		int id=1 + i*BS + j;
		double time_sum = 0;
		if (iid!=-1) id=iid;
		if (mode == 3)
		{
			if (m3d_c == 0)
				id = idx(1 + i*BS + j, m3d_x, m3d_y);
			if (m3d_c == 1)
				id = idx(m3d_x, 1 + i*BS + j, m3d_y);
			if (m3d_c == 2)
				id = idx(m3d_x, m3d_y,1 + i*BS + j);
		}
#ifdef OCL
		if (use_ocl)
			if (equation==0)
				return Sums[id];
#endif
		double kv, diff;
		if (old->size()>=2)
		{
			if (m3d_c==2)
			{
				if (sum_alg!=2)
				for (int t = old->size() - 2;t >=0 ;t--)
				{
					kv = kb_3_cache[t];
					if (old[0][t+1]!=NULL)
					{
						diff = (old[0][t + 1][id] - old[0][t][id]);
						if (sum_alg==1) // restricted summing
							if (fabs(kv)<sum_param)
							{
								// clear old saved solutions
#ifndef _OPENMP
								if (dont_clear_sum_alg1==0)
								for (t--;t>=0;t--)
									if (old[0][t]!=NULL)
									{
										delete [] old[0][t];
										old[0][t]=NULL;
									}
#endif									
								break;
							}
						time_sum += kv*diff/tau[t];
					}
					else
						break;
				}
				if (sum_alg==2) // series expansion
				{
					double mult;
					double gtj=g(Time[tstep+1]);	
					diff = (old[0][1][id] - old[0][0][id]);
					// update Sn
					mult = g(Time[tstep]);
					for (int j=0;j<sum_param;j++)
					if (row2_precalc[time_i1][0].size()>j)
					{
						Htmp[id][j]+=mult*row2_precalc[time_i1][0][j]*diff;
						mult *= g(Time[tstep]);
					}
					time_sum=_kb_row2_fixed_coefs(gtj,pow(gtj,-alpha),(double *)&Htmp[id][0],Htmp[id].size(),alpha)/tau[tstep];
				}						
				if (equation==0) b_Utimesum[id]=time_sum; // save time sum on first run
				if (equation==1) b_Ctimesum[id]=time_sum; 
				if (equation==2) b_Ttimesum[id]=time_sum; 
			}
			else // load cached time sum
			{
				if (equation==0) time_sum=b_Utimesum[id];
				if (equation==1) time_sum=b_Ctimesum[id];
				if (equation==2) time_sum=b_Ttimesum[id];
			}
		}
		return time_sum;
	}
	void Om1()
	{
		row U_;
		std::vector<double*> *old;
		double mult=1.0;
		A=1-gamma;
		get_F_oldF(&U_,&old);
		if (mode==3)
		    mult=1.0/3.0;
		// div (k Db_z F)
		for (int i = 1;i < N;i++)
		{
			if (equation == 0)
					F[i] = (KK(i)*U_[i + 1]/(dL[i]*dL[i])) - ((KK(i) / (dL[i] * dL[i])) + (KK(i - 1)/(dL[i]*dL[i-1])))*U_[i] + (KK(i-1)*U_[i - 1] / (dL[i] * dL[i - 1]));
			if (equation==1)
				F[i] = (D/sigma)*((U_[i + 1] / (dL[i] * dL[i])) - ((1.0 / (dL[i] * dL[i - 1]))+(1.0/ (dL[i] * dL[i])))*U_[i] +( U_[i-1] / (dL[i] * dL[i - 1])));
			if (equation==2)
				F[i] = (lambda/Ct)*((U_[i + 1] / (dL[i] * dL[i])) - ((1.0 / (dL[i] * dL[i - 1])) + (1.0 / (dL[i] * dL[i])))*U_[i] +( U_[i-1] / (dL[i] * dL[i - 1])));
		}
		for (int i = 0;i < NB;i++)
		{
			calc_v(i, F, BVK,Z);
			for (int j = 0;j < BS;j++)
			{
				if (equation == 0)
					BVK[j] *= inv_dw_dh(1 + i*BS + j);
				Om[1 + i*BS + j] = -Dg*BVK[j] - Da*(kkk0/ tau[tstep] )*U_[1 + i*BS + j]+Rp(1 + i*BS + j);
			}
		}		
		// div (k Db_z H)
		if (equation!=0)
		{
			for (int i = 1;i < N;i++)
			{
				if (equation==1)
					F[i] = (1.0/sigma)*((KK(i)*C[i]*U[i + 1] / (dL[i] * dL[i])) - ((KK(i)*C[i] / (dL[i] * dL[i]) )+(KK(i-1)*C[i-1] / (dL[i] * dL[i - 1])))*U[i] + (KK(i-1)*C[i-1]*U[i-1] / (dL[i] * dL[i - 1])));
				if (equation==2)
					F[i] = (p_water/Ct)*((KK(i)*T[i]*U[i + 1] / (dL[i] * dL[i])) - ((KK(i)*T[i] / (dL[i] * dL[i]) )+ (KK(i-1)*T[i-1] / (dL[i] * dL[i - 1])))*U[i] +( KK(i-1)*T[i-1]*U[i-1] / (dL[i] * dL[i - 1])));
			}
			for (int i = 0;i < NB;i++)
			{
				calc_v(i, F, BVK,Z,1);
				for (int j = 0;j < BS;j++)
					Om[1 + i*BS + j] -= Dg*BVK[j];
			}
		}
		// div (k Db_z C)
		if (nu!=0.0)
		{
			for (int i = 1;i < N;i++)
			{
				if (equation==0)
						F[i] = nu*((C[i + 1] / (dL[i] * dL[i]) )- ((1.0 / (dL[i] * dL[i ]))+(1.0 / (dL[i] * dL[i - 1])))*C[i] +( C[i-1] / (dL[i] * dL[i - 1])));
				if (equation==1)
					F[i] = (nu/sigma)*((C[i]*C[i + 1] / (dL[i] * dL[i])) - ((C[i] / (dL[i] * dL[i ]) )+(C[i-1] / (dL[i] * dL[i - 1])))*C[i] + (C[i-1]*C[i-1] / (dL[i] * dL[i - 1])));
				if (equation==2)
					F[i] = (nu*p_water/Ct)*((T[i]*C[i + 1] / (dL[i] * dL[i])) - ((T[i] / (dL[i] * dL[i ]) )+(T[i-1] / (dL[i] * dL[i - 1])))*C[i] + (T[i-1]*C[i-1] / (dL[i] * dL[i - 1])));
			}
			for (int i = 0;i < NB;i++)
			{
				calc_v(i, F, BVK,Z,1);
				for (int j = 0;j < BS;j++)
				{
					if (equation == 0)
						BVK[j] *= inv_dw_dh(1 + i*BS + j);
					Om[1 + i*BS + j] += Dg*BVK[j];
				}
			}
		}
		// div (k Db_z T)
		if (kt!=0.0)
		{
			for (int i = 1;i < N;i++)
			{
				if (equation==0)
					F[i] = kt*((T[i + 1] / (dL[i] * dL[i])) - ((1.0 / (dL[i] * dL[i ]))+(1.0 / (dL[i] * dL[i - 1])))*T[i] + (T[i-1] / (dL[i] * dL[i - 1])));
				if (equation==1)
					F[i] = (kt/sigma)*((C[i]*T[i + 1] / (dL[i] * dL[i])) - ((C[i] / (dL[i] * dL[i ]) )+(C[i-1] / (dL[i] * dL[i - 1])))*T[i] + (C[i-1]*T[i-1] / (dL[i] * dL[i - 1])));
				if (equation==2)
					F[i] = (kt*p_water/Ct)*((T[i]*T[i + 1] / (dL[i] * dL[i])) - ((T[i] / (dL[i] * dL[i ]) )+(T[i-1] / (dL[i] * dL[i - 1])))*T[i] + (T[i-1]*T[i-1] / (dL[i] * dL[i - 1])));
			}		
			for (int i = 0;i < NB;i++)
			{
				calc_v(i, F, BVK,Z,1);
				for (int j = 0;j < BS;j++)
				{
					if (equation == 0)
						BVK[j] *= inv_dw_dh(1 + i*BS + j);
					Om[1 + i*BS + j] += Dg*BVK[j];
				}
			}
		}
		// Da_t F part
		if ((rp_split_3d&1)==0)
		{
			mult=1.0;
			if (m3d_c!=2)
				return;
		}	
		if (alpha!=1.0)
		for (int i = 0;i < NB;i++)
			for (int j = 0;j < BS;j++)
				Om[1 + i*BS + j] += mult*Da*time_summing(i,j,old,U_);
	}
	// space fractional derivatives as a matrix (per row - result in R, number in row)
	void Om1SF(int r, double *R)
	{
		row U_;
		std::vector<double*> *old;
		double *points;
		A = 1 - gamma;
		get_F_oldF(&U_, &old, &points);
		// div (k Db_z F)
		double iw = inv_dw_dh(r);
		for (int i = 1;i <N;i++)
		{
			double kv = 0.0;
			if ((space_der == 0) && (i >= r))
				break;
			if ((space_der == 1) && (i == r))
				continue;
			if (A != 0.0)
			{
				if (i < r)
					kv = Dg*kb(1 - A, i - 1, i, r, points, NULL);
				else
					kv = Dg*kb(1 - A, i, i + 1, r, points, NULL);
			}
			if (equation == 0)
				kv *= iw;
			if ((space_der == 1) && (i >= r))
				kv *= -1;
			if (equation == 0)
			{
				R[i + 1] += kv*(KK(i) / (dL[i] * dL[i]));
				R[i] -= kv*((KK(i) / (dL[i] * dL[i])) + (KK(i - 1) / (dL[i] * dL[i - 1])));
				R[i - 1] += kv*(KK(i - 1) / (dL[i] * dL[i - 1]));
			}
			if (equation == 1)
			{
				R[i + 1] += kv*(D / sigma)*(1.0 / (dL[i] * dL[i]));
				R[i] -= kv*(D / sigma)*((1.0 / (dL[i] * dL[i - 1])) + (1.0 / (dL[i] * dL[i])));
				R[i - 1] += kv*(D / sigma)*(1.0 / (dL[i] * dL[i - 1]));
			}
			if (equation == 2)
			{
				R[i + 1] += kv*(lambda / Ct)*(1.0 / (dL[i] * dL[i]));
				R[i] -= kv*(lambda / Ct)*((1.0 / (dL[i] * dL[i - 1])) + (1.0 / (dL[i] * dL[i])));
				R[i - 1] += kv*(lambda / Ct)*(1.0 / (dL[i] * dL[i - 1]));
			}
		}
		if (equation != 0)
		{
			for (int i = 1;i <N;i++)
			{
				double kv = 0.0;
				if ((space_der == 0) && (i>r))
					break;
				if (A != 0.0)
				{
					if (i <= r)
						kv = Dg*kb(1 - A, i - 1, i, r, points, NULL);
					else
						kv = Dg*kb(1 - A, i, i + 1, r, points, NULL);
				}
				if ((i == r) && (A == 0.0))
					kv = Dg;
				if ((space_der == 1) && (i > r))
					kv *= -1;
				if (equation == 1)
				{
					// div (k Db_z H)
					R[i] += kv*(1.0 / sigma)*((KK(i)*U[i + 1] / (dL[i] * dL[i])) - (KK(i) / (dL[i] * dL[i]))*U[i]);
					R[i - 1] += kv*(1.0 / sigma)*(-(KK(i - 1) / (dL[i] * dL[i - 1]))*U[i] + (KK(i - 1)*U[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z C)
					R[i] -= kv*(nu / sigma)*((C[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*C[i]);
					R[i - 1] -= kv*(nu / sigma)*(-(1.0 / (dL[i] * dL[i - 1]))*C[i] + (C[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z T)		
					R[i] -= kv*(kt / sigma)*((T[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*T[i]);
					R[i - 1] -= kv*(kt / sigma)*(-(1.0 / (dL[i] * dL[i - 1]))*T[i] + (T[i - 1] / (dL[i] * dL[i - 1])));
				}
				if (equation == 2)
				{
					// div (k Db_z H)
					R[i] += kv*(p_water / Ct)*((KK(i)*U[i + 1] / (dL[i] * dL[i])) - (KK(i) / (dL[i] * dL[i]))*U[i]);
					R[i - 1] += kv*(p_water / Ct)*(-(KK(i - 1) / (dL[i] * dL[i - 1]))*U[i] + (KK(i - 1)*U[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z C)
					R[i] -= kv*(nu*p_water / Ct)*((C[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*C[i]);
					R[i - 1] -= kv*(nu*p_water / Ct)*(-(1.0 / (dL[i] * dL[i - 1]))*C[i] + (C[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z T)		
					R[i] -= kv*(kt*p_water / Ct)*((T[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*T[i]);
					R[i - 1] -= kv*(kt*p_water / Ct)*(-(1.0 / (dL[i] * dL[i - 1]))*T[i] + (T[i - 1] / (dL[i] * dL[i - 1])));
				}
			}
		}
	}
	// space fractional derivatives as a matrix - fill coeficient matrices that are multiplied on toeplitz
	void Om1T()
	{
		row U_;
		std::vector<double*> *old;
		double *points;
		get_F_oldF(&U_, &old, &points);
		// div (k Db_z F)
		for (int i = rb;i <re;i++)
		{
			if (equation == 0)
			{
				Mcoefs[2][i] = (KK(i) / (dL[i] * dL[i]));
				Mcoefs[1][i] = -((KK(i) / (dL[i] * dL[i])) + (KK(i - 1) / (dL[i] * dL[i - 1])));
				Mcoefs[0][i] = (KK(i - 1) / (dL[i] * dL[i - 1]));
			}
			if (equation == 1)
			{
				Mcoefs[2][i] = (D / sigma)*(1.0 / (dL[i] * dL[i]));
				Mcoefs[1][i] = -(D / sigma)*((1.0 / (dL[i] * dL[i - 1])) + (1.0 / (dL[i] * dL[i])));
				Mcoefs[0][i] = (D / sigma)*(1.0 / (dL[i] * dL[i - 1]));
			}
			if (equation == 2)
			{
				Mcoefs[2][i] = (lambda / Ct)*(1.0 / (dL[i] * dL[i]));
				Mcoefs[1][i] = -(lambda / Ct)*((1.0 / (dL[i] * dL[i - 1])) + (1.0 / (dL[i] * dL[i])));
				Mcoefs[0][i] = (lambda / Ct)*(1.0 / (dL[i] * dL[i - 1]));
			}
		}
		if (equation != 0)
		{
			for (int i = 1;i <N;i++)
			{
				if (equation == 1)
				{
					// div (k Db_z H)
					Mcoefs2[2][i] = 0;
					Mcoefs2[1][i] = (1.0 / sigma)*((KK(i)*U[i + 1] / (dL[i] * dL[i])) - (KK(i) / (dL[i] * dL[i]))*U[i]);
					Mcoefs2[0][i] = (1.0 / sigma)*(-(KK(i - 1) / (dL[i] * dL[i - 1]))*U[i] + (KK(i - 1)*U[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z C)
					Mcoefs2[1][i] +=-(nu / sigma)*((C[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*C[i]);
					Mcoefs2[0][i] += -(nu / sigma)*(-(1.0 / (dL[i] * dL[i - 1]))*C[i] + (C[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z T)		
					Mcoefs2[1][i] += -(kt / sigma)*((T[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*T[i]);
					Mcoefs2[0][i] += -(kt / sigma)*(-(1.0 / (dL[i] * dL[i - 1]))*T[i] + (T[i - 1] / (dL[i] * dL[i - 1])));
				}
				if (equation == 2)
				{
					// div (k Db_z H)
					Mcoefs2[2][i] = 0;
					Mcoefs2[1][i] = (p_water / Ct)*((KK(i)*U[i + 1] / (dL[i] * dL[i])) - (KK(i) / (dL[i] * dL[i]))*U[i]);
					Mcoefs2[0][i] = (p_water / Ct)*(-(KK(i - 1) / (dL[i] * dL[i - 1]))*U[i] + (KK(i - 1)*U[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z C)
					Mcoefs2[1][i] += -(nu*p_water / Ct)*((C[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*C[i]);
					Mcoefs2[0][i] += -(nu*p_water / Ct)*(-(1.0 / (dL[i] * dL[i - 1]))*C[i] + (C[i - 1] / (dL[i] * dL[i - 1])));
					// div (k Db_z T)		
					Mcoefs2[1][i] += -(kt*p_water / Ct)*((T[i + 1] / (dL[i] * dL[i])) - (1.0 / (dL[i] * dL[i]))*T[i]);
					Mcoefs2[0][i] += -(kt*p_water / Ct)*(-(1.0 / (dL[i] * dL[i - 1]))*T[i] + (T[i - 1] / (dL[i] * dL[i - 1])));
				}
			}
		}
	}
	// right part for implicit scheme
	void Om1_impl(double *Om)
	{
		row U_;
		std::vector<double*> *old;
		double mult=1.0;
		A=1-gamma;
		get_F_oldF(&U_,&old);		
		// div (k Db_z F)
		if (mode==3)
	   		mult=1.0/3.0;
		if ((mode==3)&&(implicit3d==1)&&(m3d_c!=2))
			for (int i = 0;i < NB;i++)
				for (int j = 0;j < BS;j++)
				{
				    if (((1 + i*BS + j)>=rb)&&((1 + i*BS + j)<=re))
						Om[1 + i*BS + j] = Rp(1 + i*BS + j);
				}
		else
			for (int i = 0;i < NB;i++)
				for (int j = 0;j < BS;j++)
				    if (((1 + i*BS + j)>=rb)&&((1 + i*BS + j)<=re))
						Om[1 + i*BS + j] = -Da*(kkk0/ tau[tstep] )*U_[1 + i*BS + j]+Rp(1 + i*BS + j);
		if (equation==0)
		{
			// div (k Db_z C)
			if (nu != 0.0)
			{
				for (int i = 1;i < N;i++)
					F[i] = nu*((C[i + 1] / (dL[i] * dL[i])) - ((1.0 / (dL[i] * dL[i])) + (1.0 / (dL[i] * dL[i - 1])))*C[i] + (C[i - 1] / (dL[i] * dL[i - 1])));
				for (int i = 0;i < NB;i++)
				{
					calc_v(i, F, BVK, Z, 1);
					for (int j = 0;j < BS;j++)
					{
						BVK[j] *= inv_dw_dh(1 + i*BS + j);
						Om[1 + i*BS + j] += Dg*BVK[j];
					}
				}
			}
			// div (k Db_z T)		
			if (kt != 0.0)
			{
				for (int i = 1;i < N;i++)
					F[i] = kt*((T[i + 1] / (dL[i] * dL[i])) - ((1.0 / (dL[i] * dL[i])) + (1.0 / (dL[i] * dL[i - 1])))*T[i] + (T[i - 1] / (dL[i] * dL[i - 1])));
				for (int i = 0;i < NB;i++)
				{
					calc_v(i, F, BVK, Z, 1);
					for (int j = 0;j < BS;j++)
					{
						BVK[j] *= inv_dw_dh(1 + i*BS + j);
						Om[1 + i*BS + j] += Dg*BVK[j];
					}
				}
			}
		}
		// Da_t F part
		if ((rp_split_3d&1)==0)
		{
			mult=1.0;
			if (m3d_c!=2)
				return;
		}	
		if (alpha!=1.0)
		for (int i = 0;i < NB;i++)
			for (int j = 0;j < BS;j++)
				Om[1 + i*BS + j] += mult*Da*time_summing(i,j,old,U_);
	}
	double bc0()
	{
		if (analytic_test==0)
		{
			if ((mode==1)||((mode==3)&&(m3d_c==0)))
			{
				if (equation==0)
				{
				    if (inv_dw_dh(0)!=1e20) // first order condition if saturated in 0
						return -Uc();
				    else
						return -Uc()*KK(1)/dL[0]; // E-I
				}
				if (equation==1)
					return Cu();
				if (equation==2)
					return Tu();
			}
			else
				Bt[1]=0.0;
		}
		else
		{
			if (mode==1)
				return testF(0, tstep + 1,equation);
			if (mode == 3)
			{
				if (m3d_c == 0)
					return testF3d(0, m3d_x, m3d_y, tstep + 1);
				if (m3d_c == 1)
					return testF3d(m3d_x, 0, m3d_y, tstep + 1);
				if (m3d_c == 2)
					return testF3d(m3d_x, m3d_y,0, tstep +  1);
			}
		}
		return 0.0;
	}
	double bcN()
	{
		row U_;
		std::vector<double*> *old;
		get_F_oldF(&U_,&old);					
		if (analytic_test == 0)
		{
			if ((mode == 1) || ((mode == 3) && (m3d_c == 0)))
			{
				if (bottom_cond == 1)
				{
					if (equation == 0)
						return H0 - Z[N];
					if (equation == 1)
						return C0;
					if (equation == 2)
						return T0;
				}
			}
		}
		else
		{
			if (mode == 1)
			{
				if (bottom_cond == 1)
					return testF(N, tstep + 1, equation);
				else
					return DxtestF(N, tstep + 1, equation);
			}
			if (mode == 3)
			{
				if (m3d_c == 0)
				{
					if (bottom_cond == 1)
						return testF3d(N, m3d_x, m3d_y, tstep + 1);
					else
						return DxtestF3d(N,m3d_x,m3d_y, tstep + 1);
				}
				if (m3d_c == 1)
				{
					if (bottom_cond == 1)
					   return testF3d(m3d_x, N, m3d_y, tstep +  1);
					else
						return DxtestF3d(m3d_x,N,m3d_y, tstep + 1);
				}
				if (m3d_c == 2)
				{
					if (bottom_cond == 1)
						return testF3d(m3d_x, m3d_y, N, tstep + 1);
					else
						return DxtestF3d(m3d_x,m3d_y,N, tstep +  1);	
				}
			}
		}
		return 0.0;
	}
	// beta coeffients
	void bt1()
	{
		Bt[1]=bc0();
		for (int i = 1;i < N;i++)
			Bt[i + 1] = (Al[i + 1]/B(i)) * (A_(i)*Bt[i] - Om[i]);
	}
	void implicit1_gen_matrix(double **Mat, double *Om)
	{
		// fill matrix
		for (int i = 0;i <= N;i++)
			for (int j = 0;j <= N;j++)
				Mat[i][j] = 0.0;
		for (int i = 1;i<N;i++)
		{
			Mat[i][i - 1] += A_(i);
			Mat[i][i] -= R(i);
			Mat[i][i + 1] += B(i);
			Om1SF(i, Mat[i]);
		}
		// fill right part
		Om1_impl(Om);
		// set boundary conditions
		Mat[0][0] = 1.0;
		Mat[N][N] = 1.0;
		Om[0] = bc0();
		Om[N] = bcN();
		if (bottom_cond==0)
			Om[N]*=dL[N-1];
		if (bottom_cond == 0)
			Mat[N][N - 1] = -1.0;
		if (analytic_test == 0)
		{
			if ((mode == 1) || ((mode == 3) && (m3d_c == 0)))
			{
				if (equation == 0)
					if (inv_dw_dh(0) != 1e20) // first order condition in saturated zone
						Mat[0][1] = -1.0;
			}
			else
			{
				Mat[0][1] = -1.0;
				Mat[N][N - 1] = -1.0;
			}
		}
	}
	void implicit2_gen(double **Mat,double *Om)
	{
		// fill matrix
		for (int i = ((rb!=0)?rb:1);i<re;i++)
		{
			Mat[0][i] = A_(i);
			Mat[1][i] = -R(i);
			Mat[2][i] = B(i);
		}
		Om1T();
		// fill right part
		Om1_impl(Om);
		// set boundary conditions
		Mat[1][0] = 1.0;
		Mat[1][N] = 1.0;
		Mat[0][0] = 0.0;
		Mat[0][N] = 0.0;
		Mat[2][0] = 0.0;
		Mat[2][N] = 0.0;
		// set boundary conditions
		Om[0] = bc0();
		Om[N] = bcN();
		if (bottom_cond==0)
			Om[N]*=dL[N-1];
		if (bottom_cond == 0)
			Mat[0][N] = -1.0;
		if (analytic_test == 0)
		{
			if ((mode == 1) || ((mode == 3) && (m3d_c == 0)))
			{
				if (equation == 0)
					if (inv_dw_dh(0) != 1e20) // first order condition in saturated zone
						Mat[2][0] = -1.0;
			}
			else
			{
				Mat[1][0] = -1.0;
				Mat[0][N] = -1.0;
			}
		}
	}
	// calc F
	void U1(int eq)
	{
		row U_;
		std::vector<double*> *old;
		equation=eq;
		get_F_oldF(&U_,&old);
		if (implicit==0) // explicit space fractional - tridiagonal matrix 
		{
			al1();
			Om1();
			bt1();
			// boundary condition
			if (bottom_cond == 1)
				U_[N] = bcN();
			else
			{
				if ((mode==1)||((mode==3)&&(m3d_c==0)))
				{
					if (fabs(Bt[N]) > 1e-15)
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = (Bt[N]+ bcN()*dL[N - 1]) / (1.0 - Al[N]);
						else
							U_[N] = U_[N - 1];
					}
					else
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = 0.0;
						else
							U_[N] = U_[N - 1];
					}				
				}
				else
				{
					if (fabs(Bt[N]) > 1e-15)
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = (Bt[N]+ bcN()*dL[N - 1]) / (1.0 - Al[N]);
						else
							U_[N] = U_[N - 1];
					}
					else
					{
						if (fabs(1.0 - Al[N]) > 1e-15)
							U_[N] = 0.0;
						else
							U_[N] = U_[N - 1];
					}				
				}
			}
			for (int i = N - 1;i >= 0;i--)
				U_[i] = Al[i + 1] * U_[i + 1] + Bt[i + 1];
		}
		if (implicit == 1) // implicit space fractional - fully filled matrix
		{
			implicit1_gen_matrix(Mat,Om);
			// row scaling
			if (implicit_row_scaling==1)
			for (int i = 0;i <= N;i++)
			if (Mat[i][i]!=0.0)
			{
				for (int j = 0;j <= N;j++)
					if (j!=i)
						Mat[i][j]/=Mat[i][i];
				Om[i]/=Mat[i][i];
				Mat[i][i]=1.0;
			}
			// save matrix
			if (debug_level == 3)
			{
				FILE *fi;
				if (fi = fopen("a.mtx", "wt"))
				{
					fprintf(fi, "%%%%MatrixMarket matrix coordinate real general\n");
					fprintf(fi, "%d %d %d\n", N+1, N+1, (N+1)*(N+1));
					for (int i = 0;i <= N;i++)
						for (int j = 0;j <= N;j++)
							fprintf(fi, "%d %d %g\n", i + 1, j+1, Mat[i][j]);
					fclose(fi);
				}
				if (fi = fopen("b.mtx", "wt"))
				{
					fprintf(fi, "%%%%MatrixMarket matrix array real general\n");
					fprintf(fi, "%d 1\n", N+1);
					for (int i = 0;i<=N;i++)
						fprintf(fi, "%g\n", Om[i]);
					fclose(fi);
				}
			}
			// solve
			for (int i = 0;i <= N;i++)
				Al[i] = U_[i];
			solve_mln<1>(Mat, Om, Al, N + 1, impl_err, impl_niter, vmul,0,N+1);
			for (int i = 0;i <= N;i++)
				U_[i] = Al[i];
		}
		if (implicit == 2) // implicit space fractional - split on Toeplitz matrices
		{
			implicit2_gen(Mat,Om);
			// row scaling
			if (implicit_row_scaling==1)
			for (int i = 0;i <= N;i++)
			if (Mat[1][i]!=0.0)
			{
				Om[i]/=Mat[1][i];
				Mat[0][i]/=Mat[1][i];
				Mat[2][i]/=Mat[1][i];
				RS[i]=Mat[1][i];
				Mat[1][i]=1.0;
			}
			else
				RS[i]=1.0;
			// solve
			for (int i = 0;i <= N;i++)
				Al[i] = U_[i];
			solve_mln<1>((double **)this, Om, Al, N + 1, impl_err, impl_niter, vmul_T,((rb==1)?0:rb),((re==N)?(N+1):re));
			for (int i = 0;i <= N;i++)
				U_[i] = Al[i];
		}
		U_[N + 1] = U[N];
		// save H for time-fractional derivative calculations
		if (alpha!=1.0)
			if (mode==1)
			{
				double *hs = new double[N + 2];
				if (equation==0)
					memcpy(hs, b_U, (N + 2)*sizeof(double));
				if (equation==1)
					memcpy(hs, b_C, (N + 2)*sizeof(double));
				if (equation==2)
					memcpy(hs, b_T, (N + 2)*sizeof(double));
				old->push_back(hs);
			}
	}
	// thread local arrays allocation
	void set_local_vars()
	{
#ifdef _OPENMP
		int id=omp_get_thread_num();
#pragma omp critical		
		{
		while (Als.size()<=id)
		{
			Als.push_back(NULL);
			Bts.push_back(NULL);
			Oms.push_back(NULL);
			RSs.push_back(NULL);
			Fs.push_back(NULL);
			BVKs.push_back(NULL);
			Mats.push_back(NULL);
			Mat1ds.push_back(NULL);
			Mcoefss.push_back(NULL);
			Mcoefs2s.push_back(NULL);
		}
		if (Als[id]==NULL)
		{
			Als[id]=new double[sN+_M+K+2];
			Bts[id]=new double[sN+_M+K+2];
			if (implicit3d==0)
				Oms[id]=new double[sN+_M+K+2];
			else
				Oms[id] =sOm;
			if (implicit!=2)
				RSs[id]=new double[sN+_M+K+2];
			else
			{
				if (implicit3d==1)
					RSs[id]=sRS;
				else
					RSs[id]=new double[sN+_M+K+2];
			}
			Fs[id]=new double[N + _M+K+2];
			BVKs[id]=new double[BS];
			if (implicit3d==0)
			{
				if (implicit==1)
				{
					Mats[id]=new double*[sN+_M+K + 2];
					for (int i=0;i<sN+_M+K + 2;i++)
						Mats[id][i]=new double[sN+_M+K + 2];
				}
				else
				{
					Mats[id]=new double*[3];
					for (int i=0;i<3;i++)
						Mats[id][i]=new double[sN+_M+K + 2];
				}				
			}	
			else
			{
				if (implicit==1)
				{
					Mat1ds[id]=new double*[sN+_M+K + 2];
					for (int i=0;i<sN+_M+K + 2;i++)
						Mat1ds[id][i]=new double[sN+_M+K + 2];
					Mats[id]=sMat;
				}
				if (implicit==2)
				{
					Mats[id] = new double*[3];
					for (int i = 0;i<3;i++)
						Mats[id][i] = new double[sN + _M + K + 2];
				}
			}
			if (implicit==2)
			{
				Mcoefss[id] = new double*[3];
				for (int i = 0;i<3;i++)
					Mcoefss[id][i] = new double[sN + _M + K + 2];
				Mcoefs2s[id] = new double*[3];
				for (int i = 0;i<3;i++)
					Mcoefs2s[id][i] = new double[sN + _M + K + 2];
			}
		}
		}
		Al=Als[id];
		Bt=Bts[id];
		Om=Oms[id];
		RS=RSs[id];
		F=Fs[id];
		BVK=BVKs[id];
		Mat=Mats[id];
		Mat1d=Mat1ds[id];
		Mcoefs=Mcoefss[id];
		Mcoefs2=Mcoefs2s[id];		
#endif
	}
	// copy parts of B processed by current process to A
	// zeroize other parts of A
	void copy_processed(double *A,double *B)
	{
#ifdef USE_MPI
		for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
			A[i] = 0.0;
#ifndef USE_REPARTITIONING
		// block red-black partitioning
		if (K>2)
		for (int bb=0;bb<mpi_size;bb++)
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			if (bb==st)
			{
				rb=bb*(K/mpi_size);
				re=rb+(K/mpi_size);
				if (bb==0) rb=0;
				if (bb==(mpi_size-1)) re=K+1;
				for (int k=rb;k<re;k++)
					A[idx(i,j,k)]=B[idx(i,j,k)];
			}
		}
		if (_M>2)
		for (int bb=0;bb<mpi_size;bb++)
		for (int i = 0;i <= sN ;i++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			if (bb==st)
			{					
				rb=bb*(_M/mpi_size);
				re=rb+(_M/mpi_size);
				if (bb==0) rb=0;
				if (bb==(mpi_size-1)) re=_M+1;
				for (int j=rb;j<re;j++)
					A[idx(i,j,k)]=B[idx(i,j,k)];
			}
		}
		if (sN>2)
		for (int bb=0;bb<mpi_size;bb++)
		for (int j = 0;j <= _M ;j++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=j/(_M/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			if (bb==st)
			{			
				rb=bb*(sN/mpi_size);
				re=rb+(sN/mpi_size);
				if (bb==0) rb=0;
				if (bb==(mpi_size-1)) re=sN+1;
				for (int i=rb;i<re;i++)
					A[idx(i,j,k)]=B[idx(i,j,k)];
			}
		}
#else		
		// partitioning by N
		if (((K>2)&&(_M>2)&&(sN>2))||((K<=2)||(_M<=2)))
		{
			int st,end;
			st=mpi_rank*(sN/mpi_size);
			end=st+(sN/mpi_size);
			if (mpi_rank==0) st=0;
			if (mpi_rank==(mpi_size-1)) end=sN+1;
			for (int j = 0;j <= _M ;j++)	
				for (int k = 0;k <= K ;k++)
					for (int i=st;i<end;i++)
						A[idx(i,j,k)]=B[idx(i,j,k)];
		}
		else
		{
		// if N<=2 - by M
			int st,end;
			st=mpi_rank*(_M/mpi_size);
			end=st+(_M/mpi_size);
			if (mpi_rank==0) st=0;
			if (mpi_rank==(mpi_size-1)) end=_M+1;
			for (int i = 0;i <= sN ;i++)
				for (int k = 0;k <= K ;k++)	
					for (int j=st;j<end;j++)
						A[idx(i,j,k)]=B[idx(i,j,k)];
		}
#endif		
#endif		
	}
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	// repartition for row block partitioning
	// m3d_c=2 - from N to K | M<=2 - from N to K | N<=2 - from M to K
	// m3d_c=1 - from K to M | K<=2 - from N to M | N<=2 - from K to M
	// m3d_c=0 - from M to N | M<=2 - from K to N | K<=2 - from M to N
	// full - with 0 and N,M,K
	void repartition(int m3d_c,double *U,int full=0)
	{
		MPI_Request *rqs=new MPI_Request[2*mpi_size];
		int *sizes=new int[mpi_size];
		MPI_Status st;
		double *sb,*sb2,*p1,*p2;
		int Sidx[2],SS[3]={sN+full,_M+full,K+full},SR[3][2];		
		if (((m3d_c==2)&&(K>2)&&(_M>2)&&(sN>2))||((m3d_c==2)&&(K>2)&&(_M<=2)&&(sN>2))) { Sidx[0]=0; Sidx[1]=2;}
		if ((m3d_c==0)&&(_M<=2)&&(K>2)&&(sN>2))	{ Sidx[0]=2; Sidx[1]=0;}
		if (((m3d_c==1)&&(K>2)&&(_M>2)&&(sN>2))||((m3d_c==1)&&(K>2)&&(_M>2)&&(sN<=2))) { Sidx[0]=2; Sidx[1]=1;}
		if ((m3d_c==2)&&(_M>2)&&(K>2)&&(sN<=2)) { Sidx[0]=1; Sidx[1]=2;}
		if (((m3d_c==0)&&(K>2)&&(_M>2)&&(sN>2))||((m3d_c==0)&&(_M>2)&&(K<=2)&&(sN>2))) { Sidx[0]=1; Sidx[1]=0;}
		if ((m3d_c==1)&&(_M>2)&&(K<=2)&&(sN>2)) { Sidx[0]=0; Sidx[1]=1;}
		int st1,end1,st2,end2;		
		st1=mpi_rank*((SS[Sidx[0]]-full)/mpi_size);
		end1=st1+((SS[Sidx[0]]-full)/mpi_size);
		if (mpi_rank==0) st1=1-full;
		if (mpi_rank==(mpi_size-1)) end1=SS[Sidx[0]];
		st2=mpi_rank*((SS[Sidx[1]]-full)/mpi_size);
		end2=st2+((SS[Sidx[1]]-full)/mpi_size);
		if (mpi_rank==0) st2=1-full;
		if (mpi_rank==(mpi_size-1)) end2=SS[Sidx[1]];
		sb=new double[(SS[(Sidx[0]+1)%3]+2)*(SS[(Sidx[0]+2)%3]+2)*(end1-st1+1)];
		sb2=new double[(SS[(Sidx[1]+1)%3]+2)*(SS[(Sidx[1]+2)%3]+2)*(end1-st1+1)];
		// send
		p1=p2=sb;
		for (int r=0;r<mpi_size;r++)
		if (r!=mpi_rank)
		{
			int rst,rend;
			rst=r*((SS[Sidx[1]]-full)/mpi_size);
			rend=rst+((SS[Sidx[1]]-full)/mpi_size);
			if (r==0) rst=1-full;
			if (r==(mpi_size-1)) rend=SS[Sidx[1]];
			SR[Sidx[0]][0]=st1;
			SR[Sidx[0]][1]=end1;
			SR[Sidx[1]][0]=rst;
			SR[Sidx[1]][1]=rend;
			SR[3-Sidx[0]-Sidx[1]][0]=1-full;
			SR[3-Sidx[0]-Sidx[1]][1]=SS[3-Sidx[0]-Sidx[1]];			
			for (int i=SR[0][0];i<SR[0][1];i++)
				for (int j=SR[1][0];j<SR[1][1];j++)
					for (int k=SR[2][0];k<SR[2][1];k++)
						(p2++)[0]=U[idx(i,j,k)];
			sizes[r]=(p2-p1)*sizeof(double);
			_MPI_Isend(&sizes[r],sizeof(int),MPI_BYTE,r,0,MPI_COMM_WORLD,&rqs[2*r+0]);
			_MPI_Isend(p1,sizes[r],MPI_BYTE,r,0,MPI_COMM_WORLD,&rqs[2*r+1]);
			p1=p2;
		}
		// receive
		p1=p2=sb2;
		for (int r=0;r<mpi_size;r++)
		if (r!=mpi_rank)
		{
			int rst,rend;
			rst=r*((SS[Sidx[0]]-full)/mpi_size);
			rend=rst+((SS[Sidx[0]]-full)/mpi_size);
			if (r==0) rst=1-full;
			if (r==(mpi_size-1)) rend=SS[Sidx[0]];
			_MPI_Recv(&sizes[r],sizeof(int),MPI_BYTE,r,0,MPI_COMM_WORLD,&st);
			_MPI_Recv(p1,sizes[r],MPI_BYTE,r,0,MPI_COMM_WORLD,&st);
			SR[Sidx[0]][0]=rst;
			SR[Sidx[0]][1]=rend;
			SR[Sidx[1]][0]=st2;
			SR[Sidx[1]][1]=end2;
			SR[3-Sidx[0]-Sidx[1]][0]=1-full;
			SR[3-Sidx[0]-Sidx[1]][1]=SS[3-Sidx[0]-Sidx[1]];			
			for (int i=SR[0][0];i<SR[0][1];i++)
				for (int j=SR[1][0];j<SR[1][1];j++)
					for (int k=SR[2][0];k<SR[2][1];k++)
						U[idx(i,j,k)]=(p2++)[0];
			p1=p2;
		}
		// wait
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			_MPI_Wait(&rqs[2*i+0],&st);
			_MPI_Wait(&rqs[2*i+1],&st);
		}
		// clean up
		delete [] rqs;
		delete [] sb;
		delete [] sb2;
		delete [] sizes;
	}
#endif	
	void calc_step(double tau_m)
	{
		init_time_step(tau_m);
		// save H for time-fractional derivative calculations
		if (tstep==0)
			if (alpha != 1.0)
			{
				double *hs;
				hs = new double[(N + 2)*(_M + 2)*(K + 2)];
				if (mode==3)
					memcpy(hs, b_U, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
				else
					memcpy(hs, b_U, (N + 2)*sizeof(double));
				oldH.push_back(hs);
				if (s_is_c)
				{
					hs = new double[(N + 2)*(_M + 2)*(K + 2)];
					if (mode==3)
						memcpy(hs, b_C, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
					else
						memcpy(hs, b_C, (N + 2)*sizeof(double));
					oldC.push_back(hs);
				}
				if (s_is_t)
				{
					hs = new double[(N + 2)*(_M + 2)*(K + 2)];
					if (mode==3)
						memcpy(hs, b_T, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
					else
						memcpy(hs, b_T, (N + 2)*sizeof(double));
					oldT.push_back(hs);
				}
			}
		if (mode==1) // 1d
		{
			U1(0);
			if ((analytic_test==0)||(analytic_test==3))
			{
				if ((nu!=0.0)||(is_r_C))
					U1(1);
				if (kt!=0.0)
					U1(2);
			}
		}
		if (mode==3) // 3d
		{
			// per equation solution
			for (equation=0;equation<3;equation++)
			{
				if (equation==1)
					if (!((nu != 0.0) || (is_r_C)))
						continue;
				if (equation==2)
					if (kt == 0.0)
						continue;
#ifdef OCL
				if (use_ocl) // precalc time derivative sums using opencl
					fr_ocl_call_sum();
#endif				
				// zeroize matrices and vectors
				if (implicit3d==1) 
				{
					for (int i = 0;i <= (sN+2)*(_M+2)*(K+2);i++)
					{
						if (implicit==1)
							for (int j = 0;j <= (sN+2)*(_M+2)*(K+2);j++)
								Mat[i][j] = 0.0;
						if (implicit==2)
						{
							for (int j=0;j<7;j++)
								Mat_3d[j][i]=0.0;
							for (int j=0;j<3;j++)
								for (int k=0;k<3;k++)
								{
									Mcoefs_3d[j][k][i]=0.0;
									Mcoefs2_3d[j][k][i]=0.0;
								}
						}
						Om[i]=0.0;
					}
				}
				// fill matrix or solve equation my locally-onedimensional scheme
#if !defined(USE_MPI) || defined(USE_REPARTITIONING)
#pragma omp parallel
#else
#pragma omp parallel num_threads(mpi_size*mpi_mp_add_threads) proc_bind(spread)
#endif				
				{
#pragma omp single		
				{
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				int id1_st=1,id1_end=sN;									
				int id2_st=1,id2_end=_M;
				if (implicit3d==0)
				{
					if (sN>2)
					{
						id1_st=mpi_rank*(sN/mpi_size);
						id1_end=id1_st+(sN/mpi_size);
						if (mpi_rank==0) id1_st=1;
						if (mpi_rank==(mpi_size-1)) id1_end=sN;
					}
					else
					{
						id2_st=mpi_rank*(_M/mpi_size);
						id2_end=id2_st+(_M/mpi_size);
						if (mpi_rank==0) id2_st=1;
						if (mpi_rank==(mpi_size-1)) id2_end=_M;
					}
				}
#endif					
				if ((K>2)||(implicit3d==1))
				for (int bb=0;bb<mpi_size;bb++)
				{
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
#pragma omp task firstprivate(bb) 
#endif 
				{
				int nrun=0;
				for (int i = 1;i < sN ;i++)
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((i>=id1_st)&&(i<id1_end))
#endif
				for (int j = 1;j < _M ;j++)	
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((j>=id2_st)&&(j<id2_end))
#endif
				{
					int bi=1,bj=1;
					if (sN/mpi_size) bi=i/(sN/mpi_size);
					if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
					if (bi>=mpi_size) bi=mpi_size-1;
					if (bj>=mpi_size) bj=mpi_size-1;
					int st=bi-bj; // processing stage of the block by current process
					if (st<0) st+=mpi_size;
					st=(st+mpi_rank)%mpi_size;
					if (bb==st)
					{
#pragma omp task firstprivate(bb,i,j)
					{
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_start_thread();
#endif						
					rb=bb*(K/mpi_size);
					re=rb+(K/mpi_size);
					if ((K/mpi_size)==0)
					{
						rb=1;
						re=K;
					}
					if (bb==0) rb=1;
					if (bb==(mpi_size-1)) re=K;
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
					{
						rb=1;
						re=K;
					}
#endif						
					m3d_x=i;
					m3d_y=j;
					m3d_c=2;
					U.set(b_U,m3d_x,m3d_y,m3d_c);
					C.set(b_C,m3d_x,m3d_y,m3d_c);
					T.set(b_T,m3d_x,m3d_y,m3d_c);
					set_local_vars();
					dL = dLy;
					Z=Y;
					N=K;
					NB = KB;
					if (implicit3d==0) // locally-onedimensional scheme
					{
						U1(0);
						if (analytic_test == 0)
						{
							if ((nu != 0.0) || (is_r_C))
								U1(1);
							if (kt != 0.0)
								U1(2);
						}
					}
					if (implicit3d==1)
					{
						if (implicit==1)
						{
							implicit1_gen_matrix(Mat1d, Al);
							for (int i1 = 0;i1 <= K ;i1++)
							{
								for (int j1 = 0;j1 <=K ;j1++)
									Mat[idx(i,j,i1)][idx(i,j,j1)]+=Mat1d[i1][j1];
								Om[idx(i,j,i1)]+=Al[i1];
							}
						}
						if (implicit==2)
						{
							implicit2_gen(Mat, Al);
							for (int j1 = 0;j1 <= K ;j1++)
							{
								Mat_3d[0][idx(i,j,j1)]+=Mat[0][j1];
								Mat_3d[1][idx(i,j,j1)]+=Mat[1][j1];
								Mat_3d[2][idx(i,j,j1)]+=Mat[2][j1];
								Mcoefs_3d[2][0][idx(i,j,j1)]+=Mcoefs[0][j1];
								Mcoefs_3d[2][1][idx(i,j,j1)]+=Mcoefs[1][j1];
								Mcoefs_3d[2][2][idx(i,j,j1)]+=Mcoefs[2][j1];
								Mcoefs2_3d[2][0][idx(i,j,j1)]+=Mcoefs2[0][j1];
								Mcoefs2_3d[2][1][idx(i,j,j1)]+=Mcoefs2[1][j1];
								Mcoefs2_3d[2][2][idx(i,j,j1)]+=Mcoefs2[2][j1];
								Om[idx(i,j,j1)]+=Al[j1];
							}
						}
					}
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_end_thread();
#endif						
					}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
					if ((++nrun)==mpi_mp_add_threads)
					{
#pragma omp taskwait
						nrun=0;						
					}
#endif					
				    }
				}
				}
				}
				}
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
#pragma omp single				
				if (implicit3d==0) 
				    if (K>2)
				    {
					double *s=b_Utimesum;
#ifdef OCL
					s=Sums;
#endif					
					 repartition(2,b_U);
					 repartition(2,s);
				    }
#endif					
#pragma omp single	
				{
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				int id1_st=1,id1_end=K;
				int id2_st=1,id2_end=sN;
				if (implicit3d==0)
				{
					if (K>2)
					{
						id1_st=mpi_rank*(K/mpi_size);
						id1_end=id1_st+(K/mpi_size);
						if (mpi_rank==0) id1_st=1;
						if (mpi_rank==(mpi_size-1)) id1_end=K;
					}
					else
					{
						id2_st=mpi_rank*(sN/mpi_size);
						id2_end=id2_st+(sN/mpi_size);
						if (mpi_rank==0) id2_st=1;
						if (mpi_rank==(mpi_size-1)) id2_end=sN;
					}				
				}
#endif					
				if ((_M>2)||(implicit3d==1))
				for (int bb=0;bb<mpi_size;bb++)
				{
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
#pragma omp task firstprivate(bb) 
#endif 
				{
				int nrun=0;
				for (int k = 1;k < K ;k++)
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((k>=id1_st)&&(k<id1_end))
#endif
				for (int i = 1;i < sN ;i++)
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((i>=id2_st)&&(i<id2_end))
#endif
				{
					int bi=1,bj=1;
					if (sN/mpi_size) bi=i/(sN/mpi_size);
					if (K/mpi_size) bj=k/(K/mpi_size);// block of (i,j)
					if (bi>=mpi_size) bi=mpi_size-1;
					if (bj>=mpi_size) bj=mpi_size-1;
					int st=bi-bj; // processing stage of the block by current process
					if (st<0) st+=mpi_size;
					st=(st+mpi_rank)%mpi_size;
					if (bb==st)
					{
#pragma omp task firstprivate(bb,i,k)
					{					
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_start_thread();
#endif						
					rb=bb*(_M/mpi_size);
					re=rb+(_M/mpi_size);
					if ((_M/mpi_size)==0)
					{
						rb=1;
						re=_M;
					}
					if (bb==0) rb=1;
					if (bb==(mpi_size-1)) re=_M;
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
					{
						rb=1;
						re=_M;
					}
#endif						
					m3d_x=i;
					m3d_y=k;
					m3d_c=1;
					U.set(b_U,m3d_x,m3d_y,m3d_c);
					C.set(b_C,m3d_x,m3d_y,m3d_c);
					T.set(b_T,m3d_x,m3d_y,m3d_c);
					set_local_vars();
					dL = dLx;
					Z=X;
					N=_M;
					NB = MB;
					if (implicit3d==0) // locally-onedimensional scheme
					{
						U1(0);
						if (analytic_test == 0)
						{
							if ((nu != 0.0) || (is_r_C))
								U1(1);
							if (kt != 0.0)
								U1(2);
						}
					}
					if (implicit3d==1)
					{
						if (implicit==1)
						{
							implicit1_gen_matrix(Mat1d, Al);
							for (int i1 = 0;i1 <= _M ;i1++)
							{
								for (int j1 = 0;j1 <=_M ;j1++)
									Mat[idx(i,i1,k)][idx(i,j1,k)]+=Mat1d[i1][j1];
								Om[idx(i,i1,k)]+=Al[i1];
							}
						}
						if (implicit==2)
						{
							implicit2_gen(Mat, Al);
							for (int j1 = 0;j1 <= _M ;j1++)
							{
								Mat_3d[3][idx(i,j1,k)]+=Mat[0][j1];
								Mat_3d[1][idx(i,j1,k)]+=Mat[1][j1];
								Mat_3d[4][idx(i,j1,k)]+=Mat[2][j1];
								Mcoefs_3d[1][0][idx(i,j1,k)]+=Mcoefs[0][j1];
								Mcoefs_3d[1][1][idx(i,j1,k)]+=Mcoefs[1][j1];
								Mcoefs_3d[1][2][idx(i,j1,k)]+=Mcoefs[2][j1];
								Mcoefs2_3d[1][0][idx(i,j1,k)]+=Mcoefs2[0][j1];
								Mcoefs2_3d[1][1][idx(i,j1,k)]+=Mcoefs2[1][j1];
								Mcoefs2_3d[1][2][idx(i,j1,k)]+=Mcoefs2[2][j1];
								Om[idx(i,j1,k)]+=Al[j1];
							}
						}
					}
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_end_thread();
#endif						
					}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
					if ((++nrun)==mpi_mp_add_threads)
					{
#pragma omp taskwait
						nrun=0;						
					}
#endif					
				    }
				}
				}
				}
				}
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
#pragma omp single
				if (implicit3d==0) 
				    if (_M>2)
				    {
					double *s=b_Utimesum;
#ifdef OCL
					s=Sums;
#endif					
					 repartition(1,b_U);
					 repartition(1,s);
				    }
#endif					
#pragma omp single	
				{
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				int id1_st=1,id1_end=_M;
				int id2_st=1,id2_end=K;
				if (implicit3d==0)
				{
					if (_M>2)
					{
						id1_st=mpi_rank*(_M/mpi_size);
						id1_end=id1_st+(_M/mpi_size);
						if (mpi_rank==0) id1_st=1;
						if (mpi_rank==(mpi_size-1)) id1_end=_M;
					}
					else
					{
						id2_st=mpi_rank*(K/mpi_size);
						id2_end=id2_st+(K/mpi_size);
						if (mpi_rank==0) id2_st=1;
						if (mpi_rank==(mpi_size-1)) id2_end=K;
					}				
				}
#endif					
				if ((sN>2)||(implicit3d==1))
				for (int bb=0;bb<mpi_size;bb++)
				{
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
#pragma omp task firstprivate(bb)
#endif 
				{
				int nrun=0;
				for (int j = 1;j < _M ;j++)
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((j>=id1_st)&&(j<id1_end))
#endif
				for (int k = 1;k < K ;k++)
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if ((k>=id2_st)&&(k<id2_end))
#endif
				{
					int bi=1,bj=1;
					if (_M/mpi_size) bi=j/(_M/mpi_size);
					if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
					if (bi>=mpi_size) bi=mpi_size-1;
					if (bj>=mpi_size) bj=mpi_size-1;
					int st=(bj+bi)%mpi_size; // processing stage of the block by current process
					st=(mpi_size+st-mpi_rank)%mpi_size;
					if (bb==st)
					{
#pragma omp task firstprivate(bb,k,j)
					{			
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_start_thread();
#endif						
					rb=bb*(sN/mpi_size);
					re=rb+(sN/mpi_size);
					if ((sN/mpi_size)==0)
					{
						rb=1;
						re=sN;
					}
					if (bb==0) rb=1;
					if (bb==(mpi_size-1)) re=sN;
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
					{
						rb=1;
						re=sN;
					}
#endif						
					m3d_x=j;
					m3d_y=k;
					m3d_c=0;
					U.set(b_U,m3d_x,m3d_y,m3d_c);
					C.set(b_C,m3d_x,m3d_y,m3d_c);
					T.set(b_T,m3d_x,m3d_y,m3d_c);
					set_local_vars();
					dL = sdL;
					Z=sZ;
					N=sN;
					NB = sNB;
					if (implicit3d==0) // locally-onedimensional scheme
					{
						U1(0);
						if (analytic_test == 0)
						{
							if ((nu != 0.0) || (is_r_C))
								U1(1);
							if (kt != 0.0)
								U1(2);
						}
					}
					if (implicit3d==1)
					{
						if (implicit==1)
						{
							implicit1_gen_matrix(Mat1d, Al);
							for (int i1 = 0;i1 <= sN ;i1++)
							{
								for (int j1 = 0;j1 <=sN ;j1++)
									Mat[idx(i1,j,k)][idx(j1,j,k)]+=Mat1d[i1][j1];
								Om[idx(i1,j,k)]+=Al[i1];
							}
						}
						if (implicit==2)
						{
							implicit2_gen(Mat, Al);
							for (int j1 = 0;j1 <= sN ;j1++)
							{
								Mat_3d[5][idx(j1,j,k)]+=Mat[0][j1];
								Mat_3d[1][idx(j1,j,k)]+=Mat[1][j1];
								Mat_3d[6][idx(j1,j,k)]+=Mat[2][j1];
								Mcoefs_3d[0][0][idx(j1,j,k)]+=Mcoefs[0][j1];
								Mcoefs_3d[0][1][idx(j1,j,k)]+=Mcoefs[1][j1];
								Mcoefs_3d[0][2][idx(j1,j,k)]+=Mcoefs[2][j1];
								Mcoefs2_3d[0][0][idx(j1,j,k)]+=Mcoefs2[0][j1];
								Mcoefs2_3d[0][1][idx(j1,j,k)]+=Mcoefs2[1][j1];
								Mcoefs2_3d[0][2][idx(j1,j,k)]+=Mcoefs2[2][j1];
								Om[idx(j1,j,k)]+=Al[j1];
							}
						}
					}
#ifdef USE_MPI
					if (implicit3d==0) mpi_set_end_thread();
#endif						
					}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
					if ((++nrun)==mpi_mp_add_threads)
					{
#pragma omp taskwait
						nrun=0;						
					}
#endif					
					}
				}
				}
				}
				}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
#pragma omp taskwait
#endif					
				}
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
				if (implicit3d==0) 
				    if (sN>2)
				    {
					double *s=b_Utimesum;
#ifdef OCL
					s=Sums;
#endif					
					 repartition(0,b_U);
					 repartition(0,s);
				    }
#endif					
				// solve linear system for implicit three-dimensional schemes
				if (implicit3d==1) 
				{
					Al=sAl;
					Bt=sBt;
					if (implicit==1) // fully filled matrix
					{
						// row scaling
						if (implicit_row_scaling==1)
						for (int i = 0;i <= (sN+2)*(_M+2)*(K+2);i++)
						if (Mat[i][i]!=0.0)
						{
							for (int j = 0;j <= (sN+2)*(_M+2)*(K+2);j++)
								if (j!=i)
									Mat[i][j]/=Mat[i][i];
							Om[i]/=Mat[i][i];
							Mat[i][i]=1.0;
						}
						// solution
						if (equation == 0)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_U[i];
						if (equation == 1)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_C[i];
						if (equation == 2)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_T[i];
						solve_mln<1>(Mat, Om, Al, (sN+2)*(_M+2)*(K+2) + 1, impl_err, impl_niter, vmul,0,(sN+2)*(_M+2)*(K+2) + 1);
						if (equation == 0)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_U[i] = Al[i];
						if (equation == 1)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_C[i] = Al[i];
						if (equation == 2)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_T[i] = Al[i];
					}
					if (implicit == 2) // implicit space fractional - split on Toeplitz matrices
					{
						// row scaling
						if (implicit_row_scaling==1)
						for (int i = 0;i <= (sN+2)*(_M+2)*(K+2);i++)
						if (Mat_3d[1][i]!=0.0)
						{
							Om[i]/=Mat_3d[1][i];
							for (int j=0;j<7;j++)
								if (j!=1)
									Mat_3d[j][i]/=Mat_3d[1][i];
							RS[i]=Mat_3d[1][i];
							Mat_3d[1][i]=1.0;
						}
						else
							RS[i]=1.0;
						// solution
						if (equation == 0)
						{
#ifndef USE_MPI
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_U[i];
#else							
							copy_processed(Al,b_U);
							copy_processed(Bt,Om);
							copy_processed(Om,Bt);
#endif							
						}
						if (equation == 1)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_C[i];
						if (equation == 2)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								Al[i] = b_T[i];
						solve_mln<1>((double **)this, Om, Al, (sN+2)*(_M+2)*(K+2) + 1, impl_err, impl_niter, vmul_T_3d,0,(sN+2)*(_M+2)*(K+2) + 1);
						if (equation == 0)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_U[i] = Al[i];
						if (equation == 1)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_C[i] = Al[i];
						if (equation == 2)
							for (int i = 0;i <= (sN + 2)*(_M + 2)*(K + 2);i++)
								b_T[i] = Al[i];
					}
				}
			}
			// save H for time-fractional derivative calculations
			if (alpha != 1.0)
			{
				double *hs;
				hs = new double[(N + 2)*(_M + 2)*(K + 2)];
				if (mode==3)
					memcpy(hs, b_U, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
				else
					memcpy(hs, b_U, (N + 2)*sizeof(double));
				if ((sum_alg!=2)||(oldH.size()!=2))
					oldH.push_back(hs);
				else
				{
					delete [] oldH[0];
					oldH[0]=oldH[1];
					oldH[1]=hs;
				}
				if (s_is_c)
				{
					hs = new double[(N + 2)*(_M + 2)*(K + 2)];
					if (mode==3)
						memcpy(hs, b_C, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
					else
						memcpy(hs, b_C, (N + 2)*sizeof(double));
					if ((sum_alg!=2)||(oldC.size()!=2))
						oldC.push_back(hs);
					else
					{
						delete [] oldC[0];
						oldC[0]=oldC[1];
						oldC[1]=hs;
					}
				}
				if (s_is_t)
				{
					hs = new double[(N + 2)*(_M + 2)*(K + 2)];
					if (mode==3)
						memcpy(hs, b_T, (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
					else
						memcpy(hs, b_T, (N + 2)*sizeof(double));
					if ((sum_alg!=2)||(oldT.size()!=2))
						oldT.push_back(hs);
					else
					{
						delete [] oldT[0];
						oldT[0]=oldT[1];
						oldT[1]=hs;
					}
				}
			}
		}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
		if (implicit3d==0)
		{
			for (int i=0;i<mpi_size;i++)
				if (i!=mpi_rank)
					make_sends(i);
			for (int ii=0;ii<mpi_size;ii++)
				if (ii!=mpi_rank)
				{
					for (int i=0;i<mpi_sends[ii].size();i++)
						delete [] (char *)mpi_sends[ii][i].buf;
					for (int i=0;i<mpi_recvs[ii].size();i++)
					{
						delete mpi_recvs[ii][i].cv;
						delete mpi_recvs[ii][i].cv_m;
					}
					mpi_sends[ii].clear();
					mpi_recvs[ii].clear();
					mpi_received_msgs[ii].clear();
					last_recvs[ii]=last_sends[ii]=last_msgs[ii]=0;
				}
		}
#endif
	}
	// copy all data from s2 to this solver
	void copy(H_solver *s2)
	{
		clear();
		this[0]=s2[0];
		alloc(s_is_c,s_is_t);
		if (mode==1)
		{
			memcpy(b_U,s2->b_U,(N+2)*sizeof(double));
			memcpy(b_Utimesum,s2->b_Utimesum,(N+2)*sizeof(double));
			if (s_is_c)
			{
				memcpy(b_C,s2->b_C,(N+2)*sizeof(double));
				memcpy(b_Ctimesum,s2->b_Ctimesum,(N+2)*sizeof(double));
			}
			if (s_is_t)
			{
				memcpy(b_T,s2->b_T,(N+2)*sizeof(double));
				memcpy(b_Ttimesum,s2->b_Ttimesum,(N+2)*sizeof(double));
			}
			U.set(b_U,0,0,3);
			C.set(b_C,0,0,3);
			T.set(b_T,0,0,3);
		}
		memcpy(dL, s2->dL, (N + 2)*sizeof(double));
		memcpy(sdL, s2->sdL, (N + 2)*sizeof(double));
		memcpy(Z, s2->Z, (N + 2)*sizeof(double));
		memcpy(gZ, s2->gZ, (N + 2)*sizeof(double));
		memcpy(gZg, s2->gZg, (N + 2)*sizeof(double));
		memcpy(sZ, s2->sZ, (N + 2)*sizeof(double));
		if (mode==3)
		{
			memcpy(b_U,s2->b_U,((sN+2)*(_M+2)*(K+2))*sizeof(double));
			memcpy(b_Utimesum,s2->b_Utimesum,((sN+2)*(_M+2)*(K+2))*sizeof(double));
			if (s_is_c)
			{
				memcpy(b_C,s2->b_C,((sN+2)*(_M+2)*(K+2))*sizeof(double));
				memcpy(b_Ctimesum,s2->b_Ctimesum,((sN+2)*(_M+2)*(K+2))*sizeof(double));
			}
			if (s_is_t)
			{
				memcpy(b_T,s2->b_T,((sN+2)*(_M+2)*(K+2))*sizeof(double));
				memcpy(b_Ttimesum,s2->b_Ttimesum,((sN+2)*(_M+2)*(K+2))*sizeof(double));
			}
			memcpy(dLx,s2->dLx,(_M+2)*sizeof(double));
			memcpy(dLy,s2->dLy,(K+2)*sizeof(double));
			memcpy(X,s2->X,(_M+2)*sizeof(double));
			memcpy(Y,s2->Y,(K+2)*sizeof(double));
			memcpy(gX,s2->gX,(_M+2)*sizeof(double));
			memcpy(gY,s2->gY,(K+2)*sizeof(double));
			memcpy(gXg,s2->gXg,(_M+2)*sizeof(double));
			memcpy(gYg,s2->gYg,(K+2)*sizeof(double));
			for (int i = 0;i < _M + 2;i++)
			{
				memcpy(rp_mult[i],s2->rp_mult[i], (K + 2)*sizeof(double));
				memcpy(spr[i],s2->spr[i], (K + 2)*sizeof(double));
			}
		}
		memcpy(Al,s2->Al,(sN+_M+K+2)*sizeof(double));
		memcpy(Bt,s2->Bt,(sN+ _M + K + 2)*sizeof(double));
		memcpy(Om,s2->Om,(sN+ _M + K + 2)*sizeof(double));
		oldH.clear();
		oldC.clear();
		oldT.clear();
		for (int i = 0;i < s2->oldH.size();i++)
		{
			if (mode == 1)
			{
				double *hs = new double[(N + 2)];
				memcpy(hs, s2->oldH[i], (N + 2)*sizeof(double));
				oldH.push_back(hs);
			}
			if (mode == 3)
			{
				double *hs = new double[(N + 2)*(_M + 2)*(K + 2)];
				memcpy(hs, s2->oldH[i], (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
				oldH.push_back(hs);
			}
		}
		for (int i = 0;i < s2->oldC.size();i++)
		{
			if (mode == 1)
			{
				double *hs = new double[(N + 2)];
				memcpy(hs, s2->oldC[i], (N + 2)*sizeof(double));
				oldC.push_back(hs);
			}
			if (mode == 3)
			{
				double *hs = new double[(N + 2)*(_M + 2)*(K + 2)];
				memcpy(hs, s2->oldC[i], (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
				oldC.push_back(hs);
			}
		}
		for (int i = 0;i < s2->oldT.size();i++)
		{
			if (mode == 1)
			{
				double *hs = new double[(N + 2)];
				memcpy(hs, s2->oldT[i], (N + 2)*sizeof(double));
				oldT.push_back(hs);
			}
			if (mode == 3)
			{
				double *hs = new double[(N + 2)*(_M + 2)*(K + 2)];
				memcpy(hs, s2->oldT[i], (N + 2)*(_M + 2)*(K + 2)*sizeof(double));
				oldT.push_back(hs);
			}
		}
		Time.clear();
		tau.clear();
		for(int i=0;i<s2->tau.size();i++)
		    tau.push_back(s2->tau[i]);
		for(int i=0;i<s2->Time.size();i++)
		    Time.push_back(s2->Time[i]);
		// diagonals
		int i;
		for(i=0;i<Tm_diagonals[0].size();i++)
		{
			double *n=new double[sN+2];
			memcpy(n,s2->Tm_diagonals[0][i],(sN+2)*sizeof(double));
			Tm_diagonals[0][i]=n;
		}
		for(i=0;i<Tm_diagonals[1].size();i++)
		{
			double *n=new double[_M+2];
			memcpy(n,s2->Tm_diagonals[1][i],(_M+2)*sizeof(double));
			Tm_diagonals[1][i]=n;
		}
		for(i=0;i<Tm_diagonals[2].size();i++)
		{
			double *n=new double[K+2];
			memcpy(n,s2->Tm_diagonals[2][i],(K+2)*sizeof(double));
			Tm_diagonals[2][i]=n;
		}
		for(i=0;i<r_Tm_diagonals[0].size();i++)
		{
			double *n=new double[sN+2];
			memcpy(n,s2->r_Tm_diagonals[0][i],(sN+2)*sizeof(double));
			r_Tm_diagonals[0][i]=n;
		}
		for(i=0;i<r_Tm_diagonals[1].size();i++)
		{
			double *n=new double[_M+2];
			memcpy(n,s2->r_Tm_diagonals[1][i],(_M+2)*sizeof(double));
			r_Tm_diagonals[1][i]=n;
		}
		for(i=0;i<r_Tm_diagonals[2].size();i++)
		{
			double *n=new double[K+2];
			memcpy(n,s2->r_Tm_diagonals[2][i],(K+2)*sizeof(double));
			r_Tm_diagonals[2][i]=n;
		}
		if (s2->RS)
		{
			if ((mode==3)&&(implicit == 2))
				memcpy(RS,s2->RS,sizeof(double)*(sN+2)*(_M+2)*(K+2));
			else
				memcpy(RS,s2->RS,sizeof(double)*(sN+_M+K+2));
		}
		if ((sum_alg==2)||(sum_alg==3))
				for (int i=0;i<((mode==1)?(N+2):((sN+2)*(_M+2)*(K+2)));i++)
					for (int j=0;j<Htmp[i].size();j++)
						if (j<s2->Htmp[i].size())
							Htmp[i][j]=s2->Htmp[i][j];

		for (int i=0;i<kb_cache_pts.size();i++)
		{
			if (kb_cache_pts[i]==s2->X) kb_cache_pts[i]=X;
			if (kb_cache_pts[i]==s2->Y) kb_cache_pts[i]=Y;
			if (kb_cache_pts[i]==s2->Z) kb_cache_pts[i]=Z;
			if (kb_cache_pts[i]==(double *)&s2->Time) kb_cache_pts[i]=(double *)&Time;
		}
		if (last_points==s2->X) last_points=X;
		if (last_points==s2->Y) last_points=Y;
		if (last_points==s2->Z) last_points=Z;
		if (last_points==(double *)&s2->Time) last_points=(double *)&Time;
		if (last_points)
		{
			if (last_vi1) last_vi1=&kb_cache[last_i1];
			if (last_vi1a) if (last_a!=-1) last_vi1a=&((*last_vi1)[last_a][0]);
		}
	}
	void clear()
	{
		for (int i = 0;i < oldH.size();i++)
			delete oldH[i];
		for (int i = 0;i < oldC.size();i++)
			delete oldC[i];
		for (int i = 0;i < oldT.size();i++)
			delete oldT[i];
		delete [] b_U;
		delete [] b_Utimesum;
		if (s_is_c)
		{
			delete [] b_C;
			delete [] b_Ctimesum;
		}
		if (s_is_t)
		{
			delete [] b_T;
			delete [] b_Ttimesum;
		}
		delete [] Al;
		delete [] Bt;
		delete [] Om;
		delete [] sdL;
		delete [] sZ; 
		delete [] RS;
		if ((sum_alg==2)||(sum_alg==3))
			delete [] Htmp;
		if (mode == 1)
		{
			delete[] dL;
			delete[] Z;			
			delete[] gZ;			
			delete[] gZg;			
		}
		if (mode == 3)
		{
			for (int i = 0;i < _M + 2;i++)
			{
				delete[] rp_mult[i];
				delete[] spr[i];
			}
			delete[] rp_mult;
			delete[] spr;
			delete []dLx;
			delete []dLy;
			delete []X;
			delete []Y;
			delete []gX;
			delete []gY;
			delete []gXg;
			delete []gYg;
		}
		if (implicit==1)
		{
			for (int i=0;i<sN+ _M + K + 2;i++)
				delete [] Mat[i];
			delete [] Mat;
			if (implicit3d==1)
			{
				for (int i=0;i<sN+ _M + K + 2;i++)
					delete [] Mat1d[i];
				delete [] Mat1d;
			}
		}
		if (implicit == 2) 
		{
			for (int i = 0;i < 3;i++)
				delete[] Mat[i];
			for (int i = 0;i<3;i++)
				delete[] Mcoefs[i];
			for (int i = 0;i<3;i++)
				delete[] Mcoefs2[i];
			delete [] Mat;
			delete[] Mcoefs;
			delete[] Mcoefs2;
			if (mode==3)
			{
				for (int i = 0;i < 7;i++)
					delete[] Mat_3d[i];
				for (int j=0;j<3;j++)
				{
					for (int i = 0;i<3;i++)
						delete[] Mcoefs_3d[j][i];
					for (int i = 0;i<3;i++)
						delete[] Mcoefs2_3d[j][i];
					delete[] Mcoefs_3d[j];
					delete[] Mcoefs2_3d[j];
				}
				delete [] Mat_3d;
			}
			if (vart_main==0)
			{
				for (int l = 0;l < 3;l++)
				for (int j = 0;j < Tm_diagonals[l].size();j++)
					delete[] Tm_diagonals[l][j];
				for (int j = 0;j < 3;j++)
					delete [] Tm[j];
				if (space_der == 1)
				{
					for (int l = 0;l < 3;l++)
						for (int j = 0;j < r_Tm_diagonals[l].size();j++)
							delete[] r_Tm_diagonals[l][j];
				}
			}
		}
#ifdef OCL
		if (use_ocl)
		{
			delete [] Sums;
			//delete prg;
		}
#endif		
	}
	~H_solver()
	{
		clear();
	}
};
#if defined(_OPENMP)||defined(USE_MPI)
	row H_solver::U,H_solver::C,H_solver::T;
	double *H_solver::Al, *H_solver::Bt,*H_solver::sAl, *H_solver::sBt,*H_solver::Om,*H_solver::sOm,*H_solver::F,*H_solver::BVK,*H_solver::RS,*H_solver::sRS;
	int H_solver::m3d_x,H_solver::m3d_y,H_solver::m3d_c; 
	int H_solver::a3k;
	double *H_solver::last_points;
	int H_solver::last_i1;
	int H_solver::last_a;
	double* H_solver::last_vi1a;
	int H_solver::last_vi1a_size;
	std::vector< std::vector<double> > *H_solver::last_vi1;
	double **H_solver::Mat,**H_solver::sMat, **H_solver::Mat1d,**H_solver::Mcoefs,**H_solver::Mcoefs2;
	int H_solver::rb,H_solver::re;
#endif
#ifdef USE_MPI
double parallel_sum(double v)
{
	double s=v;
	MPI_Status st;
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	if (implicit3d==0)
		return v;
#endif						
	if (mpi_rank==0)
	{
		double r;
		for (int i=1;i<mpi_size;i++)
		{
	    	_MPI_Recv(&r,sizeof(double),MPI_BYTE,i,H_solver::m3d_x*mpi_mult+H_solver::m3d_y,MPI_COMM_WORLD,&st);
			s+=r;
		}
		for (int i=1;i<mpi_size;i++)
	    	_MPI_Send(&s,sizeof(double),MPI_BYTE,i,H_solver::m3d_x*mpi_mult+H_solver::m3d_y,MPI_COMM_WORLD);
	}
	else
	{
	   	_MPI_Send(&v,sizeof(double),MPI_BYTE,0,H_solver::m3d_x*mpi_mult+H_solver::m3d_y,MPI_COMM_WORLD);
	   	_MPI_Recv(&s,sizeof(double),MPI_BYTE,0,H_solver::m3d_x*mpi_mult+H_solver::m3d_y,MPI_COMM_WORLD,&st);
	}
	//if (mpi_rank==0) printf("par sum e %g %d %d %d\n",s,mpi_rank,omp_get_thread_num(),H_solver::m3d_x*mpi_mult+H_solver::m3d_y);
	return s;
}
#endif
// z=T~v 
//full=1, space_der=1 (T~=(-T[N-1],...,-T[1],T[0],T[1],...,T[N-1])
//full=0, space_der=1 (T~=(-T[N-1],...,-T[1],0.0,T[1],...,T[N-1])
//full=1, space_der=0 (T~=(0.0,...,0.0,T[0],T[1],...,T[N-1])
//full=0, space_der=0 (T~=(0.0,...,0.0,0.0,T[1],...,T[N-1])
void Toeplitz_mult(double *T, double *v, double *z,int N,int full)
{
	static double *tmps[4] = { NULL,NULL,NULL,NULL }; // 2n
	static double *old_T=NULL;
	static int tmp_n = 0;
	if (tmp_n != N)
	{		
		for (int i = 0;i < 4;i++)
		{
			if (tmps[i]) delete[] tmps[i];
			tmps[i] = new double[5*N];
			memset(tmps[i], 0, 5 * N*sizeof(double));
		}
		tmp_n = N;
	}
	if (old_T!=T)
	{
	    for (int i = 0;i < N;i++)
	    {
			tmps[full][2 * i] = T[i];
			tmps[full][2 * i+1] = 0.0;
			if (space_der == 1)
				tmps[full][2 * (N + i)] = -T[N - i];
			else
				tmps[full][2 * (N + i)] = 0.0;
			tmps[full][2 * (N + i)+1] = 0.0;
	    }
	    if (full == 0)
			tmps[full][0] = 0.0;
	    tmps[full][2 * N] = 0.0;
	    tmps[full][2 * N+1] = 0.0;
	    // c=FFT(c)
	    fft(tmps[full], 2 * N );
	}
	// t1=(v,0)
	for (int i = 0;i < N;i++)
	{
		tmps[2][2*i] = v[i];
		tmps[2][2*i+1] = 0;
		tmps[2][2*(N + i)] = 0;
		tmps[2][2*(N + i)+1] = 0;
	}
	// v=FFT(v)
	fft(tmps[2], 2* N);
	// t2=c x v
	for (int i = 0;i < 2 * N;i++)
	{
		tmps[3][2*i] = tmps[full][2*i] * tmps[2][2 * i] - tmps[full][2*i+1] * tmps[2][2*i+1]; // Re
		tmps[3][2*i+1] = tmps[full][2 * i] * tmps[2][2*i+1] + tmps[full][2*i+1] * tmps[2][2*i]; //Im
	}
	// t2=IFFT(t2)
	ifft(tmps[3],  2*N);
	for (int i = 0;i < 2*2 * N;i++)
		tmps[3][i] /= 2 * N;
	// z[i]=tmps[3][i],i=0,...,N-1
	for (int i = 0;i < N;i++)
		z[i] = tmps[3][2*i];
}
#ifdef USE_MPI
// synchronization of Bt in one dimension
// what - 0: do synch in 1d mode, 1 - fill what to send, 2 - save received
double **synch_BT(H_solver *s,int N,int mode,int _i,int _j,int what=0,double **bb=NULL,int *bb_ns=NULL)
{
	// synchronize Bt
	int axis=s->m3d_c;
	if (mode == 2) axis = 0;
	if (mode == 1) axis = 1;
	if (mode == 0) axis = 2;
	int dir=1.0;
	if (s->m3d_c==0) dir=-1.0;	
	MPI_Request *rrs;
	if (what==0) 
		rrs=new MPI_Request[mpi_size];
	double **bbs;
	if (what!=2)
		bbs=new double *[mpi_size];
	else 
		bbs=bb;
	if (what!=2)
	for (int p=0;p<mpi_size;p++)
		if (p!=mpi_rank)
		{
			int srb=s->rb,sre=s->re;
			if (srb==1) srb=0;
			if (sre==(N-1)) sre=N;
			int bb=srb/((N-1)/mpi_size);
			int pbl=((mpi_size+bb+dir*(p-mpi_rank))%mpi_size);
			int prb=pbl*((N-1)/mpi_size);
			int pre=prb+((N-1)/mpi_size);
			if (pbl==(mpi_size-1)) pre=N;
			// if (prb-Ndiags,pre+Ndiags) intersects with (rb,re) - send Bt on intersection
			prb-=s->Tm_ndiags[axis] + 1;
			pre+=s->Tm_ndiags[axis] + 1;
			prb-=srb;
			pre-=srb;
			bbs[p]=NULL;
			if (what==1)
			    bb_ns[p]=0;
			if ((prb<=(sre-srb))&&(pre>0))
			{
				if (prb<0) prb=0;
				if (pre>(sre-srb)) pre=sre-srb;
				double *bbb=new double[pre-prb];
				for (int e=prb;e<pre;e++)
				{
					if (mode==-1) bbb[e-prb]=s->Bt[srb+e];
					if (mode == 0) bbb[e-prb]=s->Bt[idx(_i, _j, srb+e)];
					if (mode == 1) bbb[e-prb]=s->Bt[idx(_i, srb+e,_j)];
					if (mode == 2) bbb[e-prb]=s->Bt[idx(srb+e,_i,_j)];
				}				
				bbs[p]=bbb;
				if (what==0)
				    _MPI_Isend(bbb,(pre-prb)*sizeof(double),MPI_BYTE,p,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rrs[p]);
				if (what==1)
					bb_ns[p]=pre-prb;
			}
		}
	if (what==1) 
		return bbs;
	for (int p=mpi_size-1;p>=0;p--)
		if (p!=mpi_rank)
		{
			MPI_Status sst;
			int srb=s->rb,sre=s->re;
			if (srb==1) srb=0;
			if (sre==(N-1)) sre=N;
			int bb=srb/((N-1)/mpi_size);
			int pbl=((mpi_size+bb+dir*(p-mpi_rank))%mpi_size);
			int prb=pbl*((N-1)/mpi_size);
			int pre=prb+((N-1)/mpi_size);
			if (pbl==(mpi_size-1)) pre=N;
			int mrb=srb,mre=sre;
			// if (rb-Ndiags,re+Ndiags) intersects with (prb,pre) - receive Bt on intersection
			mrb-=s->Tm_ndiags[axis] + 1;
			mre+=s->Tm_ndiags[axis] + 1;
			mrb-=prb;
			mre-=prb;
			if ((mrb<=(pre-prb))&&(mre>0))
			{
				if (mrb<0) mrb=0;
				if (mre>(pre-prb)) mre=pre-prb;
				double *bbb;
				if (what==0)
				{
					bbb=new double[mre-mrb];
			    	_MPI_Recv(bbb,(mre-mrb)*sizeof(double),MPI_BYTE,p,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&sst);
				}
				else
					bbb=bbs[p];
				for (int e=mrb;e<mre;e++)
				{
					if (mode==-1) s->Bt[prb+e]=bbb[e-mrb];
					if (mode == 0) s->Bt[idx(_i, _j, prb+e)]=bbb[e-mrb];
					if (mode == 1) s->Bt[idx(_i, prb+e,_j)]=bbb[e-mrb];
					if (mode == 2) s->Bt[idx(prb+e,_i,_j)]=bbb[e-mrb];
				}	
				if (what==0)
					delete [] bbb;
			}
		}
	if (what==0)
	for (int p=0;p<mpi_size;p++)
		if (p!=mpi_rank)
			if (bbs[p]!=NULL)
			{
				MPI_Status st;
				_MPI_Wait(&rrs[p],&st);
				delete [] bbs[p];
			}	
	if (what==0)
	{
		delete [] rrs;
		delete [] bbs;	
	}
	return NULL;
}
#endif
void Tm_diags_mul_left(int N, H_solver *s, double *&tmp, int &tmp_n, int &aux_size, double *r,int full=0, int mode = -1, int _i = 0, int _j = 0)
{
	double *gZZ = s->gZ;
	double *gZZg = s->gZg;
	double *daux, *daux2;
	int axis = s->m3d_c;
	int dir=1.0;
	if (s->m3d_c==0) dir=-1.0;
	if (mode == 2) axis = 0;
	if (mode == 1) axis = 1;
	if (mode == 0) axis = 2;
	// multiplication on integrals matrix
	if (axis == 1) {gZZ = s->gX;gZZg = s->gXg;}
	if (axis == 2) {gZZ = s->gY;gZZg = s->gYg;}
	if (s->Tm_coefs[axis].size()==0)
		return;
#ifdef USE_MPI 
#ifndef	USE_REPARTITIONING
	if (implicit3d==0)
		synch_BT(s,N+1,mode,_i,_j);
	// synchronize series coefficients
	if (implicit3d==0)
	if (s->rb!=1)
	{
	    MPI_Status st;
	    _MPI_Recv(&aux_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
	    _MPI_Recv(tmp,aux_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
	}
#endif	
#else	
	s->rb=0;
	s->re=N;
#endif	
	for (int i = s->rb;i < s->re ;i++)
	{
		double v = 0.0;
		int i0 = i - s->Tm_ndiags[axis] + 1;
		if (i0 < 1) i0 = 1;
		// calculated diagonals
		daux = &s->Tm_diagonals[axis][i][i - i0];
		if (mode==-1)
		{
			daux2=&s->Bt[i0];
			for (int j = i0;j < (full?(i+1):i);j++, daux--, daux2++)
				v += daux[0] * daux2[0];
		}
		if (mode == 0)
		{
			daux2=&s->Bt[idx(_i, _j, i0)];
			for (int j = i0;j < (full?(i+1):i);j++, daux--, daux2+=(sN+2)*(_M+2))
				v += daux[0] * daux2[0];
		}
		if (mode == 1)
		{
			daux2=&s->Bt[idx(_i, i0,_j)];
			for (int j = i0;j < (full?(i+1):i);j++, daux--, daux2+=(sN+2))
				v += daux[0] * daux2[0];
		}
		if (mode == 2)
		{
			daux2=&s->Bt[idx(i0,_i,_j)];
			for (int j = i0;j < (full?(i+1):i);j++, daux--, daux2++)
				v += daux[0] * daux2[0];
		}
		// update coefficients vector
		if (i0 != 1)
		{
			int sz = s->Tm_coefs[axis][i0 - 2].size();
			if (tmp_n < sz)
			{
				double *new_tmp = new double[sz];
				memcpy(new_tmp,tmp,aux_size*sizeof(double));
				if (tmp) delete[] tmp;
				tmp=new_tmp;
				tmp_n = sz;
			}
			while (aux_size < sz)
				tmp[aux_size++] = 0.0;
			if (sz)
			{
				daux = &s->Tm_coefs[axis][i0 - 2][0];
				double d;
				if (mode==-1) d=s->Bt[i0 - 1];
				if (mode == 0) d=s->Bt[idx(_i, _j, i0 - 1)];
				if (mode == 1) d=s->Bt[idx(_i, i0 - 1,_j)];
				if (mode == 2)d=s->Bt[idx(i0 - 1,_i,_j)];
				for (int j = 0;j < sz;j++, daux++)
					tmp[j] += daux[0] * d;
			}
		}
		// calculate sum 
		if (aux_size)
			v += _kb_row2_fixed_coefs(gZZ[i], gZZg[i],tmp, aux_size, s->gamma);
		int id = i;
		double mul=1.0;
		if (mode == 0)
			id = idx(_i, _j, i);
		if (mode == 1)
			id = idx(_i, i, _j);
		if (mode == 2)
			id = idx(i, _i, _j);
		if (implicit_row_scaling == 1)
			v /= s->RS[id];
		if (s->equation == 0)
		{
			if (mode==-1) 
				mul=s->inv_dw_dh(i);
			else
			{
				int ssc=s->m3d_c;
				s->m3d_x=_i;
				s->m3d_y=_j;
				if (mode==0)
				{
					s->m3d_c=2;
					mul=s->inv_dw_dh(i);
				}
				if (mode==1)
				{
					s->m3d_c=1;
					mul=s->inv_dw_dh(i);
				}
				if (mode==2)
				{
					s->m3d_c=0;
					mul=s->inv_dw_dh(i);
				}
				s->m3d_c=ssc;
			}
		}
		r[id] += s->Dg*v*mul;
	}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
	// synchronize series coefficients
	if (implicit3d==0)
	if (s->re!=N)
	{
	    MPI_Request rr1,rr2;
	    _MPI_Isend(&aux_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr1);
	    _MPI_Isend(tmp,aux_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr2);
	}
#endif	
}
void Tm_diags_mul_right(int N, H_solver *s, double *&tmp2, int &tmp_n2, int &aux_size2, double *r, int mode = -1, int _i = 0, int _j = 0)
{
	double *gZZ = s->gZ;
	double *gZZg = s->gZg;
	double *daux, *daux2;
	int axis = s->m3d_c;
	int dir=1.0;
	if (s->m3d_c==0) dir=-1.0;
	if (mode == 2) axis = 0;
	if (mode == 1) axis = 1;
	if (mode == 0) axis = 2;
	// multiplication on integrals matrix
	if (axis == 1) {gZZ = s->gX;gZZg = s->gXg;}
	if (axis == 2) {gZZ = s->gY;gZZg = s->gYg;}
	if (s->r_Tm_coefs[axis].size()==0)
		return;
#ifdef USE_MPI
#ifndef	USE_REPARTITIONING
	if (implicit3d==0)
		synch_BT(s,N+1,mode,_i,_j);
	// synchronize series coefficients
	if (implicit3d==0)
	if (s->rb!=1)
	{
	    MPI_Status st;
	    _MPI_Recv(&aux_size2,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
	    _MPI_Recv(tmp2,aux_size2*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
	}
#endif	
#else
	s->rb=0;
	s->re=N;
#endif	
	for (int i = s->re;i >= s->rb;i--)
	{
		double v = 0.0;
		int i0 = i + s->r_Tm_ndiags[axis] - 1;
		if (i0 > (N - 3)) i0 = N - 3;
		// calculated diagonals
		daux = &s->r_Tm_diagonals[axis][i][i0 - i];
		if (mode==-1)
		{
			daux2=&s->Bt[i0];
			for (int j = i + 1;j <= i0;j++, daux--, daux2--)
				v += -daux[0] * daux2[0];
		}
		if (mode == 0)			
		{
			daux2=&s->Bt[idx(_i, _j, i0)];
			for (int j = i + 1;j <= i0;j++, daux--, daux2-=(sN+2)*(_M+2))
				v += -daux[0] * daux2[0];
		}
		if (mode == 1)
		{
			daux2=&s->Bt[idx(_i, i0,_j)];
			for (int j = i + 1;j <= i0;j++, daux--, daux2-=(sN+2))
				v += -daux[0] * daux2[0];
		}
		if (mode == 2)
		{
			daux2=&s->Bt[idx(i0,_i,_j)];
			for (int j = i + 1;j <= i0;j++, daux--, daux2--)
				v += -daux[0] * daux2[0];
		}
		// update coefficients vector
		if (i0 != (N - 3))
		{
			int sz = s->r_Tm_coefs[axis][i0 - 2].size();
			if (tmp_n2 < sz)
			{
				double *new_tmp = new double[sz];
				memcpy(new_tmp,tmp2,aux_size2*sizeof(double));
				if (tmp2) delete[] tmp2;
				tmp2=new_tmp;
				tmp_n2 = sz;
			}
			while (aux_size2 < sz)
				tmp2[aux_size2++] = 0.0;
			if (sz)
			{
				daux = &s->r_Tm_coefs[axis][i0 - 2][0];
				double d;
				if (mode==-1) d=-s->Bt[i0 - 1];
				if (mode == 0) d=-s->Bt[idx(_i, _j, i0 - 1)];
				if (mode == 1) d=-s->Bt[idx(_i, i0 - 1,_j)];
				if (mode == 2)d=-s->Bt[idx(i0 - 1,_i,_j)];
				for (int j = 0;j < sz;j++, daux++)
					tmp2[j] += daux[0] * d;
			}
		}
		// calculate sum 
		if (aux_size2)
			v += _kb_row2_fixed_coefs(gZZ[i],gZZg[i], tmp2, aux_size2, s->gamma, 1);
		int id = i;
		double mul=1.0;
		if (mode == 0)
			id = idx(_i, _j, i);
		if (mode == 1)
			id = idx(_i, i, _j);
		if (mode == 2)
			id = idx(i, _i, _j);
		if (implicit_row_scaling == 1)
			v /= s->RS[id];
		if (s->equation == 0)
		{
			if (mode==-1) 
				mul=s->inv_dw_dh(i);
			else
			{
				int ssc=s->m3d_c;
				s->m3d_x=_i;
				s->m3d_y=_j;
				if (mode==0)
				{
					s->m3d_c=2;
					mul=s->inv_dw_dh(i);
				}
				if (mode==1)
				{
					s->m3d_c=1;
					mul=s->inv_dw_dh(i);
				}
				if (mode==2)
				{
					s->m3d_c=0;
					mul=s->inv_dw_dh(i);
				}
				s->m3d_c=ssc;
			}
		}
		r[id] += s->Dg*v*mul;
	}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
	// synchronize series coefficients
	if (implicit3d==0)
	if (s->re!=N)
	{
	    MPI_Request rr1,rr2;
	    _MPI_Isend(&aux_size2,sizeof(int),MPI_BYTE,(mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr1);
	    _MPI_Isend(tmp2,aux_size2*sizeof(double),MPI_BYTE,(mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr2);
	}
#endif	
}
// matrix-vector mult for tridiag(Mat)+sum(T*Ci,i=0,..,3)
void vmul_T(double **K, double *x, double *r, int N)
{
	static double *tmp = NULL;
	static int tmp_n = 0;
	static double *tmp2 = NULL;
	static int tmp_n2 = 0;
#pragma omp threadprivate(tmp,tmp_n,tmp2,tmp_n2)
	if (tmp_n < N)
	{
		if (tmp) delete[] tmp;
		tmp = new double[N];
		tmp_n = N;
	}
	if (tmp_n2 < N)
	{
		if (tmp2) delete[] tmp2;
		tmp2 = new double[N];
		tmp_n2 = N;
	}
	H_solver *s = (H_solver *)K; // solver in K
								 // multiply main tridiagonal
	int aux_size = 0;
	int aux_size2 = 0;
	
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
	// synchronize x
	{
	    MPI_Request rr1,rr2;
		int dir=1.0;
		if (s->m3d_c==0) dir=-1.0;
		if (s->rb!=1)
		    _MPI_Isend(&x[s->rb],sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr1);
		if (s->re!=N-1)
		    _MPI_Isend(&x[s->re-1],sizeof(double),MPI_BYTE,(mpi_size+mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&rr2);
	}
	{
	    MPI_Status st;
		int dir=1.0;
		if (s->m3d_c==0) dir=-1.0;
		if (s->re!=N-1)
		    _MPI_Recv(&x[s->re],sizeof(double),MPI_BYTE,(mpi_size+mpi_rank+dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
		if (s->rb!=1)
		    _MPI_Recv(&x[s->rb-1],sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-dir)%mpi_size,s->m3d_x*mpi_mult+s->m3d_y,MPI_COMM_WORLD,&st);
	}
#endif	
	for (int i = s->rb;i<s->re;i++)
		r[i] = s->Mat[0][i] * x[i - 1] + s->Mat[1][i] * x[i] + s->Mat[2][i] * x[i + 1];
	r[0] = s->Mat[1][0] * x[0] + s->Mat[2][0] * x[1];
	r[N-1] = s->Mat[0][N-1] * x[N - 2] + s->Mat[1][N-1] * x[N-1];
	// multiply T*C[0]
	for (int j = s->rb;j<s->re;j++)
		s->Bt[j] = s->Mcoefs[0][j] * x[j - 1] + s->Mcoefs[1][j] * x[j] + s->Mcoefs[2][j] * x[j + 1];
	s->Bt[0] = s->Mcoefs[1][0] * x[0] + s->Mcoefs[2][0] * x[1];
	s->Bt[N-1] = s->Mcoefs[0][N-1] * x[N - 2] + s->Mcoefs[1][N-1] * x[N-1];
	if (toeplitz_mult_alg == 0)
	{
		for (int i = 1;i < N-1 ;i++)
		{
			double v = 0.0;
			for (int j = 1;j < i;j++)
				v += s->Bt[j] * s->Tm[s->m3d_c][i - j + 1];
			if (space_der==1)
				for (int j = i+1;j < N-1 ;j++)
					v -= s->Bt[j] * s->Tm[s->m3d_c][j-i + 1];
			if (implicit_row_scaling==1)
				v/=s->RS[i];
			r[i] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
		}
	}
	if (toeplitz_mult_alg == 1)
	{
		Toeplitz_mult(s->Tm[s->m3d_c]+1, s->Bt+1, tmp, N-2, 0);
		for (int i = 1;i < N-1 ;i++)
		{
			if (implicit_row_scaling==1)
				tmp[i-1]/=s->RS[i];
			r[i] += tmp[i-1] * ((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
		}
	}
	if (toeplitz_mult_alg == 2)
	{
		Tm_diags_mul_left(N-1, s, tmp, tmp_n, aux_size, r);
		if (space_der == 1)
			Tm_diags_mul_right(N-1, s, tmp2, tmp_n2, aux_size2, r);
	}
	// multiply T*C[1,2,3] for C and T equations
	if (s->equation != 0)
	{
		for (int j = 1;j < N - 1;j++)
			s->Bt[j] = s->Mcoefs2[0][j] * x[j - 1] + s->Mcoefs2[1][j] * x[j] + s->Mcoefs2[2][j] * x[j + 1];
		s->Bt[0] = s->Mcoefs2[1][0] * x[0] + s->Mcoefs2[2][0] * x[1];
		s->Bt[N - 1] = s->Mcoefs2[0][N - 1] * x[N - 2] + s->Mcoefs2[1][N - 1] * x[N - 1];
		if (toeplitz_mult_alg == 0)
		{
			for (int i = 1;i < N ;i++)
			{
				double v = 0.0;
				for (int j = 1;j <=i;j++)
					v += s->Bt[j] * s->Tm[s->m3d_c][i - j + 1];
				if (space_der==1)
					for (int j = i+1;j < N ;j++)
						v -= s->Bt[j] * s->Tm[s->m3d_c][j-i + 1];
				if (implicit_row_scaling==1)
					v/=s->RS[i];
				r[i] += v;
			}
		}
		if (toeplitz_mult_alg == 1)
		{
			Toeplitz_mult(s->Tm[s->m3d_c]+1, s->Bt+1, tmp, N-1, 1);
			for (int i = 1;i < N ;i++)
			{
				if (implicit_row_scaling==1)
					tmp[i-1]/=s->RS[i];
				r[i] += tmp[i-1];
			}
		}
		if (toeplitz_mult_alg == 2)
		{
			// multiplication on integrals matrix
			aux_size=0;
			aux_size2=0;
			Tm_diags_mul_left(N, s, tmp, tmp_n, aux_size, r,1);
			if (space_der == 1)
				Tm_diags_mul_right(N, s, tmp2, tmp_n2, aux_size2, r);
		}
	}
}
// alloc and thread thread local
std::vector<double **> tmps,tmp2s;
std::vector<int*> tmp_ns,tmp_n2s;
void vmul_T_3d_set_tmp(int N,double **tmp,int *tmp_n,double **tmp2,int *tmp_n2)
{
#ifdef _OPENMP
	int id=omp_get_thread_num();
#pragma omp critical
	{
		while (tmps.size()<=id)
		{
			tmps.push_back(NULL);
			tmp2s.push_back(NULL);
			tmp_ns.push_back(NULL);
			tmp_n2s.push_back(NULL);

		}
		if (tmps[id]==NULL)
		{
			tmps[id]=new double *[3];			
			tmp2s[id]=new double *[3];
			tmp_ns[id]=new int[3];
			tmp_n2s[id]=new int[3];
			for (int i=0;i<3;i++)
			{
				tmps[id][i]=tmp2s[id][i]=NULL;
				tmp_ns[id][i]=tmp_n2s[id][i]=0;
			}
		}
		for (int i=0;i<3;i++)
		{
			if (tmp_ns[id][i] < N)
			{
				if (tmps[id][i]) delete[] tmps[id][i];
				tmps[id][i] = new double[N];
				tmp_ns[id][i] = N;
			}
			if (tmp_n2s[id][i] < N)
			{
				if (tmp2s[id][i]) delete[] tmp2s[id][i];
				tmp2s[id][i] = new double[N];
				tmp_n2s[id][i] = N;
			}
		}
		for (int i=0;i<3;i++)
		{	 
			tmp[i]=tmps[id][i];
			tmp2[i]=tmp2s[id][i];
			tmp_n[i]=tmp_ns[id][i];
			tmp_n2[i]=tmp_n2s[id][i];
		}
	}
#endif
}
// matrix-vector mult for 7-diag(Mat)+sum(T*Ci,i=0,..,3) (for 3d problems)
void vmul_T_3d(double **_K, double *x, double *r, int N)
{
	static double *tmp[3] = {NULL,NULL,NULL};
	static int tmp_n[3] = {0,0,0};
	static double *tmp2[3] = {NULL,NULL,NULL};
	static int tmp_n2[3] = {0,0,0};
#pragma omp threadprivate(tmp,tmp_n,tmp2,tmp_n2)	
	for (int i=0;i<3;i++)
	{
		if (tmp_n[i] < N)
		{
			if (tmp[i]) delete[] tmp[i];
			tmp[i] = new double[N];
			tmp_n[i] = N;
		}
		if (tmp_n2[i] < N)
		{
			if (tmp2[i]) delete[] tmp2[i];
			tmp2[i] = new double[N];
			tmp_n2[i] = N;
		}
	}
	// coefficients for mpi mode
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
	double **coefs=new double*[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	int *coefs_size=new int[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	int *coefs_n=new int[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	double **coefs2=new double*[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	int *coefs_size2=new int[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	int *coefs_n2=new int[(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2)];
	for (int i=0;i<(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2);i++)
	{
		coefs[i]=new double[20];
		coefs_size[i]=0;
		coefs_n[i]=20;
		if (space_der==1)
		{
			coefs2[i]=new double[20];
			coefs_size2[i]=0;
			coefs_n2[i]=20;
		}
	}
#endif	
	H_solver *s = (H_solver *)_K; // solver in K
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	 // multiply main tridiagonal
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	double *ZZ;
	double *daux,*daux2;
#ifndef USE_MPI
#pragma omp parallel for
	for (int i = 0;i<N;i++)
	{
		r[i] =  s->Mat_3d[1][i] * x[i];
		if (sN>2)
		{
			if ((i-1)>=0)
				r[i]+=s->Mat_3d[5][i]*x[i-1];
			if ((i+1)<N)
				r[i]+=s->Mat_3d[6][i]*x[i+1];
		}
		if (_M>2)
		{
			if ((i-(sN+2))>=0)
				r[i]+=s->Mat_3d[3][i]*x[i-(sN+2)];
			if ((i+(sN+2))<N)
				r[i]+=s->Mat_3d[4][i]*x[i+(sN+2)];
		}
		if (K>2)
		{
			if ((i-(sN+2)*(_M+2))>=0)
				r[i]+=s->Mat_3d[0][i]*x[i-(sN+2)*(_M+2)];
			if ((i+(sN+2)*(_M+2))<N)
				r[i]+=s->Mat_3d[2][i]*x[i+(sN+2)*(_M+2)];
		}
	}
#else
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// synchronization
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
	if (mpi_size!=1)
	{
		double *send_buf[2],*recv_buf[2],*sbp[2],*rbp[2]; // 0 - communications with rank-1, 1 - with rank+!
		MPI_Request rrs[2];
		MPI_Status sts[2];
		// allocate arrays
		for (int i=0;i<2;i++)
		{
			sbp[i]=send_buf[i]=new double[((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))];
			rbp[i]=recv_buf[i]=new double[((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))];
		}
		// collect data to send
		if (K>2)
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(K/mpi_size);
			s->re=s->rb+(K/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=K+1;
			if (s->rb!=0)
				sbp[0][0]=x[idx(i,j,s->rb)];
			if (s->re!=K+1)
				sbp[1][0]=x[idx(i,j,s->re-1)];
			sbp[0]++;
			sbp[1]++;
		}
		if (_M>2)
		for (int i = 0;i <= sN ;i++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(_M/mpi_size);
			s->re=s->rb+(_M/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=_M+1;
			if (s->rb!=0)
				sbp[0][0]=x[idx(i,s->rb,k)];
			if (s->re!=_M+1)
				sbp[1][0]=x[idx(i,s->re-1,k)];
			sbp[0]++;
			sbp[1]++;
		}
		if (sN>2)
		for (int j = 0;j <= _M ;j++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=j/(_M/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			s->rb=st*(sN/mpi_size);
			s->re=s->rb+(sN/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=sN+1;
			if (s->rb!=0)
				sbp[1][0]=x[idx(s->rb,j,k)];
			if (s->re!=sN+1)
				sbp[0][0]=x[idx(s->re-1,j,k)];
			sbp[0]++;
			sbp[1]++;
		}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// do send
		_MPI_Isend(send_buf[0],((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[0]);
		_MPI_Isend(send_buf[1],((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[1]);
		// do receive
		_MPI_Recv(recv_buf[1],((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[1]);
		_MPI_Recv(recv_buf[0],((sN+1)*(_M+1)+(_M+1)*(K+1)+(K+1)*(sN+1))*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[0]);
		// wait for communications to end
		_MPI_Wait(&rrs[0],&sts[0]);
		_MPI_Wait(&rrs[1],&sts[1]);
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// propagate received data
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		if (K>2)
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(K/mpi_size);
			s->re=s->rb+(K/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=K+1;
			if (s->re!=K+1)
				x[idx(i,j,s->re)]=rbp[1][0];
			if (s->rb!=0)
				x[idx(i,j,s->rb-1)]=rbp[0][0];
			rbp[1]++;
			rbp[0]++;
		}
		if (_M>2)
		for (int i = 0;i <= sN ;i++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(_M/mpi_size);
			s->re=s->rb+(_M/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=_M+1;
			if (s->re!=_M+1)
				x[idx(i,s->re,k)]=rbp[1][0];
			if (s->rb!=0)
				x[idx(i,s->rb-1,k)]=rbp[0][0];
			rbp[1]++;
			rbp[0]++;
		}
		if (sN>2)
		for (int j = 0;j <= _M ;j++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=j/(_M/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			s->rb=st*(sN/mpi_size);
			s->re=s->rb+(sN/mpi_size);
			if (st==0) s->rb=0;
			if (st==(mpi_size-1)) s->re=sN+1;
			if (s->re!=sN+1)
				x[idx(s->re,j,k)]=rbp[0][0];
			if (s->rb!=0)
				x[idx(s->rb-1,j,k)]=rbp[1][0];
			rbp[1]++;
			rbp[0]++;
		}		
		// clean up
		for (int i=0;i<2;i++)
		{
			delete [] send_buf[i];
			delete [] recv_buf[i];
		}
	}
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	//process
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#pragma omp parallel for
	for (int i = 0;i<N;i++)
		r[i] = 0;
#ifdef USE_REPARTITIONING
	int id1_st=0,id1_end=sN+1;									
	int id2_st=0,id2_end=_M+1;
	if (sN>2)
	{
		id1_st=mpi_rank*(sN/mpi_size);
		id1_end=id1_st+(sN/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=sN+1;
	}
	else
	{
		id2_st=mpi_rank*(_M/mpi_size);
		id2_end=id2_st+(_M/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=_M+1;
	}
#endif					
	if (K>2)
#pragma omp parallel for collapse(2) 
	for (int i = 0;i <= sN ;i++)
	for (int j = 0;j <= _M ;j++)	
#ifdef USE_REPARTITIONING
	if ((i>=id1_st)&&(i<id1_end))
	if ((j>=id2_st)&&(j<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (sN/mpi_size) bi=i/(sN/mpi_size);
		if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=bi-bj; // processing stage of the block by current process
		if (st<0) st+=mpi_size;
		st=(st+mpi_rank)%mpi_size;
		s->rb=st*(K/mpi_size);
		s->re=s->rb+(K/mpi_size);
		if (st==0) s->rb=0;
		if (st==(mpi_size-1)) s->re=K+1;
#ifdef USE_REPARTITIONING
		s->rb=0;
		s->re=K+1;
#endif
		for (int k=s->rb;k<s->re;k++)
		{
			r[idx(i,j,k)] =  s->Mat_3d[1][idx(i,j,k)] * x[idx(i,j,k)];
			if (k!=0)
				r[idx(i,j,k)]+=s->Mat_3d[0][idx(i,j,k)]*x[idx(i,j,k-1)];
			if (k!=K)
				r[idx(i,j,k)]+=s->Mat_3d[2][idx(i,j,k)]*x[idx(i,j,k+1)];
		}
	}
#ifdef USE_REPARTITIONING
	if (K>2) 
	{
		s->repartition(2,x,1);
		s->repartition(2,r,1);
	}
#endif					
#ifdef USE_REPARTITIONING
	id1_st=0,id1_end=K+1;
	id2_st=0,id2_end=sN+1;
	if (K>2)
	{
		id1_st=mpi_rank*(K/mpi_size);
		id1_end=id1_st+(K/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=K+1;
	}
	else
	{
		id2_st=mpi_rank*(sN/mpi_size);
		id2_end=id2_st+(sN/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=sN+1;
	}				
#endif					
	if (_M>2)
#pragma omp parallel for collapse(2) 
	for (int k = 0;k <= K ;k++)
	for (int i = 0;i <= sN ;i++)
#ifdef USE_REPARTITIONING
	if ((k>=id1_st)&&(k<id1_end))
	if ((i>=id2_st)&&(i<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (sN/mpi_size) bi=i/(sN/mpi_size);
		if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=bi-bj; // processing stage of the block by current process
		if (st<0) st+=mpi_size;
		st=(st+mpi_rank)%mpi_size;
		s->rb=st*(_M/mpi_size);
		s->re=s->rb+(_M/mpi_size);
		if (st==0) s->rb=0;
		if (st==(mpi_size-1)) s->re=_M+1;
#ifdef USE_REPARTITIONING
		s->rb=0;
		s->re=_M+1;
#endif
		for (int j=s->rb;j<s->re;j++)
		{
			if (K<=2)
				r[idx(i,j,k)] =  s->Mat_3d[1][idx(i,j,k)] * x[idx(i,j,k)];
			if (j!=0)
				r[idx(i,j,k)]+=s->Mat_3d[3][idx(i,j,k)]*x[idx(i,j-1,k)];
			if (j!=_M)
				r[idx(i,j,k)]+=s->Mat_3d[4][idx(i,j,k)]*x[idx(i,j+1,k)];
		}
	} 
#ifdef USE_REPARTITIONING
	if (_M>2) 
	{
		s->repartition(1,x,1);
		s->repartition(1,r,1);
	}
#endif					
#ifdef USE_REPARTITIONING
	id1_st=0,id1_end=_M+1;
	id2_st=0,id2_end=K+1;
	if (_M>2)
	{
		id1_st=mpi_rank*(_M/mpi_size);
		id1_end=id1_st+(_M/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=_M+1;
	}
	else
	{
		id2_st=mpi_rank*(K/mpi_size);
		id2_end=id2_st+(K/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=K+1;
	}				
#endif					
	if (sN>2)
#pragma omp parallel for collapse(2) 
	for (int j = 0;j <= _M ;j++)
	for (int k = 0;k <= K ;k++)
#ifdef USE_REPARTITIONING
	if ((j>=id1_st)&&(j<id1_end))
	if ((k>=id2_st)&&(k<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (_M/mpi_size) bi=j/(_M/mpi_size);
		if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=(bj+bi)%mpi_size; // processing stage of the block by current process
		st=(mpi_size+st-mpi_rank)%mpi_size;
		s->rb=st*(sN/mpi_size);
		s->re=s->rb+(sN/mpi_size);
		if (st==0) s->rb=0;
		if (st==(mpi_size-1)) s->re=sN+1;
#ifdef USE_REPARTITIONING
		s->rb=0;
		s->re=sN+1;
#endif
		for (int i=s->rb;i<s->re;i++)
		{
			if (i!=0)
				r[idx(i,j,k)]+=s->Mat_3d[5][idx(i,j,k)]*x[idx(i-1,j,k)];
			if (i!=sN)
				r[idx(i,j,k)]+=s->Mat_3d[6][idx(i,j,k)]*x[idx(i+1,j,k)];
		}
	}
#ifdef USE_REPARTITIONING
	if (sN>2)
	{
		s->repartition(0,x,1);
		s->repartition(0,r,1);
	}
#endif					
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///// Bt dir 2
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
if (K>2)
{
	s->m3d_c=2;
#ifndef USE_MPI
#pragma omp parallel for
	for (int i = 0;i<N;i++)
	{
		s->Bt=s->sBt;
		s->Bt[i] =  s->Mcoefs_3d[2][1][i] * x[i];
		if ((i-(sN+2)*(_M+2))>=0)
			s->Bt[i]+=s->Mcoefs_3d[2][0][i]*x[i-(sN+2)*(_M+2)];
		if ((i+(sN+2)*(_M+2))<N)
			s->Bt[i]+=s->Mcoefs_3d[2][2][i]*x[i+(sN+2)*(_M+2)];
	}
#else
#ifdef USE_REPARTITIONING
	int id1_st=0,id1_end=sN+1;									
	int id2_st=0,id2_end=_M+1;
	if (sN>2)
	{
		id1_st=mpi_rank*(sN/mpi_size);
		id1_end=id1_st+(sN/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=sN+1;
	}
	else
	{
		id2_st=mpi_rank*(_M/mpi_size);
		id2_end=id2_st+(_M/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=_M+1;
	}
#endif					
#pragma omp parallel for collapse(2) 
	for (int i = 0;i <= sN ;i++)
	for (int j = 0;j <= _M ;j++)	
#ifdef USE_REPARTITIONING
	if ((i>=id1_st)&&(i<id1_end))
	if ((j>=id2_st)&&(j<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (sN/mpi_size) bi=i/(sN/mpi_size);
		if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=bi-bj; 
		if (st<0) st+=mpi_size;
		st=(st+mpi_rank)%mpi_size;
		s->rb=st*(K/mpi_size);
		s->re=s->rb+(K/mpi_size);
		if (st==0) s->rb=1;
		if (st==(mpi_size-1)) s->re=K;
#ifdef USE_REPARTITIONING
		s->rb=1;
		s->re=K;
#endif
		s->Bt=s->sBt;
		for (int k=s->rb;k<s->re;k++)
		{
			s->Bt[idx(i,j,k)] =  s->Mcoefs_3d[2][1][idx(i,j,k)] * x[idx(i,j,k)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[2][0][idx(i,j,k)]*x[idx(i,j,k-1)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[2][2][idx(i,j,k)]*x[idx(i,j,k+1)];
		}
	}
	///////////////////////////////////////////////////////////
	// synch bt dir 2
	///////////////////////////////////////////////////////////
	s->m3d_x=s->m3d_y=0;
#ifndef USE_REPARTITIONING
	if (mpi_size!=1)
	{
		int *bb_ns=new int[mpi_size];
		MPI_Request *rrs=new MPI_Request[2*mpi_size];
		MPI_Status st;
		std::vector< std::vector<double> > to_send;
		for (int i=0;i<mpi_size;i++)
			to_send.push_back(std::vector<double>());
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; 
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(K/mpi_size);
			s->re=s->rb+(K/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=K;
			double **bbs=synch_BT(s,K+1,0,i,j,1,NULL,bb_ns);
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
			{
				to_send[ii].push_back(bb_ns[ii]);
				for (int jj=0;jj<bb_ns[ii];jj++)
					to_send[ii].push_back(bbs[ii][jj]);
				delete [] bbs[ii];
			}
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			bb_ns[i]=to_send[i].size();
			_MPI_Isend(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+0]);
			_MPI_Isend(&to_send[i][0],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+1]);		
		}
		double **bufs=new double*[mpi_size];
		double **bprs=new double *[mpi_size];
		for (int i=mpi_size-1;i>=0;i--)
		if (i!=mpi_rank)
		{
			_MPI_Recv(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
			bprs[i]=bufs[i]=new double[bb_ns[i]];
			_MPI_Recv(bufs[i],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
		}
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; 
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(K/mpi_size);
			s->re=s->rb+(K/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=K;
			double **bbs=new double *[mpi_size];				
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
			{
				int n=(bprs[ii]++)[0];
				bbs[ii]=new double[n];
				for (int jj=0;jj<n;jj++)
					bbs[ii][jj]=(bprs[ii]++)[0];
			}
			synch_BT(s,K+1,0,i,j,2,bbs,NULL);
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
				delete [] bbs[ii];
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			delete [] bufs[i];
			_MPI_Wait(&rrs[2*i+0],&st);
			_MPI_Wait(&rrs[2*i+1],&st);
		}
		delete [] bufs;
		delete [] bprs;
		delete [] bb_ns;
		delete [] rrs;
	}
#endif
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	/// multiplication dir 2 
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_MPI
#pragma omp parallel for collapse(2)
	for (int _i = 1;_i < sN;_i++)
		for (int _j = 1;_j < _M;_j++)
		{
#else
#ifndef USE_REPARTITIONING
		for (int i=0;i<(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2);i++)
			coefs_size[i]=coefs_size2[i]=0;
#endif
		for (int bb=0;bb<mpi_size;bb++)
		{
		MPI_Request rrs[6];
		MPI_Status sts[6];
		double *full_coefs_s, *full_coefs2_s;
		int *coefs_size_s,*coefs_size2_s;
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 1 dir 2
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=0)
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
		    _MPI_Recv(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[0]);
	    	_MPI_Recv(coefs_size,((_M+1)*(sN+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[1]);
			full_coefs=new double[scoefs_size];
	    	_MPI_Recv(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[2]);
			if (space_der==1)
			{
				_MPI_Recv(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[3]);
				_MPI_Recv(coefs_size2,((_M+1)*(sN+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[4]);
				full_coefs2=new double[scoefs2_size];
				_MPI_Recv(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[5]);
			}			
			_MPI_Wait(&rrs[0],&sts[0]);
			_MPI_Wait(&rrs[1],&sts[1]);
			_MPI_Wait(&rrs[2],&sts[2]);
			if (space_der==1)
			{
				_MPI_Wait(&rrs[3],&sts[3]);
				_MPI_Wait(&rrs[4],&sts[4]);
				_MPI_Wait(&rrs[5],&sts[5]);
			}
			delete [] full_coefs_s;
			delete [] coefs_size_s;
			if (space_der==1)
			{
				delete [] full_coefs2_s;
				delete [] coefs_size2_s;
			}			
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= _M ;_j++)	
			{
				if (coefs_n[_i*(_M+1)+_j]<coefs_size[_i*(_M+1)+_j])
				{
					if (coefs[_i*(_M+1)+_j]) delete [] coefs[_i*(_M+1)+_j];
					coefs[_i*(_M+1)+_j]=new double[coefs_size[_i*(_M+1)+_j]];
					coefs_n[_i*(_M+1)+_j]=coefs_size[_i*(_M+1)+_j];
				}
				memcpy(coefs[_i*(_M+1)+_j],fcp,coefs_size[_i*(_M+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(_M+1)+_j];
				if (space_der==1)
				{
					if (coefs_n2[_i*(_M+1)+_j]<coefs_size2[_i*(_M+1)+_j])
					{
						if (coefs2[_i*(_M+1)+_j]) delete [] coefs2[_i*(_M+1)+_j];
						coefs2[_i*(_M+1)+_j]=new double[coefs_size2[_i*(_M+1)+_j]];
						coefs_n2[_i*(_M+1)+_j]=coefs_size2[_i*(_M+1)+_j];
					}
					memcpy(coefs2[_i*(_M+1)+_j],fcp2,coefs_size2[_i*(_M+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(_M+1)+_j];
				}
			}
			delete [] full_coefs;
			if (space_der==1)
				delete [] full_coefs2;
		}
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// process dir 2
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifdef USE_REPARTITIONING
		int id1_st=0,id1_end=sN+1;									
		int id2_st=0,id2_end=_M+1;
		if (sN>2)
		{
			id1_st=mpi_rank*(sN/mpi_size);
			id1_end=id1_st+(sN/mpi_size);
			if (mpi_rank==0) id1_st=0;
			if (mpi_rank==(mpi_size-1)) id1_end=sN+1;
		}
		else
		{
			id2_st=mpi_rank*(_M/mpi_size);
			id2_end=id2_st+(_M/mpi_size);
			if (mpi_rank==0) id2_st=0;
			if (mpi_rank==(mpi_size-1)) id2_end=_M+1;
		}
#endif					
#pragma omp parallel for collapse(2) firstprivate(bb) 
		for (int _i = 0;_i <= sN ;_i++)
		for (int _j = 0;_j <= _M ;_j++)	
#ifdef USE_REPARTITIONING
		if ((_i>=id1_st)&&(_i<id1_end))
		if ((_j>=id2_st)&&(_j<id2_end))
#endif
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=_i/(sN/mpi_size);
			if (_M/mpi_size) bj=_j/(_M/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			if (bb==st)
			{
			s->rb=bb*(K/mpi_size);
			s->re=s->rb+(K/mpi_size);
			if (bb==0) s->rb=1;
			if (bb==(mpi_size-1)) s->re=K;
#ifdef USE_REPARTITIONING
			s->rb=1;
			s->re=K;
#endif				
#endif
			int aux_size[3] = {0,0,0};
			int aux_size2[3] = {0,0,0};
			s->m3d_c=2;
			s->m3d_x=_i;
			s->m3d_y=_j;
			s->Bt=s->sBt;
			s->RS=s->sRS;
			s->U.set(s->b_U,s->m3d_x,s->m3d_y,s->m3d_c);
			s->C.set(s->b_C,s->m3d_x,s->m3d_y,s->m3d_c);
			s->T.set(s->b_T,s->m3d_x,s->m3d_y,s->m3d_c);
			vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
			if (toeplitz_mult_alg == 0)
			{
				for (int i = 1;i < K;i++)
				{
					double v = 0.0;
					for (int j = 1;j < i;j++)
						v += s->Bt[idx(_i,_j,j)]* s->Tm[2][i - j + 1];
					if (space_der==1)
						for (int j = i+1;j < K;j++)
							v -= s->Bt[idx(_i,_j,j)] * s->Tm[2][j-i + 1];
					if (implicit_row_scaling==1)
						v/=s->RS[idx(_i,_j,i)];
					r[idx(_i,_j,i)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
				}
			}
			if (toeplitz_mult_alg == 1)
			{
				for (int i = 1;i < K;i++)
					tmp[1][i]=s->Bt[idx(_i,_j,i)];
				Toeplitz_mult(s->Tm[2]+1, tmp[1]+1, tmp[0], K-1, 0);
				for (int i = 1;i < K;i++)
				{
					if (implicit_row_scaling==1)
						tmp[0][i-1]/=s->RS[idx(_i,_j,i)];
					r[idx(_i,_j,i)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
				}
			}
			if (toeplitz_mult_alg == 2)
			{
#if !defined(USE_MPI) || defined(USE_REPARTITIONING)
				aux_size[0]=aux_size2[0]=0;
				Tm_diags_mul_left(K, s, tmp[0], tmp_n[0], aux_size[0], r, 0,0,_i,_j);
				if (space_der == 1)
					Tm_diags_mul_right(K, s, tmp2[0], tmp_n2[0], aux_size2[0], r, 0, _i, _j);
#else
				Tm_diags_mul_left(K, s, coefs[_i*(_M+1)+_j], coefs_n[_i*(_M+1)+_j], coefs_size[_i*(_M+1)+_j], r, 0,0,_i,_j);
				if (space_der == 1)
					Tm_diags_mul_right(K, s, coefs2[_i*(_M+1)+_j], coefs_n2[_i*(_M+1)+_j], coefs_size2[_i*(_M+1)+_j], r, 0, _i, _j);
#endif			
			}
		}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 2 dir 2
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifdef USE_MPI			
		}
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=(mpi_size-1))
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
			int *cs_copy,*cs2_copy;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= _M ;_j++)	
			{
				scoefs_size+=coefs_size[_i*(_M+1)+_j];
				if (space_der==1)
					scoefs2_size+=coefs_size2[_i*(_M+1)+_j];
			}
			full_coefs=new double[scoefs_size];
			if (space_der==1)
				full_coefs2=new double[scoefs2_size];
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= _M ;_j++)	
			{
				memcpy(fcp,coefs[_i*(_M+1)+_j],coefs_size[_i*(_M+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(_M+1)+_j];
				if (space_der==1)
				{
					memcpy(fcp2,coefs2[_i*(_M+1)+_j],coefs_size2[_i*(_M+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(_M=1)+_j];
				}
			}
			cs_copy=new int[(_M+1)*(sN+1)];
			memcpy(cs_copy,coefs_size,((_M+1)*(sN+1))*sizeof(int));
		    _MPI_Isend(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[0]);
	    	_MPI_Isend(cs_copy,((_M+1)*(sN+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[1]);
	    	_MPI_Isend(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[2]);
			if (space_der==1)
			{
				cs2_copy=new int[(_M+1)*(sN+1)];
				memcpy(cs2_copy,coefs_size2,((_M+1)*(sN+1))*sizeof(int));
				_MPI_Isend(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[3]);
				_MPI_Isend(cs2_copy,((_M+1)*(sN+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[4]);
				_MPI_Isend(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[5]);
			}			
			full_coefs_s=full_coefs;
			full_coefs2_s=full_coefs2;
			coefs_size_s=cs_copy;
			coefs_size2_s=cs2_copy;
		}
#endif			
		}		
#endif
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	s->repartition(2,x,1);
	s->repartition(2,r,1);
#endif					
}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// Bt dir 1
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
if (_M>2)
{
	s->m3d_c=1;
#ifndef USE_MPI
#pragma omp parallel for
	for (int i = 0;i<N;i++)
	{
		s->Bt=s->sBt;
		s->Bt[i] =  s->Mcoefs_3d[1][1][i] * x[i];
		if ((i-(sN+2))>=0)
			s->Bt[i]+=s->Mcoefs_3d[1][0][i]*x[i-(sN+2)];
		if ((i+(sN+2))<N)
			s->Bt[i]+=s->Mcoefs_3d[1][2][i]*x[i+(sN+2)];
	}
#else
#ifdef USE_REPARTITIONING
	int id1_st=0,id1_end=K+1;
	int id2_st=0,id2_end=sN+1;
	if (K>2)
	{
		id1_st=mpi_rank*(K/mpi_size);
		id1_end=id1_st+(K/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=K+1;
	}
	else
	{
		id2_st=mpi_rank*(sN/mpi_size);
		id2_end=id2_st+(sN/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=sN+1;
	}				
#endif					
#pragma omp parallel for collapse(2) 
	for (int i = 0;i <= sN ;i++)
	for (int k = 0;k <= K ;k++)
#ifdef USE_REPARTITIONING
	if ((k>=id1_st)&&(k<id1_end))
	if ((i>=id2_st)&&(i<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (sN/mpi_size) bi=i/(sN/mpi_size);
		if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=bi-bj; // processing stage of the block by current process
		if (st<0) st+=mpi_size;
		st=(st+mpi_rank)%mpi_size;
		s->rb=st*(_M/mpi_size);
		s->re=s->rb+(_M/mpi_size);
		if (st==0) s->rb=1;
		if (st==(mpi_size-1)) s->re=_M;
		s->Bt=s->sBt;
#ifdef USE_REPARTITIONING
		s->rb=1;
		s->re=_M;
#endif
		for (int j=s->rb;j<s->re;j++)
		{
			s->Bt[idx(i,j,k)] =  s->Mcoefs_3d[1][1][idx(i,j,k)] * x[idx(i,j,k)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[1][0][idx(i,j,k)]*x[idx(i,j-1,k)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[1][2][idx(i,j,k)]*x[idx(i,j+1,k)];
		}
	}
	///////////////////////////////////////////////////////////
	// synch bt dir 1
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
	if (mpi_size!=1)
	{
		int *bb_ns=new int[mpi_size];
		MPI_Request *rrs=new MPI_Request[2*mpi_size];
		MPI_Status st;
		std::vector< std::vector<double> > to_send;
		for (int i=0;i<mpi_size;i++)
			to_send.push_back(std::vector<double>());
		for (int i = 0;i <= sN ;i++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(_M/mpi_size);
			s->re=s->rb+(_M/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=_M;
			double **bbs=synch_BT(s,_M+1,1,i,k,1,NULL,bb_ns);
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
			{
				to_send[ii].push_back(bb_ns[ii]);
				for (int j=0;j<bb_ns[ii];j++)
					to_send[ii].push_back(bbs[ii][j]);
				delete [] bbs[ii];
			}
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			bb_ns[i]=to_send[i].size();
			_MPI_Isend(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+0]);
			_MPI_Isend(&to_send[i][0],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+1]);		
		}
		double **bufs=new double*[mpi_size];
		double **bprs=new double *[mpi_size];
		for (int i=mpi_size-1;i>=0;i--)
		if (i!=mpi_rank)
		{
			_MPI_Recv(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
			bprs[i]=bufs[i]=new double[bb_ns[i]];
			_MPI_Recv(bufs[i],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
		}
		for (int i = 0;i <= sN ;i++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=i/(sN/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			s->rb=st*(_M/mpi_size);
			s->re=s->rb+(_M/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=_M;
			double **bbs=new double *[mpi_size];				
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
			{
				int n=(bprs[ii]++)[0];
				bbs[ii]=new double[n];
				for (int jj=0;jj<n;jj++)
					bbs[ii][jj]=(bprs[ii]++)[0];
			}
			synch_BT(s,_M+1,1,i,k,2,bbs,NULL);
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
				delete [] bbs[ii];
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			delete [] bufs[i];
			_MPI_Wait(&rrs[2*i+0],&st);
			_MPI_Wait(&rrs[2*i+1],&st);
		}
		delete [] bufs;
		delete [] bprs;
		delete [] bb_ns;
		delete [] rrs;
	}
#endif	
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// multiplication dir 1
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_MPI
#pragma omp parallel for collapse(2)				
	for (int _i = 1;_i < sN;_i++)
		for (int _k = 1;_k < K;_k++)
		{
#else
#ifndef USE_REPARTITIONING
		for (int i=0;i<(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2);i++)
			coefs_size[i]=coefs_size2[i]=0;
#endif
		for (int bb=0;bb<mpi_size;bb++)
		{
		MPI_Request rrs[6];
		MPI_Status sts[6];
		double *full_coefs_s, *full_coefs2_s;
		int *coefs_size_s,*coefs_size2_s;
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 1 dir 1
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=0)
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
		    _MPI_Recv(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[0]);
	    	_MPI_Recv(coefs_size,((sN+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[1]);
			full_coefs=new double[scoefs_size];
	    	_MPI_Recv(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[2]);
			if (space_der==1)
			{
				_MPI_Recv(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[3]);
				_MPI_Recv(coefs_size2,((sN+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[4]);
				full_coefs2=new double[scoefs2_size];
				_MPI_Recv(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&sts[5]);
			}			
			_MPI_Wait(&rrs[0],&sts[0]);
			_MPI_Wait(&rrs[1],&sts[1]);
			_MPI_Wait(&rrs[2],&sts[2]);
			if (space_der==1)
			{
				_MPI_Wait(&rrs[3],&sts[3]);
				_MPI_Wait(&rrs[4],&sts[4]);
				_MPI_Wait(&rrs[5],&sts[5]);
			}
			delete [] full_coefs_s;
			delete [] coefs_size_s;
			if (space_der==1)
			{
				delete [] full_coefs2_s;
				delete [] coefs_size2_s;
			}			
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				if (coefs_n[_i*(K+1)+_j]<coefs_size[_i*(K+1)+_j])
				{
					if (coefs[_i*(K+1)+_j]) delete [] coefs[_i*(K+1)+_j];
					coefs[_i*(K+1)+_j]=new double[coefs_size[_i*(K+1)+_j]];
					coefs_n[_i*(K+1)+_j]=coefs_size[_i*(K+1)+_j];
				}
				memcpy(coefs[_i*(K+1)+_j],fcp,coefs_size[_i*(K+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
				{
					if (coefs_n2[_i*(K+1)+_j]<coefs_size2[_i*(K+1)+_j])
					{
						if (coefs2[_i*(K+1)+_j]) delete [] coefs2[_i*(K+1)+_j];
						coefs2[_i*(K+1)+_j]=new double[coefs_size2[_i*(K+1)+_j]];
						coefs_n2[_i*(K+1)+_j]=coefs_size2[_i*(K+1)+_j];
					}
					memcpy(coefs2[_i*(K+1)+_j],fcp2,coefs_size2[_i*(K+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(K+1)+_j];
				}
			}
			delete [] full_coefs;
			if (space_der==1)
				delete [] full_coefs2;
		}
#endif			
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// process dir 1
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifdef USE_REPARTITIONING
		int id1_st=0,id1_end=K+1;
		int id2_st=0,id2_end=sN+1;
		if (K>2)
		{
			id1_st=mpi_rank*(K/mpi_size);
			id1_end=id1_st+(K/mpi_size);
			if (mpi_rank==0) id1_st=0;
			if (mpi_rank==(mpi_size-1)) id1_end=K+1;
		}
		else
		{
			id2_st=mpi_rank*(sN/mpi_size);
			id2_end=id2_st+(sN/mpi_size);
			if (mpi_rank==0) id2_st=0;
			if (mpi_rank==(mpi_size-1)) id2_end=sN+1;
		}				
#endif					
#pragma omp parallel for collapse(2) 
		for (int _i = 0;_i <= sN ;_i++)
		for (int _k = 0;_k <= K ;_k++)
#ifdef USE_REPARTITIONING
		if ((_k>=id1_st)&&(_k<id1_end))
		if ((_i>=id2_st)&&(_i<id2_end))
#endif
		{
			int bi=1,bj=1; // block of (i,j)
			if (sN/mpi_size) bi=_i/(sN/mpi_size);
			if (K/mpi_size) bj=_k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=bi-bj; // processing stage of the block by current process
			if (st<0) st+=mpi_size;
			st=(st+mpi_rank)%mpi_size;
			if (bb==st)
			{					
			s->rb=bb*(_M/mpi_size);
			s->re=s->rb+(_M/mpi_size);
			if (bb==0) s->rb=1;
			if (bb==(mpi_size-1)) s->re=_M;
#ifdef USE_REPARTITIONING
			s->rb=1;
			s->re=_M;
#endif
#endif
			int aux_size[3] = {0,0,0};
			int aux_size2[3] = {0,0,0};
			s->m3d_c=1;
			s->m3d_x=_i;
			s->m3d_y=_k;
			s->Bt=s->sBt;
			s->RS=s->sRS;
			s->U.set(s->b_U,s->m3d_x,s->m3d_y,s->m3d_c);
			s->C.set(s->b_C,s->m3d_x,s->m3d_y,s->m3d_c);
			s->T.set(s->b_T,s->m3d_x,s->m3d_y,s->m3d_c);
			vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
			if (toeplitz_mult_alg == 0)
			{
				for (int i = 1;i < _M;i++)
				{
					double v = 0.0;
					for (int j = 1;j < i;j++)
						v += s->Bt[idx(_i,j,_k)] * s->Tm[1][i - j + 1];
					if (space_der==1)
						for (int j = i+1;j < _M;j++)
							v -= s->Bt[idx(_i,j,_k)] * s->Tm[1][j-i + 1];
					if (implicit_row_scaling==1)
						v/=s->RS[idx(_i,i,_k)];
					r[idx(_i,i,_k)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
				}
			}
			if (toeplitz_mult_alg == 1)
			{
				for (int i = 1;i < _M;i++)
					tmp[1][i]=s->Bt[idx(_i,i,_k)];
				Toeplitz_mult(s->Tm[1]+1, tmp[1]+1, tmp[0], _M-1, 0);
				for (int i = 1;i < _M;i++)
				{
					if (implicit_row_scaling==1)
						tmp[0][i-1]/=s->RS[idx(_i,i,_k)];
					r[idx(_i,i,_k)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
				}
			}
			if (toeplitz_mult_alg == 2)
			{
#if !defined(USE_MPI) || defined(USE_REPARTITIONING)
				aux_size[1]=aux_size2[1]=0;
				Tm_diags_mul_left(_M, s, tmp[1], tmp_n[1], aux_size[1], r, 0, 1, _i, _k);
				if (space_der == 1)
					Tm_diags_mul_right(_M, s, tmp2[1], tmp_n2[1], aux_size2[1], r, 1, _i, _k);
#else
				Tm_diags_mul_left(_M, s, coefs[_i*(K+1)+_k], coefs_n[_i*(K+1)+_k], coefs_size[_i*(K+1)+_k], r, 0,1,_i,_k);
				if (space_der == 1)
					Tm_diags_mul_right(_M, s, coefs2[_i*(K+1)+_k], coefs_n2[_i*(K+1)+_k], coefs_size2[_i*(K+1)+_k], r, 1, _i, _k);
#endif				
			}
		}
#ifdef USE_MPI
		}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 2 dir 1
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=(mpi_size-1))
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
			int *cs_copy,*cs2_copy;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				scoefs_size+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
					scoefs2_size+=coefs_size2[_i*(K+1)+_j];
			}
			full_coefs=new double[scoefs_size];
			if (space_der==1)
				full_coefs2=new double[scoefs2_size];
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= sN ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				memcpy(fcp,coefs[_i*(K+1)+_j],coefs_size[_i*(K+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
				{
					memcpy(fcp2,coefs2[_i*(K+1)+_j],coefs_size2[_i*(K+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(K+1)+_j];
				}
			}
			cs_copy=new int[(K+1)*(sN+1)];
			memcpy(cs_copy,coefs_size,((K+1)*(sN+1))*sizeof(int));
		    _MPI_Isend(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[0]);
	    	_MPI_Isend(cs_copy,((sN+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[1]);
	    	_MPI_Isend(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[2]);
			if (space_der==1)
			{
				cs2_copy=new int[(K+1)*(sN+1)];
				memcpy(cs2_copy,coefs_size2,((K+1)*(sN+1))*sizeof(int));
				_MPI_Isend(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[3]);
				_MPI_Isend(cs2_copy,((sN+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[4]);
				_MPI_Isend(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&rrs[5]);
			}			
			full_coefs_s=full_coefs;
			full_coefs2_s=full_coefs2;
			coefs_size_s=cs_copy;
			coefs_size2_s=cs2_copy;
		}
#endif			
		}
#endif
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	s->repartition(1,x,1);
	s->repartition(1,r,1);
#endif					
}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// Bt dir 0
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
if (sN>2)
{
	s->m3d_c=0;
#ifndef USE_MPI
#pragma omp parallel for
	for (int i = 0;i<N;i++)
	{
		s->Bt=s->sBt;
		s->Bt[i] =  s->Mcoefs_3d[0][1][i] * x[i];
		if ((i-1)>=0)
			s->Bt[i]+=s->Mcoefs_3d[0][0][i]*x[i-1];
		if ((i+1)<N)
			s->Bt[i]+=s->Mcoefs_3d[0][2][i]*x[i+1];
	}
#else
#ifdef USE_REPARTITIONING
	int id1_st=0,id1_end=_M+1;
	int id2_st=0,id2_end=K+1;
	if (_M>2)
	{
		id1_st=mpi_rank*(_M/mpi_size);
		id1_end=id1_st+(_M/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=_M+1;
	}
	else
	{
		id2_st=mpi_rank*(K/mpi_size);
		id2_end=id2_st+(K/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=K+1;
	}				
#endif					
#pragma omp parallel for collapse(2) 
	for (int j = 0;j <= _M ;j++)
	for (int k = 0;k <= K ;k++)
#ifdef USE_REPARTITIONING
	if ((j>=id1_st)&&(j<id1_end))
	if ((k>=id2_st)&&(k<id2_end))
#endif
	{
		int bi=1,bj=1; // block of (i,j)
		if (_M/mpi_size) bi=j/(_M/mpi_size);
		if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=(bj+bi)%mpi_size; // processing stage of the block by current process
		st=(mpi_size+st-mpi_rank)%mpi_size;
		s->rb=st*(sN/mpi_size);
		s->re=s->rb+(sN/mpi_size);
		if (st==0) s->rb=1;
		if (st==(mpi_size-1)) s->re=sN;
		s->Bt=s->sBt;
#ifdef USE_REPARTITIONING
		s->rb=1;
		s->re=sN;
#endif
		for (int i=s->rb;i<s->re;i++)
		{
			s->Bt[idx(i,j,k)] =  s->Mcoefs_3d[0][1][idx(i,j,k)] * x[idx(i,j,k)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[0][0][idx(i,j,k)]*x[idx(i-1,j,k)];
			s->Bt[idx(i,j,k)]+=s->Mcoefs_3d[0][2][idx(i,j,k)]*x[idx(i+1,j,k)];
		}
	}
	///////////////////////////////////////////////////////////
	// synch bt dir 0
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
	if (mpi_size!=1)
	{
		int *bb_ns=new int[mpi_size];
		MPI_Request *rrs=new MPI_Request[2*mpi_size];
		MPI_Status st;
		std::vector< std::vector<double> > to_send;
		for (int i=0;i<mpi_size;i++)
			to_send.push_back(std::vector<double>());
		for (int j = 0;j <= _M ;j++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=j/(_M/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			s->rb=st*(sN/mpi_size);
			s->re=s->rb+(sN/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=sN;
			double **bbs=synch_BT(s,sN+1,2,j,k,1,NULL,bb_ns);
			for (int i=0;i<mpi_size;i++)
			if (i!=mpi_rank)
			{
				to_send[i].push_back(bb_ns[i]);
				for (int jj=0;jj<bb_ns[i];jj++)
					to_send[i].push_back(bbs[i][jj]);
				delete [] bbs[i];
			}
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			bb_ns[i]=to_send[i].size();
			_MPI_Isend(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+0]);
			_MPI_Isend(&to_send[i][0],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&rrs[2*i+1]);		
		}
		double **bufs=new double*[mpi_size];
		double **bprs=new double *[mpi_size];
		for (int i=mpi_size-1;i>=0;i--)
		if (i!=mpi_rank)
		{
			_MPI_Recv(&bb_ns[i],sizeof(int),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
			bprs[i]=bufs[i]=new double[bb_ns[i]];
			_MPI_Recv(bufs[i],bb_ns[i]*sizeof(double),MPI_BYTE,i,0,MPI_COMM_WORLD,&st);
		}
		for (int j = 0;j <= _M ;j++)
		for (int k = 0;k <= K ;k++)
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=j/(_M/mpi_size);
			if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			s->rb=st*(sN/mpi_size);
			s->re=s->rb+(sN/mpi_size);
			if (st==0) s->rb=1;
			if (st==(mpi_size-1)) s->re=sN;
			double **bbs=new double *[mpi_size];				
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
			{
				int n=(bprs[ii]++)[0];
				bbs[ii]=new double[n];
				for (int jj=0;jj<n;jj++)
					bbs[ii][jj]=(bprs[ii]++)[0];
			}
			synch_BT(s,sN+1,2,j,k,2,bbs,NULL);
			for (int ii=0;ii<mpi_size;ii++)
			if (ii!=mpi_rank)
				delete [] bbs[ii];
			delete [] bbs;
		}
		for (int i=0;i<mpi_size;i++)
		if (i!=mpi_rank)
		{
			delete [] bufs[i];
			_MPI_Wait(&rrs[2*i+0],&st);
			_MPI_Wait(&rrs[2*i+1],&st);
		}
		delete [] bufs;
		delete [] bprs;
		delete [] bb_ns;
		delete [] rrs;
	}
#endif
#endif
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// multiplication dir 0
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_MPI
#pragma omp parallel for collapse(2)				
	for (int _j = 1;_j < _M;_j++)
		for (int _k = 1;_k < K;_k++)
		{
#else
#ifndef USE_REPARTITIONING
		for (int i=0;i<(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2);i++)
			coefs_size[i]=coefs_size2[i]=0;
#endif			
		for (int bb=0;bb<mpi_size;bb++)
		{
		MPI_Request rrs[6];
		MPI_Status sts[6];
		double *full_coefs_s, *full_coefs2_s;
		int *coefs_size_s,*coefs_size2_s;
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 1 dir 0
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=0)
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
		    _MPI_Recv(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[0]);
	    	_MPI_Recv(coefs_size,((_M+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[1]);
			full_coefs=new double[scoefs_size];
	    	_MPI_Recv(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[2]);
			if (space_der==1)
			{
				_MPI_Recv(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[3]);
				_MPI_Recv(coefs_size2,((_M+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[4]);
				full_coefs2=new double[scoefs2_size];
				_MPI_Recv(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_rank+1)%mpi_size,0,MPI_COMM_WORLD,&sts[5]);
			}			
			_MPI_Wait(&rrs[0],&sts[0]);
			_MPI_Wait(&rrs[1],&sts[1]);
			_MPI_Wait(&rrs[2],&sts[2]);
			if (space_der==1)
			{
				_MPI_Wait(&rrs[3],&sts[3]);
				_MPI_Wait(&rrs[4],&sts[4]);
				_MPI_Wait(&rrs[5],&sts[5]);
			}
			delete [] full_coefs_s;
			delete [] coefs_size_s;
			if (space_der==1)
			{
				delete [] full_coefs2_s;
				delete [] coefs_size2_s;
			}			
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= _M ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				if (coefs_n[_i*(K+1)+_j]<coefs_size[_i*(K+1)+_j])
				{
					if (coefs[_i*(K+1)+_j]) delete [] coefs[_i*(K+1)+_j];
					coefs[_i*(K+1)+_j]=new double[coefs_size[_i*(K+1)+_j]];
					coefs_n[_i*(K+1)+_j]=coefs_size[_i*(K+1)+_j];
				}
				memcpy(coefs[_i*(K+1)+_j],fcp,coefs_size[_i*(K+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
				{
					if (coefs_n2[_i*(K+1)+_j]<coefs_size2[_i*(K+1)+_j])
					{
						if (coefs2[_i*(K+1)+_j]) delete [] coefs2[_i*(K+1)+_j];
						coefs2[_i*(K+1)+_j]=new double[coefs_size2[_i*(K+1)+_j]];
						coefs_n2[_i*(K+1)+_j]=coefs_size2[_i*(K+1)+_j];
					}
					memcpy(coefs2[_i*(K+1)+_j],fcp2,coefs_size2[_i*(K+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(K+1)+_j];
				}
			}
			delete [] full_coefs;
			if (space_der==1)
				delete [] full_coefs2;
		}
#endif			
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// process dir 0
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifdef USE_REPARTITIONING
		id1_st=0,id1_end=_M+1;
		id2_st=0,id2_end=K+1;
		if (_M>2)
		{
			id1_st=mpi_rank*(_M/mpi_size);
			id1_end=id1_st+(_M/mpi_size);
			if (mpi_rank==0) id1_st=0;
			if (mpi_rank==(mpi_size-1)) id1_end=_M+1;
		}
		else
		{
			id2_st=mpi_rank*(K/mpi_size);
			id2_end=id2_st+(K/mpi_size);
			if (mpi_rank==0) id2_st=0;
			if (mpi_rank==(mpi_size-1)) id2_end=K+1;
		}				
#endif					
#pragma omp parallel for collapse(2) 
		for (int _j = 0;_j <= _M ;_j++)
		for (int _k = 0;_k <= K ;_k++)
#ifdef USE_REPARTITIONING
		if ((_j>=id1_st)&&(_j<id1_end))
		if ((_k>=id2_st)&&(_k<id2_end))
#endif
		{
			int bi=1,bj=1; // block of (i,j)
			if (_M/mpi_size) bi=_j/(_M/mpi_size);
			if (K/mpi_size) bj=_k/(K/mpi_size); // block of (i,j)
			if (bi>=mpi_size) bi=mpi_size-1;
			if (bj>=mpi_size) bj=mpi_size-1;
			int st=(bj+bi)%mpi_size; // processing stage of the block by current process
			st=(mpi_size+st-mpi_rank)%mpi_size;
			if (bb==st)
			{			
				s->rb=bb*(sN/mpi_size);
				s->re=s->rb+(sN/mpi_size);
				if (bb==0) s->rb=1;
				if (bb==(mpi_size-1)) s->re=sN;
#ifdef USE_REPARTITIONING
				s->rb=1;
				s->re=sN;
#endif
#endif				
			int aux_size[3] = {0,0,0};
			int aux_size2[3] = {0,0,0};
			s->m3d_c=0;
			s->m3d_x=_j;
			s->m3d_y=_k;
			s->Bt=s->sBt;
			s->RS=s->sRS;
			s->U.set(s->b_U,s->m3d_x,s->m3d_y,s->m3d_c);
			s->C.set(s->b_C,s->m3d_x,s->m3d_y,s->m3d_c);
			s->T.set(s->b_T,s->m3d_x,s->m3d_y,s->m3d_c);
			vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
			if (toeplitz_mult_alg == 0)
			{
				for (int i = 1;i < sN;i++)
				{
					double v = 0.0;
					for (int j = 1;j < i;j++)
						v += s->Bt[idx(j,_j,_k)] * s->Tm[0][i - j + 1];
					if (space_der==1)
						for (int j = i+1;j < sN;j++)
							v -= s->Bt[idx(j,_j,_k)] * s->Tm[0][j-i + 1];
					if (implicit_row_scaling==1)
						v/=s->RS[idx(i,_j,_k)];
					r[idx(i,_j,_k)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
				}
			}
			if (toeplitz_mult_alg == 1)
			{
				for (int i = 1;i < sN;i++)
					tmp[1][i]=s->Bt[idx(i,_j,_k)];
				Toeplitz_mult(s->Tm[0]+1, tmp[1]+1, tmp[0], sN-1, 0);
				for (int i = 1;i < sN;i++)
				{
					if (implicit_row_scaling==1)
						tmp[0][i-1]/=s->RS[idx(i,_j,_k)];
					r[idx(i,_j,_k)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
				}
			}
			if (toeplitz_mult_alg == 2)
			{
#if !defined(USE_MPI) || defined(USE_REPARTITIONING)
				aux_size[2]=aux_size2[2]=0;
				Tm_diags_mul_left(sN, s, tmp[2], tmp_n[2], aux_size[2], r, 0, 2, _j, _k);
				if (space_der == 1)
					Tm_diags_mul_right(sN, s, tmp2[1], tmp_n2[2], aux_size2[2], r, 2, _j, _k);
#else
				Tm_diags_mul_left(K, s, coefs[_j*(K+1)+_k], coefs_n[_j*(K+1)+_k], coefs_size[_j*(K+1)+_k], r, 0,2,_j,_k);
				if (space_der == 1)
					Tm_diags_mul_right(K, s, coefs2[_j*(K+1)+_k], coefs_n2[_j*(K+1)+_k], coefs_size2[_j*(K+1)+_k], r, 2, _j, _k);
#endif				
			}
		}
#ifdef USE_MPI
		}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
		// synch 2 dir 0
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
#ifndef USE_REPARTITIONING
		if (mpi_size!=1)
		if (bb!=(mpi_size-1))
		{
			int scoefs_size=0,scoefs2_size=0;
			double *full_coefs, *full_coefs2;
			int *cs_copy,*cs2_copy;
			for (int _i = 0;_i <= _M ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				scoefs_size+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
					scoefs2_size+=coefs_size2[_i*(K+1)+_j];
			}
			full_coefs=new double[scoefs_size];
			if (space_der==1)
				full_coefs2=new double[scoefs2_size];
			double *fcp=full_coefs,*fcp2=full_coefs2;
			for (int _i = 0;_i <= _M ;_i++)
			for (int _j = 0;_j <= K ;_j++)	
			{
				memcpy(fcp,coefs[_i*(K+1)+_j],coefs_size[_i*(K+1)+_j]*sizeof(double));
				fcp+=coefs_size[_i*(K+1)+_j];
				if (space_der==1)
				{
					memcpy(fcp2,coefs2[_i*(K+1)+_j],coefs_size2[_i*(K+1)+_j]*sizeof(double));
					fcp2+=coefs_size2[_i*(K+1)+_j];
				}
			}
			cs_copy=new int[(K+1)*(_M+1)];
			memcpy(cs_copy,coefs_size,((K+1)*(_M+1))*sizeof(int));
		    _MPI_Isend(&scoefs_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[0]);
	    	_MPI_Isend(cs_copy,((_M+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[1]);
	    	_MPI_Isend(full_coefs,scoefs_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[2]);
			if (space_der==1)
			{
				cs2_copy=new int[(K+1)*(_M+1)];
				memcpy(cs2_copy,coefs_size2,((K+1)*(_M+1))*sizeof(int));
				_MPI_Isend(&scoefs2_size,sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[3]);
				_MPI_Isend(cs2_copy,((_M+1)*(K+1))*sizeof(int),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[4]);
				_MPI_Isend(full_coefs2,scoefs2_size*sizeof(double),MPI_BYTE,(mpi_size+mpi_rank-1)%mpi_size,0,MPI_COMM_WORLD,&rrs[5]);
			}			
			full_coefs_s=full_coefs;
			full_coefs2_s=full_coefs2;
			coefs_size_s=cs_copy;
			coefs_size2_s=cs2_copy;
		}
#endif			
		}
#endif
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	s->repartition(0,x,1);
	s->repartition(0,r,1);
#endif					
}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	// multiply T*C[1,2,3] for C and T equations - no MPI and 2D mode
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	if (s->equation != 0)
	{
#pragma omp parallel for
		for (int i = 0;i<N;i++)
		{
			s->Bt=s->sBt;
			s->Bt[i] =  s->Mcoefs2_3d[2][1][i] * x[i];
			if ((i-(sN+2)*(_M+2))>=0)
				s->Bt[i]+=s->Mcoefs2_3d[2][0][i]*x[i-(sN+2)*(_M+2)];
			if ((i+(sN+2)*(_M+2))<N)
				s->Bt[i]+=s->Mcoefs2_3d[2][2][i]*x[i+(sN+2)*(_M+2)];
		}
		s->m3d_c=2;
#pragma omp parallel for collapse(2)				
		for (int _i = 1;_i < sN;_i++)
			for (int _j = 1;_j < _M;_j++)
			{
				int aux_size[3] = {0,0,0};
				int aux_size2[3] = {0,0,0};
				s->m3d_x=_i;
				s->m3d_y=_j;
				s->Bt=s->sBt;
				vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
				if (toeplitz_mult_alg == 0)
				{
					for (int i = 1;i < K ;i++)
					{
						double v = 0.0;
						for (int j = 1;j <= i;j++)
							v += s->Bt[idx(_i,_j,j)] * s->Tm[2][i - j + 1];
						if (space_der==1)
							for (int j = i+1;j < K ;j++)
								v -= s->Bt[idx(_i,_j,j)] * s->Tm[2][j-i + 1];
						if (implicit_row_scaling==1)
							v/=s->RS[idx(_i,_j,i)];
						r[idx(_i,_j,i)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
					}
				}
				if (toeplitz_mult_alg == 1)
				{
					for (int i = 1;i < K ;i++)
						tmp[1][i]=s->Bt[idx(_i,_j,i)];
					Toeplitz_mult(s->Tm[2]+1, tmp[1]+1, tmp[0], K-1, 1);
					for (int i = 1;i < K ;i++)
					{
						if (implicit_row_scaling==1)
							tmp[0][i-1]/=s->RS[idx(_i,_j,i)];
						r[idx(_i,_j,i)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
					}
				}
				if (toeplitz_mult_alg == 2)
				{
					aux_size[0]=aux_size2[0]=0;
					Tm_diags_mul_left(K, s, tmp[0], tmp_n[0], aux_size[0], r, 1,0,_i,_j);
					if (space_der == 1)
						Tm_diags_mul_right(K, s, tmp2[0], tmp_n2[0], aux_size2[0], r, 0, _i, _j);
				}
			}
#pragma omp parallel for
		for (int i = 0;i<N;i++)
		{
			s->Bt=s->sBt;
			s->Bt[i] =  s->Mcoefs2_3d[1][1][i] * x[i];
			if ((i-(sN+2))>=0)
				s->Bt[i]+=s->Mcoefs2_3d[1][0][i]*x[i-(sN+2)];
			if ((i+(sN+2))<N)
				s->Bt[i]+=s->Mcoefs2_3d[1][2][i]*x[i+(sN+2)];
		}
		s->m3d_c=1;
#pragma omp parallel for collapse(2)				
		for (int _i = 1;_i < sN;_i++)
			for (int _k = 1;_k < K;_k++)
			{
				int aux_size[3] = {0,0,0};
				int aux_size2[3] = {0,0,0};
				s->m3d_x=_i;
				s->m3d_y=_k;
				s->Bt=s->sBt;
				vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
				if (toeplitz_mult_alg == 0)
				{
					for (int i = 1;i < _M ;i++)
					{
						double v = 0.0;
						for (int j = 1;j <= i;j++)
							v += s->Bt[idx(_i,j,_k)] * s->Tm[1][i - j + 1];
						if (space_der==1)
							for (int j = i+1;j < _M ;j++)
								v -= s->Bt[idx(_i,j,_k)] * s->Tm[1][j-i + 1];
						if (implicit_row_scaling==1)
							v/=s->RS[idx(_i,i,_k)];
						r[idx(_i,i,_k)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
					}
				}
				if (toeplitz_mult_alg == 1)
				{
					for (int i = 1;i < _M ;i++)
						tmp[1][i]=s->Bt[idx(_i,i,_k)];
					Toeplitz_mult(s->Tm[1]+1, tmp[1]+1, tmp[0], _M-1, 1);
					for (int i = 1;i < _M ;i++)
					{
						if (implicit_row_scaling==1)
							tmp[0][i-1]/=s->RS[idx(_i,i,_k)];
						r[idx(_i,i,_k)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
					}
				}
				if (toeplitz_mult_alg == 2)
				{
					aux_size[1]=aux_size2[1]=0;
					Tm_diags_mul_left(_M, s, tmp[1], tmp_n[1], aux_size[1], r, 1, 1, _i, _k);
					if (space_der == 1)
						Tm_diags_mul_right(_M, s, tmp2[1], tmp_n2[1], aux_size2[1], r, 1, _i, _k);
				}
			}
#pragma omp parallel for
		for (int i = 0;i<N;i++)
		{
			s->Bt=s->sBt;
			s->Bt[i] =  s->Mcoefs2_3d[0][1][i] * x[i];
			if ((i-1)>=0)
				s->Bt[i]+=s->Mcoefs2_3d[0][0][i]*x[i-1];
			if ((i+1)<N)
				s->Bt[i]+=s->Mcoefs2_3d[0][2][i]*x[i+1];
		}
		s->m3d_c=0;
#pragma omp parallel for collapse(2)				
		for (int _j = 1;_j < _M;_j++)
			for (int _k = 1;_k < K;_k++)
			{
				int aux_size[3] = {0,0,0};
				int aux_size2[3] = {0,0,0};
				s->m3d_x=_j;
				s->m3d_y=_k;
				s->Bt=s->sBt;
				vmul_T_3d_set_tmp(N,tmp,tmp_n,tmp2,tmp_n2);
				if (toeplitz_mult_alg == 0)
				{
					for (int i = 1;i < sN;i++)
					{
						double v = 0.0;
						for (int j = 1;j <= i;j++)
							v += s->Bt[idx(j,_j,_k)] * s->Tm[0][i - j + 1];
						if (space_der==1)
							for (int j = i+1;j < sN;j++)
								v -= s->Bt[idx(j,_j,_k)] * s->Tm[0][j-i + 1];
						if (implicit_row_scaling==1)
							v/=s->RS[idx(i,_j,_k)];
						r[idx(i,_j,_k)] += v*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);
					}
				}
				if (toeplitz_mult_alg == 1)
				{
					for (int i = 1;i < sN;i++)
						tmp[1][i]=s->Bt[idx(i,_j,_k)];
					Toeplitz_mult(s->Tm[0]+1, tmp[1]+1, tmp[0], sN-1, 1);
					for (int i = 1;i < sN;i++)
					{
						if (implicit_row_scaling==1)
							tmp[0][i-1]/=s->RS[idx(i,_j,_k)];
						r[idx(i,_j,_k)] += tmp[0][i-1]*((s->equation == 0) ? s->inv_dw_dh(i) : 1.0);;
					}
				}
				if (toeplitz_mult_alg == 2)
				{
					aux_size[2]=aux_size2[2]=0;
					Tm_diags_mul_left(sN, s, tmp[2], tmp_n[2], aux_size[2], r, 1, 2, _j, _k);
					if (space_der == 1)
						Tm_diags_mul_right(sN, s, tmp2[1], tmp_n2[2], aux_size2[2], r, 2, _j, _k);
				}
			}
	}
#if defined(USE_MPI) && !defined(USE_REPARTITIONING)
	for (int i=0;i<(sN+2)*(_M+2)+(_M+2)*(K+2)+(K+2)*(sN+2);i++)
	{
		if (coefs[i]) delete [] coefs[i];
		if (space_der==1)
			if (coefs2[i]) delete [] coefs2[i];
	}
	delete [] coefs;
	delete [] coefs_size;
	delete [] coefs_n;
	delete [] coefs2;
	delete [] coefs_size2;
	delete [] coefs_n2;
#endif	
#if defined(USE_MPI) && defined(USE_REPARTITIONING)
	// zeroize r in non-processed region
	id1_st=0,id1_end=sN+1;									
	id2_st=0,id2_end=_M+1;
	if (sN>2)
	{
		id1_st=mpi_rank*(sN/mpi_size);
		id1_end=id1_st+(sN/mpi_size);
		if (mpi_rank==0) id1_st=0;
		if (mpi_rank==(mpi_size-1)) id1_end=sN+1;
		for (int i = 0;i <= sN ;i++)
		if ((i<id1_st)||(i>=id1_end))
		for (int j = 0;j <= _M ;j++)	
			for (int k=0;k<=K;k++)
				r[idx(i,j,k)]=0;
	}
	else
	{
		id2_st=mpi_rank*(_M/mpi_size);
		id2_end=id2_st+(_M/mpi_size);
		if (mpi_rank==0) id2_st=0;
		if (mpi_rank==(mpi_size-1)) id2_end=_M+1;
		for (int i = 0;i <= sN ;i++)
		for (int j = 0;j <= _M ;j++)	
		if ((j<id2_st)||(j>=id2_end))
			for (int k=0;k<=K;k++)
				r[idx(i,j,k)]=0;
	}
#endif	
	s->m3d_x=s->m3d_y=0;
}
double solution_difference(H_solver *s1,H_solver *s2)
{
	double err=0.0;
	if (s1->mode==1)
	for (int i=0;i<=N;i++)
		{
			double ee=(s1->U[i]-s2->U[i])*(s1->U[i]-s2->U[i]);
			if (s1->s_is_c)
				ee+=(s1->C[i]-s2->C[i])*(s1->C[i]-s2->C[i]);
			if (s1->s_is_t)
				ee+=(s1->T[i]-s2->T[i])*(s1->T[i]-s2->T[i]);
			err+=ee;
		}
	if (s1->mode==3)
		for (int i = 1;i < sN;i++)
		for (int j = 1;j < _M;j++)
		for (int k = 1;k < K;k++)
			{
				double ee=(s1->b_U[idx(i, j, k)]-s2->b_U[idx(i, j, k)])*(s1->b_U[idx(i, j, k)]-s2->b_U[idx(i, j, k)]);
				if (s1->s_is_c)
					ee+=(s1->b_C[idx(i, j, k)]-s2->b_C[idx(i, j, k)])*(s1->b_C[idx(i, j, k)]-s2->b_C[idx(i, j, k)]);
				if (s1->s_is_t)
					ee+=(s1->b_T[idx(i, j, k)]-s2->b_T[idx(i, j, k)])*(s1->b_T[idx(i, j, k)]-s2->b_T[idx(i, j, k)]);
				err += ee;
			}
	return err;
}
double solution_norm(H_solver *s1)
{	
	double err=0.0;
	if (s1->mode==1)
	for (int i=0;i<=N;i++)
		{
			double ee=s1->U[i]*s1->U[i];
			if (s1->s_is_c)
				ee+=s1->C[i]*s1->C[i];
			if (s1->s_is_t)
				ee+=s1->T[i]*s1->T[i];
			err+=ee;
		}
	if (s1->mode==3)
		for (int i = 1;i < sN;i++)
		for (int j = 1;j < _M;j++)
		for (int k = 1;k < K;k++)
			{
				double ee=s1->b_U[idx(i, j, k)]*s1->b_U[idx(i, j, k)];
				if (s1->s_is_c)
					ee+=s1->b_C[idx(i, j, k)]*s1->b_C[idx(i, j, k)];
				if (s1->s_is_t)
					ee+=s1->b_T[idx(i, j, k)]*s1->b_T[idx(i, j, k)];
				err += ee;
			}
	return err;
}
// solves s1 with step tau, s2 - two steps with step tau/2, saves initial s1 into s3 and returns ||s1-s2||
double solve_T_half_step_and_check(H_solver *s1,H_solver *s2,H_solver *s3,double tau)
{
	double err;
	s3->copy(s1);
	// calc s1 - 1 step tau
	s1->second_step=0;
	s1->calc_step(tau);
	// calc s2 - 2 step tau/2
	s2->second_step=0;
	s2->calc_step(tau/2.0);
	s2->second_step=1;
	s2->calc_step(tau/2.0);
	s2->second_step=0;
	// calc err=||s1-s2||
	return solution_difference(s1,s2);
}
void collect_results(H_solver *s)
{
#ifdef USE_MPI
	MPI_Status stt;
#ifndef USE_REPARTITIONING
	if (sN>2)
	for (int j = 1;j < _M ;j++)
	for (int k = 1;k < K ;k++)
	{
		int bi=1,bj=1;
		if (_M/mpi_size) bi=j/(_M/mpi_size);	
		if (K/mpi_size) bj=k/(K/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=(bj+bi)%mpi_size; // processing stage of the block by current process
		if (mpi_rank==0)
			for (int i=1;i<mpi_size;i++)
			{
				int stg=(mpi_size+st-i)%mpi_size;
				int rb=stg*(sN/mpi_size);
				int re=rb+(sN/mpi_size);
				if (stg==(mpi_size-1)) re=sN;
	    		MPI_Recv(&s->b_U[idx(rb,j,k)],(re-rb)*sizeof(double),MPI_BYTE,i,j*mpi_mult+k,MPI_COMM_WORLD,&stt);
			}
		else
		{
			int stg=(mpi_size+st-mpi_rank)%mpi_size;
			int rb=stg*(sN/mpi_size);
			int re=rb+(sN/mpi_size);
			if (stg==(mpi_size-1)) re=sN;
    		MPI_Send(&s->b_U[idx(rb,j,k)],(re-rb)*sizeof(double),MPI_BYTE,0,j*mpi_mult+k,MPI_COMM_WORLD);
		}
	}
	if (sN<=2)
	for (int i = 1;i < sN ;i++)
	for (int j = 1;j < _M ;j++)	
	{
		int bi=1,bj=1;
		if (sN/mpi_size) bi=i/(sN/mpi_size);
		if (_M/mpi_size) bj=j/(_M/mpi_size); // block of (i,j)
		if (bi>=mpi_size) bi=mpi_size-1;
		if (bj>=mpi_size) bj=mpi_size-1;
		int st=bi-bj; // processing stage of the block by current process
		if (st<0) st+=mpi_size;
		if (mpi_rank==0)
			for (int ii=1;ii<mpi_size;ii++)
			{
				int stg=(mpi_size+st+ii)%mpi_size;
				int rb=stg*(K/mpi_size);
				int re=rb+(K/mpi_size);
				if (stg==(mpi_size-1)) re=K;
				double *arr=new double[re-rb];
	    			MPI_Recv(arr,(re-rb)*sizeof(double),MPI_BYTE,ii,i*mpi_mult+j,MPI_COMM_WORLD,&stt);
				for (int k=rb;k<re;k++)
				    s->b_U[idx(i,j,k)]=arr[k-rb];
	    			delete [] arr;
			}
		else
		{
			int stg=(mpi_size+st+mpi_rank)%mpi_size;
			int rb=stg*(K/mpi_size);
			int re=rb+(K/mpi_size);
			if (stg==(mpi_size-1)) re=K;
			double *arr=new double[re-rb];
			for (int k=rb;k<re;k++)
			    arr[k-rb]=s->b_U[idx(i,j,k)];
	    		MPI_Send(arr,(re-rb)*sizeof(double),MPI_BYTE,0,i*mpi_mult+j,MPI_COMM_WORLD);
	    		delete [] arr;
		}
	}
#else
	// paritioned by N
	if (((K>2)&&(_M>2)&&(sN>2))||((K<=2)||(_M<=2)))
	{
		int st,end;
		st=mpi_rank*(sN/mpi_size);
		end=st+(sN/mpi_size);
		if (mpi_rank==0) st=1;
		if (mpi_rank==(mpi_size-1)) end=sN;
		for (int j = 1;j < _M ;j++)	
			for (int k = 1;k < K ;k++)
				if (mpi_rank==0)				
					for (int r=1;r<mpi_size;r++)
					{
						st=r*(sN/mpi_size);
						end=st+(sN/mpi_size);
						if (r==0) st=1;
						if (r==(mpi_size-1)) end=sN;
						double *arr=new double[end-st];
						MPI_Recv(arr,(end-st)*sizeof(double),MPI_BYTE,r,j*mpi_mult+k,MPI_COMM_WORLD,&stt);
						for (int i=st;i<end;i++)
						    s->b_U[idx(i,j,k)]=arr[i-st];
						delete [] arr;
					}
				else
				{
    					double *arr=new double[end-st];
					for (int i=st;i<end;i++)
					    arr[i-st]=s->b_U[idx(i,j,k)];
					MPI_Send(arr,(end-st)*sizeof(double),MPI_BYTE,0,j*mpi_mult+k,MPI_COMM_WORLD);
					delete [] arr;
				}
	}
	else
	{
	// if N<=2 - by M
		int st,end;
		st=mpi_rank*(_M/mpi_size);
		end=st+(_M/mpi_size);
		if (mpi_rank==0) st=1;
		if (mpi_rank==(mpi_size-1)) end=_M;
		for (int i = 1;i < sN ;i++)
			for (int k = 1;k < K ;k++)	
				if (mpi_rank==0)
					for (int r=1;r<mpi_size;r++)
					{
						st=r*(_M/mpi_size);
						end=st+(_M/mpi_size);
						if (r==0) st=1;
						if (r==(mpi_size-1)) end=_M;
    						double *arr=new double[end-st];
						MPI_Recv(arr,(end-st)*sizeof(double),MPI_BYTE,r,i*mpi_mult+k,MPI_COMM_WORLD,&stt);
						for (int j=st;j<end;j++)
						    s->b_U[idx(i,j,k)]=arr[j-st];
						delete [] arr;
					}
				else
				{
    					double *arr=new double[end-st];
					for (int j=st;j<end;j++)
					    arr[j-st]=s->b_U[idx(i,j,k)];
					MPI_Send(arr,(end-st)*sizeof(double),MPI_BYTE,0,i*mpi_mult+k,MPI_COMM_WORLD);
					delete [] arr;
				}
	}
#endif	
#endif	
}
void solve(double t,double save_tau,double out_tau,int analytic_test,int irr,int bc,double tau_m,int m,double eps,double a=1.0,double b=1.0)
{
	int nsave = 1;
	int nout=1;
	int d,oldd=-1;
	double tau=tau_m;
	int first=1;
	unsigned int best_time,cur_time=0,t0;
	int best_sum_alg=-1;
	double norm0;
	FILE *fi1,*fi2,*fi3;
	fi1 = fopen("log.txt", "at");
	log_file=fi1;
	fi2 = fopen("results.txt", "at");
	if (m==3)
    	    fi3 = fopen("results3d.txt", "wb");
	printf("(%d,%d,%d) mode %d bc %d sd %d eps %g(%g) a %g b %g vT %d vZ %d(%g) vXY %d(%g) vniter %d impl %d(%d)[%d][%d] FK %d(%g) split3d %d sum %d(%g) gpgpu %d(%d,%d)\n", N,_M,K,m, bc, space_der, eps,global_eps2, a, b, varT, varZ, Zvar_coef, varXY, XYvar_coef,varZ_niter, implicit,toeplitz_mult_alg,implicit3d,tma2_all_diags,func_in_kernel,func_power,rp_split_3d,sum_alg,sum_param,use_ocl,ocl_vector,oclBS);
	fprintf(fi1,"(%d,%d,%d) mode %d bc %d sd %d eps %g(%g) a %g b %g vT %d vZ %d(%g) vXY %d(%g) vniter %d impl %d(%d)[%d][%d] FK %d(%g) split3d %d sum %d(%g)\n", N,_M,K,m, bc, space_der, eps,global_eps2, a, b, varT, varZ, Zvar_coef, varXY, XYvar_coef, varZ_niter, implicit,toeplitz_mult_alg,implicit3d,tma2_all_diags,func_in_kernel,func_power,rp_split_3d,sum_alg,sum_param);
	H_solver *s0=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
	H_solver *s2;	
	H_solver *s3;	
	s0->vart_main=1;
	if (varT)
	{
		s2=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);	
		s3=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);	
	}
	if (sum_alg==3) // set maximal number of terms series for automatic algorithm selection
		sum_param=75;
#ifdef _OPENMP
	if (varT)
	{ 
		printf("varT do not work with openmp\n"); 
		varT=0;
	}
#endif
	while (s0->Time[s0->Time.size()-1]<t)
	{
		struct timespec ti1,ti2;
		clock_gettime(CLOCK_REALTIME,&ti1);
		long long t1 = GetTickCount(),t2;
		double err;
		// do initial check
		if (s0->tau.size()!=0)
			tau=s0->tau[s0->tau.size()-1];
		// variable time step
		if (varT)
		{
			err = solve_T_half_step_and_check(s0, s2, s3, tau);
			// if err>eps - decrease step until err>eps, otherwise - increase step
			if (err > eps)
			{
				int init_tau=tau;
				while (err > eps)
				{
					//restore s1,s2
					s0->copy(s3);
					s2->copy(s3);
					// decrease tau
					tau /= 2.0;
					// do check
					err = solve_T_half_step_and_check(s0, s2, s3, tau);
					if (debug_level==1)
						if (log_file) fprintf(log_file,"tau %g err %g\n",tau,err);
					if (tau < 1e-10)
					{
					    tau=init_tau;
					    s0->calc_step(tau);
					    break;
					}
				}
			}
			else
			{
				if (tau < tau_m)
					while (err <= eps)
					{
						//restore s1,s2
						s0->copy(s3);
						s2->copy(s3);
						// increase tau
						tau *= 2.0;
						// do check
						err = solve_T_half_step_and_check(s0, s2, s3, tau);
						if (debug_level == 1)
							if (log_file) fprintf(log_file,"tau %g err %g\n",tau,err);
						if (tau > tau_m)
							break;
					}
			}
		}		
		else
		{
			// authomatic algorithm selection
			if (sum_alg==3) 
			{
				int best_fd_scheme=0; // 0 - implicit3d, 1 - explicit split0, 2 - explicit split1, 3 - implicit split0, 4 - implicit split1
				int check_implicit=0; // 0 - don't check, 1 - check split0, 2 - check split1
				int best_implicit=0; // 0 - full mult, 1 - toeplitz, 2 - series
				// on the first step find the best space derivative approximation algorithms and finite-difference schemes
				if (first==1)
				{
					double diff00,diff01,diff011,diff001;
					unsigned int time0,time00,time01,time011,time001,t0;
					delete s0;
					printf("finding best space approximation algorithm / factors - %g %g\n",difference_factor,tma22_factor);
					H_solver *s1;
					H_solver *s00;
					global_eps2=1e-5;
					// s0 - most accurate method - implicit f-d scheme
					implicit = 2;
					implicit3d=1;
					toeplitz_mult_alg = 2;
					tma2_all_diags=1;
					rp_split_3d=0;
					s00=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					t0=GetTickCount();
					s00->calc_step(tau);
					time0=GetTickCount()-t0;
					norm0=solution_norm(s00);
					// splitting scheme 1 explicit
					implicit = 0;
					implicit3d=0;
					rp_split_3d=0;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time00=GetTickCount()-t0;
					diff00=solution_difference(s00,s1);
					// splitting scheme 2 explicit
					delete s1;
					implicit = 0;
					implicit3d=0;
					rp_split_3d=1;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time01=GetTickCount()-t0;
					diff01=solution_difference(s00,s1);
					// splitting scheme 1 implicit
					delete s1;
					implicit = 2;
					implicit3d=0;
					toeplitz_mult_alg = 2;
					tma2_all_diags=1;
					rp_split_3d=0;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time001=GetTickCount()-t0;
					diff001=solution_difference(s00,s1);
					// splitting scheme 2 implicit
					delete s1;
					implicit = 2;
					implicit3d=0;
					toeplitz_mult_alg = 2;
					tma2_all_diags=1;
					rp_split_3d=1;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time011=GetTickCount()-t0;
					diff011=solution_difference(s00,s1);
					delete s1;
					// select the best f.-d. scheme
					printf("impl3d time %d norm %g / expl0 %d diff %g / expl1 %d %g / impl0 %d %g / impl1 %d %g\n",time0,norm0,time00,diff00,time01,diff01,time001,diff001,time011,diff011);
					best_time=time0;
					if ((diff00<difference_factor*norm0)&&(time00<best_time))
					{
						best_fd_scheme=1;
						best_time=time00;
					}
					if ((diff01<difference_factor*norm0)&&(time01<best_time))
					{
						best_fd_scheme=2;
						best_time=time01;
					}
					if (diff001<difference_factor*norm0)
					{
						check_implicit=1;
						if (time001<best_time)
						{
							best_fd_scheme=3;
							best_time=time001;
						}
					}
					if (diff011<difference_factor*norm0)
					{
						if ((check_implicit==0)||((check_implicit==1)&&(time011<time001)))
							check_implicit=2;
						if (time011<best_time)
						{
							best_fd_scheme=4;
							best_time=time011;
						}
					}
					if (best_time==0) best_time=1;
					printf("best scheme %d best time %d check_implicit %d\n",best_fd_scheme,best_time,check_implicit);
					if (check_implicit)
					{
						double diffT,diffS,ref_diff;
						int timeT,timeS=0,otimeS;
						double sge=global_eps2;
						double oge=sge; // previous
						double factor=10.0;
						if (check_implicit==1) ref_diff=diff001; else ref_diff=diff011;
						printf("checking implicit %d %g\n",check_implicit,ref_diff);
						// implicit Toeplitz
						implicit = 2;
						implicit3d=0;
						toeplitz_mult_alg = 1;
						tma2_all_diags=0;
						rp_split_3d=check_implicit-1;
						s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
						t0=GetTickCount();
						s1->calc_step(tau);
						timeT=GetTickCount()-t0;
						diffT=solution_difference(s00,s1);
						delete s1;
						printf("Toeplitz %d %g\n",timeT,diffT);
						// implicit series - find maximal global_eps2 (starting from the current value) that do not change solution difference mode than on a given factor
						while (1)
						{
							implicit = 2;
							implicit3d=0;
							toeplitz_mult_alg = 2;
							tma2_all_diags=0;
							rp_split_3d=check_implicit-1;
							s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
							t0=GetTickCount();
							s1->calc_step(tau);
							timeS=GetTickCount()-t0;
							diffS=solution_difference(s00,s1);
							delete s1;
							printf("series(%g) %d %g old (%g %d) factor %g\n",global_eps2,timeS,diffS,oge,otimeS,factor);
							// make eps lower if difference do not change
							if ((diffS<tma22_factor*ref_diff)&&(diffS>(2.0-tma22_factor)*ref_diff))
							{
									oge=(global_eps2*=factor);
									otimeS=timeS;
							}
							else
							{
								oge=global_eps2/factor;
								global_eps2=0.5*(oge+global_eps2);
								factor=(1.0+factor)/2.0;
								if (factor<tma22_minstep)
								{
									global_eps2=oge;
									timeS=otimeS;
									printf("final - eps %g time %d\n",oge,otimeS);
									break;
								}
							}
							if (oge>=1)
							{
								printf("final - eps %g time %d\n",oge,otimeS);
								break;
							}							    
						}
						if (timeT<best_time)
						{
							best_fd_scheme=2+check_implicit;
							best_implicit=1;
							best_time=timeT;
							printf("best scheme changed to implicit Toeplitz %d %d\n",best_fd_scheme,best_time);
						}
						if (timeS<best_time)
						{
							best_fd_scheme=2+check_implicit;
							best_implicit=2;
							best_time=timeS;							
							printf("best scheme changed to implicit series(%g) %d %d\n",global_eps2,best_fd_scheme,best_time);
						}
						else
							global_eps2=sge;
						if (best_time==0) best_time=1;
					}
					// clean up
					implicit = 2;
					implicit3d=1;
					toeplitz_mult_alg = 2;
					tma2_all_diags=1;
					rp_split_3d=0;
					delete s00;
					if (best_fd_scheme==0)
					{
						implicit = 0;
						implicit3d=1;
						rp_split_3d=0;
					}
					if (best_fd_scheme==1)
					{
						implicit = 0;
						implicit3d=0;
						rp_split_3d=0;
					}
					if (best_fd_scheme==2)
					{
						implicit = 0;
						implicit3d=0;
						rp_split_3d=1;
					}
					if (best_fd_scheme==3)
					{
						implicit = 2;
						toeplitz_mult_alg = 2;
						implicit3d=0;
						rp_split_3d=0;
					}
					if (best_fd_scheme==4)
					{
						implicit = 2;
						toeplitz_mult_alg = 2;
						implicit3d=0;
						rp_split_3d=1;
					}
					if (best_implicit==0)
						tma2_all_diags=1;
					if (best_implicit==1)
					{
						toeplitz_mult_alg = 1;
						tma2_all_diags=0;
					}
					if (best_implicit==2)
						tma2_all_diags=0;
					s0=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					first=0;
				}
				// calc step
				if (best_sum_alg!=-1)
					sum_alg=best_sum_alg;
				int tt=0;
				if (((best_sum_alg==-1)&&(cur_time>time_min)&&(((double)cur_time/best_time)>timegr_factor))&&(s0->tstep>timegr_min))
				{
					s2=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);	
					s2->copy(s0);
					tt=1;
				}
				t0=GetTickCount();
				s0->calc_step(tau);
				cur_time=GetTickCount()-t0;
				sum_alg=3;
				// find the best time derivative approximation scheme
				if (tt==1)
				{
					double time_forecast=0.5*((cur_time-best_time)/(double)s0->tstep)*(t*t/(s0->tau[0]*s0->tau[0])-s0->tstep*s0->tstep)+best_time*((t/s0->tau[0])-s0->tstep);
					double diff10,diff11,diff12,p1,p2,p3;
					double time10,time11,time12;
					double opt_time_1,opt_param_1;
					double opt_time_2,opt_param_2;
					H_solver *s1;
					sum_alg=1;
					dont_clear_sum_alg1=1;
					printf("finding the best time derivative approximation scheme %d %d %d time_forecast %g (norm %g)\n",(int)(t/s0->tau[0]),s0->tstep,cur_time,time_forecast,norm0);
					// calculate fixed summing for three kb values
					p1=sum_param=s0->kb(s0->alpha, s0->tstep, s0->tstep+1, s0->tstep+1, &s0->Time[0], (void *)&s0->Time);
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time10=GetTickCount()-t0;
					if (time10==0) time10=1;
					diff10=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep); // linearly extrapolate to the ending point
					printf("alg1(%g) - %g %g\n",p1,time10,diff10);
					p2=sum_param=s0->kb(s0->alpha, s0->tstep*2/3.0, (s0->tstep*2/3.0)+1, s0->tstep+1, &s0->Time[0], (void *)&s0->Time);
					delete s1;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time11=GetTickCount()-t0;
					if (time11==0) time11=1;
					diff11=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep); // linearly extrapolate to the ending point
					printf("alg1(%g) - %g %g\n",p2,time11,diff11);
					p3=sum_param=s0->kb(s0->alpha, s0->tstep*1/3, (s0->tstep*1/3)+1, s0->tstep+1, &s0->Time[0], (void *)&s0->Time);
					delete s1;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time12=GetTickCount()-t0;
					if (time12==0) time12=1;
					diff12=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep); // linearly extrapolate to the ending point
					printf("alg1(%g) - %g %g\n",p3,time12,diff12);
					// calculate a*exp(b/x) approximation 
					double ax1,bx1,ax2,bx2,ax3,bx3;
					bx1=log(diff12/diff11)/((1.0/p3)-(1.0/p2));
					ax1=diff12/exp((1.0/p3)*bx1);
					bx2=log(diff12/diff10)/((1.0/p3)-(1.0/p1));
					ax2=diff12/exp((1.0/p3)*bx2);
					bx3=log(diff11/diff10)/((1.0/p2)-(1.0/p1));
					ax3=diff11/exp((1.0/p2)*bx3);
					printf("d((%g %g),(%g %g),(%g %g))\n",ax1,bx1,ax2,bx2,ax3,bx3);
					// calculate ax^b approximation of time
					double at1,bt1,at2,bt2,at3,bt3;
					bt1=log(time12/time11)/log(p3/p2);
					at1=time12/pow(p3,bt1);
					bt2=log(time12/time10)/log(p3/p1);
					at2=time12/pow(p3,bt2);
					bt3=log(time11/time10)/log(p2/p1);
					at3=time11/pow(p2,bt3);
					// calc x for diff0+ax^b=diff_factor*norm0 and t forecast for this x
					if ((diff10-difference_factor*norm0)<0.0)
						opt_param_1=p1;
					else
					{					    
						opt_param_1=bx1/log(difference_factor*norm0/ax1);
						opt_param_1+=bx2/log(difference_factor*norm0/ax2);
						opt_param_1+=bx3/log(difference_factor*norm0/ax3);
						opt_param_1/=3.0;
					}
					if (opt_param_1<0.0) opt_param_1=0.0;
					opt_time_1=((at1*pow(opt_param_1,bt1))+(at2*pow(opt_param_1,bt2))+(at3*pow(opt_param_1,bt3)))*((t/s0->tau[0])-s0->tstep)/3.0;
					printf("t((%g %g),(%g %g),(%g %g)) -> %g %g\n",at1,bt1,at2,bt2,at3,bt3,opt_param_1,opt_time_1);
					// check
					sum_param=opt_param_1;
					delete s1;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					t0=GetTickCount();
					s1->calc_step(tau);
					time12=GetTickCount()-t0;
					if (time12==0) time12=1;
					diff12=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep); // linearly extrapolate to the ending point
					printf("alg1 opt (%g) - %g %g\n",sum_param,time12,diff12);
					if (diff12==0.0) opt_time_1=time_forecast; // don't use fixed memory if eps<min kb
					
					dont_clear_sum_alg1=0;
					///////////////////////////////////
					// calculate series summing for 25,50,75
					delete s1;
					sum_alg=2;
					p1=sum_param=25;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					s1->time_summing_alg0_to_alg2();
					t0=GetTickCount();
					s1->calc_step(tau);
					time10=GetTickCount()-t0;
					diff10=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep);
					if (time10==0) time10=1;
					printf("alg2(%g) - %g %g\n",p1,time10,diff10);
					delete s1;
					p2=sum_param=50;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					s1->time_summing_alg0_to_alg2();
					t0=GetTickCount();
					s1->calc_step(tau);
					time11=GetTickCount()-t0;
					diff11=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep);
					if (time11==0) time11=1;
					printf("alg2(%g) - %g %g\n",p2,time11,diff11);
					delete s1;
					p3=sum_param=75;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					s1->time_summing_alg0_to_alg2();
					t0=GetTickCount();
					s1->calc_step(tau);
					time12=GetTickCount()-t0;
					diff12=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep);
					if (time12==0) time12=1;
					delete s1;
					printf("alg2(%g) - %g %g\n",p3,time12,diff12);
					// calculate ax^b approximation of difference
					bx1=log(diff12/diff11)/log(p3/p2);
					ax1=diff12/pow(p3,bx1);
					bx2=log(diff12/diff10)/log(p3/p1);
					ax2=diff12/pow(p3,bx2);
					bx3=log(diff11/diff10)/log(p2/p1);
					ax3=diff11/pow(p2,bx3);
					printf("d((%g %g),(%g %g),(%g %g))\n",ax1,bx1,ax2,bx2,ax3,bx3);
					// calculate ax^b approximation of time
					bt1=log(time12/time11)/log(p3/p2);
					at1=time12/pow(p3,bt1);
					bt2=log(time12/time10)/log(p3/p1);
					at2=time12/pow(p3,bt2);
					bt3=log(time11/time10)/log(p2/p1);
					at3=time11/pow(p2,bt3);
					printf("t((%g %g),(%g %g),(%g %g))\n",at1,bt1,at2,bt2,at3,bt3);
					// calc x for ax^b=diff_factor*norm0 and t for this x
					opt_param_2=(int)((exp(log((difference_factor*norm0)/ax1)/bx1)+exp(log((difference_factor*norm0)/ax2)/bx2)+exp(log((difference_factor*norm0)/ax3)/bx3))/3.0);
					if (opt_param_2<1) opt_param_2=1;
					if (opt_param_2>75) opt_param_2=75;
					opt_time_2=((at1*pow(opt_param_2,bt1))+(at2*pow(opt_param_2,bt2))+(at3*pow(opt_param_2,bt3)))*((t/s0->tau[0])-s0->tstep)/3.0;
					printf("opt - %g %g\n",opt_param_2,opt_time_2);
					// check
					sum_param=opt_param_2;
					s1=new H_solver(analytic_test,1,5,irr,bc,a,b,0,0,0,m,fi1,2);
					s1->copy(s2);
					s1->time_summing_alg0_to_alg2();
					t0=GetTickCount();
					s1->calc_step(tau);
					time12=GetTickCount()-t0;
					diff12=solution_difference(s0,s1)*((t/s0->tau[0])-s0->tstep);
					if (time12==0) time12=1;
					delete s1;
					printf("alg2 opt (%g) - %g %g\n",sum_param,time12,diff12);
					sum_alg=3;
					delete s2;
					// select the best algorithm
					if (opt_time_1<time_forecast)
					{
						best_sum_alg=1;
						sum_param=opt_param_1;
						time_forecast=opt_time_1;
						printf("best - alg1(%g)\n",sum_param);
					}
					if (opt_time_2<time_forecast)
					{
						best_sum_alg=2;
						sum_param=opt_param_2;
						sum_alg=2;
						s0->time_summing_alg0_to_alg2();
						sum_alg=3;
						printf("best - alg2(%g)\n",sum_param);
					}
					if (best_sum_alg==-1)
						sum_alg=best_sum_alg=3;
						
				}
			}
			else
				s0->calc_step(tau);	// normal solution
		}
		t2 = GetTickCount();
		clock_gettime(CLOCK_REALTIME,&ti2);
		double tt=s0->Time[s0->Time.size()-1];
		d=tt/(24*3600);
		if (full_test==0)
			if (d!=oldd) printf("%d\n",d);
		double vmu,rss;
		process_mem_usage(vmu,rss);
		int ns=0;
		for (int i=0;i<s0->oldH.size();i++)
			if (s0->oldH[i]!=NULL)
				ns++;
#ifdef USE_MPI
	if ((analytic_test)|| (tt > nout*out_tau)||((tt > nsave*save_tau)||(s0->Time[s0->Time.size()-1]>=t)))
		collect_results(s0);
	if (mpi_rank==0) {
#endif			
		if (analytic_test)
		{
			fprintf(fi1,"d %2.2d h %2.2g %g time %lld %d vmu %g rs %g ns %d a_err",d, (tt-d*24*3600)/3600.0,tau,t2-t1,(int)(ti2.tv_nsec-ti1.tv_nsec),vmu-st_vmu,rss-st_rss,ns);
			s0->analytic_compare(fi1);
			fprintf (fi1,"\n");
			fflush(fi1);
		}
		// out errors
		if (tt > nout*out_tau)
		{
			printf("d %2.2d h %2.2g tau %g time %lld vmu %g rs %g ns %d ",d, (tt-d*24*3600)/3600.0,tau,t2-t1,vmu-st_vmu,rss-st_rss,ns);
			printf("rlw s0 %g %g norm %g",s0->avg_root_layer_wetness(),s0->total_water_content(),solution_norm(s0));
			if (analytic_test) s0->analytic_compare(fi1);
			printf("\n");
			fprintf(fi1,"d %2.2d h %2.2g tau %g ",d, (tt-d*24*3600)/3600.0,tau);
			fprintf(fi1,"rlw s0 %g %g norm %g",s0->avg_root_layer_wetness(),s0->total_water_content(),solution_norm(s0));
			fprintf(fi1,"\n");
			fflush(fi1);
			fflush(stdout);
			nout++;
		}
		// save result
		if ((tt > nsave*save_tau)||(s0->Time[s0->Time.size()-1]>=t))
		{
			for (int i = 0;i < N + 1;i++)
			{
			    if (m==1)
			    {
				fprintf(fi2, "%g %g %g %g %g %g ", tt,s0->Z[i],s0->U[i],(s0->b_C?s0->C[i]:0.0),(s0->b_T?s0->T[i]:0.0),s0->wetness(i));
				if (analytic_test)
					fprintf(fi2,"%g",s0->testF(i,s0->tstep+1));
				if (analytic_test==3)
					fprintf(fi2," %g %g",s0->testF(i,s0->tstep+1,1),s0->testF(i,s0->tstep+1,2));
			    }
			    if (m==3)
			    {
				fprintf(fi2, "%g %g %g %g %g %g %g %g ", tt,s0->Z[i],s0->X[_M/2],s0->Y[K/2],s0->b_U[idx(i,_M/2,K/2)],(s0->b_C?s0->b_C[idx(i,_M/2,K/2)]:0.0),(s0->b_T?s0->b_T[idx(i,_M/2,K/2)]:0.0),s0->wetness(i));
				if (analytic_test)
					fprintf(fi2,"%g",s0->testF3d(i,_M/2,K/2,s0->tstep+1));
			    }
			    fprintf(fi2,"\n");				
			}
			if (m==3)
			{
			    
			    fwrite(s0->b_U,(sN+2)*(_M+2)*(K+2),sizeof(double),fi3);
				if (s0->b_C)
			    	fwrite(s0->b_C,(sN+2)*(_M+2)*(K+2),sizeof(double),fi3);
				if (s0->b_T)
			    	fwrite(s0->b_T,(sN+2)*(_M+2)*(K+2),sizeof(double),fi3);
			}
			fflush(fi2);
			nsave++;
		}
#ifdef USE_MPI
	}
#endif			
		oldd=d;
	}
#ifdef USE_MPI
	collect_results(s0);
	if (mpi_rank==0)
#endif			
	if (analytic_test)
	{
		printf("%d %d %d %g last errs: ",N,_M,K,tau_m);
		s0->analytic_compare(stdout);
		printf("\n");
		fflush(stdout);
	}
	delete s0;
}
/// main/////////////////
int main(int argc,char **argv)
{
	double Tm = 24 * 5 * 3600.0 + 1000.0;
	double Sm = 2 * 3600.0;
	double Om = 2.0*3600.0;
	int analytic=0;
	int irr=0,bc=0,m=1;
	double tau_m=1000.0/20.0;
	double eps=1e-5;
	double aa=1.0,bb=1.0;
	process_mem_usage(st_vmu,st_rss);
	if (argc==1)
	{
		printf("Tm : end time\n");
		printf("Sm : save time\n");
		printf("Om : log output time\n");
		printf("BS : grid block size\n");
		printf("NB : number of grid blocks\n");
		printf("A : analytic test\n");
		printf("I : 1 - irrigation disabled\n");
		printf("B : bottom boundary condition - 0 -dU/dn=0, 1 - U=H0\n");
		printf("Tau : tau multiplier\n");
		printf("mode : 1D/3D\n");
		printf("_M : x blocks for 3D\n");
		printf("K : y blocks for 3D\n");
		printf("SD : space derivative: 0 - one-sided, 1 - two-sided\n");
		printf("EPS : varT eps\n");
		printf("var{T,Z,XY} : variable steps on/off\n");
		printf("FT: testing on all analytic tests\n");
		printf("aa: time fractional derivative coef\n");
		printf("bb: space fractional derivative coef\n");
		printf("XYcoef,Zcoef: XY and Z variable steps coef\n");	
		printf("Zniter: number of iteration for variable steps\n");
		printf("avg: 0 or 1-order averaging\n");
		printf("impl: 0 - explicit, 1 - implicit with full matrix, 2 - implicit with decomp. of integral matrices \n");
		printf("impl3d: 3d problem - 0 - locally-onedimensional, 1 - implicit\n");
		printf("TMA: integral matrix-vector mult alg - 0: simple, 1: toeplitz through FFT, 2: through series decomp.\n");
		printf("debug: debug level (0,1,2)\n");
		printf("FK: func in fractional derivative kernel - 0 - x, 1 - x^1/2, 2 - x^2, 3 - x^k\n");
		printf("FP: power for FK3\n");
		printf("int_niter: maximal number of iteration while calculating integrals\n");
		printf("initer: implicit niter\n");
		printf("ieps: implicit eps\n");
		printf("irs: implicit row scaling\n");
		printf("split3d: split rp in locally-onedimensional scheme in 3d\n");
	}
	for (int i=1;i<argc;i+=2)
	{
		if (strcmp(argv[i],"Tm")==0)
			Tm=atof(argv[i+1]);
		if (strcmp(argv[i],"Sm")==0)
			Sm=atof(argv[i+1]);
		if (strcmp(argv[i],"Om")==0)
			Om=atof(argv[i+1]);
		if (strcmp(argv[i],"BS")==0)
			BS=atoi(argv[i+1]);
		if (strcmp(argv[i],"NB")==0)
			NB=atoi(argv[i+1]);
		if (strcmp(argv[i],"A")==0)
		{
			analytic=atoi(argv[i+1]);
			if (analytic==10)
			{
				analytic=5;
				no_t=1;
			}
			if (analytic==11)
			{
				analytic=5;
				no_x=1;
			}
		}
		if (strcmp(argv[i],"I")==0)
			irr=atoi(argv[i+1]);
		if (strcmp(argv[i],"B")==0)
			bc=atoi(argv[i+1]);
		if (strcmp(argv[i],"Tau")==0)
			tau_m=atof(argv[i+1]);
		if (strcmp(argv[i],"mode")==0)
			m=atoi(argv[i+1]);
		if (strcmp(argv[i],"M")==0)
			MB=atoi(argv[i+1]);
		if (strcmp(argv[i],"K")==0)
			KB=atoi(argv[i+1]);
		if (strcmp(argv[i], "EPS") == 0)
			global_eps=eps = atof(argv[i + 1]);
		if (strcmp(argv[i], "EPS2") == 0)
			global_eps2=atof(argv[i + 1]);
		if (strcmp(argv[i], "varT") == 0)
			varT = atoi(argv[i + 1]);
		if (strcmp(argv[i], "varZ") == 0)
			varZ = atoi(argv[i + 1]);
		if (strcmp(argv[i], "varXY") == 0)
			varXY = atoi(argv[i + 1]);
		if (strcmp(argv[i], "FT") == 0)
			full_test = atoi(argv[i + 1]);
		if (strcmp(argv[i], "aa") == 0)
			aa = atof(argv[i + 1]);
		if (strcmp(argv[i], "bb") == 0)
			bb = atof(argv[i + 1]);
		if (strcmp(argv[i], "XYcoef") == 0)
			XYvar_coef = atof(argv[i + 1]);
		if (strcmp(argv[i], "Zcoef") == 0)
			Zvar_coef = atof(argv[i + 1]);
		if (strcmp(argv[i], "Zniter") == 0)
			varZ_niter = atof(argv[i + 1]);
		if (strcmp(argv[i], "avg") == 0)
			linear_avg = atoi(argv[i + 1]);
		if (strcmp(argv[i], "impl") == 0)
			implicit = atoi(argv[i + 1]);
		if (strcmp(argv[i], "impl3d") == 0)
			implicit3d = atoi(argv[i + 1]);
		if (strcmp(argv[i], "debug") == 0)
			debug_level = atoi(argv[i + 1]);
		if (strcmp(argv[i], "FK") == 0)
		{
			func_in_kernel = atoi(argv[i + 1]);
			if (func_in_kernel==0) func_power=1;
			if (func_in_kernel==1) func_power=0.5;
			if (func_in_kernel==2) func_power=2;
		}
		if (strcmp(argv[i], "FP") == 0)
			func_power = atof(argv[i + 1]);
		if (strcmp(argv[i], "int_niter") == 0)
			integr_max_niter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "TMA") == 0)
			toeplitz_mult_alg = atoi(argv[i + 1]);		
		if (strcmp(argv[i], "initer") == 0)
			impl_niter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "ieps") == 0)
			impl_err = atof(argv[i + 1]);
		if (strcmp(argv[i], "irs") == 0)
			implicit_row_scaling = atoi(argv[i + 1]);		
		if (strcmp(argv[i], "SD") == 0)
			space_der = atoi(argv[i + 1]);
		if (strcmp(argv[i], "split3d") == 0)
			rp_split_3d = atoi(argv[i + 1]);
		if (strcmp(argv[i], "sum_alg") == 0)
			sum_alg = atoi(argv[i + 1]);
		if (strcmp(argv[i], "sum_param") == 0)
			sum_param = atof(argv[i + 1]);
		if (strcmp(argv[i], "TMA2ad") == 0)
			tma2_all_diags = atoi(argv[i + 1]);
		if (strcmp(argv[i], "auto_df") == 0)
			difference_factor = atof(argv[i + 1]);
		if (strcmp(argv[i], "auto_tma22f") == 0)
			tma22_factor = atof(argv[i + 1]);
		if (strcmp(argv[i], "auto_tf") == 0)
			timegr_factor = atof(argv[i + 1]);
		if (strcmp(argv[i], "auto_tma22m") == 0)
			tma22_minstep = atof(argv[i + 1]);
		if (strcmp(argv[i], "auto_time_min") == 0)
			time_min = atoi(argv[i + 1]);
		if (strcmp(argv[i], "auto_iter_min") == 0)
			timegr_min = atoi(argv[i + 1]);
		if (strcmp(argv[i], "a5coef") == 0)
			a5coef = atof(argv[i + 1]);
#ifdef OCL
		if (strcmp(argv[i], "double_ext") == 0)
			double_ext = atoi(argv[i + 1]);
		if (strcmp(argv[i], "use_ocl") == 0)
			use_ocl = atoi(argv[i + 1]);
		if (strcmp(argv[i], "device") == 0)
			device = atoi(argv[i + 1]);
		if (strcmp(argv[i], "oclBS") == 0)
			oclBS = atoi(argv[i + 1]);
		if (strcmp(argv[i], "ocl_vector") == 0)
			ocl_vector = atoi(argv[i + 1]);
#endif
#ifdef USE_MPI
		if (strcmp(argv[i], "mpi_add_thr") == 0)
			mpi_mp_add_threads = atoi(argv[i + 1]);
#endif		
	}
	if ((KB*BS<=2)&&(m==3))
	{
		printf("K cannot be <=2. To do calculations in 2D mode set N or M to 2\n");
		exit(0);
	}
#ifdef USE_MPI
	int prov;
#ifndef _OPENMP
	printf("MPI mode only works with OpenMP enables\n");
	exit(0);
#endif
	if (implicit3d==0)
	{
	    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&prov);
	    if (prov!=MPI_THREAD_MULTIPLE)
	    {
		printf("mpi_thread_multiple not granted\n");
		return 1;
	    }
	    omp_set_dynamic(0);
	    omp_set_num_threads(mpi_size*mpi_mp_add_threads);
	}
	else
	    MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	printf("size %d rank %d nadd_thr %d\n",mpi_size,mpi_rank,mpi_mp_add_threads);
	omp_init_lock(&mpi_lock);
	proc_locks=new omp_lock_t[mpi_size];
	send_locks=new omp_lock_t[mpi_size];
	recv_locks=new omp_lock_t[mpi_size];
	proc_send_locks=new omp_lock_t[mpi_size];
	for (int i=0;i<mpi_size;i++)
	{
		omp_init_lock(&proc_locks[i]);
		omp_init_lock(&proc_send_locks[i]);
		omp_init_lock(&recv_locks[i]);
		omp_init_lock(&send_locks[i]);
	}
	// initial alloc
	if (sizes==NULL) sizes=new int[mpi_size];
	if (ns==NULL) ns=new int[mpi_size];
	if (bs==NULL) bs=new char *[mpi_size];
	if (bufs==NULL)
	{
		bufs=new char *[mpi_size];
		memset(bufs,0,mpi_size*sizeof(char *));
	}
	if (rqs==NULL) rqs=new MPI_Request[2*mpi_size];	
	if (sizes2==NULL) sizes2=new int[mpi_size];
	if (bufs2==NULL)
	{
		bufs2=new char *[mpi_size];
		memset(bufs2,0,mpi_size*sizeof(char *));
	}
	mpi_sends=new std::vector<mpi_req>[mpi_size];
	mpi_recvs=new std::vector<mpi_req>[mpi_size];
	mpi_received_msgs=new std::vector<mpi_msg>[mpi_size];
	last_sends=new int[mpi_size];
	memset(last_sends,0,mpi_size*sizeof(int));
	last_recvs=new int[mpi_size];
	memset(last_recvs,0,mpi_size*sizeof(int));
	last_msgs=new int[mpi_size];
	memset(last_msgs,0,mpi_size*sizeof(int));
#ifdef OCL	
	printf("OpenCL code is not parallelized for distributed computations");
#endif		   
#endif
	if (full_test)
	{
		double Ns[]={2,10};
		int nn=2;
		double taus[]={7200,3600};
		int ntaus=2;
#ifdef _OPENMP	
		H_solver::F = new double[BS*1000];
		H_solver::BVK=new double[BS];
#else		
		F = new double[BS*1000];
		BVK=new double[BS];
#endif
		for (int i=0;i<nn;i++)
			for (int j=0;j<ntaus;j++)
			{
				NB=Ns[i];
				sN=N=(BS*NB);
				sNB = NB;		
				_M = (BS*MB);
				K = (BS*KB);
				printf("test1\n");
				solve(2*24*3600 , 2*24*3600,2*24*3600,1,0,0,taus[j],1,eps,1.0,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,1,0,0,taus[j],1,eps,0.95,0.95);
				solve(2*24*3600 , 2*24*3600,2*24*3600,1,0,0,taus[j],1,eps,0.95,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,1,0,0,taus[j],1,eps,1.0,0.95);
				printf("test2\n");
				solve(2*24*3600 , 2*24*3600,2*24*3600,2,0,0,taus[j],1,eps,1.0,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,2,0,0,taus[j],1,eps,0.95,0.95);
				solve(2*24*3600 , 2*24*3600,2*24*3600,2,0,0,taus[j],1,eps,0.95,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,2,0,0,taus[j],1,eps,1.0,0.95);
				printf("test3\n");
				solve(2*24*3600 , 2*24*3600,2*24*3600,3,0,0,taus[j],1,eps,1.0,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,3,0,0,taus[j],1,eps,0.95,0.95);
				solve(2*24*3600 , 2*24*3600,2*24*3600,3,0,0,taus[j],1,eps,0.95,1.0);
				solve(2*24*3600 , 2*24*3600,2*24*3600,3,0,0,taus[j],1,eps,1.0,0.95);
			}
		NB = 2;
		sN = N = (BS*NB);
		sNB = NB;
		MB = 3;
		KB = 4;
		_M = (BS*MB);
		K = (BS*KB);
		printf("test1 3d\n");
		eps = 1e-5;
		solve(0.001 * 3600, 0.001 * 3600, 0.001 * 3600, 1, 0, 0, 0.01, 3, eps, 1.0, 1.0);
		solve(0.001 * 3600, 0.001 * 3600, 0.001 * 3600, 1, 0, 0, 0.01, 3, eps, 0.95, 0.95);
		solve(0.001 * 3600, 0.001 * 3600, 0.001 * 3600, 1, 0, 0, 0.01, 3, eps, 0.95, 1.0);
		solve(0.001 * 3600, 0.001 * 3600, 0.001 * 3600, 1, 0, 0, 0.01, 3, eps, 1.0, 0.95);
		return 0;
	}
	sN=N=(BS*NB);
	sNB = NB;
	_M = (BS*MB);
	K = (BS*KB);
	if (m==1)
		printf("1D - N %d (%d*%d) tend %g tsave %g \n",N,BS,NB,Tm,Sm);
	if (m==3)
		printf("3D - (%d,%d,%d) tend %g tsave %g \n",N,_M,K,Tm,Sm);
	fflush(stdout);		
#ifdef _OPENMP	
	if (m==1)
		H_solver::F=new double[N+2];
	if (m==3)
		H_solver::F = new double[N + _M+K+2];
	H_solver::BVK=new double[BS];
#else
	if (m==1)
		F=new double[N+2];
	if (m==3)
		F = new double[N + _M+K+2];
	BVK=new double[BS];
#endif
#ifdef USE_MPI
	mpi_mult=sN;
	if (_M>mpi_mult) mpi_mult=_M;
	if (K>mpi_mult) mpi_mult=K;
#endif
	solve(Tm , Sm,Om,analytic,irr,bc,tau_m,m,eps,aa,bb);
#ifdef USE_MPI
	MPI_Finalize();
	printf("%d nsends %d send size %d sum send time %d\n",mpi_rank,ncalls[0],ncalls[2],mpi_times[0]);
	printf("%d nrecvs %d recv size %d sum recv time %d\n",mpi_rank,ncalls[1],ncalls[3],mpi_times[1]);
	printf("%d niters %d\n",mpi_rank,mpi_times[4]);
	omp_destroy_lock(&mpi_lock);
	for (int i=0;i<mpi_size;i++)
	{
		omp_destroy_lock(&proc_locks[i]);		
		omp_destroy_lock(&proc_send_locks[i]);		
		omp_destroy_lock(&recv_locks[i]);
		omp_destroy_lock(&send_locks[i]);
	}
	delete [] recv_locks;
	delete [] send_locks;
	delete [] proc_locks;
	delete [] proc_send_locks;
	if (bufs)
	for (int i=0;i<mpi_size;i++)
	if (bufs[i])
		delete [] bufs[i];
	for (int i=0;i<mpi_size;i++)
	if (bufs2[i])
		delete [] bufs2[i];
	if (sizes) delete [] sizes;
	if (sizes2) delete [] sizes2;
	if (ns) delete [] ns;
	if (bufs) delete [] bufs;
	if (bufs2) delete [] bufs2;
	if (bs) delete [] bs;
	if (rqs) delete [] rqs;
	delete [] mpi_sends;
	delete [] mpi_recvs;
	delete [] mpi_received_msgs;
	delete [] last_sends;
	delete [] last_recvs;
	delete [] last_msgs;
#endif	
	return 0;
}
