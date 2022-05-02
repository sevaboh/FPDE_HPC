/* Author: Vsevolod Bohaienko */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/times.h>
#include "fract_common.h"
#include <map>
// timing
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}
///////////////////////////////////////////////////////////////
///////////// computing the values of DO derivative ///////////
///////////////////////////////////////////////////////////////
// arrays
T *dF=NULL; // values of dF/dt
T *bkk=NULL; // values of int(t_i,t_i+1)k^a for diff i and a
T *WA=NULL; // values of the weight function for the distributed order derivative * a step
T *DF=NULL; // values of fractional derivative of F

///////////// CPU code ///////////

// function to differentiate
double f(double x) 
{
	return x*x;
}
// df/dx
double df(double x)
{
	return 2*x;
}
// weight function for distributed order derivative (w(0)=0, int(w,a=0,..,1)=1
double w(double a) 
{
	return Gamma(3.0-a);
}
// analytic form of DoC derivative
double analytic(double x)
{
	if (x==1.0) return 2.0;
	return 2*(x-1.0)*x/log(x);
}
// fill arrays before computations
void fill_arrays(T tau,int ntau,T a0,T a1,int na,double (*ww)(double)=w,double (*ff)(double)=f)
{
	if (ff) if (dF) delete [] dF;
	if (bkk) delete [] bkk;
	if (WA) delete [] WA;
	if (DF) delete [] DF;
	if (ff) dF=new T[ntau];
	bkk=new T[ntau*na];
	DF=new T[ntau];
	WA=new T[na];
#pragma omp parallel for
	for (int i=0;i<ntau;i++)
	{
		if (ff) dF[i]=(ff(tau*(i+1))-ff(tau*i))/tau; // df/dx
		for (int j=0;j<na;j++)
			bkk[i*na+j]=bk((i+1)*tau,i*tau,(na==1)?a0:(a0+(a1-a0)*j/(T)(na-1))); // (t_i+1)^(1-a) - t_i^(1-a)
	}
	if ((na>1)&&(ww))
#pragma omp parallel for
	for (int j=0;j<na;j++)
		WA[j]=ww(a0+(a1-a0)*j/(T)(na-1))*(1.0/(T)(na-1)); // weight function
}
// fills DF with Caputo derivative values (int(df/dx / (xi-x)^a dxi, xi,0,t)
void Cder(int ntau,T tau,T a,T *F=NULL,int maxt=-1)
{
	T g2a=Gamma(2.0-a);
	if (F==NULL) fill_arrays(tau,ntau,a,a,1);
	for (int t=1;t<ntau;t++)
	{
	    T sum=0.0;
	    for (int i=0;i<((maxt==-1)?t:((maxt<t)?maxt:t));i++)
		sum+=((F==NULL)?dF[i]:((F[i+1]-F[i])/tau))*bkk[t-i-1];
	    sum/=g2a;
	    DF[t]=sum;
	}
}
// fills DF with distributed order Caputo derivative values (int(w(a) Da f da,a,0,1))
// do computations for time points from t0 to ntau with summing from j0 to maxt (if it is not -1) in inner cycle
void DO_Cder(int ntau,T tau,int asteps,T *F=NULL,int maxt=-1,int t0=1,int j0=0,int pt=0)
{
	int t1=GetTickCount();
	T *CD=new T[asteps];
	if (F==NULL) fill_arrays(tau,ntau,0.0,1.0,asteps);
	for (int t=t0;t<ntau;t++)
	{
	    T sum=0.0;
#pragma omp parallel for
	    for (int i=0;i<asteps;i++) // calculate Caputo derivative values for different a
	    {
		T sum2=0.0;
		for (int j=j0;j<((maxt==-1)?t:((t>maxt)?maxt:t));j++)
		if (j>=0)
		    sum2+=((F==NULL)?dF[j]:((F[j+1]-F[j])/tau))*bkk[(t-j-1)*asteps+i];
		sum2/=Gamma(2.0-(i/(T)(asteps-1)));
		CD[i]=sum2;
	    }
	    for (int i=0;i<asteps-1;i++) // integrate over a
		sum+=0.5*(WA[i]*CD[i]+WA[i+1]*CD[i+1]);
	    DF[t]=sum;
	}
	delete [] CD;
	if (F==NULL) printf("ntau %d tau %g asteps %d time %d\n",ntau,tau,asteps,GetTickCount()-t1);
}

////////////// CUDA code for distributed order Caputo derivative /////////////////

#ifdef OCL
#include "opencl_cuda_class.h"
#define cuda_text_macro(T) "\n\
#include \"cuda_fp16.h\" \n\
#include \"mma.h\" \n\
#include \"math_constants.h\" \n\
using namespace nvcuda;\n\
inline __device__ "#T" f("#T" x)\n\
{\n\
	return x*x;\n\
}\n\
// integral values for Caputo derivatives\n\
inline __device__ "#T" bk("#T" d1,"#T" d2,"#T" beta)\n\
{\n\
    if (d1==0.0) return pow(d2,("#T")1.0-beta);\n\
    if (d2==0.0) return pow(d1,("#T")1.0-beta);\n\
    return pow(d1,("#T")1.0-beta)-pow(d2,("#T")1.0-beta);\n\
}\n\
// fills array before computation\n\
extern \"C\" __global__ void Fill(double _tau,double _na,"#T" *df,"#T" *bkk)\n\
{\n\
	int na=_na;\n\
	"#T" tau=_tau;\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x; \n\
	df[id]=(f(tau*(id+1))-f(tau*id))/tau; \n\
	for (int j=0;j<na;j++)\n\
		bkk[id*na+j]=bk((id+1)*tau,id*tau,(na==1)?0:(j/("#T")(na-1))); \n\
}\n\
// calculates the values of Caputo derivatives (in do1)\n\
extern \"C\" __global__ void DOC(double _mode,double _ntau,double _tau,double _na,double _t0,double _j0,double _maxt,double _use_f,"#T" *F,"#T" *df,"#T" *bkk,"#T" *wa,"#T" *do1,"#T" *gs,double _total_ntau)\n\
{\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x; \n\
	int ntau=_ntau, na=_na;\n\
	int mode=_mode,use_f=_use_f;\n\
	int pt=id/na;\n\
	int t0=_t0,j0=_j0,maxt=_maxt,total_ntau=_total_ntau;\n\
	id=id-pt*na; \n\
	"#T" tau=_tau;\n\
	// simple mode \n\
	if ((mode==0)||(mode==2))\n\
	for (int t=t0;t<ntau;t++)\n\
	{\n\
		"#T" sum=0.0; \n\
		for (int j=j0;j<((maxt==-1)?t:((t>maxt)?maxt:t));j++) \n\
		if (j>=0)\n\
		{\n\
		    "#T" f;\n\
		    if (use_f==0)\n\
			    f=df[j];\n\
		    else \n\
			    f=(F[pt*total_ntau+j+1]-F[pt*total_ntau+j])/tau;\n\
		    sum+=f*bkk[(t-j-1)*na+id]; \n\
		}\n\
		sum/=gs[id]; \n\
		do1[pt*na*ntau+na*t+id]=sum;\n\
	}\n\
	// tensor cores \n\
	if (mode==1)\n\
	{\n\
		int ll=threadIdx.x;\n\
		__shared__ __align__(8) float C[16*16],C2[16*16];\n\
		__shared__ __align__(8) half A[16*16],A2[16*16];\n\
		__shared__ __align__(8) half B[16*16];\n\
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;\n\
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;\n\
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag,c_frag2;\n\
		int maxtblock=16*(maxt/16);\n\
		if (maxt%16) maxtblock++;\n\
		for (int tblock=(t0/16);tblock<(ntau/16);tblock++) \n\
		{ \n\
			// Initialize the output to zero\n\
			wmma::fill_fragment(c_frag, 0.0f);\n\
			wmma::fill_fragment(c_frag2, 0.0f);\n\
			for (int jblock=(j0/16);jblock<=((maxt==-1)?tblock:((tblock>maxtblock)?maxtblock:tblock));jblock++)\n\
			{\n\
				// fill matrices \n\
				if (ll<16) for (int i=0;i<16;i++) A[ll*16+i]=bkk[((tblock+1)*16-jblock*16-i-1)*na+id];\n\
				if (ll>=16) for (int i=0;i<16;i++) A2[(ll-16)*16+i]=bkk[((tblock+1)*16-jblock*16-i-1)*na+id];\n\
				if (use_f==0) // df/dt from df array \n\
				{ \n\
					if (ll<16) for (int i=0;i<8;i++) \n\
						B[i*16+ll]=(((((jblock-1)*16+i+ll)>=0)&&\n\
							    ((maxt==-1)||(((jblock-1)*16+i+ll)<maxt)))?\n\
								    df[(jblock-1)*16+i+ll]:0.0);\n\
					if (ll>=16) for (int i=8;i<16;i++) \n\
						B[i*16+(ll-16)]=(((((jblock-1)*16+i+(ll-16))>=0)&&\n\
							    ((maxt==-1)||((((jblock-1)*16+i+(ll-16))<maxt))))?\n\
								    df[(jblock-1)*16+i+(ll-16)]:0.0);\n\
				} \n\
				else // df/dt as f+1-f0 / tau from F array \n\
				{ \n\
					if (ll<16) for (int i=0;i<8;i++) \n\
						B[i*16+ll]=(((((jblock-1)*16+i+ll)>=0)&&\n\
						((maxt==-1)||(((jblock-1)*16+i+ll)<maxt)))?\n\
						((F[pt*total_ntau+(jblock-1)*16+i+ll+1]-F[pt*total_ntau+(jblock-1)*16+i+ll])/tau):0.0);\n\
					if (ll>=16) for (int i=8;i<16;i++) \n\
						B[i*16+(ll-16)]=(((((jblock-1)*16+i+(ll-16))>=0)&&\n\
						((maxt==-1)||(((jblock-1)*16+i+(ll-16))<maxt)))?\n\
						((F[pt*total_ntau+(jblock-1)*16+i+(ll-16)+1]-F[pt*total_ntau+(jblock-1)*16+i+(ll-16)])/tau):0.0);\n\
				} \n\
				__syncthreads();\n\
				// perform the matrix multiplication\n\
				wmma::load_matrix_sync(a_frag, A, 16);\n\
				wmma::load_matrix_sync(b_frag, B, 16);\n\
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);\n\
				wmma::load_matrix_sync(a_frag, A2, 16);\n\
				wmma::mma_sync(c_frag2, a_frag, b_frag, c_frag2);\n\
			}\n\
			// Store the output\n\
			wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);\n\
			wmma::store_matrix_sync(C2, c_frag2, 16, wmma::mem_row_major);\n\
			// Change do1\n\
			for (int i=tblock*16;i<(tblock+1)*16;i++)\n\
			    if (ll<16) \n\
				do1[pt*na*ntau+na*i+id]=C[ll*16+(i-tblock*16)]/gs[id];\n\
			    else\n\
				do1[pt*na*ntau+na*i+id]=C2[(ll-16)*16+(i-tblock*16)]/gs[id];\n\
		} \n\
	}\n\
}\n\
// calculates DO derivative (in do2) as weighted (weights in wa) sum of previously calculated Caputo derivatives (in do1) \n\
extern \"C\" __global__ void Sum(double _ntau,double _ntau2,double _t0,double _na,double _append,"#T" *do1,"#T" *do2,"#T" *wa)\n\
{\n\
	int ntau=_ntau,t0=_t0,na=_na,ntau2=_ntau2;\n\
	int append=_append;\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x; \n\
	int pt=id/(ntau-t0);\n\
	id=id-pt*(ntau-t0)+t0; \n\
	"#T" sum=0.0; \n\
	for (int i=0;i<na-1;i++) \n\
		sum+=0.5*(wa[i]*do1[pt*na*ntau+na*id+i]+wa[i+1]*do1[pt*na*ntau+na*id+i+1]); \n\
	if (append==0) \n\
		do2[pt*ntau2+id]=sum;\n\
	else \n\
		do2[pt*ntau2+id]+=sum;\n\
}\n\
/////////////////////////////// equation solution \n\
// source function\n\
inline __device__ "#T" eq_ff("#T" t,"#T" x,double ex)\n\
{\n\
	if (ex==0.0)\n\
	    return sin(2*CUDART_PI_F*x)*(800*CUDART_PI_F*CUDART_PI_F*t*t*t*t+(2400*(t*t*t*t-t*t*t)/log(t))); \n\
	if (ex==1.0)\n\
	    return 0.0;\n\
	return 0.0;\n\
}\n\
// calculates right part of linear equation system\n\
extern \"C\" __global__ void eq_F(double nx,double _eq_Nt,double eq_D,double eq_b0,double eq_tau,double eq_h,double _j,"#T" *C,"#T" *DF,"#T" *F,double ex)\n\
{\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x;\n\
	int j=(int)_j;\n\
	int eq_Nt=(int)_eq_Nt;\n\
	if ((id==0)||(id==(int)(nx-1)))\n\
		F[id]=0.0;\n\
	else\n\
	{\n\
		"#T" rp=-0.5*(eq_D/(eq_h*eq_h))*(C[(id-1)*eq_Nt+j]-2.0*C[id*eq_Nt+j]+C[(id+1)*eq_Nt+j]);\n\
		"#T" lp=-(eq_b0*C[id*eq_Nt+j]/eq_tau)-0.5*(eq_ff(j*eq_tau,id*eq_h,ex)+eq_ff((j+1)*eq_tau,id*eq_h,ex));\n\
		"#T" nlp=DF[id*eq_Nt+j+1];\n\
		if (!isfinite(lp))\n\
			lp=-(eq_b0*C[id*eq_Nt+j]/eq_tau)-eq_ff((j+1)*eq_tau,id*eq_h,ex);\n\
		F[id]=rp+lp+nlp;\n\
	}\n\
}\n\
// tridiagonal system solver through a known inverse matrix (inv)\n\
extern \"C\" __global__ void eq_Solve(double _t,double _nt,double _nx,double A,"#T" *C,"#T" *inv,"#T" *F)\n\
{\n\
	int id=threadIdx.x+blockIdx.x*blockDim.x;\n\
	int ll=threadIdx.x;\n\
	__shared__ __align__(8) "#T" sF[32];\n\
	int nx=(int)_nx,t=(int)_t,nt=(int)_nt;\n\
	"#T" sum=0.0;\n\
	for (int i=0;i<nx;i+=32)\n\
	{\n\
		sF[ll]=F[i+ll];\n\
		__syncthreads();\n\
		for (int j=0;j<32;j++)\n\
			sum+=inv[id*nx+i+j]*sF[j];\n\
	}\n\
	C[id*nt+t]=sum/A;\n\
}\n\
"
#if Ti == 0
const char *cuda_text=cuda_text_macro(float);
#endif
#if Ti == 1
const char *cuda_text=cuda_text_macro(double);
#endif
OpenCL_program *prg=NULL;
OpenCL_commandqueue *queue;
OpenCL_prg *prog;
OpenCL_kernel *kFill,*kDO,*kSum,*k_eq_F,*k_eq_solve;
OpenCL_buffer *bDF,*bBKK,*bWA,*bDOF,*bgs,*bDO1,*bC,*b_eq_F,*b_eq_inv_m=NULL;
int DO_CUDA_init(int ntau,int asteps,int use_f,double (*ww)(double),T *F=NULL)
{
	int nf=1;
	if (use_f>0) nf=use_f;
	int t1;
	if (prg) delete prg;
	prg = new OpenCL_program(1);
	t1=GetTickCount();
	queue = prg->create_queue(0, 0);
	prog = prg->create_program(cuda_text);
	kFill = prg->create_kernel(prog, "Fill"); // fill constant coefs
	kDO = prg->create_kernel(prog, "DOC"); // calculate Dalpha
	kSum = prg->create_kernel(prog, "Sum"); // sum Dalpaha into Ddistributes
	k_eq_F = prg->create_kernel(prog, "eq_F"); // calc right part for eq solution
	k_eq_solve = prg->create_kernel(prog, "eq_Solve"); // solve tridiagonal (a b c)=f
	bDF = prg->create_buffer(CL_MEM_READ_WRITE, ntau*sizeof(T), NULL);
	bC = prg->create_buffer(CL_MEM_READ_WRITE, nf*ntau*sizeof(T), NULL);// to save C values while solving 1d equation
	b_eq_F = prg->create_buffer(CL_MEM_READ_WRITE, nf*sizeof(T), NULL);// to save right-part values while solving 1d equation
	bBKK = prg->create_buffer(CL_MEM_READ_WRITE, ntau*asteps*sizeof(T), NULL);
	bDOF = prg->create_buffer(CL_MEM_READ_WRITE, nf*ntau*sizeof(T), NULL); // solutions for each time point
	bDO1 = prg->create_buffer(CL_MEM_READ_WRITE, nf*ntau*asteps*sizeof(T), NULL); // asteps DaF values for each time point
	bWA = prg->create_buffer(CL_MEM_READ_WRITE, asteps*sizeof(T), NULL);
	bgs = prg->create_buffer(CL_MEM_READ_WRITE, asteps*sizeof(T), NULL);
	// write initial F
	if (F && use_f)
		queue->EnqueueWriteBuffer(bC, F, 0,nf*ntau*sizeof(T));
	// precalc gammas
	T *gs=new T[asteps];
#pragma omp parallel for
	for (int i=0;i<asteps;i++)
		gs[i]=Gamma(2.0-(i/(T)(asteps-1)));
	queue->EnqueueWriteBuffer(bgs, gs, 0,asteps*sizeof(T));
	// precalc W
	WA=new T[asteps];
	if (asteps>1)
#pragma omp parallel for
	for (int j=0;j<asteps;j++)
		WA[j]=ww(j/(T)(asteps-1))*(1.0/(T)(asteps-1)); // weight function
	queue->EnqueueWriteBuffer(bWA, WA, 0,  asteps*sizeof(T));
	queue->Finish();
	delete [] gs;
	delete [] WA;
	return t1;
}
// fills DF with distributed order Caputo derivative values using CUDA
// mode=0 - simple, 1 - with the usage of tensor cores
// use_f ==0 - F from function f
// use_f>0 - F from F array for a grid with number of points equal to use_f
// time step number is limited to [t0,ntau]
// summation for Caputo derivative is limited to [j0,t] if maxt=-1 and to [j0,maxt] if not 
// if init_ntau!=-1 GPU memory is allocated for number of time steps equal to init_ntau
int DO_Cder_CUDA(int mode,int ntau,double tau,int asteps,int append=0,T *F=NULL,int use_f=0,int maxt=-1,int t0=1,int j0=0,double (*ww)(double)=w,int init_ntau=-1)
{
	int t1=GetTickCount();
	// number of a and time steps must be a multiply of 16
	if (mode==1) ntau=16*((ntau/16)+((ntau%16)?1:0));
	asteps=32*((asteps/32)+((asteps%32)?1:0));
	// initialize
	if (prg==NULL) t1=DO_CUDA_init(((init_ntau==-1)?ntau:init_ntau),asteps,use_f,ww,F);
	// fill arrays
	int err=0;
	double dtau=tau;
	double da1=asteps;
	err |= kFill->SetArg(0, sizeof(double), &dtau);
	err |= kFill->SetArg(1, sizeof(double), &da1);
	err |= kFill->SetBufferArg(bDF, 2);
	err |= kFill->SetBufferArg(bBKK, 3);
	if (err) { printf("Error: Failed to set kernels args"); exit(0); }
	size_t nth=ntau,lsize=1; // one thread per one array element
	queue->ExecuteKernel(kFill, 1, &nth, &lsize);
	// compute DaF
	err=0;	
	double da0=ntau;
	double dm=mode;
	double dt0=t0,dj0=j0,dmaxt=maxt,duse_f=use_f,dtnt=((init_ntau==-1)?ntau:init_ntau);
	err |= kDO->SetArg(0, sizeof(double), &dm);
	err |= kDO->SetArg(1, sizeof(double), &da0);
	err |= kDO->SetArg(2, sizeof(double), &tau);
	err |= kDO->SetArg(3, sizeof(double), &da1);
	err |= kDO->SetArg(4, sizeof(double), &dt0);
	err |= kDO->SetArg(5, sizeof(double), &dj0);
	err |= kDO->SetArg(6, sizeof(double), &dmaxt);
	err |= kDO->SetArg(7, sizeof(double), &duse_f);
	err |= kDO->SetBufferArg(bC, 8);
	err |= kDO->SetBufferArg(bDF, 9);
	err |= kDO->SetBufferArg(bBKK, 10);
	err |= kDO->SetBufferArg(bWA, 11);
	err |= kDO->SetBufferArg(bDO1, 12);
	err |= kDO->SetBufferArg(bgs, 13);
	err |= kDO->SetArg(14, sizeof(double), &dtnt);
	if (err) { printf("Error: Failed to set kernels args"); exit(0); }
	nth=asteps*((use_f>0)?use_f:1); // one thread per a that calculates DaF for all time points
	lsize=32;
	queue->ExecuteKernel(kDO, 1, &nth, &lsize);
	// sum to get DOC
	err=0;
	double dap=append;
	double dnt2=((init_ntau==-1)?ntau:init_ntau);
	err |= kSum->SetArg(0, sizeof(double), &da0);
	err |= kSum->SetArg(1, sizeof(double), &dnt2);
	err |= kSum->SetArg(2, sizeof(double), &dt0);
	err |= kSum->SetArg(3, sizeof(double), &da1);
	err |= kSum->SetArg(4, sizeof(double), &dap);
	err |= kSum->SetBufferArg(bDO1, 5);
	err |= kSum->SetBufferArg(bDOF, 6);
	err |= kSum->SetBufferArg(bWA, 7);
	if (err) { printf("Error: Failed to set kernels args"); exit(0); }
	nth=(ntau-t0)*((use_f>0)?use_f:1); // one thread per time point
	lsize=1;
	queue->ExecuteKernel(kSum, 1, &nth, &lsize);
	// get data
	if (use_f==0)
	{
		if (DF) delete [] DF;
		DF=new T[ntau];
		queue->EnqueueBuffer(bDOF, DF, 0,  ntau*sizeof(T));
		queue->Finish();
		printf("ntau %d tau %g asteps %d time %d\n",ntau,tau,asteps,GetTickCount()-t1);
	}
	return t1;
}
#endif
///////////////////////////////////////////////////////////////
///////////// solving DOu=Dd^2u/dx^2 + f //////////////////////
///////////////////////////////////////////////////////////////
int eq_ex=0;
//eq_ex==0 -  testing example from doi.org/10.1007/s10915-017-0360-8
T eq_D=2.0;
T eq_max_x=0.5;
T eq_max_t=0.5;
// eq_ex==1 - testing example with analytical solution in the form of a row from DOI 10.1007/s10559-022-00436-3
T eq_ex1_power=2.0;
T eq_ex1_err=1e-10;
void set_for_ex()
{
    eq_D=2.0;
    if (eq_ex==0)
    {
	eq_max_x=0.5;
	eq_max_t=0.5;
    }
    if (eq_ex==1)
    {
	eq_max_x=1.0;
	eq_max_t=1.0;
    }
}
// right part
T eq_f(T t,T x)
{
    if (eq_ex==0)
	return sin(2*M_PI*x)*(800*M_PI*M_PI*t*t*t*t+(2400*(t*t*t*t-t*t*t)/log(t)));
    if (eq_ex==1)
        return 0.0;
    return 0.0;
}
// DO derivative weight function
double eq_w(double a)
{
    if (eq_ex==0)
	return Gamma(5.0-a);
    if (eq_ex==1)
	return pow(a,eq_ex1_power)*(1.0+eq_ex1_power);
    return 0.0;
}
double eq_C0(double x)
{
    if (eq_ex==0)
	return 0.0;
    if (eq_ex==1)
	return x*(1.0-x);
    return 0.0;
}
////////////////////////////////////
// analytical solution
// for the problem Du=k(1+tau_r D)d2u/dx2, u(0,t)=u(1,t)=0
// x from [0,1]
////////////////////////////////////
int int_max_iter=10;
double eq_tau_r=0.0; // to match diffusion equation
// iterative subdivision with 5-point open Newton-Cotes quadrature to calculate int(f1*f2,x=a,..,b)
double integrate_ab(double a,double b,double (*f1)(double),double (*f2)(double,double),double p1,double err,int nst=1,int iter=0)
{
    double v1=0.0,v2=0.0;
    double st1=(b-a)/nst,st2=st1/2.0;
#pragma omp parallel for reduction (+:v1)    
    for (int i=0;i<nst;i++)
	v1=v1+(1.0/20.0)*st1*(11.0*f1((i+(1.0/6.0))*st1)*f2((i+(1.0/6.0))*st1,p1)-14.0*f1((i+(2.0/6.0))*st1)*f2((i+(2.0/6.0))*st1,p1)+26.0*f1((i+(3.0/6.0))*st1)*f2((i+(3.0/6.0))*st1,p1)-14.0*f1((i+(4.0/6.0))*st1)*f2((i+(4.0/6.0))*st1,p1)+11.0*f1((i+(5.0/6.0))*st1)*f2((i+(5.0/6.0))*st1,p1));
#pragma omp parallel for reduction (+:v2)    
    for (int i=0;i<2*nst;i++)
	v2=v2+(1.0/20.0)*st2*(11.0*f1((i+(1.0/6.0))*st2)*f2((i+(1.0/6.0))*st2,p1)-14.0*f1((i+(2.0/6.0))*st2)*f2((i+(2.0/6.0))*st2,p1)+26.0*f1((i+(3.0/6.0))*st2)*f2((i+(3.0/6.0))*st2,p1)-14.0*f1((i+(4.0/6.0))*st2)*f2((i+(4.0/6.0))*st2,p1)+11.0*f1((i+(5.0/6.0))*st2)*f2((i+(5.0/6.0))*st2,p1));
    if ((fabs(v2-v1)<err)||((iter+1)>=int_max_iter))
	return v2;
    double i=integrate_ab(a,b,f1,f2,p1,err,nst*2,iter+1);
    return i;
}
// quadrature to calculate int(f1*f2,x=a,..,inf) [ mid-point rule with exponential growth of step ]
double integrate_ainf(double a,double (*f1)(double,double),double (*f2)(double,double,double),double p1,double p2,double p3,double err)
{
    double v1=0.0,v,x=0;
    double st=err*1e-10;
    do
    {
	v=f1(x+0.5*st,p1)*f2(x+0.5*st,p2,p3);
	v1+=st*v;
	x=x+st;
	st*=1.05;
    }
    while (fabs(v)>err);
    return v1;
}
double lambda_n(double n)
{
    return n*n*M_PI*M_PI;
}
double fi_n(double x,double n)
{
    return sin(sqrt(lambda_n(n))*x);
}
double omega_n(double n)
{
    return eq_D*lambda_n(n)/(1.0+eq_tau_r*eq_D*lambda_n(n));
}
double eq_C0_n(double n,double err)
{
    double i=integrate_ab(0.0,1.0,eq_C0,fi_n,n,err);
    return i;
}
double eq_r_b1(double b,double r)
{
    return pow(r,b)*cos(M_PI*b);
}
double eq_r_b2(double b,double r)
{
    return pow(r,b)*sin(M_PI*b);
}
std::map<double,double> As,Bs; 
double eq_A_r(double r,double err)
{
    std::map<double,double>::iterator it;
    if ((it=As.find(r))!=As.end())
	return it->second;
    double i=integrate_ab(0.0,1.0,eq_w,eq_r_b1,r,err);
    As.insert(std::pair<double,double>(r,i));
    return i;
}
double eq_B_r(double r,double err)
{
    std::map<double,double>::iterator it;
    if ((it=Bs.find(r))!=Bs.end())
	return it->second;
    double i=integrate_ab(0.0,1.0,eq_w,eq_r_b2,r,err);
    Bs.insert(std::pair<double,double>(r,i));
    return i;
}
double eq_G_n_r(double r,double n,double err)
{
    double A=eq_A_r(r,err);
    double B=eq_B_r(r,err);
    double om=omega_n(n);
    double i=om*B/((A+om)*(A+om)+B*B);
    return i;
} 
double eq_S_f1(double r,double t)
{
    return (1.0/M_PI)*exp(-r*t)/r;
}
std::map<std::pair<double,double>,double >Sns;
double eq_S_n_t(double t,double n, double err)
{
    char str[1024];
    double ft;
    sprintf(str,"%lg",t);
    sscanf(str,"%lg",&ft);
    std::map<std::pair<double,double>,double>::iterator it;
    if ((it=Sns.find(std::pair<double,double>(ft,n)))!=Sns.end())
	return it->second;
    double i=integrate_ainf(0.0,eq_S_f1,eq_G_n_r,t,n,err,err);
    Sns.insert(std::pair<std::pair<double,double>,double >(std::pair<double,double>(ft,n),i));
    // write to file
    FILE *fi=fopen("s_n_t.txt","at");
    fprintf(fi,"%lg %lg %lg\n",t,n,i);
    fclose(fi);
    return i;
}
double eq_anal_u_ex1(double t, double x,double err)
{
    static int first=1;
    if (first) // read S_n_t values from file
    {
	char str[1024];
	FILE *fi=fopen("s_n_t.txt","rt");
	if (fi)
	{
	    while(fgets(str,1024,fi))
	    {
		double t,n,s;
		if (sscanf(str,"%lg %lg %lg\n",&t,&n,&s)==3)
		    Sns.insert(std::pair<std::pair<double,double>,double >(std::pair<double,double>(t,n),s));	    
	    }
	    fclose(fi);
	}
	first=0;
    }
    double v,sum=0.0;
    int n=1;
    do
    {
	double v1=fi_n(x,n),v2=eq_C0_n(n,err),v3=eq_S_n_t(t,n,err);
	v=2.0*v1*v2*v3;
	sum+=v;
	n++;
    }
    while (fabs(v)>err);
    return sum;
}
T eq_anal_u(T t,T x)
{
    if (eq_ex==0)
	return 100*t*t*t*t*sin(2*M_PI*x);
    if (eq_ex==1)
	return eq_anal_u_ex1(t,x,eq_ex1_err);
    return 0.0;
}
////////////////////////////////////
// numerical solution
// linear equation system Ax_i-1^j+1-Sx_i^j+1+Bx_i+1^j+1=F_i^j
// Crank-Nicholson scheme on uniform grid
////////////////////////////////////
T *eq_C=NULL; // solutions
T *eq_A=NULL,*eq_B=NULL; // coefficients for Thomas algorithm
T *eq_DF=NULL; // non-local parts of DO derivative
T *eq_inv_m=NULL; // inverse of linear system's matrix
int eq_Nx=101; // number of grid nodes
int eq_Na=64; // number of integration nodes for DO derivative
int eq_tblock=16; // t block size
int eq_eNt=160; // needed number of t steps
int eq_Nt=(eq_eNt/eq_tblock)*eq_tblock; // number of t steps
T eq_b0;
T eq_h=eq_max_x/(eq_Nx-1);
T eq_tau=eq_max_t/eq_Nt;
// A,B,S of linear equation system
T eq_Aa=0.5*eq_D/(eq_h*eq_h); 
T eq_Bb=0.5*eq_D/(eq_h*eq_h);
T eq_Ss;
// right part of linear equation system
T inline eq_Ff(int i,int j)
{
	if ((i==0)||(i==(eq_Nx-1))) return 0.0;
	T rp=-0.5*(eq_D/(eq_h*eq_h))*(eq_C[(i-1)*eq_Nt+j]-2.0*eq_C[i*eq_Nt+j]+eq_C[(i+1)*eq_Nt+j]); // second derivative
	T lp=-(eq_b0*eq_C[i*eq_Nt+j]/eq_tau)-0.5*(eq_f(j*eq_tau,i*eq_h)+eq_f((j+1)*eq_tau,i*eq_h)); // local part of t derivative and f
	T nlp=eq_DF[i*eq_tblock+((j+1)%eq_tblock)]; // non-local part of t derivative
	if (!finite(lp))
		lp=-(eq_b0*eq_C[i*eq_Nt+j]/eq_tau)-eq_f((j+1)*eq_tau,i*eq_h);
	return rp+lp+nlp;
}
void eq_init()
{
	set_for_ex();
	eq_Nt=(eq_eNt/eq_tblock)*eq_tblock;
	eq_h=eq_max_x/(eq_Nx-1);
	eq_tau=eq_max_t/eq_Nt;
	eq_Aa=0.5*eq_D/(eq_h*eq_h);
	eq_Bb=0.5*eq_D/(eq_h*eq_h);
	// alloc
	if (eq_A) delete [] eq_A;
	if (eq_B) delete [] eq_B;
	if (eq_C) delete [] eq_C;
	if (eq_DF) delete [] eq_DF;
	eq_A=new T[eq_Nx];
	eq_B=new T[eq_Nx];
	eq_C=new T[eq_Nx*eq_Nt];
	eq_DF=new T[eq_Nx*eq_Nt];
	memset(eq_C,0,eq_Nx*eq_Nt*sizeof(T)); // u(x,0)
	for (int i=0;i<eq_Nx;i++)
		eq_C[i*eq_Nt]=eq_C0(i*eq_h);
	// local coefficient for C[i]
	fill_arrays(eq_tau,eq_Nt,0,1,eq_Na,eq_w,NULL);
	eq_b0=0;
	for (int a=0;a<eq_Na-1;a++)
		eq_b0+=0.5*(WA[a]*bkk[a]/Gamma(2.0-(a/(T)(eq_Na-1)))+WA[a+1]*bkk[a+1]/Gamma(2.0-((a+1)/(T)(eq_Na-1))));
	eq_Ss=(eq_D/(eq_h*eq_h))+(eq_b0/eq_tau);
	// A in Thomas algorithm is fixed
	eq_A[1]=0.0; // u(t,0) = 0
	for (int i=1;i<eq_Nx-1;i++)
		eq_A[i+1]=eq_Bb/(eq_Ss-eq_Aa*eq_A[i]);
#ifdef OCL
	// compute inverse matrix
	T *Mk=new T[eq_Nx-1]; // determinants
	eq_inv_m=new T[(eq_Nx)*(eq_Nx)]; // inverse matrix
	Mk[0]=1;
	Mk[1]=-eq_Ss/eq_Aa;
	for (int i=2;i<eq_Nx-1;i++)
		Mk[i]=(-eq_Ss/eq_Aa)*Mk[i-1]-Mk[i-2];
	for (int j=1;j<eq_Nx-1;j++)
		for (int i=j;i<eq_Nx-1;i++)
			eq_inv_m[i*eq_Nx+j]=pow(-1.0,(T)(i+j))*Mk[j-1]*Mk[eq_Nx-2-i]/Mk[eq_Nx-2];
	for (int j=1;j<eq_Nx-1;j++)
		for (int i=1;i<j;i++)
			eq_inv_m[i*eq_Nx+j]=eq_inv_m[j*eq_Nx+i];
	eq_inv_m[0]=eq_inv_m[(eq_Nx-1)*eq_Nx+eq_Nx-1]=eq_Aa;
	eq_inv_m[eq_Nx-1]=eq_inv_m[(eq_Nx-1)*eq_Nx]=0.0;
	for (int i=1;i<eq_Nx-1;i++)
	{
	    eq_inv_m[i*eq_Nx]=-eq_inv_m[i*eq_Nx+1]*eq_Aa;
	    eq_inv_m[i*eq_Nx+eq_Nx-1]=-eq_inv_m[i*eq_Nx+eq_Nx-2]*eq_Aa;
	}
	for (int j=1;j<eq_Nx-1;j++)
	    eq_inv_m[j]=eq_inv_m[(eq_Nx-1)*eq_Nx+j]=0.0;
	delete [] Mk;
#endif
	printf("Nx %d Na %d tblock %d eNt %d example %d ex1_power %g ex1_err %g int_max_iter %d\n",eq_Nx,eq_Na,eq_tblock,eq_eNt,eq_ex,eq_ex1_power,eq_ex1_err,int_max_iter);
}
// Thomas solver
void eq_one_step(int j)
{
	eq_B[1]=0.0;
	for (int i=1;i<eq_Nx-1;i++)
		eq_B[i+1]=(eq_A[i+1]/eq_Bb)*(eq_Aa*eq_B[i]-eq_Ff(i,j-1));
	eq_C[j+eq_Nt*(eq_Nx-1)]=0.0; // u(max_x,t)=0 
	for (int i=eq_Nx-2;i>=0;i--)
		eq_C[j+eq_Nt*i]=eq_A[i+1]*eq_C[j+eq_Nt*(i+1)]+eq_B[i+1];
}
void eq_solve(int total)
{
	int t1=GetTickCount(),t2;
	eq_init();
	for (int i=0;i<eq_Nt;i+=eq_tblock)
	{
		// calculate part of DO derivative in all cells
		for (int j=0;j<eq_Nx;j++)
		{
			DO_Cder(i+eq_tblock,eq_tau,eq_Na,eq_C+j*eq_Nt,((i==0)?i:i-1),i,0,j);
			for (int k=0;k<eq_tblock;k++)
				eq_DF[j*eq_tblock+k]=DF[i+k];
		}
		// solve for each time step in a block
		for (int j=((i==0)?1:0);j<eq_tblock;j++)
		{
			// solve linear equation system
			eq_one_step(i+j);
			// change eq_DF
			if (j!=(eq_tblock-1))
			for (int jj=0;jj<eq_Nx;jj++)
			{
				DO_Cder(i+j+2,eq_tau,eq_Na,eq_C+jj*eq_Nt,i+j,i+j+1,i-1,jj);
				eq_DF[jj*eq_tblock+j+1]+=DF[i+j+1];
			}
			if (total==2)
			    printf("t %g x[%g]=%g\n",(i+j)*eq_tau,(eq_Nx/2)*eq_h,eq_C[(eq_Nx/2)*eq_Nt+i+j]);
		}
	}
	// check on a final t step
	t2=GetTickCount();
	double err=0.0;
	int i=eq_Nt-1;
	for (int jj=0;jj<eq_Nx;jj++)
	{
		double a=eq_anal_u(i*eq_tau,jj*eq_h);
		double diff=(eq_C[jj*eq_Nt+i]-a);
		err+=diff*diff;
		if (total) printf("t %g x %g N %g A %g\n",i*eq_tau,jj*eq_h,eq_C[jj*eq_Nt+i],a);
	}
	err=sqrt(err/eq_Nx);
	printf("CPU t %g err %g time %d time anal %d\n",i*eq_tau,err,t2-t1,GetTickCount()-t2);
}

////////////// CUDA code /////////////////

#ifdef OCL
// threediagonal system inversion - http://iopscience.iop.org/0305-4470/29/7/020
void eq_one_step_cuda(int j)
{
	// put inv matrix into the gpu memory
	if (b_eq_inv_m==NULL)
	{
		b_eq_inv_m = prg->create_buffer(CL_MEM_READ_WRITE, eq_Nx*eq_Nx*sizeof(T), NULL);
		queue->EnqueueWriteBuffer(b_eq_inv_m, eq_inv_m, 0,eq_Nx*eq_Nx*sizeof(T));
	}
	// fill right part vector
	int err=0;
	double dnx=eq_Nx,dnt=eq_Nt,dj=j-1;
	double db0=eq_b0,dtau=eq_tau,dh=eq_h,dD=eq_D,dex=eq_ex;
	err |= k_eq_F->SetArg(0, sizeof(double), &dnx);
	err |= k_eq_F->SetArg(1, sizeof(double), &dnt);
	err |= k_eq_F->SetArg(2, sizeof(double), &dD);
	err |= k_eq_F->SetArg(3, sizeof(double), &db0);
	err |= k_eq_F->SetArg(4, sizeof(double), &dtau);
	err |= k_eq_F->SetArg(5, sizeof(double), &dh);
	err |= k_eq_F->SetArg(6, sizeof(double), &dj);
	err |= k_eq_F->SetBufferArg(bC, 7);
	err |= k_eq_F->SetBufferArg(bDOF, 8);
	err |= k_eq_F->SetBufferArg(b_eq_F, 9);
	err |= k_eq_F->SetArg(10, sizeof(double), &dex);
	if (err) { printf("Error: Failed to set kernels args"); exit(0); }
	size_t nth=eq_Nx; // one thread per one x node
	size_t lsize=1;
	queue->ExecuteKernel(k_eq_F, 1, &nth, &lsize);
	// solve linear equation system
	err=0;
	dj=j;
	double da=eq_Aa;
	err |= k_eq_solve->SetArg(0, sizeof(double), &dj);
	err |= k_eq_solve->SetArg(1, sizeof(double), &dnt);
	err |= k_eq_solve->SetArg(2, sizeof(double), &dnx);
	err |= k_eq_solve->SetArg(3, sizeof(double), &da);
	err |= k_eq_solve->SetBufferArg(bC, 4);
	err |= k_eq_solve->SetBufferArg(b_eq_inv_m, 5);
	err |= k_eq_solve->SetBufferArg(b_eq_F, 6);
	if (err) { printf("Error: Failed to set kernels args"); exit(0); }
	nth=eq_Nx; // one thread per one x node
	lsize=32;
	queue->ExecuteKernel(k_eq_solve, 1, &nth, &lsize);
}
void eq_solve_cuda(int total)
{
	int t1=GetTickCount(),t2=-1,t3=-1,t4;
	eq_tblock=16*((eq_tblock/16)+((eq_tblock%16)?1:0));
	eq_Nt=16*((eq_Nt/16)+((eq_Nt%16)?1:0));
	eq_Na=32*((eq_Na/32)+((eq_Na%32)?1:0));
	eq_Nx=32*((eq_Nx/32)+((eq_Nx%32)?1:0));
	eq_init();
	for (int i=0;i<eq_Nt;i+=eq_tblock)
	{
		if (t2==-1) t2=GetTickCount();
		// calculate part of DO derivative in all cells
		t4=DO_Cder_CUDA(1,i+eq_tblock,eq_tau,eq_Na,0,eq_C,eq_Nx,((i==0)?i:i-1),i,0,eq_w,eq_Nt); // results in bDOF (corresponds to eq_DF
		if (t3==-1) t3=t4;
		// solve for each time step on a block
		for (int j=((i==0)?1:0);j<eq_tblock;j++)
		{
			// solve linear equation system
			eq_one_step_cuda(i+j);
			// change eq_DF
			if (j!=(eq_tblock-1))
				DO_Cder_CUDA(0,i+j+2,eq_tau,eq_Na,1,eq_C,eq_Nx,i+j,i+j+1,i-1,eq_w,eq_Nt);
		}
	}
	// get results and check on a final t step
	queue->EnqueueBuffer(bC, eq_C, 0,  eq_Nx*eq_Nt*sizeof(T));
	queue->Finish();
	t4=GetTickCount();
	double err=0.0;
	int i=eq_Nt-1;
	for (int jj=0;jj<eq_Nx;jj++)
	{
		double a=eq_anal_u(i*eq_tau,jj*eq_h);
		double diff=(eq_C[jj*eq_Nt+i]-a);
		err+=diff*diff;
		if (total) printf("t %g x %g N %g A %g\n",i*eq_tau,jj*eq_h,eq_C[jj*eq_Nt+i],a);
	}
	err=sqrt(err/eq_Nx);
	printf("GPU t %g err %g time %d time anal %d\n",i*eq_tau,err,t2-t1+t4-t3,GetTickCount()-t4);
}
#endif
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    T *dF0,*dF05,*dF1,*DOF;
    int N=1000;
    int aN=32;
    int total=1; // output for each time step
    int mode=0; // cuda mode: 0 - simple, 1 - with tensor cores usage
    if (argc>=2) N=atoi(argv[1]); 
    if (argc>=3) aN=atoi(argv[2]);
    if (argc>=4) total=atoi(argv[3]);
    if (argc>=5) mode=atoi(argv[4]);
    // test equation solution
    if (N<0)
    {
	if (argc>=3) eq_Nx=atoi(argv[2]);
	if (argc>=4) eq_Na=atoi(argv[3]);
	if (argc>=5) eq_tblock=atoi(argv[4]);
	if (argc>=6) eq_eNt=atoi(argv[5]);
	if (argc>=7) total=atoi(argv[6]);
	if (argc>=8) eq_ex=atoi(argv[7]);
	if (argc>=9) eq_ex1_power=atof(argv[8]);
	if (argc>=10) eq_ex1_err=atof(argv[9]);
	if (argc>=11) int_max_iter=atoi(argv[10]);
	if (N==-1) eq_solve(total);
#ifdef OCL
	if (N==-2) eq_solve_cuda(total);
#endif
	return 0;
    }
    printf("N %d aN %d\n",N,aN);
    // test derivative computation
    T tt=3.0;
    T tau=tt/(N-1);
    dF0=new T[N];
    dF05=new T[N];
    dF1=new T[N];
    DOF=new T[N];
    Cder(N,tau,0.0);
    memcpy(dF0,DF,N*sizeof(T));
    Cder(N,tau,0.5);
    memcpy(dF05,DF,N*sizeof(T));
    Cder(N,tau,1.0);
    memcpy(dF1,DF,N*sizeof(T));
    DO_Cder(N,tau,aN);
    memcpy(DOF,DF,N*sizeof(T));
#ifdef OCL
    DO_Cder_CUDA(mode,N,tau,aN);
#endif
    if (total==1)
	for (int i=1;i<N;i++)
	    printf("t %g F %g dF0 %g dF05 %g dF1 %g df %g DO %g DOCUDA %g DOanal %g\n",i*tau,f(i*tau),dF0[i],dF05[i],dF1[i],df(i*tau),DOF[i],DF[i],analytic(i*tau));
    double err1=0.0,err2=0.0;	    
    for (int i=1;i<N;i++)
    {
	err1+=(analytic(i*tau)-DOF[i])*(analytic(i*tau)-DOF[i]);
	err2+=(analytic(i*tau)-DF[i])*(analytic(i*tau)-DF[i]);
    }
    err1=sqrt(err1/N);
    err2=sqrt(err2/N);
    printf("DOerr %g DOCUDAerr %g\n",err1,err2);
    return 0;
}
