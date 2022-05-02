/* Author: Vsevolod Bohaienko */
/* basic opencl/cuda operations */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <string>
#include <map>
#include "opencl_cuda_class.h"
int opencl_debug=0;
#define SWARN(s) {printf("%s\n",s);}
#define SWARN_INT(s,i) {printf("%s %d\n",s,i);}
#define SERROR(s) {printf("%s\n",s);exit(0);}
#define SERROR_INT(s,i) {printf("%s %d\n",s,i);exit(0);}
#define SWARN_HEX(s,i) {printf("%s %x\n",s,i);}
#define SERROR_HEX(s,i) {printf("%s %x\n",s,i);exit(0);}
using namespace std;

#ifndef CUDA
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////// operncl ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
OpenCL_buffer::OpenCL_buffer(cl_context *ct,cl_mem_flags flags,size_t sz,void * mem)
	{
	   cl_int err;
	   buffer = clCreateBuffer(ct[0],flags,size=sz,mem,&err);
	   if (err!=CL_SUCCESS) SERROR_INT("OpenCL failed to create buffer",err);
		if (opencl_debug)
	    {
		char str[1024];
		sprintf(str,"OpenCL buffer created: size %d, flags %d",(int)sz,(int)flags);
		SWARN(str);
	    }
	}
OpenCL_buffer::~OpenCL_buffer()
	{
	   clReleaseMemObject(buffer); 
	}
//////////////////////////////////////////////////////////
/////////////// hash func  //////////////////////////////
/*
  Name  : CRC-32
  Poly  : 0x04C11DB7    x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 
                       + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
  Init  : 0xFFFFFFFF
  Revert: true
  XorOut: 0xFFFFFFFF
  Check : 0xCBF43926 ("123456789")
  MaxLen: 268 435 455 байт (2 147 483 647 бит) - обнаружение
   одинарных, двойных, пакетных и всех нечетных ошибок
*/
unsigned int Crc32(const unsigned char *buf, size_t len)
{
    unsigned int crc_table[256];
    unsigned int crc; int i, j;
 
    for (i = 0; i < 256; i++)
    {
        crc = i;
        for (j = 0; j < 8; j++)
            crc = crc & 1 ? (crc >> 1) ^ 0xEDB88320UL : crc >> 1;
 
        crc_table[i] = crc;
    };
 
    crc = 0xFFFFFFFFUL;
 
    while (len--) 
        crc = crc_table[(crc ^ *buf++) & 0xFF] ^ (crc >> 8);
 
    return crc ^ 0xFFFFFFFFUL;
}
OpenCL_prg::OpenCL_prg(cl_context *ct,cl_device_id device_id, const char *source)
{
		int err;
		char *buffer;
		char *buffer2;
		size_t len;
	    hContext=ct;
		// try to find cached binary source
		{
			FILE *fi=fopen("res/opencl_cache","rb");
			if (fi)
			{
				char dev_name[1024];
				size_t l;
				int crc;
				int rcrc,rs;
				char *data=NULL;
				// form string to compare
				string s=string(source);
				clGetDeviceInfo(device_id,CL_DEVICE_NAME,1024,dev_name,&l);
				s+=string(dev_name);
				// get crc
				crc=Crc32((const unsigned char *)s.c_str(),s.length());
				// file format - hash-size-data
				// read file, search for crc
				while(fread(&rcrc,1,sizeof(int),fi))
				{
					fread(&rs,1,sizeof(int),fi);
					if (rcrc==crc)  // try to create program from binary source
					{
						data=new char[rs];
						size_t srs=rs;
						fread(data,1,rs,fi);
						hProgram = clCreateProgramWithBinary(hContext[0],1,&device_id,&srs,(const unsigned char **)&data,NULL,&err);
						if (!hProgram || err != CL_SUCCESS)
							goto end;
						err = clBuildProgram(hProgram, 0, NULL, NULL, NULL, NULL);
						if (err != CL_SUCCESS)
						{
							clReleaseProgram(hProgram);
							goto end;
						}
						delete [] data;
						fclose(fi);
						return;
					}
					else
						fseek(fi,rs,SEEK_CUR);
				}
end:
				if (data) delete [] data;
				fclose(fi);
			}
		}
		hProgram = clCreateProgramWithSource(hContext[0], 1, (const char **) &source, NULL, &err);
		if (!hProgram || err != CL_SUCCESS) SERROR_INT("OpenCL failed to Create program with source",err);
		err = clBuildProgram(hProgram, 0, NULL, NULL, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			clGetProgramBuildInfo(hProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			buffer=new char[len+1];
			buffer2=new char[len+1024];
			clGetProgramBuildInfo(hProgram, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
			sprintf(buffer2,"OpenCL failed to build program executable: %s\n", buffer);
			SERROR(buffer2);
		}
		// write binary to cache file
		FILE *fi=fopen("res/opencl_cache","ab");
		if (fi || opencl_debug)
		{
			char dev_name[1024];
			size_t l;
			int crc;
			int ndevs;
			size_t *rs;
			char **data;
			// form string to compare
			string s = string(source);
			clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, dev_name, &l);
			s += string(dev_name);
			// get crc
			crc = Crc32((const unsigned char *)s.c_str(), s.length());
			// write data
			if (fi) fwrite(&crc, 1, sizeof(int), fi);
			// get binary
			clGetProgramInfo(hProgram, CL_PROGRAM_NUM_DEVICES, sizeof(int), &ndevs, NULL);
			rs=new size_t[ndevs];
			data=new char*[ndevs];
			clGetProgramInfo(hProgram, CL_PROGRAM_BINARY_SIZES, ndevs*sizeof(size_t), rs, NULL);
			for (int i=0;i<ndevs;i++)
			    data[i] = new char[rs[i]];
			clGetProgramInfo(hProgram, CL_PROGRAM_BINARIES, ndevs*sizeof(char *), data, NULL);
			// write binary
			int r=rs[0];
			if (fi) fwrite(&r, 1, sizeof(int), fi);
			if (fi) fwrite(data[0], 1, rs[0], fi);
			if (opencl_debug)
			    SWARN(data[0]);
			delete [] rs;
			for (int i=0;i<ndevs;i++) delete [] data[i];
			delete[] data;
			if (fi)fclose(fi);
		}
}	
OpenCL_prg::~OpenCL_prg()
{
		clReleaseProgram(hProgram);
}
OpenCL_kernel::OpenCL_kernel(cl_context* ct, OpenCL_prg *hprog,const char *name)
	{
		int err;
		hProgram=&hprog->hProgram;
		hContext=ct;
		// create kernel
	    hKernel = clCreateKernel(hProgram[0], name, &err);
		if (!hKernel || err != CL_SUCCESS)	SERROR_INT("OpenCL failed to create kernel",err);
		if (opencl_debug)
		{
			nm=new char[strlen(name)+1];
			strcpy(nm,name);
			{
			char str[1024];
			sprintf(str,"OpenCL kernel created: name %s",name);
			SWARN(str);
			}
		}
	}
OpenCL_kernel::~OpenCL_kernel()
	{
	    clReleaseKernel(hKernel);
 	}
cl_int OpenCL_kernel::SetBufferArg(OpenCL_buffer *buf,int idx)
	{
		return clSetKernelArg(hKernel, idx, sizeof(cl_mem), (void *)&buf->buffer);
	}
cl_int OpenCL_kernel::SetArg(int idx,int size,void *buf)
	{
		return clSetKernelArg(hKernel, idx, size, buf);
	}
OpenCL_commandqueue::OpenCL_commandqueue(cl_context *ctxt,cl_device_id did,cl_command_queue_properties pr)
	{
	   cl_int err;
	   hContext=ctxt;
	   device_id=did;
	   props=pr;
	   if (opencl_debug) props|=CL_QUEUE_PROFILING_ENABLE;
	   hCmdQueue = clCreateCommandQueue(hContext[0], device_id, props, &err);
	   if (err!=CL_SUCCESS) SERROR_INT("OpenCL clCreateCommandQueue failed",err);
	}
OpenCL_commandqueue::~OpenCL_commandqueue()
	{
	   clReleaseCommandQueue(hCmdQueue);
	}
cl_event OpenCL_commandqueue::ExecuteKernel(OpenCL_kernel *krnl,int ndims,size_t *dims,size_t *ldims,int nwaitkernels,cl_event *evs,int noevent)
	{
		int err,local_ldims=0,i;
		size_t nthr;
		static size_t local=16;
		static int got_local=0;
		cl_event e,*ep=&e;
		static size_t *alloced_ldims=NULL;
		static int alloced_ldims_size=0;
		// check for 0 dims
		for (int i=0;i<ndims;i++)
			if (dims[i]==0) return e;
		if (ldims==NULL)
		{
		    if (alloced_ldims_size<ndims)
		    {
		      if (alloced_ldims) delete [] alloced_ldims;
		      alloced_ldims_size=ndims;
		      alloced_ldims=new size_t[ndims];
		    }
	            ldims=alloced_ldims;
	  	    local_ldims=1;
		}
		// Get the maximum work-group size for executing the kernel on the device
		if (got_local==0)
		{
    		    err = clGetKernelWorkGroupInfo(krnl->hKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);		    
		    if (err != CL_SUCCESS)	SERROR_INT("OpenCL clGetKernelWorkGroupInfo Failed",err);
		    got_local=1;
		}
		// adjust work-group size
		if (local_ldims==1)
		{
			int j=0;
			// initial set and adjust dims%ldims
			for (i=0;i<ndims;i++) ldims[i]=((dims[i]<local)?dims[i]:local);
			for (i=0;i<ndims;i++) 
			  while (dims[i]%ldims[i]) ldims[i]--;
			// adjust nthreads
			nthr=1;
			for (i=0;i<ndims;i++) nthr*=ldims[i];
			if (nthr>local)
			{
				do
				{
					if (ldims[j]<2) j=((j+1)%ndims);
					ldims[j]--;
					while (dims[j]%ldims[j]) ldims[j]--;
					j=((j+1)%ndims);
					nthr=1;
					for (i=0;i<ndims;i++) nthr*=ldims[i];
				}
				while (nthr>local);
			}
		}
		// execute kernel
		{
			// clear nulls from event list
			int newevssize=nwaitkernels;
			cl_event *evs0=evs;
			for (i=0;i<nwaitkernels;i++)
				if (evs[i]==NULL)
					newevssize--;
			int j=0;
			for (i=0;i<newevssize;i++)
			{
				while (evs[j]==NULL) j++;
				evs[i]=evs[j];
				j++;
			}
			if (newevssize==0) evs0=NULL;
			if (noevent) ep=NULL;
			if (opencl_debug)
			{
				char str[1024];
				sprintf(str,"OpenCL kernel is scheduled for execution (waiting %d events)",newevssize);
	    		SWARN(str);
	    		for (i=0;i<ndims;i++)
	    		{
	    		  sprintf(str,"dim %d - (%d,%d)",i,dims[i],ldims[i]);
	    		  SWARN(str);
	    		}
			}
			err = clEnqueueNDRangeKernel(hCmdQueue, krnl->hKernel, ndims, NULL, dims, ldims, newevssize, evs0, ep);
			if (opencl_debug)
			{
				ReleaseEvent(e);
				Finish();
			}
		}
		if (err != CL_SUCCESS) SERROR_INT("OpenCL clEnqueueNDRangeKernel Failed",err);
		return e;
	}
void OpenCL_commandqueue::ReleaseEvent(cl_event e)
{
	clWaitForEvents(1,&e);
	if (props&CL_QUEUE_PROFILING_ENABLE) // print profiling info
	{
		cl_int err;
		cl_ulong info[4];
		err=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&info[0],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong),&info[1],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&info[2],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&info[3],NULL);
		if (err!=CL_SUCCESS) SERROR_INT("OpenCL cGetEventProfilingInfo failed",err);
		{
			char str[4096];
			sprintf(str,"OpenCL profiling info for event %llu - (%llu,%llu,%llu,%llu) exec time - %llu nanoseconds",(cl_ulong)(e),info[0],info[1],info[2],info[3],info[3]-info[2]);
			SWARN(str);
		}
	}
	clReleaseEvent(e);
}
cl_int OpenCL_commandqueue::EnqueueBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		if (size==0) size=b->size;
		return clEnqueueReadBuffer(hCmdQueue, b->buffer, CL_TRUE, offset, size, mem, 0, NULL, NULL);
	}
cl_int OpenCL_commandqueue::EnqueueWriteBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		if (size==0) size=b->size;
		return clEnqueueWriteBuffer(hCmdQueue, b->buffer, CL_TRUE, offset, size, mem, 0, NULL, NULL);
	}
void OpenCL_commandqueue::Finish()
	{
	   clFinish(hCmdQueue);
	}
OpenCL_program::OpenCL_program(int gpu)
   {
	    int err;
		char buffer[2048];
		char buffer2[2048];
		size_t len;
		cl_platform_id *pids;
		cl_uint np,cp;		
		err = clGetPlatformIDs( 1, &platform, &np );
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get platform IDs",err);
		pids=new cl_platform_id[np];
		err = clGetPlatformIDs( np, pids, NULL );
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get platform IDs",err);
		cp=0;
a10:
		platform=pids[cp++];
		err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
		if (err != CL_SUCCESS)
		{
			if (gpu==1)
			{
				if (cp==np)
				{
				    gpu=0;
				    cp=0;
				}
				goto a10;
			}			
			SERROR_INT("OpenCL failed to get number of devices",err);
		}
		device_ids=new cl_device_id[n_devices];
		err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, n_devices, device_ids, NULL);
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get device ID",err);
		for (int i=0;i<n_devices;i++)
		{
			err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(buffer), buffer, &len);
			sprintf(buffer2,"CL_DEVICE_NAME: %s\n", buffer);
			SWARN(buffer2);
			err = clGetDeviceInfo(device_ids[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, &len);
			sprintf(buffer2,"CL_DEVICE_VENDOR: %s\n", buffer);
			SWARN(buffer2);
		}
		// create OpenCL device & context
		hContext = clCreateContext(0, n_devices, device_ids, NULL, NULL, &err);
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to create context",err);
		// initialize vectors
		buffers=new vector<OpenCL_buffer*>[1];
		kernels=new vector<OpenCL_kernel*>[1];
		queues=new vector<OpenCL_commandqueue*>[1];
		progs=new vector<OpenCL_prg *>[1];
   }
int OpenCL_program::get_ndevices()
{
	return (int)n_devices;
}
cl_device_type OpenCL_program::get_device_type(int d)
{
	cl_device_type ret;
	cl_int err=clGetDeviceInfo(device_ids[d],CL_DEVICE_TYPE,sizeof(cl_device_type),&ret,0);
	if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get device type",err);
	return ret;
}
OpenCL_program::~OpenCL_program()
   {
	   int i;
	   for (i=0;i<((vector<OpenCL_buffer*>*)buffers)->size();i++) delete ((vector<OpenCL_buffer*>*)buffers)->operator[](i);
	   for (i=0;i<((vector<OpenCL_prg*>*)progs)->size();i++) delete ((vector<OpenCL_prg*>*)progs)->operator[](i);
	   for (i=0;i<((vector<OpenCL_kernel*>*)kernels)->size();i++) delete ((vector<OpenCL_kernel*>*)kernels)->operator[](i);
	   for (i=0;i<((vector<OpenCL_commandqueue*>*)queues)->size();i++) delete ((vector<OpenCL_commandqueue*>*)queues)->operator[](i);
	   ((vector<OpenCL_commandqueue*>*)queues)->clear();
	   ((vector<OpenCL_buffer*>*)buffers)->clear();
	   ((vector<OpenCL_prg*>*)progs)->clear();
	   ((vector<OpenCL_kernel*>*)kernels)->clear();
	   delete [] ((vector<OpenCL_commandqueue*>*)queues);
	   delete [] ((vector<OpenCL_buffer*>*)buffers);
	   delete [] ((vector<OpenCL_prg*>*)progs);
	   delete [] ((vector<OpenCL_kernel*>*)kernels);
	   clReleaseContext(hContext);
   }
OpenCL_buffer *OpenCL_program::create_buffer(cl_mem_flags flags,size_t sz,void * mem)
   {
	   OpenCL_buffer *buf=new OpenCL_buffer(&hContext,flags,sz,mem);
	  ((vector<OpenCL_buffer*>*)buffers)->push_back(buf);
	   return buf;
   }
OpenCL_prg *OpenCL_program::create_program(const char *src)
   {
	   OpenCL_prg *prg=new OpenCL_prg(&hContext,device_ids[0],src);
	  ((vector<OpenCL_prg*>*)progs)->push_back(prg);
	   return prg;
   }
OpenCL_kernel *OpenCL_program::create_kernel(OpenCL_prg *prog,const char *name)
   {
	   OpenCL_kernel *krnl=new OpenCL_kernel(&hContext,prog,name);
	   ((vector<OpenCL_kernel*>*)kernels)->push_back(krnl);
	   return krnl;
   }
OpenCL_commandqueue *OpenCL_program::create_queue(int device,cl_command_queue_properties props)
   {
	   OpenCL_commandqueue *queue=new OpenCL_commandqueue(&hContext,device_ids[device],props);
	   ((vector<OpenCL_commandqueue*>*)queues)->push_back(queue);
	   return queue;
   }
#else
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////// cuda ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
#include <time.h>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
OpenCL_buffer::OpenCL_buffer(cl_context *ct,cl_mem_flags flags,size_t sz,void * mem)
	{
	   cudaError_t err=cudaMalloc((void **)&buffer,size=sz);
	   if (err!=cudaSuccess) SERROR_INT("CUDA - failed to create buffer",(int)err);
	   if (mem)
	     if ((err=cudaMemcpy((void *)buffer,mem,sz,cudaMemcpyHostToDevice))!=cudaSuccess)
	        SERROR_INT("CUDA - failed copying data to device while creating buffer",(int)err);
	    if (opencl_debug)
	    {
		char str[1024];
		sprintf(str,"CUDA - buffer created: size %d - %lx (%lx)",(int)sz,(long int)((void *)buffer),(long int)mem);
		SWARN(str);
	    }
	}
OpenCL_buffer::~OpenCL_buffer()
	{
	    int err;
	    if (opencl_debug)
	    {
		char str[1024];
		sprintf(str,"CUDA - freeing buffer %lx",(long int)((void *)buffer));
		SWARN(str);
	    }
	    if ((err=cudaFree((void *)buffer))!=cudaSuccess) SERROR_INT("CUDA - failed freeing buffer",err);
	}
OpenCL_prg::OpenCL_prg(cl_context *ct,cl_device_id device_id, const char *source) // no caching of compiled code for CUDA , only one program can exist
{
	// Create an instance of nvrtcProgram
	nvrtcProgram prog;
	int err;
	if ((err=nvrtcCreateProgram(&prog, source, "function", 0, NULL, NULL))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error creating program",err);
	// Compile the program 
	const char *opts[] = {"--gpu-architecture=compute_70","-rdc=true","-I/usr/local/cuda/include","--fmad=true" };
	nvrtcResult compileResult = nvrtcCompileProgram(prog, 4, opts);
	// Obtain compilation log from the program.
	size_t logSize;
	if ((err=nvrtcGetProgramLogSize(prog, &logSize))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error getting compilation log size",err);
	if (logSize > 1)
	{
		char *log=new char[logSize];
		if ((err=nvrtcGetProgramLog(prog, log))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error getting compilation log size",err);
		SWARN(log);
		delete [] log;
	}
	if (compileResult != NVRTC_SUCCESS)
		exit(1);
	// Obtain PTX from the program.
	size_t ptxSize;
	if ((err=nvrtcGetPTXSize(prog, &ptxSize))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error getting PTX size",err);
	char *ptx = new char[ptxSize];
	if ((err=nvrtcGetPTX(prog, ptx))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error getting PTX",err);
	// Destroy the program.
	if ((err=nvrtcDestroyProgram(&prog))!=NVRTC_SUCCESS) SERROR_INT("CUDA - error destroying program",err);
	// Load precompiled relocatable source with call to external function
	// and link it together with NVRTC-compiled function.
	CUlinkState linker;
	if((err=cuLinkCreate(0, NULL, NULL, &linker))!=CUDA_SUCCESS) SERROR_INT("CUDA - error creating linker",err);
	if((err=cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void*)ptx, ptxSize, "function.ptx", 0, NULL, NULL))!=CUDA_SUCCESS) SERROR_INT("CUDA - error starting linking",err);
	void* cubin;
	if((err=cuLinkComplete(linker, &cubin, NULL))!=CUDA_SUCCESS) SERROR_INT("CUDA - error finishing linking",err);
	if((err=cuModuleLoadDataEx((CUmodule *)ct, cubin, 0, NULL, NULL))!=CUDA_SUCCESS) SERROR_INT("CUDA - error loading data by linker",err);
	if((err=cuLinkDestroy(linker))!=CUDA_SUCCESS) SERROR_INT("CUDA - error destroying linker",err);
	if (opencl_debug) 
	{
	    //SWARN(ptx);
	    SWARN_INT("CUDA - program compiled successfully - ptx size",(int)ptxSize);
	    SWARN_HEX("CUDA - program compiled successfully - module",(long long int)((void *)ct[0]));
	}
}	
OpenCL_prg::~OpenCL_prg()
{
}
struct KernelConfig
{
	std::map<int,std::pair<int,void *> > args;
	std::vector<char> arguments;
	CUfunction kernel;
	int max_idx;
};
OpenCL_kernel::OpenCL_kernel(cl_context* ct, OpenCL_prg *hprog,const char *name)
	{
		int err;
		struct KernelConfig *kk=new struct KernelConfig[1];
		hKernel=(cl_kernel)kk;
		kk->max_idx=0;
		if ((err=cuModuleGetFunction(&kk->kernel, (CUmodule)((void *)ct[0]), name))!=CUDA_SUCCESS)
		{
		     SWARN_INT("CUDA - error creating kernel - error",err);
		     SWARN(name);
		     SERROR_HEX("CUDA - error creating kernel - module",(long long int)((void *)ct[0]));
		}
		if (opencl_debug)
		{
			nm=new char[strlen(name)+1];
			strcpy(nm,name);
			{
			char str[1024];
			sprintf(str,"CUDA - kernel created: name %s",name);
			SWARN(str);
			}
		}
	}
OpenCL_kernel::~OpenCL_kernel()
	{
 	}
cl_int OpenCL_kernel::SetArg(int idx,int size,void *buf)
	{
		char *bb;
		bb=new char[size];
		memcpy(bb,buf,size);
		struct KernelConfig *kk=(struct KernelConfig *) hKernel;
		std::map<int,std::pair<int,void *> >::iterator i;
		if ((i=kk->args.find(idx))!=kk->args.end())
		{
			(*i).second.first=size;
			delete [] (char *)((*i).second.second);
			(*i).second.second=(void *)bb;
		}
		else
			kk->args.insert(std::pair<int,std::pair<int,void *> >(idx,std::pair<int,void *>(size,bb)));
		if (idx>kk->max_idx) kk->max_idx=idx;
		return 0;
	}
cl_int OpenCL_kernel::SetBufferArg(OpenCL_buffer *buf,int idx)
	{
		return SetArg(idx,sizeof(void *),(void *)&buf->buffer);
	}
OpenCL_commandqueue::OpenCL_commandqueue(cl_context *ctxt,cl_device_id did,cl_command_queue_properties pr)
	{
	}
OpenCL_commandqueue::~OpenCL_commandqueue()
	{
	}
cl_event OpenCL_commandqueue::ExecuteKernel(OpenCL_kernel *krnl,int ndims,size_t *dims,size_t *ldims,int nwaitkernels,cl_event *evs,int noevent) // no waiting for other kernels, synchronious
	{
		int local_ldims=0,i;
		size_t nthr;
		static size_t local=16;
		cl_event e;
		static size_t *alloced_ldims=NULL;
		static int alloced_ldims_size=0;
		// check for 0 dims
		for (int i=0;i<ndims;i++)
			if (dims[i]==0) return e;
		if (ldims==NULL)
		{
		    if (alloced_ldims_size<ndims)
		    {
		      if (alloced_ldims) delete [] alloced_ldims;
		      alloced_ldims_size=ndims;
		      alloced_ldims=new size_t[ndims];
		    }
	            ldims=alloced_ldims;
	  	    local_ldims=1;
		}
		// adjust work-group size
		if (local_ldims==1)
		{
			int j=0;
			// initial set and adjust dims%ldims
			for (i=0;i<ndims;i++) ldims[i]=((dims[i]<local)?dims[i]:local);
			for (i=0;i<ndims;i++) 
			  while (dims[i]%ldims[i]) ldims[i]--;
			// adjust nthreads
			nthr=1;
			for (i=0;i<ndims;i++) nthr*=ldims[i];
			if (nthr>local)
			{
				do
				{
					if (ldims[j]<2) j=((j+1)%ndims);
					ldims[j]--;
					while (dims[j]%ldims[j]) ldims[j]--;
					j=((j+1)%ndims);
					nthr=1;
					for (i=0;i<ndims;i++) nthr*=ldims[i];
				}
				while (nthr>local);
			}
		}
		// execute kernel
		struct KernelConfig *kk=(struct KernelConfig *) krnl->hKernel;
		int gd[3]={1,1,1},sd[3]={1,1,1};
		if (ndims>=1) {gd[0]=dims[0]/ldims[0];sd[0]=ldims[0];}
		if (ndims>=2) {gd[1]=dims[1]/ldims[1];sd[1]=ldims[1];}
		if (ndims>=3) {gd[2]=dims[2]/ldims[2];sd[2]=ldims[2];}
		if (ndims>=4) SERROR("CUDA - ndims>3 is not supported");
		if (opencl_debug)
		{
		    char str[1024];
		    sprintf(str,"CUDA - executing kernel (%d %d %d)-(%d %d %d)",gd[0],gd[1],gd[2],sd[0],sd[1],sd[2]);
		    SWARN(str);
		    sprintf(str,"CUDA - executing kernel - nargs %d",kk->max_idx);
		    SWARN(str);
		    for (int i=0;i<=kk->max_idx;i++)
		    {
			int size=0;
			void *arg=NULL;
			std::map<int,std::pair<int,void *> >::iterator ii;
			if ((ii=kk->args.find(i))!=kk->args.end())
			{
				size=(*ii).second.first;
				arg=(*ii).second.second;
				sprintf(str,"CUDA - executing kernel - args %d - (%d %x)",i,size,((int*)arg)[0]);
				SWARN(str);
			}
		    }
		}
		// convert arguments map to char vector
		int offset=0;
		kk->arguments.clear();
		for (int i=0;i<=kk->max_idx;i++)
		{
			int size=0;
			void *arg=NULL;
			std::map<int,std::pair<int,void *> >::iterator ii;
			if ((ii=kk->args.find(i))!=kk->args.end())
			{
				size_t currSize = kk->arguments.size();
				size=(*ii).second.first;
				arg=(*ii).second.second;
				if (currSize < offset + size)
					kk->arguments.resize(offset + size);
				memcpy(&kk->arguments[0] + offset, arg, size);
			}
			else
				SERROR("CUDA - kernel argument is not set");
			if (size<8) size=8;
			offset+=size;
		}
		size_t szarguments = kk->arguments.size();
		void *config[] = {
			CU_LAUNCH_PARAM_BUFFER_POINTER, &kk->arguments[0],
			CU_LAUNCH_PARAM_BUFFER_SIZE, &szarguments,
			CU_LAUNCH_PARAM_END
		};
		if (opencl_debug)
		    SWARN_INT("CUDA - executing kernel - arg size",(int)szarguments);
		struct timespec t1, t2; 
		clock_gettime(CLOCK_REALTIME, &t1); 
		CUresult err=cuLaunchKernel(kk->kernel,gd[0],gd[1],gd[2],sd[0],sd[1],sd[2], 1024*16, 0, NULL, config);
		if (err != CUDA_SUCCESS) SERROR_INT("CUDA - launch kernel failed",(int)err);
		int e1;
		if ((e1=cudaDeviceSynchronize())!=CUDA_SUCCESS) SERROR_INT("CUDA - error executing kernel",(int)e1);
                clock_gettime(CLOCK_REALTIME, &t2); 
		if (opencl_debug)		
			SWARN_INT("CUDA - kernel completed - time (nanoseconds) - ",(int)(t2.tv_nsec-t1.tv_nsec));
		return e;
	}
void OpenCL_commandqueue::ReleaseEvent(cl_event e)
{
}
cl_int OpenCL_commandqueue::EnqueueBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		int err;
		if (size==0) size=b->size;
		if (opencl_debug)
		{
		    char str[1024];
		    sprintf(str,"CUDA - reading buffer %lx+%d->%lx (%d)",(long int)((char *)b->buffer),offset,(long int)mem,size);
		    SWARN(str);
		}
		Finish();
		err=cudaMemcpy(mem,((char *)b->buffer)+offset,size,cudaMemcpyDeviceToHost);
		if (err!=cudaSuccess)
			SERROR_INT("CUDA - failed copying data to host",(int)err);
		Finish();
	    	return err;
	}
cl_int OpenCL_commandqueue::EnqueueWriteBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		int err;
		if (size==0) size=b->size;
		if (opencl_debug)
		{
		    char str[1024];
		    sprintf(str,"CUDA - writing buffer %lx+%d<-%lx (%d)",(long int)((char *)b->buffer),offset,(long int)mem,size);
		    SWARN(str);
		}
		Finish();
		err=cudaMemcpy(((char *)b->buffer)+offset,mem,size,cudaMemcpyHostToDevice);
		    if (err!=cudaSuccess)
			SERROR_INT("CUDA - failed copying data to device",(int)err);
		Finish();
	    	return err;
	}
void OpenCL_commandqueue::Finish()
	{
	    int err;
	    if ((err=cudaDeviceSynchronize())!=CUDA_SUCCESS) SERROR_INT("CUDA - error while finish",(int)err);
	}
OpenCL_program::OpenCL_program(int gpu)
   {
		CUdevice cuDevice;
		CUcontext cuContext;
		int err;
		// Initialize CUDA driver.
		if ((err=cuInit(0))!= CUDA_SUCCESS) SERROR_INT("CUDA - initialization error",err);
		if ((err=cuDeviceGet(&cuDevice, 0))!= CUDA_SUCCESS) SERROR_INT("CUDA - get device error",err);
		if ((err=cuCtxCreate(&cuContext, 0, cuDevice))!= CUDA_SUCCESS) SERROR_INT("CUDA - create context error",err);
		hContext=(cl_context )new void *[1];
		// initialize vectors
		buffers=new vector<OpenCL_buffer*>[1];
		kernels=new vector<OpenCL_kernel*>[1];
		queues=new vector<OpenCL_commandqueue*>[1];
		progs=new vector<OpenCL_prg *>[1];
		device_ids=new cl_device_id[1];
   }
int OpenCL_program::get_ndevices()
{
	return 1;
}
cl_device_type OpenCL_program::get_device_type(int d)
{
	return CL_DEVICE_TYPE_GPU;
}
OpenCL_program::~OpenCL_program()
   {
	   int i;
	   for (i=0;i<((vector<OpenCL_buffer*>*)buffers)->size();i++) delete ((vector<OpenCL_buffer*>*)buffers)->operator[](i);
	   for (i=0;i<((vector<OpenCL_prg*>*)progs)->size();i++) delete ((vector<OpenCL_prg*>*)progs)->operator[](i);
	   for (i=0;i<((vector<OpenCL_kernel*>*)kernels)->size();i++) delete ((vector<OpenCL_kernel*>*)kernels)->operator[](i);
	   for (i=0;i<((vector<OpenCL_commandqueue*>*)queues)->size();i++) delete ((vector<OpenCL_commandqueue*>*)queues)->operator[](i);
	   ((vector<OpenCL_commandqueue*>*)queues)->clear();
	   ((vector<OpenCL_buffer*>*)buffers)->clear();
	   ((vector<OpenCL_prg*>*)progs)->clear();
	   ((vector<OpenCL_kernel*>*)kernels)->clear();
	   delete [] ((vector<OpenCL_commandqueue*>*)queues);
	   delete [] ((vector<OpenCL_buffer*>*)buffers);
	   delete [] ((vector<OpenCL_prg*>*)progs);
	   delete [] ((vector<OpenCL_kernel*>*)kernels);
   }
OpenCL_buffer *OpenCL_program::create_buffer(cl_mem_flags flags,size_t sz,void * mem)
   {
	   OpenCL_buffer *buf=new OpenCL_buffer(&hContext,flags,sz,mem);
	  ((vector<OpenCL_buffer*>*)buffers)->push_back(buf);
	   return buf;
   }
OpenCL_prg *OpenCL_program::create_program(const char *src)
   {
	   OpenCL_prg *prg=new OpenCL_prg(&hContext,device_ids[0],src);
	  ((vector<OpenCL_prg*>*)progs)->push_back(prg);
	   return prg;
   }
OpenCL_kernel *OpenCL_program::create_kernel(OpenCL_prg *prog,const char *name)
   {
	   OpenCL_kernel *krnl=new OpenCL_kernel(&hContext,prog,name);
	   ((vector<OpenCL_kernel*>*)kernels)->push_back(krnl);
	   return krnl;
   }
OpenCL_commandqueue *OpenCL_program::create_queue(int device,cl_command_queue_properties props)
   {
	   OpenCL_commandqueue *queue=new OpenCL_commandqueue(&hContext,device_ids[device],props);
	   ((vector<OpenCL_commandqueue*>*)queues)->push_back(queue);
	   return queue;
   }
#endif
