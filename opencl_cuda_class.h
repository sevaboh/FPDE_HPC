/* Author: Vsevolod Bohaienko */
/* basic opencl/cuda operations */
#include <CL/opencl.h>
class  OpenCL_buffer
{
public:
    cl_mem buffer;
	int size;
	OpenCL_buffer(cl_context *ct,cl_mem_flags flags,size_t sz,void * mem);
	~OpenCL_buffer();
};
class  OpenCL_prg
{
public:
    cl_program hProgram;
    cl_context *hContext;
    OpenCL_prg(cl_context *ct,cl_device_id did,const char *source);
    ~OpenCL_prg();
};
class  OpenCL_kernel
{
public:
    char *nm;
    cl_program *hProgram;
    cl_kernel hKernel;
    cl_context *hContext;
    cl_device_id device_id;
	OpenCL_kernel(cl_context* ct, OpenCL_prg *hprog,const char *name);
	~OpenCL_kernel();
	cl_int SetBufferArg(OpenCL_buffer *buf,int idx);
	cl_int SetArg(int idx,int size,void *buf);
};
class  OpenCL_commandqueue
{
   cl_device_id device_id;
   cl_context *hContext;
   cl_command_queue_properties props;
public:
	cl_command_queue hCmdQueue;
	OpenCL_commandqueue(cl_context *ctxt,cl_device_id did,cl_command_queue_properties props=0);
	~OpenCL_commandqueue();
	cl_event ExecuteKernel(OpenCL_kernel *krnl,int ndims,size_t *dims,size_t *ldims=NULL,int nwaitkernels=0,cl_event *evs=0,int noevent=0);
	cl_int EnqueueBuffer(OpenCL_buffer *b,void *mem,int offset=0,int size=0);
	cl_int EnqueueWriteBuffer(OpenCL_buffer *b,void *mem,int offset=0,int size=0);
	void ReleaseEvent(cl_event);
	void Finish();
};
class  OpenCL_program
{
   cl_platform_id platform;
   cl_device_id *device_ids;
   cl_uint n_devices;
   cl_context hContext;
   void *buffers; // pointers to vector<...>
   void *kernels;
   void *queues;
   void *progs;
public:
   OpenCL_program(int gpu);
   ~OpenCL_program();
   OpenCL_buffer *create_buffer(cl_mem_flags flags,size_t sz,void * mem);
   OpenCL_prg *create_program(const char *source);
   OpenCL_kernel *create_kernel(OpenCL_prg *hprog,const char *name);
   OpenCL_commandqueue *create_queue(int device,cl_command_queue_properties props=0);
   int get_ndevices();
   cl_device_type get_device_type(int device);
};
