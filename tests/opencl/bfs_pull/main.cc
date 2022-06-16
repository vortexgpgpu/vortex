#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>
#define v 1024 //no of vertices
#define e 32208 //no of edges
#define m 0 //source node

#define MAX_THREADS_PER_BLOCK 256
#define r v/(MAX_THREADS_PER_BLOCK)

// OpenCL variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;
static cl_int err;
cl_int status;
uint8_t *kernel_bin = NULL;

int shutdown()
{
    // release resources
    if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
    if( context ) clReleaseContext( context );
    if( device_list ) delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;

}

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
	 shutdown();			                                                     \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   shutdown();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}

int main(void){
//Reading the file
int *mat_rp = (int*)malloc( sizeof(int)*(v+1));
int *col = (int*)malloc( sizeof(int)*e);

FILE *fp;

  fp = fopen("1024.indices", "r"); // read mode
   if (fp == NULL)
   {
      printf("Error while opening the file.\n");
     }
   for(int i = 0; i< e ; i++){
fscanf(fp,"%d\n",&col[i]);
   }
   fclose(fp);
   fp = fopen("1024.off", "r"); // read mode
   if (fp == NULL)
   {
      printf("Error while opening the file.\n");
     }
  for(int i = 0; i< (v+1) ; i++){
  fscanf(fp,"%d\n",&mat_rp[i]);
}
//1. Allocate and initialize host memory
int *vec = (int*)malloc( sizeof(int)*v);
int flag;
int *maski = (int*)malloc( sizeof(int)*v);
int *disti = (int*)malloc( sizeof(int)*v);
int *active = (int*)malloc( sizeof(int)*(v));

for(int i = 0; i< v ; i++){
   disti[i]=-1;
   maski[i]=1;
   vec[i]=0;
   active[i]=0;
}

//source node values
disti[m]=0; 
maski[m]=0; 
vec[m]=1; 
int iter = 0;
//2. Get platform and device id + setup 
	  cl_platform_id platform_id;
	  cl_device_id device_id = NULL;
	  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
	  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
	
	
//3. Create context
	  //cl_context context = NULL;
	  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

//4. Create command queue
	 cmd_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &_err)); 

//5. Create memory buffers
	 //create device side buffers
    cl_mem mask, dist, mat, col_ind, vec_d, active_d, iter_d;
    
 mask = clCreateBuffer(context, CL_MEM_WRITE_ONLY, v*sizeof(int), NULL, &status);
        dist = clCreateBuffer(context, CL_MEM_READ_WRITE, v*sizeof(int), NULL, &status);
        mat = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (v+1)*sizeof(int), NULL, &status);
        col_ind =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, e*sizeof(int), NULL, &status);
        vec_d =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, v*sizeof(int), NULL, &status);
        active_d =  clCreateBuffer(context, CL_MEM_READ_WRITE, v*sizeof(int), NULL, &status);
        iter_d =  clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &status);

//6. Copy host variables to device

cl_event event;

status = clEnqueueWriteBuffer(cmd_queue, mask, CL_TRUE, 0, v*sizeof(int), maski, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, dist, CL_TRUE, 0, v*sizeof(int),  disti, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, mat, CL_TRUE, 0, (v+1)*sizeof(int),  mat_rp, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, col_ind, CL_TRUE, 0, e*sizeof(int), col, 0, NULL, &event);
	/*status = clEnqueueWriteBuffer(cmd_queue, vec, CL_TRUE, 0, v*sizeof(int), vec, 0, NULL, &event);
	status = clEnqueueWriteBuffer(cmd_queue, active, CL_TRUE, 0, v*sizeof(int), active, 0, NULL, &event);
	 status = clEnqueueWriteBuffer(cmd_queue, iter, CL_TRUE, 0, sizeof(int), iter, 0, NULL, &event);*/
	 
//7. Use the kernel function and read kernel, check if success
	// read kernel binary from file  
	size_t kernel_size;
	if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size)){
	    return -1;}

//8. Create program from kernel (with binary)
		printf("Create program from kernel source\n");
		cl_int binary_status;
		 cl_program prog = NULL;
		prog = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &_err));
	if (prog == NULL) {
	    shutdown();
	    return -1;
  }

//9. Build program and create opencl kernel
	  // Build program
	  CL_CHECK(clBuildProgram(prog, 1, &device_id, NULL, NULL, NULL));
	  cl_kernel kernel1;
	  char * kernelpr1  = "BFS";
	  kernel1 = clCreateKernel(prog, kernelpr1, &err);  
    if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 1 => %d\n", err); return -1; }

	clReleaseProgram(prog);

//10. Declare global and local worksize
        int num_of_blocks = 1;
        int num_of_threads_per_block = v;
if(v>MAX_THREADS_PER_BLOCK) {
            num_of_blocks = (int)ceil(v/(double)MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }
int work_group_size = num_of_threads_per_block;

//Variables for execution time
cl_event myEvent; 
cl_ulong startTime, endTime;
cl_ulong totaltime=0, temptime; 

//11. Set kernel args and execute opencl kernel 

do {

status = clEnqueueWriteBuffer(cmd_queue, vec_d, CL_TRUE, 0, v*sizeof(int), vec, 0, NULL, &event);
status = clEnqueueWriteBuffer(cmd_queue, active_d, CL_TRUE, 0, v*sizeof(int), active, 0, NULL, &event);
status = clEnqueueWriteBuffer(cmd_queue, iter_d, CL_TRUE, 0, sizeof(int), &iter, 0, NULL, &event);
if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: Writing variables (%d)\n", err); return -1; }
//printf("\n*********************\niter outside is %d\n*************\n", iter);

//11. a. Global and local work size
int work_items = v;
if(work_items%work_group_size != 0)	{
        work_items = work_items + (work_group_size-(work_items%work_group_size));}
    size_t local_work_size[2] = {work_group_size, 1};
    size_t global_work_size[2] = {work_items, 1};

//11. b. a. Set args kernel 
 clSetKernelArg(kernel1, 0, sizeof(mask), &mask);
 clSetKernelArg(kernel1, 1, sizeof(dist), &dist);
 clSetKernelArg(kernel1, 2, sizeof(mat), &mat);
 clSetKernelArg(kernel1, 3, sizeof(col_ind), &col_ind);
 clSetKernelArg(kernel1, 4, sizeof(vec_d), &vec_d);
 clSetKernelArg(kernel1, 5, sizeof(active_d), &active_d);
 clSetKernelArg(kernel1, 6, sizeof(int), &iter);

//11. b. b. Invoke kernel 1
	status = clEnqueueNDRangeKernel(cmd_queue, kernel1, 1, NULL,                       global_work_size, local_work_size, 0, 0, &myEvent);	
    if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", err); return -1; }
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
temptime = endTime - startTime;
totaltime = totaltime + temptime;
//11. d. Copy active to vec, make active 0

status = clEnqueueReadBuffer(cmd_queue, active_d, CL_TRUE, 0, v*sizeof(int), active, 0,0, 0);
//for(int i = 0; i< 10 ; i++){
//   printf("active %d => %d\n", i, active[i]);
//}
for(int i = 0; i< v ; i++){
   vec[i]=active[i];
   active[i]=0;
}

//11. e. Increment iter
iter++; 

} while(iter<12);

clFinish(cmd_queue);

//clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL); clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL); 
unsigned long kernelExecTime = totaltime;
printf("Execution time : %lu\n", kernelExecTime);

status = clEnqueueReadBuffer(cmd_queue, dist, CL_TRUE, 0, v*sizeof(int), disti, 0,0, 0);
    if(status != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadBuffer()=>%d failed\n", err); return -1; }

for(int i = 0; i< 10 ; i++){
   printf("dist %d => %d\n", i, disti[i]);
}

if(mask!= NULL) clReleaseMemObject(mask);
 if(dist!= NULL) clReleaseMemObject(dist);
 if(mat!= NULL) clReleaseMemObject(mat);
 if(col_ind!= NULL) clReleaseMemObject(col_ind);
if(vec_d!= NULL) clReleaseMemObject(vec_d);
if(active_d!= NULL) clReleaseMemObject(active_d);
if(iter_d!= NULL) clReleaseMemObject(iter_d);
 shutdown();

free(disti);
free(maski);
free(vec);
free(mat_rp);
free(active);
free(col);


return 0;
}
