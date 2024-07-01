
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <assert.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <parboil.h>
#include "file.h"
#include "ocl.h"
#include <math.h>

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp)
{
	int s=0;
	int i,j,k;
	for(i=0;i<nz;i++)
	{
		for(j=0;j<ny;j++)
		{
			for(k=0;k<nx;k++)
			{
        fread(A0+s,sizeof(float),1,fp);
				s++;
			}
		}
	}
	return 0;
}

static char* replaceFilenameExtension(const char* filename, const char* ext) {
  const char* dot = strrchr(filename, '.');
  int baseLen = dot ? (dot - filename) : strlen(filename);
  char* sz_out = (char*)malloc(baseLen + strlen(ext) + 1);
  if (!sz_out)
    return NULL;
  strncpy(sz_out, filename, baseLen);
  strcpy(sz_out + baseLen, ext);
  return sz_out;
}

static float* read_output_file(const char* filename, int* out_size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    int size;
    // Read the size (number of floats)
    if (fread(&size, sizeof(int), 1, file) != 1) {
        fclose(file);
        perror("Error reading size from file");
        return NULL;
    }
    // Allocate memory for the floats
    float* floats = (float*)malloc(size * sizeof(float));
    if (floats == NULL) {
        fclose(file);
        perror("Memory allocation failed");
        return NULL;
    }
    // Read the float data
    if (fread(floats, sizeof(float), size, file) != size) {
        fclose(file);
        free(floats);
        perror("Error reading floats from file");
        return NULL;
    }
    // Close the file
    fclose(file);
    // If out_size is not NULL, store the size there
    if (out_size != NULL) {
        *out_size = size;
    }
    return floats;
}

static int compare_floats(const float* src, const float* gold, int count) {
  int num_errors = 0;
  float abstol = 0.0f;
  float max_value = 0.0f;
  // Find the maximum magnitude in the gold array for absolute tolerance calculation
  for (int i = 0; i < count; i++) {
    if (fabs(gold[i]) > max_value)
      max_value = fabs(gold[i]);
  }
  // Absolute tolerance is 0.01% of the maximum magnitude of gold array
  abstol = 1e-4 * max_value;
  // Compare each pair of floats
  for (int i = 0; i < count; i++) {
      float diff = fabs(gold[i] - src[i]);
      if (!(diff <= abstol || diff < 0.002 * fabs(gold[i]))) {
          if (num_errors < 10)
              printf("Fail at row %d: (gold) %f != %f (computed)\n", i, gold[i], src[i]);
          num_errors++;
      }
  }
  return num_errors;
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return CL_INVALID_VALUE;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return CL_INVALID_VALUE;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return CL_SUCCESS;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;

	printf("OpenCL accelerated 7 points stencil codes****\n");
	printf("Author: Li-Wen Chang <lchang20@illinois.edu>\n");

	parameters = pb_ReadParameters(&argc, argv);

	/*parameters->inpFiles = (char **)malloc(sizeof(char *) * 2);
  parameters->inpFiles[0] = (char *)malloc(100);
  parameters->inpFiles[1] = NULL;
  strncpy(parameters->inpFiles[0], "128x128x32.bin", 100);*/

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	//declaration
	int nx,ny,nz;
	int size;
	int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5)
    	{
	     printf("Usage: probe nx ny nz t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
	     "t: the iteration time\n");
	     return -1;
	}

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

  cl_int clStatus;
  cl_context clContext;
  cl_device_id clDevice;
  cl_platform_id clPlatform;

  // Below is the new interface, coupled with Parboil runtime.
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found.");
    return -1;
  }

  // okay, let's deliver actual variables
  clPlatform = (cl_platform_id) pb_context->clPlatformId;
  clContext = (cl_context) pb_context->clContext;
  clDevice = (cl_device_id) pb_context->clDeviceId;

	cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
	CHECK_ERROR("clCreateCommandQueue")

  	pb_SetOpenCL(&clContext, &clCommandQueue);

	//const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
	//cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
	uint8_t *kernel_bin = NULL;
	size_t kernel_size;
	cl_int binary_status = 0;
  cl_program clProgram;

  clStatus = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
  CHECK_ERROR("read_kernel_file")
	clProgram = clCreateProgramWithSource(
        clContext, 1, (const char**)&kernel_bin, &kernel_size, &clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

	char clOptions[50];
	sprintf(clOptions,"-I src/opencl_base");
	clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
	CHECK_ERROR("clBuildProgram")

	cl_kernel clKernel = clCreateKernel(clProgram,"naive_kernel",&clStatus);
	CHECK_ERROR("clCreateKernel")

	//host data
	float *h_A0;
	float *h_Anext;

	//device
	cl_mem d_A0;
	cl_mem d_Anext;

	//load data from files

	size=nx*ny*nz;

	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	FILE *fp = fopen(parameters->inpFiles[0], "rb");
	read_data(h_A0, nx,ny,nz, fp);
  fclose(fp);
 	memcpy (h_Anext,h_A0,sizeof(float)*size);

	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

	//memory allocation
	d_A0 = clCreateBuffer(clContext,CL_MEM_READ_WRITE,size*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_Anext = clCreateBuffer(clContext,CL_MEM_READ_WRITE,size*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")

	//memory copy
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_A0,CL_FALSE,0,size*sizeof(float),h_A0,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_Anext,CL_TRUE,0,size*sizeof(float),h_Anext,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	//only use 1D thread block
  int tx = 128;
	size_t block[3] = {1,1,1}; // {tx,1,1}
	size_t grid[3] = {(nx-2+tx-1)/tx*tx,ny-2,nz-2};
  	//size_t grid[3] = {nx-2,ny-2,nz-2};
  	size_t offset[3] = {1,1,1};
  	printf("grid size in x/y/z = %ld %ld %ld\n",grid[0],grid[1],grid[2]);
	printf("block size in x/y/z = %ld %ld %ld\n",block[0],block[1],block[2]);

  	printf ("blocks = %ld\n", (grid[0]/block[0])*(grid[1]/block[1])*(grid[2]*block[2]));

	clStatus = clSetKernelArg(clKernel,0,sizeof(float),(void*)&c0);
	clStatus = clSetKernelArg(clKernel,1,sizeof(float),(void*)&c1);
	clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),(void*)&d_A0);
	clStatus = clSetKernelArg(clKernel,3,sizeof(cl_mem),(void*)&d_Anext);
	clStatus = clSetKernelArg(clKernel,4,sizeof(int),(void*)&nx);
	clStatus = clSetKernelArg(clKernel,5,sizeof(int),(void*)&ny);
	clStatus = clSetKernelArg(clKernel,6,sizeof(int),(void*)&nz);
	CHECK_ERROR("clSetKernelArg")

	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

	int t;
	for(t=0;t<iteration;t++)
	{
		clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,3,NULL,grid,block,0,NULL,NULL);
    //printf("iteration %d\n",t)
		CHECK_ERROR("clEnqueueNDRangeKernel")

    cl_mem d_temp = d_A0;
    d_A0 = d_Anext;
    d_Anext = d_temp;
    clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),(void*)&d_A0);
    clStatus = clSetKernelArg(clKernel,3,sizeof(cl_mem),(void*)&d_Anext);
	}

	cl_mem d_temp = d_A0;
	d_A0 = d_Anext;
	d_Anext = d_temp;

	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	clStatus = clEnqueueReadBuffer(clCommandQueue,d_Anext,CL_TRUE,0,size*sizeof(float),h_Anext,0,NULL,NULL);
	CHECK_ERROR("clEnqueueReadBuffer")

  clStatus = clReleaseMemObject(d_A0);
	clStatus = clReleaseKernel(clKernel);
	clStatus = clReleaseProgram(clProgram);
	clStatus = clReleaseCommandQueue(clCommandQueue);
	clStatus = clReleaseContext(clContext);
	CHECK_ERROR("clReleaseContext")
  clStatus = clReleaseDevice(clDevice);

	int output_size = nx*ny*nz;

	if (parameters->outFile) {
		pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,output_size);
	}

	// verify output
  int gold_size;
  char* gold_file = replaceFilenameExtension(parameters->inpFiles[0], ".gold");
  float* gold_data = read_output_file(gold_file, &gold_size);
  if (!gold_data)
    return -1;
  if (gold_size != output_size) {
    printf("error: gold data size mismatch: current=%d, expected=%d\n", output_size, gold_size);
    return -1;
  }
  int errors = compare_floats(h_Anext, gold_data, gold_size);
  if (errors > 0) {
    printf("FAILED!\n");
  } else {
    printf("PASSED!\n");
  }
  free(gold_data);
  free(gold_file);

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	//free((void*)clSource[0]);
  if (kernel_bin) free(kernel_bin);

	free(h_A0);
	free(h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return errors;
}
