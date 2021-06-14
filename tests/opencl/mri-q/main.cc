/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <sys/time.h>
#include <parboil.h>
#include <CL/cl.h>

#include "ocl.h"
#include "file.h"
#include "macros.h"
#include "computeQ.h"

static void
setupMemoryGPU(int num, int size, cl_mem* dev_ptr, float* host_ptr,clPrmtr* clPrm)
{
  cl_int clStatus;
  *dev_ptr = clCreateBuffer(clPrm->clContext,CL_MEM_READ_ONLY,num*size,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer");
  clStatus = clEnqueueWriteBuffer(clPrm->clCommandQueue,*dev_ptr,CL_TRUE,0,num*size,host_ptr,0,NULL,NULL);
  CHECK_ERROR("clEnequeueWriteBuffer");
}

static void
cleanupMemoryGPU(int num, int size, cl_mem* dev_ptr, float* host_ptr, clPrmtr* clPrm)
{
  cl_int clStatus;
  clStatus = clEnqueueReadBuffer(clPrm->clCommandQueue,*dev_ptr,CL_TRUE,0,num*size,host_ptr,0,NULL,NULL);
  CHECK_ERROR("clEnqueueReadBuffer")
  clStatus = clReleaseMemObject(*dev_ptr);
  CHECK_ERROR("clReleaseMemObject")
}

int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  
  params->inpFiles = (char **)malloc(sizeof(char *) * 2);
  params->inpFiles[0] = (char *)malloc(100);
  params->inpFiles[1] = NULL;
  strncpy(params->inpFiles[0], "32_32_32_dataset.bin", 100);

  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
    
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  printf("OK\n");

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  clPrmtr clPrm;

  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(params);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found.");
    return -1;
  }

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  clPrm.clContext = (cl_context) pb_context->clContext;

  clPrm.clCommandQueue = clCreateCommandQueue(clPrm.clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&(clPrm.clContext), &(clPrm.clCommandQueue));

  printf("OK\n");

  //const char* clSource[] = {readFile("src/opencl_base/kernels.cl")};
  //cl_program clProgram = clCreateProgramWithSource(clPrm.clContext,1,clSource,NULL,&clStatus);
  cl_program clProgram = clCreateProgramWithBuiltInKernels(
      clPrm.clContext, 1, &clDevice, "ComputePhiMag_GPU;ComputeQ_GPU", &clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char options[50];
  sprintf(options,"-I src/opencl_nvidia");
  clStatus = clBuildProgram(clProgram,0,NULL,options,NULL,NULL);
  if (clStatus != CL_SUCCESS) {
    char buf[4096];
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 4096, buf, NULL);
    printf ("%s\n", buf);
    CHECK_ERROR("clBuildProgram")
  }

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

  /* GPU section 1 (precompute PhiMag) */
  {
    clPrm.clKernel = clCreateKernel(clProgram,"ComputePhiMag_GPU",&clStatus);
    CHECK_ERROR("clCreateKernel")    

    /* Mirror several data structures on the device */
    cl_mem phiR_d;
    cl_mem phiI_d;
    cl_mem phiMag_d;

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    
    setupMemoryGPU(numK,sizeof(float),&phiR_d,phiR,&clPrm);
    setupMemoryGPU(numK,sizeof(float),&phiI_d,phiI,&clPrm);
    phiMag_d = clCreateBuffer(clPrm.clContext,CL_MEM_WRITE_ONLY,numK*sizeof(float),NULL,&clStatus);
    CHECK_ERROR("clCreateBuffer")

    clStatus = clFinish(clPrm.clCommandQueue);
    CHECK_ERROR("clFinish")

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d, &clPrm);

    clStatus = clFinish(clPrm.clCommandQueue);
    CHECK_ERROR("clFinish")

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    cleanupMemoryGPU(numK,sizeof(float),&phiMag_d,phiMag,&clPrm);

    clStatus = clReleaseMemObject(phiR_d);
    CHECK_ERROR("clReleaseMemObject")
    clStatus = clReleaseMemObject(phiI_d);
    CHECK_ERROR("clReleaseMemObject")
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));

  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  free(phiMag);
  
  clStatus = clReleaseKernel(clPrm.clKernel);

  /* GPU section 2 */
  {
    clPrm.clKernel = clCreateKernel(clProgram,"ComputeQ_GPU",&clStatus);
    CHECK_ERROR("clCreateKernel")

    cl_mem x_d;
    cl_mem y_d;
    cl_mem z_d;
    cl_mem Qr_d;
    cl_mem Qi_d;

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    setupMemoryGPU(numX,sizeof(float),&x_d,x,&clPrm);
    setupMemoryGPU(numX,sizeof(float),&y_d,y,&clPrm);
    setupMemoryGPU(numX,sizeof(float),&z_d,z,&clPrm);

    Qr_d = clCreateBuffer(clPrm.clContext,CL_MEM_READ_WRITE,numX*sizeof(float),NULL,&clStatus);
    CHECK_ERROR("clCreateBuffer")
    clMemSet(&clPrm,Qr_d,0,numX*sizeof(float));
    Qi_d = clCreateBuffer(clPrm.clContext,CL_MEM_READ_WRITE,numX*sizeof(float),NULL,&clStatus);
    CHECK_ERROR("clCreateBuffer")
    clMemSet(&clPrm,Qi_d,0,numX*sizeof(float));

    clStatus = clFinish(clPrm.clCommandQueue);
    CHECK_ERROR("clFinish")

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    computeQ_GPU(numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d, &clPrm);

    clStatus = clFinish(clPrm.clCommandQueue);
    CHECK_ERROR("clFinish")

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    clStatus = clReleaseMemObject(x_d);
    CHECK_ERROR("clReleaseMemObject")
    clStatus = clReleaseMemObject(y_d);
    CHECK_ERROR("clReleaseMemObject")
    clStatus = clReleaseMemObject(z_d);
    CHECK_ERROR("clReleaseMemObject")
    cleanupMemoryGPU(numX,sizeof(float),&Qr_d,Qr,&clPrm);
    cleanupMemoryGPU(numX,sizeof(float),&Qi_d,Qi,&clPrm);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);

  //free((void*)clSource[0]);

  clStatus = clReleaseKernel(clPrm.clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clPrm.clCommandQueue);
  clStatus = clReleaseContext(clPrm.clContext);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);

  pb_FreeParameters(params);

  return 0;
}
