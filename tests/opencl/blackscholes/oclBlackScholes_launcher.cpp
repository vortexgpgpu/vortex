/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <oclUtils.h>
#include "oclBlackScholes_common.h"

static cl_program cpBlackScholes;   //OpenCL program
static cl_kernel  ckBlackScholes;   //OpenCL kernel
static cl_command_queue cqDefaultCommandQueue;

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (NULL == filename || NULL == data || 0 == size)
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

extern "C" void initBlackScholes(cl_context cxGPUContext, cl_command_queue cqParamCommandQueue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    /*shrLog("...loading BlackScholes.cl\n");
        char *cPathAndName = shrFindFilePath("BlackScholes.cl", argv[0]);
        shrCheckError(cPathAndName != NULL, shrTRUE);
        char *cBlackScholes = oclLoadProgSource(cPathAndName, "// My comment\n", &kernelLength);
        shrCheckError(cBlackScholes != NULL, shrTRUE);*/

    shrLog("...creating BlackScholes program\n");
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    cl_device_id device_id = oclGetFirstDev(cxGPUContext);

    ciErrNum = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
    shrCheckError(ciErrNum, CL_SUCCESS);
    cpBlackScholes = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&kernel_bin, &kernel_size, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    
    shrLog("...building BlackScholes program\n");
        ciErrNum = clBuildProgram(cpBlackScholes, 0, NULL, "-cl-fast-relaxed-math -Werror", NULL, NULL);

        if(ciErrNum != CL_BUILD_SUCCESS){
            shrLog("*** Compilation failure ***\n");

            size_t deviceNum;
            cl_device_id *cdDevices;
            ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceNum);
            shrCheckError(ciErrNum, CL_SUCCESS);

            cdDevices = (cl_device_id *)malloc(deviceNum * sizeof(cl_device_id));
            shrCheckError(cdDevices != NULL, shrTRUE);

            ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, deviceNum * sizeof(cl_device_id), cdDevices, NULL);
            shrCheckError(ciErrNum, CL_SUCCESS);

            size_t logSize;
            char *logTxt;

            ciErrNum = clGetProgramBuildInfo(cpBlackScholes, cdDevices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
            shrCheckError(ciErrNum, CL_SUCCESS);

            logTxt = (char *)malloc(logSize);
            shrCheckError(logTxt != NULL, shrTRUE);

            ciErrNum = clGetProgramBuildInfo(cpBlackScholes, cdDevices[0], CL_PROGRAM_BUILD_LOG, logSize, logTxt, NULL);
            shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog("%s\n", logTxt);
            shrLog("*** Exiting ***\n");
            free(logTxt);
            free(cdDevices);
            exit(1);
        }

    //Save ptx code to separate file
    //oclLogPtx(cpBlackScholes, oclGetFirstDev(cxGPUContext), "BlackScholes.ptx");

    shrLog("...creating BlackScholes kernels\n");
        ckBlackScholes = clCreateKernel(cpBlackScholes, "BlackScholes", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    cqDefaultCommandQueue = cqParamCommandQueue;
    //free(cBlackScholes);
    //free(cPathAndName);
}

extern "C" void closeBlackScholes(void){
    cl_int ciErrNum;
    ciErrNum  = clReleaseKernel(ckBlackScholes);
    ciErrNum |= clReleaseProgram(cpBlackScholes);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL Black-Scholes kernel launcher
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholes(
    cl_command_queue cqCommandQueue,
    cl_mem d_Call, //Call option price
    cl_mem d_Put,  //Put option price
    cl_mem d_S,    //Current stock price
    cl_mem d_X,    //Option strike price
    cl_mem d_T,    //Option years
    cl_float R,    //Riskless rate of return
    cl_float V,    //Stock volatility
    cl_uint optionCount
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQueue;

    ciErrNum  = clSetKernelArg(ckBlackScholes, 0, sizeof(cl_mem),   (void *)&d_Call);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 1, sizeof(cl_mem),   (void *)&d_Put);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 2, sizeof(cl_mem),   (void *)&d_S);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 3, sizeof(cl_mem),   (void *)&d_X);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 4, sizeof(cl_mem),   (void *)&d_T);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 5, sizeof(cl_float), (void *)&R);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 6, sizeof(cl_float), (void *)&V);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 7, sizeof(cl_uint),  (void *)&optionCount);
    shrCheckError(ciErrNum, CL_SUCCESS);

    //Run the kernel
    size_t globalWorkSize = 128;//60 * 1024;
    size_t localWorkSize = 1;//128;
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBlackScholes, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}
