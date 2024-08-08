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

/* Matrix transpose with Cuda
 * Host code.

 * This example transposes arbitrary-size matrices.  It compares a naive
 * transpose kernel that suffers from non-coalesced writes, to an optimized
 * transpose with fully coalesced memory access and no bank conflicts.  On
 * a G80 GPU, the optimized transpose can be more than 10x faster for large
 * matrices.
 */

// standard utility and system includes
#include "oclUtils.h"
#include "shrQATest.h"

#define BLOCK_DIM 4

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 1;

// global variables
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_context cxGPUContext;
cl_kernel ckKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];
cl_program rv_program;

// forward declarations
// *********************************************************************
int runTest( int argc, const char** argv);
extern "C" void computeGold( float* reference, float* idata,
                         const unsigned int size_x, const unsigned int size_y );

// Main Program
// *********************************************************************
int main( int argc, const char** argv)
{
    shrQAStart(argc, (char **)argv);

    // set logfile name and start logs
    shrSetLogFileName ("oclTranspose.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    // run the main test
    int result = runTest(argc, argv);
    oclCheckError(result, 0);
}

static double transposeGPU(const char* kernelName, bool useLocalMem,  cl_uint ciDeviceCount, float* h_idata, float* h_odata, unsigned int size_x, unsigned int size_y)
{
    cl_mem d_odata[MAX_GPU_COUNT];
    cl_mem d_idata[MAX_GPU_COUNT];
    cl_kernel ckKernel[MAX_GPU_COUNT];

    size_t szGlobalWorkSize[2];
    size_t szLocalWorkSize[2];
    cl_int ciErrNum;

    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    size_t sizePerGPU = shrRoundUp(BLOCK_DIM, (size_x+ciDeviceCount-1) / ciDeviceCount);

    // size of memory required to store the matrix
    const size_t mem_size = sizeof(float) * size_x * size_y;

    for(unsigned int i = 0; i < ciDeviceCount; ++i){
        // allocate device memory and copy host to device memory
        d_idata[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    mem_size, h_idata, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        // create buffer to store output
        d_odata[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY ,
                                    sizePerGPU*size_y*sizeof(float), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        // create the naive transpose kernel
        ckKernel[i] = clCreateKernel(rv_program, kernelName, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        // set the args values for the naive kernel
        size_t offset = i * sizePerGPU;
        ciErrNum  = clSetKernelArg(ckKernel[i], 0, sizeof(cl_mem), (void *) &d_odata[i]);
        ciErrNum |= clSetKernelArg(ckKernel[i], 1, sizeof(cl_mem), (void *) &d_idata[0]);
        ciErrNum |= clSetKernelArg(ckKernel[i], 2, sizeof(int), &offset);
        ciErrNum |= clSetKernelArg(ckKernel[i], 3, sizeof(int), &size_x);
        ciErrNum |= clSetKernelArg(ckKernel[i], 4, sizeof(int), &size_y);
        if (useLocalMem) {
            ciErrNum |= clSetKernelArg(ckKernel[i], 5, (BLOCK_DIM + 1) * BLOCK_DIM * sizeof(float), 0 );
        }
    }
    oclCheckError(ciErrNum, CL_SUCCESS);

    // set up execution configuration
    szLocalWorkSize[0] = BLOCK_DIM;
    szLocalWorkSize[1] = BLOCK_DIM;
    szGlobalWorkSize[0] = sizePerGPU;
    szGlobalWorkSize[1] = shrRoundUp(BLOCK_DIM, size_y);

    // execute the kernel numIterations times
    //int numIterations = 100;
    int numIterations = 1;
    shrLog("\nProcessing a %d by %d matrix of floats...\n\n", size_x, size_y);
    for (int i = -1; i < numIterations; ++i) {
        if (i == 0)
            shrDeltaT(0);
        for (unsigned int k=0; k < ciDeviceCount; ++k) {
            ciErrNum |= clEnqueueNDRangeKernel(commandQueue[k], ckKernel[k], 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        }
        oclCheckError(ciErrNum, CL_SUCCESS);
    }

    // Block CPU till GPU is done
    for(unsigned int k=0; k < ciDeviceCount; ++k){
        ciErrNum |= clFinish(commandQueue[k]);
    }
    double time = shrDeltaT(0)/(double)numIterations;
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Copy back to host
    for(unsigned int i = 0; i < ciDeviceCount; ++i){
        size_t offset = i * sizePerGPU;
        size_t size = MIN(size_x - i * sizePerGPU, sizePerGPU);

        ciErrNum |= clEnqueueReadBuffer(commandQueue[i], d_odata[i], CL_TRUE, 0,
                                size * size_y * sizeof(float), &h_odata[offset * size_y],
                                0, NULL, NULL);
    }
    oclCheckError(ciErrNum, CL_SUCCESS);

    for(unsigned int i = 0; i < ciDeviceCount; ++i){
        ciErrNum |= clReleaseMemObject(d_idata[i]);
        ciErrNum |= clReleaseMemObject(d_odata[i]);
        ciErrNum |= clReleaseKernel(ckKernel[i]);
    }
    oclCheckError(ciErrNum, CL_SUCCESS);

    return time;
}

uint8_t *kernel_bin = NULL;

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

//! Run a simple test for CUDA
// *********************************************************************
int runTest( const int argc, const char** argv)
{
    cl_int ciErrNum;
    cl_uint ciDeviceCount;
    //unsigned int size_x = 2048;
    //unsigned int size_y = 2048;
    unsigned int size_x = 128;
    unsigned int size_y = 128;

    int temp;
    if( shrGetCmdLineArgumenti( argc, argv,"width", &temp) ){
        size_x = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"height", &temp) ){
        size_y = temp;
    }

    if ((size_x / BLOCK_DIM) * BLOCK_DIM != size_x) {
        printf("Error: size_x must be a multiple of %d\n", BLOCK_DIM);
        return -1;
    }

    if ((size_y / BLOCK_DIM) * BLOCK_DIM != size_y) {
        printf("Error: size_y must be a multiple of %d\n", BLOCK_DIM);
        return -1;
    }

    // size of memory required to store the matrix
    const size_t mem_size = sizeof(float) * size_x * size_y;

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &uiNumDevices);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, uiNumDevices, cdDevices, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevices, cdDevices, NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        ciDeviceCount = 0;
        // User specified GPUs
        char* deviceList;
        char* deviceStr;

        shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

        #ifdef WIN32
            char* next_token;
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif
        ciDeviceCount = 0;
        while(deviceStr != NULL)
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
	    if( device == (cl_device_id)-1 ) {
                shrLog(" Invalid Device: %s\n\n", deviceStr);
                return -1;
	    }

            shrLog("Device %d: ", atoi(deviceStr));
            oclPrintDevName(LOGBOTH, device);
            shrLog("\n");

            // create command queue
            commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }

            ++ciDeviceCount;

            #ifdef WIN32
                deviceStr = strtok_s (NULL," ,.-", &next_token);
            #else
                deviceStr = strtok (NULL," ,.-");
            #endif
        }

        free(deviceList);
    }
    else
    {
        // Find out how many GPU's to compute on all available GPUs
        size_t nDeviceBytes;
        ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
        ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            shrLog(" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        }

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i)
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            shrLog("Device %d: ", i);
            oclPrintDevName(LOGBOTH, device);
            shrLog("\n");

            // create command queue
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
        }
    }

    // allocate and initalize host memory
    float* h_idata = (float*)malloc(mem_size);
    float* h_odata = (float*) malloc(mem_size);
    srand(15235911);
    shrFillArray(h_idata, (size_x * size_y));

    // create the program
    //rv_program = clCreateProgramWithSource(cxGPUContext, 1,
                     // (const char **)&source, &program_length, &ciErrNum);
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;

    ciErrNum = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
    if (ciErrNum != CL_SUCCESS) {
        shrLog(" Error %i in read_kernel_file call !!!\n\n", ciErrNum);
        return ciErrNum;
    }
    rv_program = clCreateProgramWithSource(
        cxGPUContext, 1, (const char**)&kernel_bin, &kernel_size, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        shrLog(" Error %i in clCreateProgramWithSource call !!!\n\n", ciErrNum);
        return ciErrNum;
    }

    // build the program
    ciErrNum = clBuildProgram(rv_program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        // write out standard error, Build Log and PTX, then return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(rv_program, oclGetFirstDev(cxGPUContext));
        oclLogPtx(rv_program, oclGetFirstDev(cxGPUContext), "oclTranspose.ptx");
        return(EXIT_FAILURE);
    }

    // Run Naive Kernel
#ifdef GPU_PROFILING
    // Matrix Copy kernel runs to measure reference performance.
    //double uncoalescedCopyTime = transposeGPU("uncoalesced_copy", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    //double simpleCopyTime = transposeGPU("simple_copy", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    //double sharedCopyTime = transposeGPU("shared_copy", true, ciDeviceCount, h_idata, h_odata, size_x, size_y);
#endif

    double naiveTime = transposeGPU("transpose_naive", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    //double optimizedTime = transposeGPU("transpose", true, ciDeviceCount, h_idata, h_odata, size_x, size_y);

#ifdef GPU_PROFILING
    // log times

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-simple copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-9 * double(size_x * size_y * sizeof(float))/simpleCopyTime), simpleCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM);

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-shared memory copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-9 * double(size_x * size_y * sizeof(float))/sharedCopyTime), sharedCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM);

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-uncoalesced copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-9 * double(size_x * size_y * sizeof(float))/uncoalescedCopyTime), uncoalescedCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM);

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-naive, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-9 * double(size_x * size_y * sizeof(float))/naiveTime), naiveTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM);

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-optimized, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-9 * double(size_x * size_y * sizeof(float))/optimizedTime), optimizedTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM);

#endif

    // compute reference solution and cross check results
    float* reference = (float*)malloc( mem_size);
    computeGold( reference, h_idata, size_x, size_y);
    shrLog("\nComparing results with CPU computation... \n\n");
    shrBOOL res = shrComparef( reference, h_odata, size_x * size_y);

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    //free(source);
    //free(source_path);

    // cleanup OpenCL
    ciErrNum = clReleaseProgram(rv_program);
    for(unsigned int i = 0; i < ciDeviceCount; ++i)
    {
        ciErrNum |= clReleaseCommandQueue(commandQueue[i]);
    }
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // pass or fail (cumulative... all tests in the loop)
    shrQAFinishExit(argc, (const char **)argv, (1 == res) ? QA_PASSED : QA_FAILED);

    return 0;
}
