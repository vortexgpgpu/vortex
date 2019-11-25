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

/*
    Parallel reduction

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value.

    Reductions are a very common computation in parallel algorithms.  Any time
    an array of values needs to be reduced to a single value using a binary 
    associative operator, a reduction can be used.  Example applications include
    statistics computaions such as mean and standard deviation, and image 
    processing applications such as finding the total luminance of an
    image.

    This code performs sum reductions, but any associative operator such as
    min() or max() could also be used.

    It assumes the input size is a power of 2.

    COMMAND LINE ARGUMENTS

    "--shmoo":         Test performance for 1 to 32M elements with each of the 7 different kernels
    "--n=<N>":         Specify the number of elements to reduce (default 1048576)
    "--threads=<N>":   Specify the number of threads per block (default 128)
    "--kernel=<N>":    Specify which kernel to run (0-6, default 6)
    "--maxblocks=<N>": Specify the maximum number of thread blocks to launch (kernel 6 only, default 64)
    "--cpufinal":      Read back the per-block results and do final sum of block sums on CPU (default false)
    "--cputhresh=<N>": The threshold of number of blocks sums below which to perform a CPU final reduction (default 1)
    
*/

// Common system and utility includes 
#include <oclUtils.h>
#include <shrQATest.h>

// additional includes
#include <sstream>
#include <oclReduction.h>

// Forward declarations and sample-specific defines
// *********************************************************************
enum ReduceType
{
    REDUCE_INT,
    REDUCE_FLOAT,
    REDUCE_DOUBLE
};

template <class T>
bool runTest( int argc, const char** argv, ReduceType datatype);

#define MAX_BLOCK_DIM_SIZE 65535

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

cl_kernel getReductionKernel(ReduceType datatype, int whichKernel, int blockSize, int isPowOf2);

// Main function 
// *********************************************************************
int main( int argc, const char** argv) 
{
    shrQAStart(argc, (char **)argv);

    // start logs 
    shrSetLogFileName ("oclReduction.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    char *typeChoice;
    shrGetCmdLineArgumentstr(argc, argv, "type", &typeChoice);

    // determine type of array from command line args
    if (0 == typeChoice)
    {
        typeChoice = (char*)malloc(7 * sizeof(char));
        #ifdef WIN32
            strcpy_s(typeChoice, 7 * sizeof(char) + 1, "int");
        #else
            strcpy(typeChoice, "int");
        #endif
    }
    ReduceType datatype = REDUCE_INT;

    #ifdef WIN32
        if (!_strcmpi(typeChoice, "float"))
            datatype = REDUCE_FLOAT;
        else if (!_strcmpi(typeChoice, "double"))
            datatype = REDUCE_DOUBLE;
        else
            datatype = REDUCE_INT;
    #else
        if (!strcmp(typeChoice, "float"))
            datatype = REDUCE_FLOAT;
        else if (!strcmp(typeChoice, "double"))
            datatype = REDUCE_DOUBLE;
        else
            datatype = REDUCE_INT;
    #endif

    shrLog("Reducing array of type %s.\n", typeChoice);

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &uiNumDevices);
    //oclCheckError(ciErrNum, CL_SUCCESS);
    cl_device_id *cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, uiNumDevices, cdDevices, NULL);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevices, cdDevices, NULL, NULL, &ciErrNum);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    // get and log the device info
    if( shrCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, (const char**)argv, "device", &device_nr);
	  if( device_nr < uiNumDevices ) {
		device = oclGetDev(cxGPUContext, device_nr);
	  } else {
		shrLog("Invalid Device %d Requested.\n", device_nr);
		shrExitEX(argc, argv, EXIT_FAILURE);
	  }
    } else {
      device = oclGetMaxFlopsDev(cxGPUContext);
    }
    oclPrintDevName(LOGBOTH, device);
    shrLog("\n");

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    source_path = shrFindFilePath("oclReduction_kernel.cl", argv[0]);

    bool bSuccess = false;
    switch (datatype)
    {
    default:
    case REDUCE_INT:
        bSuccess = runTest<int>( argc, argv, datatype);
        break;
    case REDUCE_FLOAT:
        bSuccess = runTest<float>( argc, argv, datatype);
        break;
    }
    
    // finish
    shrQAFinishExit(argc, (const char **)argv, bSuccess ? QA_PASSED : QA_FAILED);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//! 
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;              
    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;  
        T t = sum + y;      
        c = (t - sum) - y;  
        sum = t;            
    }
    return sum;
}

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
        

    if (whichKernel == 6)
        blocks = MIN(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and 
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T profileReduce(ReduceType datatype,
                  cl_int  n, 
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  int  whichKernel, 
                  int  testIterations,
                  bool cpuFinalReduction,
                  int  cpuFinalThreshold,
                  double* dTotalTime,
                  T* h_odata,
                  cl_mem d_idata, 
                  cl_mem d_odata)
{


    T gpu_result = 0;
    bool needReadBack = true;
    cl_kernel finalReductionKernel[10];
    int finalReductionIterations=0;

    //shrLog("Profile Kernel %d\n", whichKernel);

    cl_kernel reductionKernel = getReductionKernel(datatype, whichKernel, numThreads, isPow2(n) );
    clSetKernelArg(reductionKernel, 0, sizeof(cl_mem), (void *) &d_idata);
    clSetKernelArg(reductionKernel, 1, sizeof(cl_mem), (void *) &d_odata);
    clSetKernelArg(reductionKernel, 2, sizeof(cl_int), &n);
    clSetKernelArg(reductionKernel, 3, sizeof(T) * numThreads, NULL);

    if( !cpuFinalReduction ) {
        int s=numBlocks;
        int threads = 0, blocks = 0;
        int kernel = (whichKernel == 6) ? 5 : whichKernel;
        
        while(s > cpuFinalThreshold) 
        {
            getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

            finalReductionKernel[finalReductionIterations] = getReductionKernel(datatype, kernel, threads, isPow2(s) );
            clSetKernelArg(finalReductionKernel[finalReductionIterations], 0, sizeof(cl_mem), (void *) &d_odata);
            clSetKernelArg(finalReductionKernel[finalReductionIterations], 1, sizeof(cl_mem), (void *) &d_odata);
            clSetKernelArg(finalReductionKernel[finalReductionIterations], 2, sizeof(cl_int), &n);
            clSetKernelArg(finalReductionKernel[finalReductionIterations], 3, sizeof(T) * numThreads, NULL);
            
            if (kernel < 3)
                s = (s + threads - 1) / threads;
            else
                s = (s + (threads*2-1)) / (threads*2);

            finalReductionIterations++;
        }
    }
    
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    for (int i = 0; i < testIterations; ++i)
    {
        gpu_result = 0;

        clFinish(cqCommandQueue);
        if(i>0) shrDeltaT(1);

        // execute the kernel
        globalWorkSize[0] = numBlocks * numThreads;
        localWorkSize[0] = numThreads;
	
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue,reductionKernel, 1, 0, globalWorkSize, localWorkSize,
                                          0, NULL, NULL);               

        // check if kernel execution generated an error        
        //oclCheckError(ciErrNum, CL_SUCCESS);

        if (cpuFinalReduction)
        {
            // sum partial sums from each block on CPU        
            // copy result from device to host
            clEnqueueReadBuffer(cqCommandQueue, d_odata, CL_TRUE, 0, numBlocks * sizeof(T), 
                                h_odata, 0, NULL, NULL);

            for(int i=0; i<numBlocks; i++) 
            {
                gpu_result += h_odata[i];
            }

            needReadBack = false;
        }
        else
        {
            // sum partial block sums on GPU
            int s=numBlocks;
            int kernel = (whichKernel == 6) ? 5 : whichKernel;
            int it = 0;
            

            while(s > cpuFinalThreshold) 
            {
                int threads = 0, blocks = 0;
                getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

                globalWorkSize[0] = threads * blocks;
                localWorkSize[0] = threads;
                
                ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, finalReductionKernel[it], 1, 0,
                                                  globalWorkSize, localWorkSize, 0, NULL, NULL);               
                //oclCheckError(ciErrNum, CL_SUCCESS);
                
                if (kernel < 3)
                    s = (s + threads - 1) / threads;
                else
                    s = (s + (threads*2-1)) / (threads*2);

                it++;
            }

            if (s > 1)
            {
                // copy result from device to host
                clEnqueueReadBuffer(cqCommandQueue, d_odata, CL_TRUE, 0, s * sizeof(T), 
                                    h_odata, 0, NULL, NULL);

                for(int i=0; i < s; i++) 
                {
                    gpu_result += h_odata[i];
                }

                needReadBack = false;
            }
        }

        clFinish(cqCommandQueue);
        if(i>0) *dTotalTime += shrDeltaT(1); 
    }

    if (needReadBack)
    {
        // copy final sum from device to host
        clEnqueueReadBuffer(cqCommandQueue, d_odata, CL_TRUE, 0, sizeof(T), 
                            &gpu_result, 0, NULL, NULL);
    }

    // Release the kernels
    clReleaseKernel(reductionKernel);
    if( !cpuFinalReduction ) {
        for(int it=0; it<finalReductionIterations; ++it) {
            clReleaseKernel(finalReductionKernel[it]);
        }
        
    }

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// This function calls profileReduce multple times for a range of array sizes
// and prints a report in CSV (comma-separated value) format that can be used for
// generating a "shmoo" plot showing the performance for each kernel variation
// over a wide range of input sizes.
////////////////////////////////////////////////////////////////////////////////
template <class T>
void shmoo(int minN, int maxN, int maxThreads, int maxBlocks, ReduceType datatype)
{ 
    // create random input data on CPU
    unsigned int bytes = maxN * sizeof(T);

    T* h_idata = (T*)malloc(bytes);

    for(int i = 0; i < maxN; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        if (datatype == REDUCE_INT)
            h_idata[i] = (T)(rand() & 0xFF);
        else
            h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
    }

    int maxNumBlocks = MIN( maxN / maxThreads, MAX_BLOCK_DIM_SIZE);

    // allocate mem for the result on host side
    T* h_odata = (T*) malloc(maxNumBlocks*sizeof(T));

    // allocate device memory and data
    cl_mem d_idata = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_idata, NULL);
    cl_mem d_odata = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, maxNumBlocks * sizeof(T), NULL, NULL);

    int testIterations = 100;
    double dTotalTime = 0.0;
    
    // print headers
    shrLog("Time in seconds for various numbers of elements for each kernel\n");
    shrLog("\n\n");
    shrLog("Kernel");
    for (int i = minN; i <= maxN; i *= 2)
    {
        shrLog(", %d", i);
    }
   
    for (int kernel = 0; kernel < 7; kernel++)
    {
        shrLog("\n");
        shrLog("%d", kernel);
        for (int i = minN; i <= maxN; i *= 2)
        {
            int numBlocks = 0;
            int numThreads = 0;
            getNumBlocksAndThreads(kernel, i, maxBlocks, maxThreads, numBlocks, numThreads);
            
            double reduceTime;
            if( numBlocks <= MAX_BLOCK_DIM_SIZE ) {
                profileReduce(datatype, i, numThreads, numBlocks, maxThreads, maxBlocks, kernel, 
                                testIterations, false, 1, &dTotalTime, h_odata, d_idata, d_odata);
                reduceTime = dTotalTime/(double)testIterations;
            } else {                
                reduceTime = -1.0;
            }
            shrLog(", %.4f m", reduceTime);
        }
    }

    // cleanup
    free(h_idata);
    free(h_odata);
    clReleaseMemObject(d_idata);
    clReleaseMemObject(d_odata);
}

////////////////////////////////////////////////////////////////////////////////
// The main function whihc runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
template <class T>
bool
runTest( int argc, const char** argv, ReduceType datatype) 
{
    int size = 1<<24;    // number of elements to reduce
    int maxThreads;

    cl_kernel reductionKernel = getReductionKernel(datatype, 0, 64, 1);        
    clReleaseKernel(reductionKernel);

    if (smallBlock) 
      maxThreads = 64;  // number of threads per block
    else
      maxThreads = 128;

    int whichKernel = 6;
    int maxBlocks = 64;
    bool cpuFinalReduction = false;
    int cpuFinalThreshold = 1;

    shrGetCmdLineArgumenti( argc, (const char**) argv, "n", &size);
    shrGetCmdLineArgumenti( argc, (const char**) argv, "threads", &maxThreads);
    shrGetCmdLineArgumenti( argc, (const char**) argv, "kernel", &whichKernel);
    shrGetCmdLineArgumenti( argc, (const char**) argv, "maxblocks", &maxBlocks);
    
    shrLog(" %d elements\n", size);
    shrLog(" %d threads (max)\n", maxThreads);

    cpuFinalReduction = (shrCheckCmdLineFlag( argc, (const char**) argv, "cpufinal") == shrTRUE);
    shrGetCmdLineArgumenti( argc, (const char**) argv, "cputhresh", &cpuFinalThreshold);

    bool runShmoo = (shrCheckCmdLineFlag(argc, (const char**) argv, "shmoo") == shrTRUE);

#ifdef GPU_PROFILING
    if (runShmoo)
    {
        shmoo<T>(1, 33554432, maxThreads, maxBlocks, datatype);
        return true;
    }
    else
#endif
    {
        // create random input data on CPU
        unsigned int bytes = size * sizeof(T);
        T* h_idata = (T*)malloc(bytes);

        for(int i=0; i<size; i++) 
        {
            // Keep the numbers small so we don't get truncation error in the sum
            if (datatype == REDUCE_INT)
                h_idata[i] = (T)(rand() & 0xFF);
            else
                h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
        }

        int numBlocks = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);
        if (numBlocks == 1) cpuFinalThreshold = 1;
        shrLog(" %d blocks\n\n", numBlocks);

        // allocate mem for the result on host side
        T* h_odata = (T*)malloc(numBlocks * sizeof(T));

        // allocate device memory and data
        cl_mem d_idata = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_idata, NULL);
        cl_mem d_odata = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, numBlocks * sizeof(T), NULL, NULL);
      
        int testIterations = 100;
        double dTotalTime = 0.0;
        T gpu_result = 0;
        gpu_result = profileReduce<T>(datatype, size, numThreads, numBlocks, maxThreads, maxBlocks,
                                        whichKernel, testIterations, cpuFinalReduction, 
                                        cpuFinalThreshold, &dTotalTime,
                                        h_odata, d_idata, d_odata);

#ifdef GPU_PROFILING
        double reduceTime = dTotalTime/(double)testIterations;
        shrLogEx(LOGBOTH | MASTER, 0, "oclReduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n", 
               1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, numThreads);
#endif

        // compute reference solution
        shrLog("\nComparing against Host/C++ computation...\n"); 
        T cpu_result = reduceCPU<T>(h_idata, size);
        if (datatype == REDUCE_INT)
        {
            shrLog(" GPU result = %d\n", gpu_result);
            shrLog(" CPU result = %d\n\n", cpu_result);
            shrLog("%s\n\n", (gpu_result == cpu_result) ? "PASSED" : "FAILED");
        }
        else
        {
            shrLog(" GPU result = %.9f\n", gpu_result);
            shrLog(" CPU result = %.9f\n\n", cpu_result);

            double threshold = (datatype == REDUCE_FLOAT) ? 1e-8 * size : 1e-12;
            double diff = abs((double)gpu_result - (double)cpu_result);
            shrLog("%s\n\n", (diff < threshold) ? "PASSED" : "FAILED");
        }
      
        // cleanup
        free(h_idata);
        free(h_odata);
        clReleaseMemObject(d_idata);
        clReleaseMemObject(d_odata);

        return (gpu_result == cpu_result);
    }
}

// Helper function to create and build program and kernel
// *********************************************************************
cl_kernel getReductionKernel(ReduceType datatype, int whichKernel, int blockSize, int isPowOf2)
{
    // compile cl program
    size_t program_length;
    char *source; 

    std::ostringstream preamble;   

    // create the program
    // with type specification depending on datatype argument
    switch (datatype)
    {
    default:
    case REDUCE_INT:
        preamble << "#define T int" << std::endl;
        break;
    case REDUCE_FLOAT:
        preamble << "#define T float" << std::endl;
        break;
    }
    
    // set blockSize at compile time
    preamble << "#define blockSize " << blockSize << std::endl;
    
    // set isPow2 at compile time
    preamble << "#define nIsPow2 " << isPowOf2 << std::endl;
    
    // Load the source code and prepend the preamble
    source = oclLoadProgSource(source_path, preamble.str().c_str(), &program_length);
    //oclCheckError(source != NULL, shrTRUE);
    
    program =
      clCreateProgramWithBuiltInKernels(context, 1, &device_id, "reduce0", NULL);
    //cl_program rv_program = clCreateProgramWithSource(cxGPUContext, 1,(const char **) &source, 
     //                                                &program_length, &ciErrNum);
    //oclCheckError(ciErrNum, CL_SUCCESS);
    free(source);

    // build the program
    ciErrNum = clBuildProgram(rv_program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(rv_program, oclGetFirstDev(cxGPUContext));
        oclLogPtx(rv_program, oclGetFirstDev(cxGPUContext), "oclReduction.ptx");
        //oclCheckError(ciErrNum, CL_SUCCESS); 
    }
    
    // create Kernel    
    std::ostringstream kernelName;
    kernelName << "reduce" << whichKernel;    
    cl_kernel ckKernel = clCreateKernel(rv_program, kernelName.str().c_str(), &ciErrNum);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    size_t wgSize;
    ciErrNum = clGetKernelWorkGroupInfo(ckKernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
    if (wgSize == 64) 
      smallBlock = true;
    else smallBlock = false;

    // NOTE: the program will get deleted when the kernel is also released
    clReleaseProgram(rv_program);
    
    return ckKernel;
}
