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

// *********************************************************************
// oclCopyComputeOverlap Notes:  
//
// OpenCL API demo application for NVIDIA CUDA GPU's that implements a
// element by element vector hyptenuse computation using 2 input float arrays
// and 1 output float array.
//
// Demonstrates host->GPU and GPU->host copies that are asynchronous/overlapped
// with respect to GPU computation (and with respect to host thread). 
//
// Because the overlap acheivable for this computation and data set on a given system depends upon the GPU being used and the
// GPU/Host bandwidth, the sample adjust the computation duration to test the most ideal case and test against a consistent standard.
// This sample should be able to achieve up to 30% overlap on GPU's arch 1.2 and 1.3, and up to 50% on arch 2.0+ (Fermi) GPU's.
//
// After setup, warmup and calibration to the system, the sample runs 4 scenarios:  
//      A) Computations with 2 command queues on GPU
//         A multiple-cycle sequence is executed, timed and compared against the host
//      B) Computations with 1 command queue on GPU
//         A multiple-cycle sequence is executed, timed and compared against the host
//
//      The 2-command queue approach ought to be substantially faster
//
// For developmental purposes, the "iInnerLoopCount" variable passes into kernel and independently 
// increases compute time without increasing data size (via a loop inside the kernel)
//
//      At some value of iInnerLoopCount, # of elements, workgroup size, etc the Overlap percentage should reach 30%:
//      (This ~naively assumes time H2D bandwidth is the same as D2H bandwidth, but this is close on most systems)
//
//      If we name the time to copy single input vector H2D (or outpute vector D2H) as "T", then the optimum comparison case is:
//        
//          Single Queue with all the data and all the work  
//             Ttot (serial)        = 4T + 4T + 2T      = 10T    
//
//          Dual Queue, where each queue has 1/2 the data and 1/2 the work 
//             Tq0  (overlap)       = 2T + 2T + T .... 
//             Tq1  (overlap)       = .... 2T + 2T + T
//
//             Ttot (elapsed, wall) = 2T + 2T + 2T + T  = 7T
//
//          Best Overlap % = 100.0 * (10T - 7T)/10T = 30.0 %	(Tesla arch 1.2 or 1.3, single copy engine)
//
//			For multiple independent cycles using arch >= 2.0 with 2 copy engines, input and output copies can also be overlapped.
//			This doesn't help for the first cycle, but theoretically can lead to 50% overlap over many independent cycles.			
// *********************************************************************

// common SDK header for standard utilities and system libs 
#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>

// Best possible and Min ratio of compute/copy overlap timing benefit to pass the test
// values greater than 0.0f represent a speed-up relative to non-overlapped
#define EXPECTED_OVERLAP 30.0f
#define EXPECTED_OVERLAP_FERMI 45.0f
#define PASS_FACTOR 0.60f
#define RETRIES_ON_FAILURE 1

// Base sizes for parameters manipulated dynamically or on the command line
#define BASE_WORK_ITEMS 64
#define BASE_ARRAY_LENGTH 40000
#define BASE_LOOP_COUNT 32

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

// Vars
// *********************************************************************
cl_platform_id cpPlatform;                          // OpenCL platform
cl_context cxGPUContext;                            // OpenCL context
cl_command_queue cqCommandQueue[2];                 // OpenCL command queues
cl_device_id* cdDevices;                            // OpenCL device list  
cl_program cpProgram;                               // OpenCL program
cl_kernel ckKernel[2];                              // OpenCL kernel, 1 per queue
cl_mem cmPinnedSrcA;                                // OpenCL pinned host source buffer A
cl_mem cmPinnedSrcB;                                // OpenCL pinned host source buffer B 
cl_mem cmPinnedResult;                              // OpenCL pinned host result buffer 
float* fSourceA = NULL;                             // Mapped pointer for pinned Host source A buffer
float* fSourceB = NULL;                             // Mapped pointer for pinned Host source B buffer
float* fResult = NULL;                              // Mapped pointer for pinned Host result buffer 
cl_mem cmDevSrcA;                                   // OpenCL device source buffer A
cl_mem cmDevSrcB;                                   // OpenCL device source buffer B 
cl_mem cmDevResult;                                 // OpenCL device result buffer 
size_t szBuffBytes;                                 // Size of main buffers
size_t szGlobalWorkSize;                            // 1D var for Total # of work items in the launched ND range
size_t szLocalWorkSize = BASE_WORK_ITEMS;           // initial # of work items in the work group	
cl_int ciErrNum;			                        // Error code var
char* cPathAndName = NULL;                          // Var for full paths to data, src, etc.
char* cSourceCL = NULL;                             // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

// demo config vars
const char* cSourceFile = "kernel.cl";         // OpenCL computation kernel source code
float* Golden = NULL;                               // temp buffer to hold golden results for cross check    
bool bNoPrompt = false;                             // Command line switch to skip exit prompt
bool bQATest = false;                               // Command line switch to test

// Forward Declarations
// *********************************************************************
double DualQueueSequence(int iCycles, unsigned int uiNumElements, bool bShowConfig);
double OneQueueSequence(int iCycles, unsigned int uiNumElements, bool bShowConfig);
int AdjustCompute(cl_device_id cdTargetDevice, unsigned int uiNumElements, int iInitialLoopCount, int iCycles); 
void VectorHypotHost(const float* pfData1, const float* pfData2, float* pfResult, unsigned int uiNumElements, int iInnerLoopCount);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int *gp_argc = 0;
const char *** gp_argv = NULL;

// Main function 
// *********************************************************************
int main(int argc, const char **argv)
{
    //Locals
    size_t szKernelLength;                      // Byte size of kernel code
    double dBuildTime;                          // Compile time
    cl_uint uiTargetDevice = 0;	                // Default Device to compute on
    cl_uint uiNumDevsUsed = 1;                  // Number of devices used in this sample   
    cl_uint uiNumDevices;                       // Number of devices available 
    int iDevCap = -1;                           // Capability of device
    int iInnerLoopCount = BASE_LOOP_COUNT;      // Varies "compute intensity" per data within the kernel 
    const int iTestCycles = 10;                 // How many times to run the external test loop 
    const int iWarmupCycles = 8;                // How many times to run the warmup sequence 
    cl_uint uiWorkGroupMultiple = 4;            // Command line var (using "workgroupmult=<n>") to optionally increase workgroup size
    cl_uint uiNumElements = BASE_ARRAY_LENGTH;  // initial # of elements per array to process (note: procesing 4 per work item)
    cl_uint uiSizeMultiple = 4;                 // Command line var (using "sizemult=<n>") to optionally increase vector sizes
    bool bPassFlag = false;                     // Var to accumulate test pass/fail
    shrBOOL bMatch = shrFALSE;                  // Cross check result
	shrBOOL bTestOverlap = shrFALSE;
	double dAvgGPUTime[2] = {0.0, 0.0};         // Average time of iTestCycles calls for 2-Queue and 1-Queue test
    double dHostTime[2] = {0.0, 0.0};           // Host computation time (2nd test is redundant but a good stability indicator)
    float fMinPassCriteria[2] = {0.0f, 0.0f};	// Test pass cireria, adjusted dependant on GPU arch

    gp_argc = &argc;
    gp_argv = &argv;

    shrQAStart(argc, (char **)argv);

    // start logs 
	cExecutableName = argv[0];
    shrSetLogFileName ("oclCopyComputeOverlap.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // get basic command line args 
    bNoPrompt = (shrTRUE == shrCheckCmdLineFlag(argc, argv, "noprompt"));
    bQATest   = (shrTRUE == shrCheckCmdLineFlag(argc, argv, "qatest"));
    shrGetCmdLineArgumentu(argc, argv, "device", &uiTargetDevice);

    // Optional Command-line multiplier for vector size 
    //   Default val of 4 gives 10.24 million float elements per vector
    //   Range of 3 - 16 (7.68 to 40.96 million floats) is reasonable range (if system and GPU have enough memory)
    shrGetCmdLineArgumentu(argc, argv, "sizemult", &uiSizeMultiple);
    uiSizeMultiple = CLAMP(uiSizeMultiple, 1, 50);  
    uiNumElements = uiSizeMultiple * BASE_ARRAY_LENGTH * BASE_WORK_ITEMS;
    shrLog("Array sizes = %u float elements\n", uiNumElements); 

    // Optional Command-line multiplier for workgroup size (x 64 work items)
    //   Default val of 4 gives szLocalWorkSize of 256.
    //   Range of 1 - 8 (resulting in workgroup sizes of 64 to 512) is reasonable range
    shrGetCmdLineArgumentu(argc, argv, "workgroupmult", &uiWorkGroupMultiple);
    uiWorkGroupMultiple = CLAMP(uiWorkGroupMultiple, 1, 10); 
    szLocalWorkSize = uiWorkGroupMultiple * BASE_WORK_ITEMS;
    shrLog("Workgroup Size = %u\n\n", szLocalWorkSize); 

    // Get the NVIDIA platform if available, otherwise use default
    shrLog("Get the Platform ID...\n\n");
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get OpenCL platform name and version
    char cBuffer[256];
    ciErrNum = clGetPlatformInfo (cpPlatform, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("Platform Name = %s\n\n", cBuffer);

    // Get all the devices
    shrLog("Get the Device info and select Device...\n");
    uiNumDevices = 1;
    cdDevices = (cl_device_id*)malloc(uiNumDevices * sizeof(cl_device_id));
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, 1, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);   

    // Set target device and check capabilities 
    shrLog(" # of Devices Available = %u\n", uiNumDevices); 
    uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    shrLog(" Using Device %u, ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);  
    /*iDevCap = oclGetDevCap(cdDevices[uiTargetDevice]);
    if (iDevCap > 0) {
       shrLog(", Capability = %d.%d\n\n", iDevCap/10, iDevCap%10);
    } else {
       shrLog("\n\n", iDevCap); 
    }
    if (strstr(cBuffer, "NVIDIA") != NULL)
    {
        if (iDevCap < 12)
        {
            shrLog("Device doesn't have overlap capability.  Skipping test...\n"); 
            Cleanup (EXIT_SUCCESS);
        }

		// Device and Platform eligible for overlap testing
		bTestOverlap = shrTRUE;

        // If device has overlap capability, proceed
        fMinPassCriteria[0] = PASS_FACTOR * EXPECTED_OVERLAP;               // 1st cycle overlap is same for 1 or 2 copy engines
        if (iDevCap != 20) 
        {
            // Single copy engine
            fMinPassCriteria[1] = PASS_FACTOR * EXPECTED_OVERLAP;            // avg of many cycles
        }
        else 
        {
            char cDevName[1024];
            clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_NAME, sizeof(cDevName), &cDevName, NULL);
            if(strstr(cDevName, "Quadro")!=0 || strstr(cDevName, "Tesla")!=0)
            {
                // Tesla or Quadro (arch = 2.0) ... Dual copy engine 
                fMinPassCriteria[1] = PASS_FACTOR * EXPECTED_OVERLAP_FERMI;  // average of many cycles
            }
            else
            {
                // Geforce ... Single copy engine
                fMinPassCriteria[1] = PASS_FACTOR * EXPECTED_OVERLAP;        // average of many cycles
            }
        }
    }*/ 

    // Create the context
    shrLog("clCreateContext...\n"); 
    cxGPUContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // Create 2 command-queues
    cqCommandQueue[0] = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateCommandQueue [0]...\n"); 
    cqCommandQueue[1] = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateCommandQueue [1]...\n"); 

    // Allocate the OpenCL source and result buffer memory objects on GPU device GMEM
    szBuffBytes = sizeof(cl_float) * uiNumElements;
    cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevResult = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateBuffer (Src A, Src B and Result GPU Device GMEM, 3 x %u floats) ...\n", uiNumElements); 

    // Allocate pinned source and result host buffers:  
    //   Note: Pinned (Page Locked) memory is needed for async host<->GPU memory copy operations ***
    cmPinnedSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmPinnedSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmPinnedResult = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateBuffer (Src A, Src B and Result Pinned Host buffers, 3 x %u floats)...\n\n", uiNumElements); 

    // Get mapped pointers to pinned input host buffers
    //   Note:  This allows general (non-OpenCL) host functions to access pinned buffers using standard pointers
    fSourceA = (cl_float*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedSrcA, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    fSourceB = (cl_float*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedSrcB, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    fResult = (cl_float*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedResult, CL_TRUE, CL_MAP_READ, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    oclCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clEnqueueMapBuffer (Pointers to 3 pinned host buffers)...\n"); 

    // Alloc temp golden buffer for cross checks
    Golden = (float*)malloc(szBuffBytes);
    oclCheckErrorEX(Golden != NULL, shrTRUE, pCleanup);

#ifdef HOSTGPU
    // Read the OpenCL kernel in from source file
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    oclCheckError(cPathAndName != NULL, shrTRUE);
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);    
    // Create the program object
    shrLog("clCreateProgramWithSource...\n");
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#else    
    uint8_t *kernel_bin = NULL;
	size_t kernel_size;
	cl_int binary_status = 0;  
	ciErrNum = read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	cpProgram = clCreateProgramWithBinary(
		cxGPUContext, 1, &cdDevices[uiTargetDevice], &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#endif
    // Build the program for the target device
    clFinish(cqCommandQueue[0]);
    shrDeltaT(0);
    ciErrNum = clBuildProgram(cpProgram, uiNumDevsUsed, &cdDevices[uiTargetDevice], "-cl-fast-relaxed-math", NULL, NULL);
    shrLog("clBuildProgram..."); 
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, (double)ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "VectorHypot.ptx");
        Cleanup(EXIT_FAILURE);
    }
    dBuildTime = shrDeltaT(0);

    // Create the kernel
    ckKernel[0] = clCreateKernel(cpProgram, "VectorHypot", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel[1] = clCreateKernel(cpProgram, "VectorHypot", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateKernel (ckKernel[2])...\n"); 

    // Offsets for 2 queues
    cl_uint uiOffset[2] = {0, uiNumElements / (2 * 4)};

    // Set the Argument values for the 1st kernel instance (queue 0)
    ciErrNum = clSetKernelArg(ckKernel[0], 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErrNum |= clSetKernelArg(ckKernel[0], 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErrNum |= clSetKernelArg(ckKernel[0], 2, sizeof(cl_mem), (void*)&cmDevResult);
    ciErrNum |= clSetKernelArg(ckKernel[0], 3, sizeof(cl_uint), (void*)&uiOffset[0]);
    ciErrNum |= clSetKernelArg(ckKernel[0], 4, sizeof(cl_int), (void*)&iInnerLoopCount);
    ciErrNum |= clSetKernelArg(ckKernel[0], 5, sizeof(cl_uint), (void*)&uiNumElements);
    shrLog("clSetKernelArg ckKernel[0] args 0 - 5...\n"); 

    // Set the Argument values for the 2d kernel instance (queue 1)
    ciErrNum |= clSetKernelArg(ckKernel[1], 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErrNum |= clSetKernelArg(ckKernel[1], 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErrNum |= clSetKernelArg(ckKernel[1], 2, sizeof(cl_mem), (void*)&cmDevResult);
    ciErrNum |= clSetKernelArg(ckKernel[1], 3, sizeof(cl_uint), (void*)&uiOffset[1]);
    ciErrNum |= clSetKernelArg(ckKernel[1], 4, sizeof(cl_int), (void*)&iInnerLoopCount);
    ciErrNum |= clSetKernelArg(ckKernel[1], 5, sizeof(cl_uint), (void*)&uiNumElements);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clSetKernelArg ckKernel[1] args 0 - 5...\n\n"); 

    //*******************************************
    // Warmup the driver with dual queue sequence
    //*******************************************

    // Warmup with dual queue sequence for iTestCycles
    shrLog("Warmup with 2-Queue sequence, %d cycles...\n", iWarmupCycles);
    DualQueueSequence(iWarmupCycles, uiNumElements, false);

    // Use single queue config to adjust compute intensity 
    shrLog("Adjust compute for GPU / system...\n");
    iInnerLoopCount = AdjustCompute(cdDevices[uiTargetDevice], uiNumElements, iInnerLoopCount, iTestCycles); 
    shrLog("  Kernel inner loop count = %d\n", iInnerLoopCount); 

    //*******************************************
    // Run and time with 2 command-queues
    //*******************************************
	for( int iRun =0; iRun <= RETRIES_ON_FAILURE; ++iRun ) {
	
	// Run the sequence iTestCycles times
    dAvgGPUTime[0] = DualQueueSequence(iTestCycles, uiNumElements, false);

    // Warmup then Compute on host iTestCycles times (using mapped standard pointer to pinned host cl_mem buffer) 
    shrLog("  Device vs Host Result Comparison\t: "); 
    VectorHypotHost(fSourceA, fSourceB, Golden, uiNumElements, iInnerLoopCount);   
    shrDeltaT(0);
    for (int i = 0; i < iTestCycles; i++)
    {
        VectorHypotHost (fSourceA, fSourceB, Golden, uiNumElements, iInnerLoopCount);   
    }
    dHostTime[0] = shrDeltaT(0)/iTestCycles;

    // Compare host and GPU results (using mapped standard pointer to pinned host cl_mem buffer) 
    bMatch = shrComparefet(Golden, fResult, uiNumElements, 0.0f, 0);
    shrLog("gpu %s cpu\n", (bMatch == shrTRUE) ? "MATCHES" : "DOESN'T MATCH"); 
    bPassFlag = (bMatch == shrTRUE);

    //*******************************************
    // Run and time with 1 command queue
    //*******************************************
    // Run the sequence iTestCycles times
    dAvgGPUTime[1] = OneQueueSequence(iTestCycles, uiNumElements, false);

    // Compute on host iTestCycles times (using mapped standard pointer to pinned host cl_mem buffer) 
    shrLog("  Device vs Host Result Comparison\t: "); 
    shrDeltaT(0);
    for (int i = 0; i < iTestCycles; i++)
    {
        VectorHypotHost(fSourceA, fSourceB, Golden, (int)uiNumElements, iInnerLoopCount);   
    }
    dHostTime[1] = shrDeltaT(0)/iTestCycles;

    // Compare host and GPU results (using mapped standard pointer to pinned host cl_mem buffer) 
    bMatch = shrComparefet(Golden, fResult, uiNumElements, 0.0f, 0);
    shrLog("gpu %s cpu\n", (bMatch == shrTRUE) ? "MATCHES" : "DOESN'T MATCH"); 
    bPassFlag &= (bMatch == shrTRUE);

    //*******************************************

    // Compare Single and Dual queue timing 
    shrLog("\nResult Summary:\n"); 

    // Log GPU and CPU Time for 2-queue scenario
    shrLog("  Avg GPU Elapsed Time for 2-Queues\t= %.5f s\n", dAvgGPUTime[0]);
    shrLog("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", dHostTime[0]);

    // Log GPU and CPU Time for 1-queue scenario
    shrLog("  Avg GPU Elapsed Time for 1-Queue\t= %.5f s\n", dAvgGPUTime[1]);
    shrLog("  Avg Host Elapsed Time\t\t\t= %.5f s\n\n", dHostTime[1]);

    // Log overlap % for GPU (comparison of 2-queue and 1 queue scenarios) and status
    double dAvgOverlap = 100.0 * (1.0 - dAvgGPUTime[0]/dAvgGPUTime[1]);
    
	if( bTestOverlap ) {
		bool bAvgOverlapOK = (dAvgOverlap >= fMinPassCriteria[1]);
		if( iRun == RETRIES_ON_FAILURE || bAvgOverlapOK ) {
			shrLog("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Measured Overlap is %s\n\n", dAvgOverlap, fMinPassCriteria[1], bAvgOverlapOK ? "Acceptable" : "NOT Acceptable");

			// Log info to master log in standard format
			shrLogEx(LOGBOTH | MASTER, 0, "oclCopyComputeOverlap-Avg, Throughput = %.4f OverlapPercent, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n", 
			dAvgOverlap, dAvgGPUTime[0], uiNumElements, uiNumDevsUsed, szLocalWorkSize); 
			
			bPassFlag &= bAvgOverlapOK;
			break;
		}
	} 

		shrLog("  Measured and (Acceptable) Avg Overlap\t= %.1f %% (%.1f %%)  -> Retry %d more time(s)...\n\n", dAvgOverlap, fMinPassCriteria[1], RETRIES_ON_FAILURE - iRun);
	}


    //*******************************************
    // Report pass/fail, cleanup and exit
    Cleanup (bPassFlag ? EXIT_SUCCESS : EXIT_FAILURE);

    return 0;
}

// Run 1 queue sequence for n cycles
// *********************************************************************
double OneQueueSequence(int iCycles, unsigned int uiNumElements, bool bShowConfig)
{
    // Use fresh source Data: (re)initialize pinned host array buffers (using mapped standard pointer to pinned host cl_mem buffer) 
    shrFillArray(fSourceA, (int)uiNumElements);
    shrFillArray(fSourceB, (int)uiNumElements);

    // Reset Global work size for 1 command-queue, and log work sizes & dimensions
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, (int)(uiNumElements/4));

    // *** Make sure queues are empty and then start timer 
    double dAvgTime = 0.0;
    clFinish(cqCommandQueue[0]);
    clFinish(cqCommandQueue[1]);  
    shrDeltaT(0);

    // Run the sequence iCycles times
    for (int i = 0; i < iCycles; i++)
    {
        // Nonblocking Write of all of input data from host to device in command-queue 0 
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcA, CL_FALSE, 0, szBuffBytes, (void*)&fSourceA[0], 0, NULL, NULL);
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcB, CL_FALSE, 0, szBuffBytes, (void*)&fSourceB[0], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Launch kernel computation, command-queue 0 
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue[0], ckKernel[0], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Non Blocking Read of output data from device to host, command-queue 0 
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue[0], cmDevResult, CL_FALSE, 0, szBuffBytes, (void*)&fResult[0], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Flush sequence to device (may not be necessary on Linux or WinXP or when using the NVIDIA Tesla Computing Cluster driver)
        clFlush(cqCommandQueue[0]);
    }
    
    // *** Assure sync to host and return average sequence time
    clFinish(cqCommandQueue[0]);
    dAvgTime = shrDeltaT(0)/(double)iCycles;

    // Log config if asked for
    if (bShowConfig)
    {
        shrLog("\n1-Queue sequence Configuration:\n");
        shrLog("  Global Work Size (per command-queue)\t= %u\n  Local Work Size \t\t\t= %u\n  # of Work Groups (per command-queue)\t= %u\n  # of command-queues\t\t\t= 1\n", 
           szGlobalWorkSize, szLocalWorkSize, szGlobalWorkSize/szLocalWorkSize); 
    }
    return dAvgTime;
}

// Run 2 queue sequence for n cycles
// *********************************************************************
double DualQueueSequence(int iCycles, unsigned int uiNumElements, bool bShowConfig)
{
    // Locals
    size_t szHalfBuffer = szBuffBytes / 2;
    size_t szHalfOffset = szHalfBuffer / sizeof(float);
    double dAvgTime = 0.0;

    // Use fresh source Data: (re)initialize pinned host array buffers (using mapped standard pointer to pinned host cl_mem buffer) 
    shrFillArray(fSourceA, (int)uiNumElements);
    shrFillArray(fSourceB, (int)uiNumElements);

    // Set Global work size for 2 command-queues, and log work sizes & dimensions
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, (int)(uiNumElements/(2 * 4)));

    // Make sure queues are empty and then start timer 
    clFinish(cqCommandQueue[0]);
    clFinish(cqCommandQueue[1]);
    shrDeltaT(0);

    for (int i = 0; i < iCycles; i++)
    {
        // Mid Phase 0 
        // Nonblocking Write of 1st half of input data from host to device in command-queue 0 
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcA, CL_FALSE, 0, szHalfBuffer, (void*)&fSourceA[0], 0, NULL, NULL);
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcB, CL_FALSE, 0, szHalfBuffer, (void*)&fSourceB[0], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Push out the write for queue 0 (and prior read from queue 1 at end of loop) to the driver 
        // (not necessary on Linux, Mac OSX or WinXP)
        clFlush(cqCommandQueue[0]);
        clFlush(cqCommandQueue[1]);

        // Start Phase 1 ***********************************

        // Launch kernel computation, command-queue 0
        // (Note:  The order MATTERS here on Fermi !  THE KERNEL IN THIS PHASE SHOULD BE LAUNCHED BEFORE THE WRITE)
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue[0], ckKernel[0], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Nonblocking Write of 2nd half of input data from host to device in command-queue 1 
        // (Note:  The order MATTERS here on Fermi !  THE KERNEL IN THIS PHASE SHOULD BE LAUNCHED BEFORE THE WRITE)
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue[1], cmDevSrcA, CL_FALSE, szHalfBuffer, szHalfBuffer, (void*)&fSourceA[szHalfOffset], 0, NULL, NULL);
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[1], cmDevSrcB, CL_FALSE, szHalfBuffer, szHalfBuffer, (void*)&fSourceB[szHalfOffset], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Push out the compute for queue 0 and write for queue 1 to the driver
        // (not necessary on Linux, Mac OSX or WinXP)
        clFlush(cqCommandQueue[0]);
        clFlush(cqCommandQueue[1]);

        // Start Phase 2 ***********************************

        // Launch kernel computation, command-queue 1 
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue[1], ckKernel[1], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Non Blocking Read of 1st half of output data from device to host, command-queue 0 
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue[0], cmDevResult, CL_FALSE, 0, szHalfBuffer, (void*)&fResult[0], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Push out the compute for queue 1 and the read for queue 0 to the driver 
        // (not necessary on Linux, Mac OSX or WinXP)
        clFlush(cqCommandQueue[0]);
        clFlush(cqCommandQueue[1]);

        // Start Phase 0 (Rolls over) ***********************************

        // Non Blocking Read of 2nd half of output data from device to host, command-queue 1 
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue[1], cmDevResult, CL_FALSE, szHalfBuffer, szHalfBuffer, (void*)&fResult[szHalfOffset], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    // *** Sync to host and get average sequence time
    clFinish(cqCommandQueue[0]);
    clFinish(cqCommandQueue[1]);    
    dAvgTime = shrDeltaT(0)/(double)iCycles;

    // Log config if asked for
    if (bShowConfig)
    {
        shrLog("\n2-Queue sequence Configuration:\n");
        shrLog("  Global Work Size (per command-queue)\t= %u\n  Local Work Size \t\t\t= %u\n  # of Work Groups (per command-queue)\t= %u\n  # of command-queues\t\t\t= 2\n", 
           szGlobalWorkSize, szLocalWorkSize, szGlobalWorkSize/szLocalWorkSize); 
    }

    return dAvgTime;
}

// Function to adjust compute task according to device capability
// This allows a consistent overlap % across a wide variety of GPU's for test purposes
// It also implitly illustrates the relationship between compute capability and overlap at fixed work size
// *********************************************************************
int AdjustCompute(cl_device_id cdTargetDevice, unsigned int uiNumElements, int iInitLoopCount, int iCycles)
{
    // Locals
    double dCopyTime, dComputeTime;
    int iComputedLoopCount; 

    // Change Source Data
    shrFillArray(fSourceA, (int)uiNumElements);
    shrFillArray(fSourceB, (int)uiNumElements);

    // Reset Global work size for 1 command-queue, and log work sizes & dimensions
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, (int)(uiNumElements/4));

    // *** Make sure queues are empty and then start timer 
    clFinish(cqCommandQueue[0]);
    clFinish(cqCommandQueue[1]);  
    shrDeltaT(0);

    // Run the copy iCycles times and measure copy time on this system
    for (int i = 0; i < iCycles; i++)
    {
        // Nonblocking Write of all of input data from host to device in command-queue 0 
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcA, CL_FALSE, 0, szBuffBytes, (void*)&fSourceA[0], 0, NULL, NULL);
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[0], cmDevSrcB, CL_FALSE, 0, szBuffBytes, (void*)&fSourceB[0], 0, NULL, NULL);
        ciErrNum |= clFlush(cqCommandQueue[0]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }
    clFinish(cqCommandQueue[0]);
    dCopyTime = shrDeltaT(0);

    // Run the compute iCycles times and measure compute time on this system
    for (int i = 0; i < iCycles; i++)
    {
        // Launch kernel computation, command-queue 0 
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue[0], ckKernel[0], 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
        ciErrNum |= clFlush(cqCommandQueue[0]);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    clFinish(cqCommandQueue[0]);
    dComputeTime = shrDeltaT(0);

    // Determine number of core loop cycles proportional to copy/compute time ratio
    dComputeTime = MAX(dComputeTime, 1.0e-6);
    iComputedLoopCount = CLAMP(2, (int)((dCopyTime/dComputeTime) * (double)iInitLoopCount), (iInitLoopCount * 4));
    ciErrNum |= clSetKernelArg(ckKernel[0], 4, sizeof(cl_int), (void*)&iComputedLoopCount);
    ciErrNum |= clSetKernelArg(ckKernel[1], 4, sizeof(cl_int), (void*)&iComputedLoopCount);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    return (iComputedLoopCount);
} 

// Cleanup/Exit function 
// *********************************************************************
void Cleanup (int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(Golden)free(Golden);
    if(ckKernel[0])clReleaseKernel(ckKernel[0]);
    if(ckKernel[1])clReleaseKernel(ckKernel[1]);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(fSourceA)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedSrcA, (void*)fSourceA, 0, NULL, NULL);
    if(fSourceB)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedSrcB, (void*)fSourceB, 0, NULL, NULL);
    if(fResult)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedResult, (void*)fResult, 0, NULL, NULL);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevResult)clReleaseMemObject(cmDevResult);
    if(cmPinnedSrcA)clReleaseMemObject(cmPinnedSrcA);
    if(cmPinnedSrcB)clReleaseMemObject(cmPinnedSrcB);
    if(cmPinnedResult)clReleaseMemObject(cmPinnedResult);
    if(cqCommandQueue[0])clReleaseCommandQueue(cqCommandQueue[0]);
    if(cqCommandQueue[1])clReleaseCommandQueue(cqCommandQueue[1]);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cdDevices)free(cdDevices);

    // Master status Pass/Fail (all tests)
    shrQAFinishExit( *gp_argc, (const char **)*gp_argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED );
}

// "Golden" Host processing vector hyptenuse function for comparison purposes
// *********************************************************************
void VectorHypotHost(const float* pfData1, const float* pfData2, float* pfResult, unsigned int uiNumElements, int iInnerLoopCount)
{
    for (unsigned int i = 0; i < uiNumElements; i++) 
    {
        float fA = pfData1[i];
        float fB = pfData2[i];
        float fC = sqrtf(fA * fA + fB * fB);

        pfResult[i] = fC;
    }
}
