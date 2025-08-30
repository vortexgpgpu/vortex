//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "main.h"								// (in the main program folder)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>									// (in path known to compiler)	needed by printf
#include <string.h>									// (in path known to compiler)	needed by strlen

#include <CL/cl.h>									// (in path specified to compiler)			needed by OpenCL types and functions

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "opencl.h"				// (in directory)							needed by device functions

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "kernel_gpu_opencl_wrapper.h"			// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	fp* image,											// input image
							int Nr,												// IMAGE nbr of rows
							int Nc,												// IMAGE nbr of cols
							long Ne,											// IMAGE nbr of elem
							int niter,											// nbr of iterations
							fp lambda,											// update step size
							long NeROI,											// ROI nbr of elements
							int* iN,
							int* iS,
							int* jE,
							int* jW,
							int iter,											// primary loop
							int mem_size_i,
							int mem_size_j)
{

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	COMMON VARIABLES
	//====================================================================================================100

	// common variables
	cl_int error;

	//====================================================================================================100
	//	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT ONE
	//====================================================================================================100

	// Get the number of available platforms
	cl_uint num_platforms;
	error = clGetPlatformIDs(	0, 
								NULL, 
								&num_platforms);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of available platforms
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(	num_platforms, 
								platforms, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the 1st platform
	cl_platform_id platform = platforms[0];

	// Get the name of the selected platform and print it (if there are multiple platforms, choose the first one)
	char pbuf[100];
	error = clGetPlatformInfo(	platform, 
								CL_PLATFORM_VENDOR, 
								sizeof(pbuf), 
								pbuf, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);

	//====================================================================================================100
	//	CREATE CONTEXT FOR THE PLATFORM
	//====================================================================================================100

	// Create context properties for selected platform
	cl_context_properties context_properties[3] = {	CL_CONTEXT_PLATFORM, 
													(cl_context_properties) platform, 
													0};

	// Create context for selected platform being GPU
	cl_context context;
	context = clCreateContextFromType(	context_properties, 
										CL_DEVICE_TYPE_ALL, 
										NULL, 
										NULL, 
										&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GET DEVICES AVAILABLE FOR THE CONTEXT, SELECT ONE
	//====================================================================================================100

	// Get the number of devices (previousely selected for the context)
	size_t devices_size;
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								0, 
								NULL, 
								&devices_size);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of devices (previousely selected for the context)
	cl_device_id *devices = (cl_device_id *) malloc(devices_size);
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								devices_size, 
								devices, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the first device (previousely selected for the context) (if there are multiple devices, choose the first one)
	cl_device_id device;
	device = devices[0];

	// Get the name of the selected device (previousely selected for the context) and print it
	error = clGetDeviceInfo(device, 
							CL_DEVICE_NAME, 
							sizeof(pbuf), 
							pbuf, 
							NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("Device: %s\n", pbuf);

	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100

	// Create a command queue
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(	context, 
											device, 
											0, 
											&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	const char *source = load_kernel_source("./kernel_gpu_opencl.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

  char clOptions[150];
//  sprintf(clOptions,"-I../../src");                                                    
  sprintf(clOptions,"-I.");
#ifdef RD_WG_SIZE
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE=%d", RD_WG_SIZE);
#endif
#ifdef RD_WG_SIZE_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0=%d", RD_WG_SIZE_0);
#endif
#ifdef RD_WG_SIZE_0_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0_0=%d", RD_WG_SIZE_0_0);
#endif

	// Compile the program
	error = clBuildProgram(	program, 
							1, 
							&device, 
							clOptions, 
							NULL, 
							NULL);
	// Print warnings and errors from compilation
	static char log[65536]; 
	memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(	program, 
							device, 
							CL_PROGRAM_BUILD_LOG, 
							sizeof(log)-1, 
							log, 
							NULL);
	printf("-----OpenCL Compiler Output-----\n");
	if (strstr(log,"warning:") || strstr(log, "error:")) 
		printf("<<<<\n%s\n>>>>\n", log);
	printf("--------------------------------\n");
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE Kernels
	//====================================================================================================100

	// Extract kernel
	cl_kernel extract_kernel;
	extract_kernel = clCreateKernel(program, 
									"extract_kernel", 
									&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Prepare kernel
	cl_kernel prepare_kernel;
	prepare_kernel = clCreateKernel(program, 
									"prepare_kernel", 
									&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Reduce kernel
	cl_kernel reduce_kernel;
	reduce_kernel = clCreateKernel(	program, 
									"reduce_kernel", 
									&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// SRAD kernel
	cl_kernel srad_kernel;
	srad_kernel = clCreateKernel(	program, 
									"srad_kernel", 
									&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// SRAD2 kernel
	cl_kernel srad2_kernel;
	srad2_kernel = clCreateKernel(	program, 
									"srad2_kernel", 
									&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Compress kernel
	cl_kernel compress_kernel;
	compress_kernel = clCreateKernel(	program, 
										"compress_kernel", 
										&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	TRIGGERING INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();		// the above does it

	//======================================================================================================================================================150
	// 	GPU VARIABLES
	//======================================================================================================================================================150

	// CUDA kernel execution parameters
	int blocks_x;

	//======================================================================================================================================================150
	// 	ALLOCATE MEMORY IN GPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// common memory size
	//====================================================================================================100

	int mem_size;															// matrix memory size
	mem_size = sizeof(fp) * Ne;												// get the size of float representation of input IMAGE

	//====================================================================================================100
	// allocate memory for entire IMAGE on DEVICE
	//====================================================================================================100

	cl_mem d_I;
	d_I = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for coordinates on DEVICE
	//====================================================================================================100

	cl_mem d_iN;
	d_iN = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size_i,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_iS;
	d_iS = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size_i,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_jE;
	d_jE = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size_j,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_jW;
	d_jW = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size_j,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for derivatives
	//====================================================================================================100

	cl_mem d_dN;
	d_dN = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_dS;
	d_dS = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_dW;
	d_dW = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_dE;
	d_dE = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for coefficient on DEVICE
	//====================================================================================================100

	cl_mem d_c;
	d_c = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for partial sums on DEVICE
	//====================================================================================================100

	cl_mem d_sums;
	d_sums = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	cl_mem d_sums2;
	d_sums2 = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COPY INPUT TO CPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// Image
	//====================================================================================================100

	error = clEnqueueWriteBuffer(	command_queue, 
									d_I, 
									1, 
									0, 
									mem_size, 
									image, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// coordinates
	//====================================================================================================100

	error = clEnqueueWriteBuffer(	command_queue, 
									d_iN, 
									1, 
									0, 
									mem_size_i, 
									iN, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clEnqueueWriteBuffer(	command_queue, 
									d_iS, 
									1, 
									0, 
									mem_size_i, 
									iS, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clEnqueueWriteBuffer(	command_queue, 
									d_jE, 
									1, 
									0, 
									mem_size_j, 
									jE, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clEnqueueWriteBuffer(	command_queue, 
									d_jW, 
									1, 
									0, 
									mem_size_j, 
									jW, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	KERNEL EXECUTION PARAMETERS
	//======================================================================================================================================================150

	// threads
	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;

	// workgroups
	int blocks_work_size;
	size_t global_work_size[1];
	blocks_x = Ne/(int)local_work_size[0];
	if (Ne % (int)local_work_size[0] != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks_work_size = blocks_x;
	global_work_size[0] = blocks_work_size * local_work_size[0];						// define the number of blocks in the grid

	printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//======================================================================================================================================================150
	// 	Extract Kernel - SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//======================================================================================================================================================150

	//====================================================================================================100
	//	set arguments
	//====================================================================================================100

	error = clSetKernelArg(	extract_kernel, 
							0, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	extract_kernel, 
							1, 
							sizeof(cl_mem), 
							(void *) &d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	launch kernel
	//====================================================================================================100
	printf("global size=%zu, local size=%zu\n", global_work_size[0], local_work_size[0]);

	error = clEnqueueNDRangeKernel(	command_queue, 
									extract_kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	Synchronization - wait for all operations in the command queue so far to finish
	//====================================================================================================100

	// error = clFinish(command_queue);
	// if (error != CL_SUCCESS) 
		// fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	WHAT IS CONSTANT IN COMPUTATION LOOP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	Prepare Kernel
	//====================================================================================================100

	error = clSetKernelArg(	prepare_kernel, 
							0, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	prepare_kernel, 
							1, 
							sizeof(cl_mem), 
							(void *) &d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	prepare_kernel, 
							2, 
							sizeof(cl_mem), 
							(void *) &d_sums);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	prepare_kernel, 
							3, 
							sizeof(cl_mem), 
							(void *) &d_sums2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	Reduce Kernel
	//====================================================================================================100

	int blocks2_x;
	int blocks2_work_size;
	size_t global_work_size2[1];
	long no;
	int mul;
	int mem_size_single = sizeof(fp) * 1;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	error = clSetKernelArg(	reduce_kernel, 
							0, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	reduce_kernel, 
							3, 
							sizeof(cl_mem), 
							(void *) &d_sums);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	reduce_kernel, 
							4, 
							sizeof(cl_mem), 
							(void *) &d_sums2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	SRAD Kernel
	//====================================================================================================100

	error = clSetKernelArg(	srad_kernel, 
							0, 
							sizeof(fp), 
							(void *) &lambda);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							1, 
							sizeof(int), 
							(void *) &Nr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							2, 
							sizeof(int), 
							(void *) &Nc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							3, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							4, 
							sizeof(cl_mem), 
							(void *) &d_iN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							5, 
							sizeof(cl_mem), 
							(void *) &d_iS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							6, 
							sizeof(cl_mem), 
							(void *) &d_jE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							7, 
							sizeof(cl_mem), 
							(void *) &d_jW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							8, 
							sizeof(cl_mem), 
							(void *) &d_dN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							9, 
							sizeof(cl_mem), 
							(void *) &d_dS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							10, 
							sizeof(cl_mem), 
							(void *) &d_dW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							11, 
							sizeof(cl_mem), 
							(void *) &d_dE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							13, 
							sizeof(cl_mem), 
							(void *) &d_c);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad_kernel, 
							14, 
							sizeof(cl_mem), 
							(void *) &d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	SRAD2 Kernel
	//====================================================================================================100

	error = clSetKernelArg(	srad2_kernel, 
							0, 
							sizeof(fp), 
							(void *) &lambda);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							1, 
							sizeof(int), 
							(void *) &Nr);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							2, 
							sizeof(int), 
							(void *) &Nc);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							3, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							4, 
							sizeof(cl_mem), 
							(void *) &d_iN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							5, 
							sizeof(cl_mem), 
							(void *) &d_iS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							6, 
							sizeof(cl_mem), 
							(void *) &d_jE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							7, 
							sizeof(cl_mem), 
							(void *) &d_jW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							8, 
							sizeof(cl_mem), 
							(void *) &d_dN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							9, 
							sizeof(cl_mem), 
							(void *) &d_dS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							10, 
							sizeof(cl_mem), 
							(void *) &d_dW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							11, 
							sizeof(cl_mem), 
							(void *) &d_dE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							12, 
							sizeof(cl_mem), 
							(void *) &d_c);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	srad2_kernel, 
							13, 
							sizeof(cl_mem), 
							(void *) &d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COMPUTATION
	//======================================================================================================================================================150

	printf("Iterations Progress: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

		printf("%d ", iter);
		fflush(NULL);

		//====================================================================================================100
		// Prepare kernel
		//====================================================================================================100
		printf("global size=%zu, local size=%zu\n", global_work_size[0], local_work_size[0]);

		// launch kernel
		error = clEnqueueNDRangeKernel(	command_queue, 
										prepare_kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// synchronize
		// error = clFinish(command_queue);
		// if (error != CL_SUCCESS) 
			// fatal_CL(error, __LINE__);

		//====================================================================================================100
		//	Reduce Kernel - performs subsequent reductions of sums
		//====================================================================================================100

		// initial values
		blocks2_work_size = blocks_work_size;							// original number of blocks
		global_work_size2[0] = global_work_size[0];
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

		// loop
		while(blocks2_work_size != 0){

			// set arguments that were uptaded in this loop
			error = clSetKernelArg(	reduce_kernel, 
									1, 
									sizeof(long), 
									(void *) &no);
			if (error != CL_SUCCESS) 
				fatal_CL(error, __LINE__);
			error = clSetKernelArg(	reduce_kernel, 
									2, 
									sizeof(int), 
									(void *) &mul);
			if (error != CL_SUCCESS) 
				fatal_CL(error, __LINE__);

			error = clSetKernelArg(	reduce_kernel, 
									5, 
									sizeof(int), 
									(void *) &blocks2_work_size);
			if (error != CL_SUCCESS) 
				fatal_CL(error, __LINE__);

			// launch kernel
			printf("reduce global size=%zu, local size=%zu\n", global_work_size2[0], local_work_size[0]);
			error = clEnqueueNDRangeKernel(	command_queue, 
											reduce_kernel, 
											1, 
											NULL, 
											global_work_size2, 
											local_work_size, 
											0, 
											NULL, 
											NULL);
			if (error != CL_SUCCESS) 
				fatal_CL(error, __LINE__);

			// synchronize
			// error = clFinish(command_queue);
			// if (error != CL_SUCCESS) 
				// fatal_CL(error, __LINE__);

			// update execution parameters
			no = blocks2_work_size;												// get current number of elements
			if(blocks2_work_size == 1){
				blocks2_work_size = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;										// update the increment
				blocks_x = blocks2_work_size/(int)local_work_size[0];			// number of blocks
				if (blocks2_work_size % (int)local_work_size[0] != 0){			// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2_work_size = blocks_x;
				global_work_size2[0] = blocks2_work_size * (int)local_work_size[0];
			}

		}

		// copy total sums to device
		error = clEnqueueReadBuffer(command_queue,
									d_sums,
									CL_TRUE,
									0,
									mem_size_single,
									&total,
									0,
									NULL,
									NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		error = clEnqueueReadBuffer(command_queue,
									d_sums2,
									CL_TRUE,
									0,
									mem_size_single,
									&total2,
									0,
									NULL,
									NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		//====================================================================================================100
		// calculate statistics
		//====================================================================================================100
		
		meanROI	= total / (fp)(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / (fp)(NeROI)) - meanROI2;							// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		//====================================================================================================100
		// execute srad kernel
		//====================================================================================================100

		// set arguments that were uptaded in this loop
		error = clSetKernelArg(	srad_kernel, 
							12, 
							sizeof(fp), 
							(void *) &q0sqr);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// launch kernel
		printf("srad global size=%zu, local size=%zu\n", global_work_size[0], local_work_size[0]);
		error = clEnqueueNDRangeKernel(	command_queue, 
										srad_kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// synchronize
		// error = clFinish(command_queue);
		// if (error != CL_SUCCESS) 
			// fatal_CL(error, __LINE__);

		//====================================================================================================100
		// execute srad2 kernel
		//====================================================================================================100

		// launch kernel
		printf("srad2 global size=%zu, local size=%zu\n", global_work_size[0], local_work_size[0]);
		error = clEnqueueNDRangeKernel(	command_queue, 
										srad2_kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// synchronize
		// error = clFinish(command_queue);
		// if (error != CL_SUCCESS) 
			// fatal_CL(error, __LINE__);

		//====================================================================================================100
		// End
		//====================================================================================================100

	}

	printf("\n");

	//======================================================================================================================================================150
	// 	Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//======================================================================================================================================================150

	//====================================================================================================100
	// set parameters
	//====================================================================================================100

	error = clSetKernelArg(	compress_kernel, 
							0, 
							sizeof(long), 
							(void *) &Ne);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clSetKernelArg(	compress_kernel, 
							1, 
							sizeof(cl_mem), 
							(void *) &d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// launch kernel
	//====================================================================================================100
	printf("global size=%zu, local size=%zu\n", global_work_size[0], local_work_size[0]);
	error = clEnqueueNDRangeKernel(	command_queue, 
									compress_kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// synchronize
	//====================================================================================================100

	error = clFinish(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COPY RESULTS BACK TO CPU
	//======================================================================================================================================================150

	error = clEnqueueReadBuffer(command_queue,
								d_I,
								CL_TRUE,
								0,
								mem_size,
								image,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// int i;
	// for(i=0; i<100; i++){
		// printf("%f ", image[i]);
	// }

	//======================================================================================================================================================150
	// 	FREE MEMORY
	//======================================================================================================================================================150

	// OpenCL structures
	error = clReleaseKernel(extract_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseKernel(prepare_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseKernel(reduce_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseKernel(srad_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseKernel(srad2_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseKernel(compress_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseProgram(program);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common_change
	error = clReleaseMemObject(d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_c);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clReleaseMemObject(d_iN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_iS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_jE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_jW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clReleaseMemObject(d_dN);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_dS);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_dE);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_dW);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	error = clReleaseMemObject(d_sums);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseMemObject(d_sums2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// OpenCL structures
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseCommandQueue(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseContext(context);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//======================================================================================================================================================150
	// 	End
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
