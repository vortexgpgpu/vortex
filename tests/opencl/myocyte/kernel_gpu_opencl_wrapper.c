// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "common.h"									// (in directory)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "timer.h"						// (in directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "solver.h"									// (in directory)
#include "opencl.h"						// (in directory)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>										// (in path known to compiler)	needed by printf
#include <string.h>										// (in path known to compiler)	needed by strlen
#include <CL/cl.h>										// (in path provided to compiler)	needed by OpenCL types and functions

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "kernel_gpu_opencl_wrapper.h"					// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
kernel_gpu_opencl_wrapper(	int xmax,
							int workload,

							fp ***y,
							fp **x,
							fp **params,
							fp *com)
{

	//======================================================================================================================================================150
	//	VARIABLES
	//======================================================================================================================================================150

	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long timecopyin = 0;
	long long timekernel = 0;
	long long timecopyout = 0;
	long long timeother;

	time0 = get_time();

	int i;

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
										CL_DEVICE_TYPE_GPU, 
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
	//	CRATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	const char *source = load_kernel_source("kernel_gpu_opencl.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Compile the program
	error = clBuildProgram(	program, 
							1, 
							&device, 
							"-I./../", 
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

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, 
							"kernel_gpu_opencl", 
							&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();

	time1 = get_time();

	//======================================================================================================================================================150
	//	ALLOCATE MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	d_initvalu_mem
	//====================================================================================================100

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	cl_mem d_initvalu;
	d_initvalu = clCreateBuffer(context,					// context
								CL_MEM_READ_WRITE,			// flags
								d_initvalu_mem,				// size of buffer
								NULL,						// host pointer (optional)
								&error );					// returned error
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	d_finavalu_mem
	//====================================================================================================100

	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	cl_mem d_finavalu;
	d_finavalu = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								d_finavalu_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	d_params_mem
	//====================================================================================================100

	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	cl_mem d_params;
	d_params = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								d_params_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	d_com_mem
	//====================================================================================================100

	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);
	cl_mem d_com;
	d_com = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							d_com_mem, 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	time2 = get_time();

	//======================================================================================================================================================150
	//	EXECUTION
	//======================================================================================================================================================150

	int status;

	for(i=0; i<workload; i++){

		status = solver(	y[i],
							x[i],
							xmax,
							params[i],
							com,

							d_initvalu,
							d_finavalu,
							d_params,
							d_com,

							command_queue,
							kernel,

							&timecopyin,
							&timekernel,
							&timecopyout);

		if(status !=0){
			printf("STATUS: %d\n", status);
		}

	}

	// // // print results
	// // int k;
	// // for(i=0; i<workload; i++){
		// // printf("WORKLOAD %d:\n", i);
		// // for(j=0; j<(xmax+1); j++){
			// // printf("\tTIME %d:\n", j);
			// // for(k=0; k<EQUATIONS; k++){
				// // printf("\t\ty[%d][%d][%d]=%13.10f\n", i, j, k, y[i][j][k]);
			// // }
		// // }
	// // }

	time3 = get_time();

	//======================================================================================================================================================150
	//	FREE GPU MEMORY
	//======================================================================================================================================================150

	// Release kernels...
	clReleaseKernel(kernel);

	// Now the program...
	clReleaseProgram(program);

	// Clean up the device memory...
	clReleaseMemObject(d_initvalu);
	clReleaseMemObject(d_finavalu);
	clReleaseMemObject(d_params);
	clReleaseMemObject(d_com);

	// Flush the queue
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);

	// ???
	clReleaseContext(context);

	time4= get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f % : CPU: GPU SETUP\n", 								(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU: ALLOCATE GPU MEMORY\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU: COMPUTATION\n", 							(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time4-time0) * 100);

	printf("\tGPU: COMPUTATION Components:\n");
	printf("\t%15.12f s, %15.12f % : GPU: COPY DATA IN\n", 							(float) (timecopyin) / 1000000, (float) (timecopyin) / (float) (time4-time0) * 100);
	printf("\t%15.12f s, %15.12f % : GPU: KERNEL\n", 								(float) (timekernel) / 1000000, (float) (timekernel) / (float) (time4-time0) * 100);
	printf("\t%15.12f s, %15.12f % : GPU: COPY DATA OUT\n", 						(float) (timecopyout) / 1000000, (float) (timecopyout) / (float) (time4-time0) * 100);
	timeother = time3-time2-timecopyin-timekernel-timecopyout;
	printf("\t%15.12f s, %15.12f % : GPU: OTHER\n", 								(float) (timeother) / 1000000, (float) (timeother) / (float) (time4-time0) * 100);

	printf("%15.12f s, %15.12f % : CPU: FREE GPU MEMORY\n", 						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time4-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 															(float) (time4-time0) / 1000000);

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0;

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
