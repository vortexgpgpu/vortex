// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <CL/cl.h>									// (in directory provided to compiler)		needed by OpenCL types and functions
#include <string.h>									// (in directory known to compiler)			needed by memset
#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "./common.h"									// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./timer.h"						// (in directory provided here)
#include "./opencl.h"
#ifdef TIMING
#include "./timing.h"
#endif

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper_2.h"				// (in directory provided here)

//========================================================================================================================================================================================================200
//	FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper_2(knode *knodes,
							long knodes_elem,
							long knodes_mem,

							int order,
							long maxheight,
							int count,

							long *currKnode,
							long *offset,
							long *lastKnode,
							long *offset_2,
							int *start,
							int *end,
							int *recstart,
							int *reclength)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

//Primitives for timing
#ifdef TIMING
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	struct timeval tv_init_end;
	struct timeval tv_h2d_start, tv_h2d_end;
	struct timeval tv_d2h_start, tv_d2h_end;
	struct timeval tv_kernel_start, tv_kernel_end;
	struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
	struct timeval tv_close_start, tv_close_end;
	float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time= 0,
		  d2h_time = 0, close_time = 0, total_time = 0;
#endif

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();

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
	cl_platform_id platform = platforms[platform_id_inuse];

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
	//	GET DEVICE INFORMATION
	//====================================================================================================100

	cl_uint devices_size;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devices_size);
	if (error != CL_SUCCESS)
		fatal_CL(error, __LINE__);
    if (devices_size == 0) {
        printf("There are no devices for Platform %d\n", platform_id_inuse);
        exit(0);
    }
    printf("Device num: %u\n", devices_size);
    // Get the list of devices (previousely selected for the context)
    cl_device_id *devices = (cl_device_id *) malloc(devices_size);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_size,
            devices, NULL);
    if (error != CL_SUCCESS) 
        fatal_CL(error, __LINE__);

	// Select the device
	cl_device_id device;
	device = devices[device_id_inuse];

	// Check device type
	error = clGetDeviceInfo(device, CL_DEVICE_TYPE,
            sizeof(device_type), (void *)&device_type, NULL);
	if (error != CL_SUCCESS)
		fatal_CL(error, __LINE__);
	if(device_type == CL_DEVICE_TYPE_GPU)
	    printf("Creating GPU Context\n");
	else if (device_type == CL_DEVICE_TYPE_CPU)
        printf("Creating CPU Context\n");
	else
        printf("This Context Type Not Supported\n");

	// Get the name of the selected device
	error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf),
            pbuf, NULL);
	if (error != CL_SUCCESS)
		fatal_CL(error, __LINE__);
	printf("Device: %s\n", pbuf);

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
										device_type, 
										NULL, 
										NULL, 
										&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100

	// Create a command queue
	cl_command_queue command_queue;
#ifdef TIMING
	command_queue = clCreateCommandQueue(context, device,
            CL_QUEUE_PROFILING_ENABLE, &error);
#else
	command_queue = clCreateCommandQueue(context, device, 0, &error);
#endif
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	const char *source = load_kernel_source("./kernel_gpu_opencl_2.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	char clOptions[110];
	//  sprintf(clOptions,"-I../../src");                                                                                 
	sprintf(clOptions,"-I./../");

#ifdef DEFAULT_ORDER_2
	sprintf(clOptions + strlen(clOptions), " -DDEFAULT_ORDER_2=%d", DEFAULT_ORDER_2);
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

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, 
							"findRangeK", 
							&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY				MALLOC
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

	cl_mem knodesD;
	knodesD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								knodes_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	currKnodeD
	//==================================================50

	cl_mem currKnodeD;
	currKnodeD = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	offsetD
	//==================================================50

	cl_mem offsetD;
	offsetD = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	lastKnodeD
	//==================================================50

	cl_mem lastKnodeD;
	lastKnodeD = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	offset_2D
	//==================================================50

	cl_mem offset_2D;
	offset_2D = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(long), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	startD
	//==================================================50

	cl_mem startD;
	startD = clCreateBuffer(context, 
								CL_MEM_READ_WRITE, 
								count*sizeof(int), 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	endD
	//==================================================50

	cl_mem endD;
	endD = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							count*sizeof(int), 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	cl_mem ansDStart;
	ansDStart = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							count*sizeof(int), 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	ansDLength
	//==================================================50

	cl_mem ansDLength;
	ansDLength = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							count*sizeof(int), 
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef  TIMING
	gettimeofday(&tv_mem_alloc_end, NULL);
	tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
	mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

    cl_event event;
	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									knodesD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									knodes_mem,				// size to be copied
									knodes,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	currKnodeD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									currKnodeD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									currKnode,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	offsetD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									offsetD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									offset,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	lastKnodeD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									lastKnodeD,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									lastKnode,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	offset_2D
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									offset_2D,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(long),		// size to be copied
									offset_2,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	startD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									startD,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(int),		// size to be copied
									start,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	endD
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									endD,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(int),		// size to be copied
									end,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									endD,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(int),		// size to be copied
									end,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
								    &event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	ansDLength
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									ansDLength,					// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									count*sizeof(int),		// size to be copied
									reclength,					// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event);					// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    h2d_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	END
	//==================================================50

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	Execution Parameters
	//====================================================================================================100

	size_t local_work_size[1];
	local_work_size[0] = order < 1024 ? order : 1024;
	size_t global_work_size[1];
	global_work_size[0] = count * local_work_size[0];

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//====================================================================================================100
	//	Kernel Arguments
	//====================================================================================================100

	clSetKernelArg(	kernel, 
					0, 
					sizeof(long), 
					(void *) &maxheight);
	clSetKernelArg(	kernel, 
					1, 
					sizeof(cl_mem), 
					(void *) &knodesD);
	clSetKernelArg(	kernel, 
					2, 
					sizeof(long), 
					(void *) &knodes_elem);

	clSetKernelArg(	kernel, 
					3, 
					sizeof(cl_mem), 
					(void *) &currKnodeD);
	clSetKernelArg(	kernel, 
					4, 
					sizeof(cl_mem), 
					(void *) &offsetD);
	clSetKernelArg(	kernel, 
					5, 
					sizeof(cl_mem), 
					(void *) &lastKnodeD);
	clSetKernelArg(	kernel, 
					6, 
					sizeof(cl_mem), 
					(void *) &offset_2D);
	clSetKernelArg(	kernel, 
					7, 
					sizeof(cl_mem), 
					(void *) &startD);
	clSetKernelArg(	kernel, 
					8, 
					sizeof(cl_mem), 
					(void *) &endD);
	clSetKernelArg(	kernel, 
					9, 
					sizeof(cl_mem), 
					(void *) &ansDStart);
	clSetKernelArg(	kernel, 
					10, 
					sizeof(cl_mem), 
					(void *) &ansDLength);

	//====================================================================================================100
	//	Kernel
	//====================================================================================================100
printf("global_work_size[0]=%d, local_work_size[0]=%d\n", (int)global_work_size[0], (int)local_work_size[0]);

	error = clEnqueueNDRangeKernel(	command_queue, 
									kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									&event);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Wait for all operations to finish NOT SURE WHERE THIS SHOULD GO
#ifdef TIMING
    kernel_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	error = clEnqueueReadBuffer(command_queue,				// The command queue.
								ansDStart,					// The image on the device.
								CL_TRUE,					// Blocking? (ie. Wait at this line until read has finished?)
								0,							// Offset. None in this case.
								count*sizeof(int),			// Size to copy.
								recstart,					// The pointer to the image on the host.
								0,							// Number of events in wait list. Not used.
								NULL,						// Event wait list. Not used.
								&event);						// Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    d2h_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	ansDLength
	//==================================================50
	error = clEnqueueReadBuffer(command_queue,				// The command queue.
								ansDLength,					// The image on the device.
								CL_TRUE,					// Blocking? (ie. Wait at this line until read has finished?)
								0,							// Offset. None in this case.
								count*sizeof(int),			// Size to copy.
								reclength,					// The pointer to the image on the host.
								0,							// Number of events in wait list. Not used.
								NULL,						// Event wait list. Not used.
								&event);						// Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    d2h_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	//==================================================50
	//	END
	//==================================================50

	//====================================================================================================100
	//	END
	//====================================================================================================100

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif

	// Release kernels...
	clReleaseKernel(kernel);

	// Now the program...
	clReleaseProgram(program);

	// Clean up the device memory...
	clReleaseMemObject(knodesD);

	clReleaseMemObject(currKnodeD);
	clReleaseMemObject(offsetD);
	clReleaseMemObject(lastKnodeD);
	clReleaseMemObject(offset_2D);
	clReleaseMemObject(startD);
	clReleaseMemObject(endD);
	clReleaseMemObject(ansDStart);
	clReleaseMemObject(ansDLength);

	// Flush the queue
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);

	// ???
	clReleaseContext(context);

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
    tvsub(&tv_close_end, &tv_total_start, &tv);
    total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

#ifdef TIMING
	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");
	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif

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
