//#ifdef __cplusplus
//extern "C" {
//#endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150
#include <string.h>

#include <CL/cl.h>					// (in library path provided to compiler)	needed by OpenCL types and functions

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "opencl.h"				// (in library path specified to compiler)	needed by for device functions
#include "timer.h"					// (in library path specified to compiler)	needed by timer
#include "timing.h"

//======================================================================================================================================================150
//	KERNEL_GPU_OPENCL_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper.h"				// (in the current directory)

//========================================================================================================================================================================================================200
//	KERNEL_GPU_OPENCL_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

extern int platform_id_inuse;
extern int device_id_inuse;

int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp_ = fopen(filename, "r");
  if (NULL == fp_) {
    fprintf(stderr, "Failed to load kernel. %s\n", filename);
    return -1;
  }
  fseek(fp_ , 0 , SEEK_END);
  long fsize = ftell(fp_);
  rewind(fp_);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp_);
  
  fclose(fp_);
  
  return 0;
}


void 
kernel_gpu_opencl_wrapper(	par_str par_cpu,
							dim_str dim_cpu,
							box_str* box_cpu,
							FOUR_VECTOR* rv_cpu,
							fp* qv_cpu,
							FOUR_VECTOR* fv_cpu)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

#ifdef TIMING
    struct timeval tv;
    struct timeval tv_total_start, tv_total_end;
    struct timeval tv_init_end;
    struct timeval tv_h2d_start, tv_h2d_end;
    struct timeval tv_d2h_start, tv_d2h_end;
    struct timeval tv_kernel_start, tv_kernel_end;
    struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
    struct timeval tv_close_start, tv_close_end;
    float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
          d2h_time = 0, close_time = 0, total_time = 0;

    gettimeofday(&tv_total_start, NULL);
#endif

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
	device = devices[device_id_inuse];

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
#ifdef TIMING
	command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
#else
	command_queue = clCreateCommandQueue(context, device, 0, &error);
#endif
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CRATE PROGRAM, COMPILE IT
	//====================================================================================================100
/*
	// Load kernel source code from file
	const char *source = load_kernel_source("./kernel/kernel_gpu_opencl.cl");
	size_t sourceSize = strlen(source);

	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
													1, 
													&source, 
													&sourceSize, 
													&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// parameterized kernel dimension
	char clOptions[110];
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
*/

  uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
      std::abort();

    cl_program program= clCreateProgramWithBinary(
          context, 1, &devices[device_id_inuse], &kernel_size,
          (const uint8_t**)&kernel_bin, &binary_status, &error);
    free(kernel_bin);
	  if(error != CL_SUCCESS) {
		    fatal_CL(error, __LINE__);}

	  error = clBuildProgram(program, 1, &devices[device_id_inuse], NULL, NULL, NULL);
	  if(error != CL_SUCCESS) { 
		    fatal_CL(error, __LINE__);}


	static char log[65536]; 
	memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(	program, 
							device, 
							CL_PROGRAM_BUILD_LOG, 
							sizeof(log)-1, 
							log, 
							NULL);
	if (strstr(log,"warning:") || strstr(log, "error:")) 
		printf("<<<<\n%s\n>>>>\n", log);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Create kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, 
							"kernel_gpu_opencl", 
							&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef  TIMING
    gettimeofday(&tv_init_end, NULL);
    tvsub(&tv_init_end, &tv_total_start, &tv);
    init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	// cudaThreadSynchronize();

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;
	size_t global_work_size[1];
	global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

	printf("# of blocks = %lu, # of threads/block = %lu (ensure that device can handle)\n", global_work_size[0]/local_work_size[0], local_work_size[0]);

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	cl_mem d_box_gpu;
	d_box_gpu = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								dim_cpu.box_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	rv
	//==================================================50

	cl_mem d_rv_gpu;
	d_rv_gpu = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								dim_cpu.space_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	qv
	//==================================================50

	cl_mem d_qv_gpu;
	d_qv_gpu = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								dim_cpu.space_mem2, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GPU MEMORY				COPY (IN & OUT)
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	cl_mem d_fv_gpu;
	d_fv_gpu = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								dim_cpu.space_mem, 
								NULL, 
								&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	//======================================================================================================================================================150
	//	GPU MEMORY				COPY IN
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

    cl_event event1, event2, event3, event4;
	error = clEnqueueWriteBuffer(	command_queue,			// command queue
									d_box_gpu,				// destination
									1,						// block the source from access until this copy operation complates (1=yes, 0=no)
									0,						// offset in destination to write to
									dim_cpu.box_mem,		// size to be copied
									box_cpu,				// source
									0,						// # of events in the list of events to wait for
									NULL,					// list of events to wait for
									&event1);				// ID of this operation to be used by waiting operations
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	rv
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue, 
									d_rv_gpu, 
									1, 
									0, 
									dim_cpu.space_mem, 
									rv_cpu, 
									0, 
									0, 
									&event2);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//==================================================50
	//	qv
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue, 
									d_qv_gpu, 
									1, 
									0, 
									dim_cpu.space_mem2, 
									qv_cpu, 
									0, 
									0, 
									&event3);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GPU MEMORY				COPY (IN & OUT)
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	error = clEnqueueWriteBuffer(	command_queue, 
									d_fv_gpu, 
									1,
									0, 
									dim_cpu.space_mem, 
									fv_cpu, 
									0, 
									0, 
									&event4);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

#ifdef TIMING
    h2d_time += probe_event_time(event1, command_queue);
    h2d_time += probe_event_time(event2, command_queue);
    h2d_time += probe_event_time(event3, command_queue);
    h2d_time += probe_event_time(event3, command_queue);
    h2d_time += probe_event_time(event4, command_queue);
#endif
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// ???
	clSetKernelArg(	kernel, 
					0, 
					sizeof(par_str), 
					(void *) &par_cpu);
	clSetKernelArg(	kernel, 
					1, 
					sizeof(dim_str), 
					(void *) &dim_cpu);
	clSetKernelArg(	kernel, 
					2, 
					sizeof(cl_mem), 
					(void *) &d_box_gpu);
	clSetKernelArg(	kernel, 
					3, 
					sizeof(cl_mem), 
					(void *) &d_rv_gpu);
	clSetKernelArg(	kernel, 
					4, 
					sizeof(cl_mem), 
					(void *) &d_qv_gpu);
	clSetKernelArg(	kernel, 
					5, 
					sizeof(cl_mem), 
					(void *) &d_fv_gpu);

	// launch kernel - all boxes
	error = clEnqueueNDRangeKernel(	command_queue, 
									kernel, 
									1, 
									NULL, 
									global_work_size, 
									local_work_size, 
									0, 
									NULL, 
									&event1);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    kernel_time += probe_event_time(event1, command_queue);
#else
	// Wait for all operations to finish NOT SURE WHERE THIS SHOULD GO
	error = clFinish(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#endif
    clReleaseEvent(event1);

	//======================================================================================================================================================150
	//	GPU MEMORY				COPY OUT
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				COPY (IN & OUT)
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	error = clEnqueueReadBuffer(command_queue,               // The command queue.
								d_fv_gpu,                         // The image on the device.
								CL_TRUE,                     // Blocking? (ie. Wait at this line until read has finished?)
								0,                           // Offset. None in this case.
								dim_cpu.space_mem, 			// Size to copy.
								fv_cpu,                       // The pointer to the image on the host.
								0,                           // Number of events in wait list. Not used.
								NULL,                        // Event wait list. Not used.
								&event1);                       // Event object for determining status. Not used.
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
#ifdef TIMING
    d2h_time += probe_event_time(event1, command_queue);
#endif
    clReleaseEvent(event1);

	// (enable for testing purposes only - prints some range of output, make sure not to initialize input in main.c with random numbers for comparison across runs)
	// int g;
	// int offset = 395;
	// for(g=0; g<10; g++){
		// printf("%f, %f, %f, %f\n", fv_cpu[offset+g].v, fv_cpu[offset+g].x, fv_cpu[offset+g].y, fv_cpu[offset+g].z);
	// }

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
	clReleaseMemObject(d_rv_gpu);
	clReleaseMemObject(d_qv_gpu);
	clReleaseMemObject(d_fv_gpu);
	clReleaseMemObject(d_box_gpu);

	// Flush the queue
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);

	// ???
	clReleaseContext(context);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

#ifdef  TIMING
	printf("Time spent in different stages of the application:\n");
    gettimeofday(&tv_close_end, NULL);
    tvsub(&tv_close_end, &tv_close_start, &tv);
    close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
    tvsub(&tv_close_end, &tv_total_start, &tv);
    total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    printf("Init: %f\n", init_time);
    printf("MemAlloc: %f\n", mem_alloc_time);
    printf("HtoD: %f\n", h2d_time);
    printf("Exec: %f\n", kernel_time);
    printf("DtoH: %f\n", d2h_time);
    printf("Close: %f\n", close_time);
    printf("Total: %f\n", total_time);
#endif
}

//========================================================================================================================================================================================================200
//	END KERNEL_GPU_OPENCL_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

//#ifdef __cplusplus
//}
//#endif
