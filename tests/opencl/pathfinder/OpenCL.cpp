#include <cstdlib>
#include "OpenCL.h"
#include "timing.h"

#define TIMING
#ifdef TIMING
	extern struct timeval tv;
	extern struct timeval tv_total_start, tv_total_end;
	extern struct timeval tv_close_start, tv_close_end;
	extern float init_time, mem_alloc_time, h2d_time, kernel_time,
		  d2h_time, close_time, total_time;
#endif

OpenCL::OpenCL(int displayOutput)
{
	VERBOSE = displayOutput;
}

OpenCL::~OpenCL()
{
	// Flush and kill the command queue...
	clFlush(command_queue);
	clFinish(command_queue);

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
	// Release each kernel in the map kernelArray
	map<string, cl_kernel>::iterator it;
	for ( it=kernelArray.begin() ; it != kernelArray.end(); it++ )
		clReleaseKernel( (*it).second );
		
	// Now the program...
	clReleaseProgram(program);
	
	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
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

size_t OpenCL::localSize()
{
	return this->lwsize;
}

cl_command_queue OpenCL::q()
{
	return this->command_queue;
}

void OpenCL::launch(string toLaunch)
{
	cl_event event;
	// Launch the kernel (or at least enqueue it).
	printf("global work size: %d, local work size: %d\n", (int)gwsize, (int)lwsize);
	ret = clEnqueueNDRangeKernel(command_queue, 
	                             kernelArray[toLaunch],
	                             1,
	                             NULL,
	                             &gwsize,
	                             &lwsize,
	                             0, 
	                             NULL, 
	                             &event);
	
	if (ret != CL_SUCCESS)
	{
		printf("\nError attempting to launch %s. Error in clCreateProgramWithSource with error code %i\n\n", toLaunch.c_str(), ret);
		exit(1);
	}
    kernel_time += probe_event_time(event, command_queue);
}

void OpenCL::gwSize(size_t theSize)
{
	this->gwsize = theSize;
}

cl_context OpenCL::ctxt()
{
	return this->context;
}

cl_kernel OpenCL::kernel(string kernelName)
{
	return this->kernelArray[kernelName];
}

void OpenCL::createKernel(string kernelName)
{
	cl_kernel kernel = clCreateKernel(this->program, kernelName.c_str(), NULL);
	kernelArray[kernelName] = kernel;
	
	// Get the kernel work group size.
	// clGetKernelWorkGroupInfo(kernelArray[kernelName], device_id[device_id_inuse], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &lwsize, NULL);
	lwsize = 32; // vortex
	if (lwsize == 0)
	{
		cout << "Error: clGetKernelWorkGroupInfo() returned a max work group size of zero!" << endl;
		exit(1);
	}
	
	// Local work size must divide evenly into global work size.
	size_t howManyThreads = lwsize;
	if (lwsize > gwsize)
	{
		lwsize = gwsize;
		printf("Using %zu for local work size. \n", lwsize);
	}
	else
	{
		while (gwsize % howManyThreads != 0)
		{
			howManyThreads--;
		}
		if (VERBOSE)
			printf("Max local threads is %zu. Using %zu for local work size. \n", lwsize, howManyThreads);

		this->lwsize = howManyThreads;
	}
}

void OpenCL::buildKernel()
{
	/* Load the source code for all of the kernels into the array source_str */
	FILE*  theFile;
	char*  source_str;
	size_t source_size;
	
	theFile = fopen("kernels.cl", "r");
	if (!theFile)
	{
		fprintf(stderr, "Failed to load kernel file.\n");
		exit(1);
	}
	// Obtain length of source file.
	fseek(theFile, 0, SEEK_END);
	source_size = ftell(theFile);
	rewind(theFile);
	// Read in the file.
	source_str = (char*) malloc(sizeof(char) * (source_size + 1));
	fread(source_str, 1, source_size, theFile);
	fclose(theFile);
	source_str[source_size] = '\0';

	// Create a program from the kernel source.
	program = clCreateProgramWithSource(context,
	                                    1,
	                                    (const char **) &source_str,
	                                    NULL,           // Number of chars in kernel src. NULL means src is null-terminated.
	                                    &ret);          // Return status message in the ret variable.

	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateProgramWithSource! Error code %i\n\n", ret);
		exit(1);
	}

	// Memory cleanup for the variable used to hold the kernel source.
	free(source_str);
	
	// Build (compile) the program.
	ret = clBuildProgram(program, NULL, NULL, NULL, NULL, NULL);
	
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clBuildProgram! Error code %i\n\n", ret);
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, device_id[device_id_inuse], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, device_id[device_id_inuse], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
		exit(1);
	}

	/* Show error info from building the program. */
	if (VERBOSE)
	{
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, device_id[device_id_inuse], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, device_id[device_id_inuse], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
	}
}

void OpenCL::getDevices()
{
	cl_uint         platforms_n = 0;
	cl_uint         devices_n   = 0;
	
	/* The following code queries the number of platforms and devices, and
	 * lists the information about both.
	 */
	clGetPlatformIDs(100, platform_id, &platforms_n);
	if (VERBOSE)
	{
		printf("\n=== %d OpenCL platform(s) found: ===\n", platforms_n);
		for (int i = 0; i < platforms_n; i++)
		{
			char buffer[10240];
			printf("  -- %d --\n", i);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 10240, buffer,
			                  NULL);
			printf("  PROFILE = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 10240, buffer,
			                  NULL);
			printf("  VERSION = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
			printf("  NAME = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
			printf("  VENDOR = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_EXTENSIONS, 10240, buffer,
			                  NULL);
			printf("  EXTENSIONS = %s\n", buffer);
		}
	}
	
	clGetDeviceIDs(platform_id[platform_id_inuse], CL_DEVICE_TYPE_ALL, 100, device_id, &devices_n);
	if (VERBOSE)
	{
		printf("Using the default platform (platform 0)...\n\n");
		printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
		for (int i = 0; i < devices_n; i++)
		{
			char buffer[10240];
			cl_uint buf_uint;
			cl_ulong buf_ulong;
			printf("  -- %d --\n", i);
			clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
			clGetDeviceInfo(device_id[i], CL_DEVICE_LOCAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  CL_DEVICE_LOCAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
		}
		printf("\n");
	}

    // Get device type
    /*
    cl_device_type device_type;
    ret = clGetDeviceInfo(device_id[device_id_inuse], CL_DEVICE_TYPE,
            sizeof(device_type), (void *)&device_type, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("ERROR: clGetDeviceIDs failed\n");
        exit(1);
    };
    */

	// Create an OpenCL context.
	context = clCreateContext(NULL, 1, &device_id[device_id_inuse], NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateContext! Error code %i\n\n", ret);
		exit(1);
	}

	// Create a command queue.
#ifdef TIMING
	command_queue = clCreateCommandQueue(context, device_id[device_id_inuse], CL_QUEUE_PROFILING_ENABLE, &ret);
#else
	command_queue = clCreateCommandQueue(context, device_id[device_id_inuse], 0, &ret);
#endif
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
		exit(1);
	}
}

void OpenCL::init()
{
	getDevices();

	buildKernel();
}
