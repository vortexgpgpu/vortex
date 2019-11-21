/*
 *  Simple OpenCL demo program
 *
 *  Copyright (C) 2009  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  gcc -o cldemo -std=gnu99 -Wall -I/usr/include/nvidia-current cldemo.c -lOpenCL
 *
 */

#include <CL/cl.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUM_DATA 65536

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

//
///
//  Retreive program binary for all of the devices attached to the
//  program an and store the one for the device passed in
//
bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName)
{
    //cl_uint numDevices = malloc(sizeof(cl_uint));
    //cl_uint* numDevices = malloc(sizeof(cl_uint));
    cl_int errNum;

    printf("try getting program info\n");
    // 1 - Query for number of devices attached to program
    /*errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                              &numDevices, NULL);
    printf("Got program_num_devices\n");
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for number of devices." << std::endl;
        return false;
    }*/

    // 2 - Get all of the Device IDs
    cl_device_id *devices = new cl_device_id[1];
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * 1,
                              devices, NULL);
    printf("Got program_devices\n");
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for devices." << std::endl;
        delete [] devices;
        return false;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = new size_t [1];
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * 1,
                              programBinarySizes, NULL);
    printf("Got program_binary_sizes\n");
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binary sizes." << std::endl;
        delete [] devices;
        delete [] programBinarySizes;
        return false;
    }

    unsigned char **programBinaries = new unsigned char*[1];
    for (cl_uint i = 0; i < 1; i++)
    {
        programBinaries[i] = new unsigned char[programBinarySizes[i]];
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * 1,
                              programBinaries, NULL);
    printf("Got program_binarys\n");
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binaries" << std::endl;

        delete [] devices;
        delete [] programBinarySizes;
        for (cl_uint i = 0; i < 1; i++)
        {
            delete [] programBinaries[i];
        }
        delete [] programBinaries;
        return false;
    }

    // 5 - Finally store the binaries for the device requested out to disk for future reading.
    for (cl_uint i = 0; i < 1; i++)
    {
        // Store the binary just for the device requested.  In a scenario where
        // multiple devices were being used you would save all of the binaries out here.
        if (devices[i] == device)
        {
            FILE *fp = fopen(fileName, "wb");
            if(fp ==NULL){
              delete [] devices;
              delete [] programBinarySizes;
              for (cl_uint i = 0; i < 1; i++)
              {
                  delete [] programBinaries[i];
              }
              delete [] programBinaries;
              return false;
            }
            printf("Opened file\n");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            printf("wrote file\n");
            fclose(fp);
            printf("close file\n");
            break;
        }
    }

    // Cleanup
    delete [] devices;
    delete [] programBinarySizes;
    for (cl_uint i = 0; i < 1; i++)
    {
        delete [] programBinaries[i];
    }
    delete [] programBinaries;
    return true;
}

///
//  Attempt to create the program object from a cached binary.  Note that
//  on first run this will fail because the binary has not yet been created.
//
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName)
{
    FILE *fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    // Determine the size of the binary
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);

    unsigned char *programBinary = new unsigned char[binarySize];
    fread(programBinary, 1, binarySize, fp);
    fclose(fp);

    cl_int errNum = 0;
    cl_program program;
    cl_int binaryStatus;

    program = clCreateProgramWithBinary(context,
                                        1,
                                        &device,
                                        &binarySize,
                                        (const unsigned char**)&programBinary,
                                        &binaryStatus,
                                        &errNum);
    delete [] programBinary;
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error loading program binary." << std::endl;
        return NULL;
    }

    if (binaryStatus != CL_SUCCESS)
    {
        std::cerr << "Invalid binary for device" << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("build errNum:%d\n", errNum);
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in program: " << std::endl;
        std::cerr << buildLog << std::endl;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

int main(int argc, char **argv)
{
  printf("enter demo main\n");
  fflush(stdout);
  putenv("POCL_VERBOSE=1");
  putenv("POCL_DEVICES=basic");
  putenv("POCL_LEAVE_TEMP_DIRS=1");
  putenv("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1");
  putenv("POCL_TEMP_DIR=pocl");
  putenv("POCL_CACHE_DIR=pocl");
  putenv("POCL_WORK_GROUP_METHOD=spmd");
  if(argc >= 2){
    printf("argv[1]:%s:\n",argv[1]);
    if(!strcmp(argv[1], "h"))
      putenv("POCL_WORK_GROUP_METHOD=spmd");
    if(!strcmp(argv[1], "c"))
      putenv("POCL_CROSS_COMPILE=1");
  }
  if(argc >= 3){
    printf("argv[2]:%s:\n",argv[2]);
    if(!strcmp(argv[2], "h"))
      putenv("POCL_WORK_GROUP_METHOD=spmd");
    if(!strcmp(argv[2], "c"))
      putenv("POCL_CROSS_COMPILE=1");
  }

  //putenv("LD_LIBRARY_PATH=/scratch/colins/build/linux/fs/lib");
  //putenv("LTDL_LIBRARY_PATH=/scratch/colins/build/linux/fs/lib");
  //lt_dlsetsearchpath("/scratch/colins/build/linux/fs/lib");
  //printf("SEARCH_PATH:%s\n",lt_dlgetsearchpath());
	cl_platform_id platforms[100];
	cl_uint platforms_n = 0;
	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

	printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
	for (int i=0; i<platforms_n; i++)
	{
		char buffer[10240];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}

	if (platforms_n == 0)
		return 1;

	cl_device_id devices[100];
	cl_uint devices_n = 0;
	// CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
	CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));

	printf("=== %d OpenCL device(s) found on platform:\n", platforms_n);
	for (int i=0; i<devices_n; i++)
	{
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
    size_t wi_size[3];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_NAME = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VENDOR = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DRIVER_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wi_size), &wi_size, NULL));
		printf("  DEVICE_MAX_WG_SIZE X=%ld,Y=%ld,Z=%ld\n", wi_size[0], wi_size[1], wi_size[2]);
	}

	if (devices_n == 0)
		return 1;

	cl_context context;
	context = CL_CHECK_ERR(clCreateContext(NULL, 1, devices+1, &pfn_notify, NULL, &_err));

	cl_command_queue queue;
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, devices[1], CL_QUEUE_PROFILING_ENABLE, &_err));

	cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0,0};


  // Create OpenCL program - first attempt to load cached binary.
  //  If that is not available, then create the program from source
  //  and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  cl_program program = CreateProgramFromBinary(context, devices[1], "kernel.cl.bin");
  if (program == NULL)
  {
      std::cout << "Binary not loaded, create from source..." << std::endl;
      program = CreateProgram(context, devices[1], "kernel.cl");
      if (program == NULL)
      {
          Cleanup(context, queue, program, kernel, memObjects);
          return 1;
      }

      std::cout << "Save program binary for future run..." << std::endl;
      if (SaveProgramBinary(program, devices[1], "kernel.cl.bin") == false)
      {
          std::cerr << "Failed to write program binary" << std::endl;
          Cleanup(context, queue, program, kernel, memObjects);
          return 1;
      }
  }
  else
  {
      std::cout << "Read program from binary." << std::endl;
  }

  printf("attempting to create input buffer\n");
  fflush(stdout);
	cl_mem input_buffer;
	input_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_DATA, NULL, &_err));

  printf("attempting to create output buffer\n");
  fflush(stdout);
	cl_mem output_buffer;
	output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*NUM_DATA, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  float factor = ((float)rand()/(float)(RAND_MAX)) * 100.0;

  printf("attempting to create kernel\n");
  fflush(stdout);
	kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy", &_err));
  printf("setting up kernel args cl_mem:%lx \n",input_buffer);
  fflush(stdout);
	CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
	CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
	CL_CHECK(clSetKernelArg(kernel, 2, sizeof(factor), &factor));

  printf("attempting to enqueue write buffer\n");
  fflush(stdout);
	for (int i=0; i<NUM_DATA; i++) {
    float in = ((float)rand()/(float)(RAND_MAX)) * 100.0;
		CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, i*sizeof(float), 4, &in, 0, NULL, NULL));
	}

	cl_event kernel_completion;
	size_t global_work_size[1] = { NUM_DATA };
  printf("attempting to enqueue kernel\n");
  fflush(stdout);
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &kernel_completion));
  printf("Enqueue'd kerenel\n");
  fflush(stdout);
  cl_ulong time_start, time_end;
  CL_CHECK(clWaitForEvents(1, &kernel_completion));
  CL_CHECK(clGetEventProfilingInfo(kernel_completion, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
  CL_CHECK(clGetEventProfilingInfo(kernel_completion, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
  double elapsed = time_end - time_start;
  printf("time(ns):%lg\n",elapsed);
	CL_CHECK(clReleaseEvent(kernel_completion));

	printf("Result:");
	for (int i=0; i<NUM_DATA; i++) {
		float data;
		CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, i*sizeof(float), 4, &data, 0, NULL, NULL));
		//printf(" %f", data);
	}
	printf("\n");

	CL_CHECK(clReleaseMemObject(memObjects[0]));
	CL_CHECK(clReleaseMemObject(memObjects[1]));

	CL_CHECK(clReleaseKernel(kernel));
	CL_CHECK(clReleaseProgram(program));
	CL_CHECK(clReleaseContext(context));

	return 0;
}

