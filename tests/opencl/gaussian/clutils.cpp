/****************************************************************************\
 * Copyright (c) 2011, Advanced Micro Devices, Inc.                           *
 * All rights reserved.                                                       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * Redistributions of source code must retain the above copyright notice,     *
 * this list of conditions and the following disclaimer.                      *
 *                                                                            *
 * Redistributions in binary form must reproduce the above copyright notice,  *
 * this list of conditions and the following disclaimer in the documentation  *
 * and/or other materials provided with the distribution.                     *
 *                                                                            *
 * Neither the name of the copyright holder nor the names of its contributors *
 * may be used to endorse or promote products derived from this software      *
 * without specific prior written permission.                                 *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               *
 *                                                                            *
 * If you use the software (in whole or in part), you shall adhere to all     *
 * applicable U.S., European, and other export laws, including but not        *
 * limited to the U.S. Export Administration Regulations (“EAR”), (15 C.F.R.  *
 * Sections 730 through 774), and E.U. Council Regulation (EC) No 1334/2000   *
 * of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR, you       *
 * hereby certify that, except pursuant to a license granted by the United    *
 * States Department of Commerce Bureau of Industry and Security or as        *
 * otherwise permitted pursuant to a License Exception under the U.S. Export  *
 * Administration Regulations ("EAR"), you will not (1) export, re-export or  *
 * release to a national of a country in Country Groups D:1, E:1 or E:2 any   *
 * restricted technology, software, or source code you receive hereunder,     *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such *
 * technology or software, if such foreign produced direct product is subject *
 * to national security controls as identified on the Commerce Control List   *
 *(currently found in Supplement 1 to Part 774 of EAR).  For the most current *
 * Country Group listings, or for additional information about the EAR or     *
 * your obligations under those regulations, please refer to the U.S. Bureau  *
 * of Industry and Security’s website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <CL/cl.h>

#include "clutils.h"
#include "utils.h"


// The following variables have file scope to simplify
// the utility functions

//! All discoverable OpenCL platforms
static cl_platform_id* platforms = NULL;
static cl_uint numPlatforms;

//! All discoverable OpenCL devices (one pointer per platform)
static cl_device_id* devices = NULL;
static cl_uint* numDevices = NULL;

//! The chosen OpenCL platform
static cl_platform_id platform = NULL;

//! The chosen OpenCL device
static cl_device_id device = NULL;

//! OpenCL context
static cl_context context = NULL;

//! OpenCL command queue
static cl_command_queue commandQueue = NULL;
static cl_command_queue commandQueueProf = NULL;
static cl_command_queue commandQueueNoProf = NULL;

//! Global status of events
static bool eventsEnabled = false;

//-------------------------------------------------------
//          Initialization and Cleanup
//-------------------------------------------------------

//! Initialize OpenCl environment on one device
/*!
    Init function for one device. Looks for supported devices and creates a context
    \return returns a context initialized
*/
/*cl_context cl_init(char devicePreference)
{
    cl_int status;

    // Discover and populate the platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_errChk(status, "Getting platform IDs", true);
    if (numPlatforms > 0)
    {
        // Get all the platforms
        platforms = (cl_platform_id*)alloc(numPlatforms *
            sizeof(cl_platform_id));

        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        cl_errChk(status, "Getting platform IDs", true);
    }
    else
    {
        // If no platforms are available, we shouldn't continue
        printf("No OpenCL platforms found\n");
        exit(-1);
    }

    // Allocate space for the device lists and lengths
    numDevices = (cl_uint*)alloc(sizeof(cl_uint)*numPlatforms);
    devices = (cl_device_id**)alloc(sizeof(cl_device_id*)*numPlatforms);

    // If a device preference was supplied, we'll limit the search of devices
    // based on type
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    if(devicePreference == 'c') {
        deviceType = CL_DEVICE_TYPE_CPU;
    }
    if(devicePreference == 'g') {
        deviceType = CL_DEVICE_TYPE_GPU;
    }

    // Traverse the platforms array printing information and
    // populating devices
    for(unsigned int i = 0; i < numPlatforms ; i++)
    {
        // Print out some basic info about the platform
        char* platformName = NULL;
        char* platformVendor = NULL;

        platformName = cl_getPlatformName(platforms[i]);
        platformVendor = cl_getPlatformVendor(platforms[i]);

        status = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices[i]);
        cl_errChk(status, "Getting device IDs", false);
        if(status != CL_SUCCESS) {
            printf("This is a known NVIDIA bug (if platform == AMD then die)\n");
            printf("Setting number of devices to 0 and continuing\n");
            numDevices[i] = 0;
        }

        printf("Platform %d (%d devices):\n", i, numDevices[i]);
        printf("\tName: %s\n", platformName);
        printf("\tVendor: %s\n", platformVendor);

        free(platformName);
        free(platformVendor);

        // Populate OpenCL devices if any exist
        if(numDevices[i] != 0)
        {
            // Allocate an array of devices of size "numDevices"
            devices[i] = (cl_device_id*)alloc(sizeof(cl_device_id)*numDevices[i]);

            // Populate Arrray with devices
            status = clGetDeviceIDs(platforms[i], deviceType, numDevices[i],
                devices[i], NULL);
            cl_errChk(status, "Getting device IDs", true);
        }

        // Print some information about each device
        for( unsigned int j = 0; j < numDevices[i]; j++)
        {
            char* deviceName = NULL;
            char* deviceVendor = NULL;

            printf("\tDevice %d:\n", j);

            deviceName = cl_getDeviceName(devices[i][j]);
            deviceVendor = cl_getDeviceVendor(devices[i][j]);

            printf("\t\tName: %s\n", deviceName);
            printf("\t\tVendor: %s\n", deviceVendor);

            free(deviceName);
            free(deviceVendor);
        }
    }

    // Hard-code in the platform/device to use, or uncomment 'scanf'
    // to decide at runtime
    cl_uint chosen_platform, chosen_device;
    // UNCOMMENT the following two lines to manually select device each time
    //printf("Enter Platform and Device No (Seperated by Space) \n");
    //scanf("%d %d", &chosen_platform, &chosen_device);
    chosen_platform = 0;
    chosen_device = 0;
    printf("Using Platform %d, Device %d \n", chosen_platform, chosen_device);

    // Do a sanity check of platform/device selection
    if(chosen_platform >= numPlatforms ||
        chosen_device >= numDevices[chosen_platform]) {
        printf("Invalid platform/device combination\n");
        exit(-1);
    }

    // Set the selected platform and device
    platform = platforms[chosen_platform];
    device = devices[chosen_platform][chosen_device];

    // Create the context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform), 0};
    context = clCreateContext(cps, 1, &device, NULL, NULL, &status);
    cl_errChk(status, "Creating context", true);

    // Create the command queue
    commandQueueProf = clCreateCommandQueue(context, device,
                            CL_QUEUE_PROFILING_ENABLE, &status);
    cl_errChk(status, "creating command queue", true);

    commandQueueNoProf = clCreateCommandQueue(context, device, 0, &status);
    cl_errChk(status, "creating command queue", true);

    if(eventsEnabled) {
        printf("Profiling enabled\n");
        commandQueue = commandQueueProf;
    }
    else {
        printf("Profiling disabled\n");
        commandQueue = commandQueueNoProf;
    }

    return context;
}*/

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


cl_context cl_init_context(int platform, int dev,int quiet) {
    int printInfo=1;
    if (platform >= 0 && dev >= 0) printInfo = 0;
	cl_int status;
	// Used to iterate through the platforms and devices, respectively

	// These will hold the platform and device we select (can potentially be
	// multiple, but we're just doing one for now)
	// cl_platform_id platform = NULL;

	/*status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (printInfo) printf("Number of platforms detected:%d\n", numPlatforms);

	// Print some information about the available platforms
	cl_platform_id *platforms = NULL;
	cl_device_id * devices = NULL;
	if (numPlatforms > 0)
	{
		// get all the platforms
		platforms = (cl_platform_id*)malloc(numPlatforms *
			sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);

		// Traverse the platforms array
		if (printInfo) printf("Checking For OpenCl Compatible Devices\n");
		for(unsigned int i = 0; i < numPlatforms ; i++)
		{
			char pbuf[100];
			if (printInfo) printf("Platform %d:\t", i);
			status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
				sizeof(pbuf), pbuf, NULL);
			if (printInfo) printf("Vendor: %s\n", pbuf);

			//unsigned int numDevices;

			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
			if(cl_errChk(status, "checking for devices",true))
				exit(1);
			if(numDevices == 0) {
				printf("There are no devices for Platform %d\n",i);
				exit(0);
			}
			else
			{
				if (printInfo) printf("\tNo of devices for Platform %d is %u\n",i, numDevices);
				//! Allocate an array of devices of size "numDevices"
				devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
				//! Populate Arrray with devices
				status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices,
					devices, NULL);
				if(cl_errChk(status, "getting device IDs",true)) {
					exit(1);
				}
			}
			for( unsigned int j = 0; j < numDevices; j++)
			{
				char dbuf[100];
				char deviceStr[100];
				if (printInfo) printf("\tDevice: %d\t", j);
				status = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(dbuf),
					deviceStr, NULL);
				cl_errChk(status, "Getting Device Info\n",true);
			    if (printInfo) printf("Vendor: %s", deviceStr);
				status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dbuf),
					dbuf, NULL);
				if (printInfo) printf("\n\t\tName: %s\n", dbuf);
			}
		}
	}
	else
	{
		// If no platforms are available, we're sunk!
		printf("No OpenCL platforms found\n");
		exit(0);
	}

	int platform_touse;
	unsigned int device_touse;
	if (printInfo) printf("Enter Platform and Device No (Seperated by Space) \n");
	if (printInfo) scanf("%d %d", &platform_touse, &device_touse);
	else {
	  platform_touse = platform;
	  device_touse = dev;
	}
	if (!quiet) printf("Using Platform %d \t Device No %d \n",platform_touse, device_touse);

	//! Recheck how many devices does our chosen platform have
	status = clGetDeviceIDs(platforms[platform_touse], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	if(device_touse > numDevices)
	{
		printf("Invalid Device Number\n");
		exit(1);
	}

	//! Populate devices array with all the visible devices of our chosen platform
	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	status = clGetDeviceIDs(platforms[platform_touse],
					CL_DEVICE_TYPE_ALL, numDevices,
					devices, NULL);
	if(cl_errChk(status,"Error in Getting Devices\n",true)) exit(1);


	//!Check if Device requested is a CPU or a GPU
	cl_device_type dtype;
	device = devices[device_touse];
	status = clGetDeviceInfo(devices[device_touse],
					CL_DEVICE_TYPE,
					sizeof(dtype),
					(void *)&dtype,
					NULL);
	if(cl_errChk(status,"Error in Getting Device Info\n",true)) exit(1);
	if(dtype == CL_DEVICE_TYPE_GPU) {
	  if (!quiet) printf("Creating GPU Context\n\n");
	}
	else if (dtype == CL_DEVICE_TYPE_CPU) {
      if (!quiet) printf("Creating CPU Context\n\n");
	}
	else perror("This Context Type Not Supported\n");

	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platforms[platform_touse]), 0};

	cl_context_properties *cprops = cps;

	context = clCreateContextFromType(
					cprops, (cl_device_type)dtype,
					NULL, NULL, &status);
	if(cl_errChk(status, "creating Context",true)) {
		exit(1);
	}*/

    // Getting platform and device information

    numPlatforms = 1;
    platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

    numDevices = (cl_uint*)malloc(sizeof(cl_uint)*numPlatforms);
    numDevices[0] = 1;
    devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices[0]);

    int platform_touse = 0;
    int device_touse = 0;

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    cl_errChk(status, "Oops!", true);
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, numDevices[0], devices, NULL);
    cl_errChk(status, "Oops!", true);
    context = clCreateContext(NULL, numDevices[0], devices, NULL, NULL,  &status);
    cl_errChk(status, "Oops!", true);

    char device_string[1024];
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("Using device: %s\n", device_string);

    device=devices[device_touse];

#ifdef PROFILING

	commandQueue = clCreateCommandQueue(context,
						devices[device_touse], CL_QUEUE_PROFILING_ENABLE, &status);

#else

	commandQueue = clCreateCommandQueue(context,
						devices[device_touse], 0, &status);

#endif // PROFILING

	if(cl_errChk(status, "creating command queue",true)) {
		exit(1);
	}
	return context;
}
/*!
    Release all resources that the user doesn't have access to.
*/
void cl_cleanup()
{
    cl_int status;

    // Free the command queue
    if (commandQueue) {
        status = clReleaseCommandQueue(commandQueue);
        cl_errChk(status, "Oops!", true);
        printf("clReleaseCommandQueue()\n");
    }

    // Free the context
    if (context) {
        status = clReleaseContext(context);
        cl_errChk(status, "Oops!", true);
        printf("clReleaseContext()\n");
    }

    for (cl_uint p = 0; p < numPlatforms; ++p) {
        for (cl_uint d = 0; d < numDevices[p]; ++d) {
            status = clReleaseDevice(devices[d]);
            cl_errChk(status, "Oops!", true);
            printf("clReleaseDevice()\n");
        }
    }

    free(devices);
    free(numDevices);
    free(platforms);
}

//! Release a kernel object
/*!
    \param mem The kernel object to release
*/
void cl_freeKernel(cl_kernel kernel)
{
    cl_int status;

    if(kernel != NULL) {
        status = clReleaseKernel(kernel);
        cl_errChk(status, "Releasing kernel object", true);
        printf("clReleaseKernel()\n");
    }
}

//! Release memory allocated on the device
/*!
    \param mem The device pointer to release
*/
void cl_freeMem(cl_mem mem)
{
    cl_int status;

    if(mem != NULL) {
        status = clReleaseMemObject(mem);
        cl_errChk(status, "Releasing mem object", true);
        printf("clReleaseMemObject()\n");
    }
}

//! Release a program object
/*!
    \param mem The program object to release
*/
void cl_freeProgram(cl_program program)
{
    cl_int status;

    if(program != NULL) {
        status = clReleaseProgram(program);
        cl_errChk(status, "Releasing program object", true);
        printf("clReleaseProgram()\n");
    }
}

//! Returns a reference to the command queue
/*!
	Returns a reference to the command queue \n
	Used for any OpenCl call that needs the command queue declared in clutils.cpp
*/
cl_command_queue cl_getCommandQueue()
{
	return commandQueue;
}

//-------------------------------------------------------
//          Synchronization functions
//-------------------------------------------------------

/*!
    Wait till all pending commands in queue are finished
*/
void cl_sync()
{
    clFinish(commandQueue);
}


//-------------------------------------------------------
//          Memory allocation
//-------------------------------------------------------

//! Allocate a buffer on a device
/*!
    \param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBuffer(size_t mem_size, cl_mem_flags flags)
{
    cl_mem mem;
    cl_int status;

    /*!
        Logging information for keeping track of device memory
    */
    static int allocationCount = 1;
    static size_t allocationSize = 0;

    allocationCount++;
    allocationSize += mem_size;

    mem = clCreateBuffer(context, flags, mem_size, NULL, &status);

    cl_errChk(status, "creating buffer", true);

    return mem;
}

//! Allocate constant memory on device
/*!
    \param mem_size Size of memory in bytes
    \param host_ptr Host pointer that contains the data
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBufferConst(size_t mem_size, void* host_ptr)
{
    cl_mem mem;
    cl_int status;

    mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         mem_size, host_ptr, &status);
    cl_errChk(status, "Error creating const mem buffer", true);

    return mem;
}

//! Allocate a buffer on device pinning the host memory at host_ptr
/*!
    \param mem_size Size of memory in bytes
    \return Returns a cl_mem object that points to pinned memory on the host
*/
cl_mem cl_allocBufferPinned(size_t mem_size)
{
    cl_mem mem;
    cl_int status;

    mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         mem_size, NULL, &status);
    cl_errChk(status, "Error allocating pinned memory", true);

    return mem;
}

//! Allocate an image on a device
/*!
    \param height Number of rows in the image
    \param width Number of columns in the image
    \param elemSize Size of the elements in the image
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocImage(size_t height, size_t width, char type, cl_mem_flags flags)
{
    cl_mem mem;
    cl_int status;

    size_t elemSize = 0;

    cl_image_format format;
    format.image_channel_order = CL_R;

    switch(type) {
    case 'f':
        elemSize = sizeof(float);
        format.image_channel_data_type = CL_FLOAT;
        break;
    case 'i':
        elemSize = sizeof(int);
        format.image_channel_data_type = CL_SIGNED_INT32;
        break;
    default:
        printf("Error creating image: Unsupported image type.\n");
        exit(-1);
    }

    /*!
        Logging information for keeping track of device memory
    */
    static int allocationCount = 1;
    static size_t allocationSize = 0;

    allocationCount++;
    allocationSize += height*width*elemSize;

    // Create the image
    mem = clCreateImage2D(context, flags, &format, width, height, 0, NULL, &status);

    //cl_errChk(status, "creating image", true);
    if(status != CL_SUCCESS) {
        printf("Error creating image: Images may not be supported for this device.\n");
        printSupportedImageFormats();
        getchar();
        exit(-1);
    }

    return mem;
}


//-------------------------------------------------------
//          Data transfers
//-------------------------------------------------------


// Copy and map a buffer
void* cl_copyAndMapBuffer(cl_mem dst, cl_mem src, size_t size) {

    void* ptr;  // Pointer to the pinned memory that will be returned

    cl_copyBufferToBuffer(dst, src, size);

    ptr = cl_mapBuffer(dst, size, CL_MAP_READ);

    return ptr;
}

// Copy a buffer
void cl_copyBufferToBuffer(cl_mem dst, cl_mem src, size_t size)
{
    cl_int status;
    status = clEnqueueCopyBuffer(commandQueue, src, dst, 0, 0, size, 0, NULL,
        NULL);
    cl_errChk(status, "Copying buffer", true);

}

//! Copy a buffer to the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param mem_size Size of data to copy
        \param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToDevice(cl_mem dst, void* src, size_t mem_size, cl_bool blocking)
{
    cl_int status;
    status = clEnqueueWriteBuffer(commandQueue, dst, blocking, 0,
        mem_size, src, 0, NULL, NULL);
    cl_errChk(status, "Writing buffer", true);

}

//! Copy a buffer to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param mem_size Size of data to copy
        \param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToHost(void* dst, cl_mem src, size_t mem_size, cl_bool blocking)
{
    cl_int status;
    status = clEnqueueReadBuffer(commandQueue, src, blocking, 0,
        mem_size, dst, 0, NULL, NULL);
    cl_errChk(status, "Reading buffer", true);

}

//! Copy a buffer to a 2D image
/*!
    \param src Valid device buffer
    \param dst Empty device image
    \param mem_size Size of data to copy
*/
void cl_copyBufferToImage(cl_mem buffer, cl_mem image, int height, int width)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    cl_int status;
    status = clEnqueueCopyBufferToImage(commandQueue, buffer, image, 0,
        origin, region, 0, NULL, NULL);
    cl_errChk(status, "Copying buffer to image", true);

}

// Copy data to an image on the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToDevice(cl_mem dst, void* src, size_t height, size_t width)
{
    cl_int status;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    status = clEnqueueWriteImage(commandQueue, dst, CL_TRUE, origin,
        region, 0, 0, src, 0, NULL, NULL);
    cl_errChk(status, "Writing image", true);
}

//! Copy an image to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToHost(void* dst, cl_mem src, size_t height, size_t width)
{
    cl_int status;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    status = clEnqueueReadImage(commandQueue, src, CL_TRUE, origin,
        region, 0, 0, dst, 0, NULL, NULL);
    cl_errChk(status, "Reading image", true);
}

//! Map a buffer into a host address
/*!
    \param mem cl_mem object
        \param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a host pointer that points to the mapped region
*/
void *cl_mapBuffer(cl_mem mem, size_t mem_size, cl_mem_flags flags)
{
    cl_int status;
    void *ptr;

    ptr = (void *)clEnqueueMapBuffer(commandQueue, mem, CL_TRUE, flags,
                                             0, mem_size, 0, NULL, NULL, &status);

    cl_errChk(status, "Error mapping a buffer", true);

    return ptr;
}

//! Unmap a buffer or image
/*!
    \param mem cl_mem object
    \param ptr A host pointer that points to the mapped region
*/
void cl_unmapBuffer(cl_mem mem, void *ptr)
{

    // TODO It looks like AMD doesn't support profiling unmapping yet. Leaving the
    //      commented code here until it's supported

    cl_int status;

    status = clEnqueueUnmapMemObject(commandQueue, mem, ptr, 0, NULL, NULL);

    cl_errChk(status, "Error unmapping a buffer or image", true);
}

void cl_writeToZCBuffer(cl_mem mem, void* data, size_t size)
{

    void* ptr;

    ptr = cl_mapBuffer(mem, size, CL_MAP_WRITE);

    memcpy(ptr, data, size);

    cl_unmapBuffer(mem, ptr);
}

//-------------------------------------------------------
//          Program and kernels
//-------------------------------------------------------

//! Convert source code file into cl_program
/*!
Compile Opencl source file into a cl_program. The cl_program will be made into a kernel in PrecompileKernels()

\param kernelPath  Filename of OpenCl code
\param compileoptions Compilation options
\param verbosebuild Switch to enable verbose Output
*/
cl_program cl_compileProgram(char* kernelPath, char* compileoptions, bool verbosebuild )
{
    cl_int status;
    FILE *fp = NULL;
    char *source = NULL;
    long int size;

    /*printf("\t%s\n", kernelPath);

    // Determine the size of the source file
#ifdef _WIN32
    fopen_s(&fp, kernelPath, "rb");
#else
    fp = fopen(kernelPath, "rb");
#endif
    if(!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if(status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }
    rewind(fp);

    // Allocate enough space for the source code
    source = (char *)alloc(size + 1);

    // fill with NULLs (just for fun)
    for (int i = 0; i < size+1; i++)  {
        source[i] = '\0';
    }

    // Read in the source code
    fread(source, 1, size, fp);
    source[size] = '\0';*/

    // read kernel binary from file
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    cl_program clProgramReturn;

    // Create the program object
    int err = read_kernel_file("kernel.cl", &kernel_bin, &kernel_size);
    cl_errChk(err, "read_kernel_file", true);
    clProgramReturn = clCreateProgramWithSource(context, 1, (const char **)&kernel_bin, &kernel_size, &status);
    free(kernel_bin);
    cl_errChk(status, "Creating program", true);

    //free(source);
    //fclose(fp);

    // Try to compile the program
    status = clBuildProgram(clProgramReturn, 0, NULL, compileoptions, NULL, NULL);
    if(cl_errChk(status, "Building program", false) || verbosebuild == 1)
    {

        cl_build_status build_status;

        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_STATUS,
            sizeof(cl_build_status), &build_status, NULL);

        if(build_status == CL_SUCCESS && verbosebuild == 0) {
            return clProgramReturn;
        }

        //char *build_log;
        size_t ret_val_size;
        printf("Device: %p",device);
        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG, 0,
            NULL, &ret_val_size);

        char *build_log = (char*)alloc(ret_val_size+1);

        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG,
            ret_val_size+1, build_log, NULL);

        // to be careful, terminate with \0
        // there's no information in the reference whether the string is 0
        // terminated or not
        build_log[ret_val_size] = '\0';

        printf("Build log:\n %s...\n", build_log);
        if(build_status != CL_SUCCESS) {
            getchar();
            exit(-1);
        }
        else
            return clProgramReturn;
    }

    // print the ptx information
    // printBinaries(clProgram);

    return clProgramReturn;
}

//! Create a kernel from compiled source
/*!
Create a kernel from compiled source

\param program  Compiled OpenCL program
\param kernel_name  Name of the kernel in the program
\return Returns a cl_kernel object for the specified kernel
*/
cl_kernel cl_createKernel(cl_program program, const char* kernel_name) {

    cl_kernel kernel;
    cl_int status;

    kernel = clCreateKernel(program, kernel_name, &status);
    cl_errChk(status, "Creating kernel", true);

    return kernel;
}

//! Set an argument for a OpenCL kernel
/*!
Set an argument for a OpenCL kernel

\param kernel The kernel for which the argument is being set
\param index The argument index
\param size The size of the argument
\param data A pointer to the argument
*/
void cl_setKernelArg(cl_kernel kernel, unsigned int index, size_t size,
                     void* data)
{
    cl_int status;
    status = clSetKernelArg(kernel, index, size, data);

    cl_errChk(status, "Setting kernel arg", true);
}


//-------------------------------------------------------
//          Profiling/events
//-------------------------------------------------------


//! Time kernel execution using cl_event
/*!
    Prints out the time taken between the start and end of an event
    \param event_time
*/
double cl_computeExecTime(cl_event event_time)
{
    cl_int status;
    cl_ulong starttime;
    cl_ulong endtime;

    double elapsed;

    status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_START,
                                          sizeof(cl_ulong), &starttime, NULL);
    cl_errChk(status, "profiling start", true);

    status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_END,
                                          sizeof(cl_ulong), &endtime, NULL);
    cl_errChk(status, "profiling end", true);

    // Convert to ms
    elapsed = (double)(endtime-starttime)/1000000.0;

    return elapsed;
}

//! Compute the elapsed time between two timer values
double cl_computeTime(cl_time start, cl_time end)
{
#ifdef _WIN32
    __int64 freq;
    int status;

    status = QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    if(status == 0) {
        perror("QueryPerformanceFrequency");
        exit(-1);
    }

    // Return time in ms
    return double(end-start)/(double(freq)/1000.0);
#else

    return end-start;
#endif
}

//! Grab the current time using a system-specific timer
void cl_getTime(cl_time* time)
{

#ifdef _WIN32
    int status = QueryPerformanceCounter((LARGE_INTEGER*)time);
    if(status == 0) {
        perror("QueryPerformanceCounter");
        exit(-1);
    }
#else
    // Use gettimeofday to get the current time
    struct timeval curTime;
    gettimeofday(&curTime, NULL);

    // Convert timeval into double
    *time = curTime.tv_sec * 1000 + (double)curTime.tv_usec/1000;
#endif
}



//-------------------------------------------------------
//          Error handling
//-------------------------------------------------------

//! OpenCl error code list
/*!
    An array of character strings used to give the error corresponding to the error code \n

    The error code is the index within this array
*/
char *cl_errs[MAX_ERR_VAL] = {
    (char *)"CL_SUCCESS",                         // 0
    (char *)"CL_DEVICE_NOT_FOUND",                //-1
    (char *)"CL_DEVICE_NOT_AVAILABLE",            //-2
    (char *)"CL_COMPILER_NOT_AVAILABLE",          //-3
    (char *)"CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
    (char *)"CL_OUT_OF_RESOURCES",                //-5
    (char *)"CL_OUT_OF_HOST_MEMORY",              //-6
    (char *)"CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
    (char *)"CL_MEM_COPY_OVERLAP",                //-8
    (char *)"CL_IMAGE_FORMAT_MISMATCH",           //-9
    (char *)"CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
    (char *)"CL_BUILD_PROGRAM_FAILURE",           //-11
    (char *)"CL_MAP_FAILURE",                     //-12
    (char *)"",                                   //-13
    (char *)"",                                   //-14
    (char *)"",                                   //-15
    (char *)"",                                   //-16
    (char *)"",                                   //-17
    (char *)"",                                   //-18
    (char *)"",                                   //-19
    (char *)"",                                   //-20
    (char *)"",                                   //-21
    (char *)"",                                   //-22
    (char *)"",                                   //-23
    (char *)"",                                   //-24
    (char *)"",                                   //-25
    (char *)"",                                   //-26
    (char *)"",                                   //-27
    (char *)"",                                   //-28
    (char *)"",                                   //-29
    (char *)"CL_INVALID_VALUE",                   //-30
    (char *)"CL_INVALID_DEVICE_TYPE",             //-31
    (char *)"CL_INVALID_PLATFORM",                //-32
    (char *)"CL_INVALID_DEVICE",                  //-33
    (char *)"CL_INVALID_CONTEXT",                 //-34
    (char *)"CL_INVALID_QUEUE_PROPERTIES",        //-35
    (char *)"CL_INVALID_COMMAND_QUEUE",           //-36
    (char *)"CL_INVALID_HOST_PTR",                //-37
    (char *)"CL_INVALID_MEM_OBJECT",              //-38
    (char *)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
    (char *)"CL_INVALID_IMAGE_SIZE",              //-40
    (char *)"CL_INVALID_SAMPLER",                 //-41
    (char *)"CL_INVALID_BINARY",                  //-42
    (char *)"CL_INVALID_BUILD_OPTIONS",           //-43
    (char *)"CL_INVALID_PROGRAM",                 //-44
    (char *)"CL_INVALID_PROGRAM_EXECUTABLE",      //-45
    (char *)"CL_INVALID_KERNEL_NAME",             //-46
    (char *)"CL_INVALID_KERNEL_DEFINITION",       //-47
    (char *)"CL_INVALID_KERNEL",                  //-48
    (char *)"CL_INVALID_ARG_INDEX",               //-49
    (char *)"CL_INVALID_ARG_VALUE",               //-50
    (char *)"CL_INVALID_ARG_SIZE",                //-51
    (char *)"CL_INVALID_KERNEL_ARGS",             //-52
    (char *)"CL_INVALID_WORK_DIMENSION ",         //-53
    (char *)"CL_INVALID_WORK_GROUP_SIZE",         //-54
    (char *)"CL_INVALID_WORK_ITEM_SIZE",          //-55
    (char *)"CL_INVALID_GLOBAL_OFFSET",           //-56
    (char *)"CL_INVALID_EVENT_WAIT_LIST",         //-57
    (char *)"CL_INVALID_EVENT",                   //-58
    (char *)"CL_INVALID_OPERATION",               //-59
    (char *)"CL_INVALID_GL_OBJECT",               //-60
    (char *)"CL_INVALID_BUFFER_SIZE",             //-61
    (char *)"CL_INVALID_MIP_LEVEL",               //-62
    (char *)"CL_INVALID_GLOBAL_WORK_SIZE"};       //-63

//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message
\return True if Error Seen, False if no error
*/
int cl_errChk(const cl_int status, const char * msg, bool exitOnErr)
{

    if(status != CL_SUCCESS) {
        printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);

        if(exitOnErr) {
            exit(-1);
        }

        return true;
    }
    return false;
}

// Queries the supported image formats for the device and prints
// them to the screen
 void printSupportedImageFormats()
{
    cl_uint numFormats;
    cl_int status;

    status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D,
        0, NULL, &numFormats);
    cl_errChk(status, "getting supported image formats", true);

    cl_image_format* imageFormats = NULL;
    imageFormats = (cl_image_format*)alloc(sizeof(cl_image_format)*numFormats);

    status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D,
        numFormats, imageFormats, NULL);

    printf("There are %d supported image formats\n", numFormats);

    cl_uint orders[]={CL_R,  CL_A, CL_INTENSITY, CL_LUMINANCE, CL_RG,
        CL_RA, CL_RGB, CL_RGBA, CL_ARGB, CL_BGRA};
    char  *orderstr[]={(char *)"CL_R", (char *)"CL_A",(char *)"CL_INTENSITY", (char *)"CL_LUMINANCE", (char *)"CL_RG",
        (char *)"CL_RA", (char *)"CL_RGB", (char *)"CL_RGBA", (char *)"CL_ARGB", (char *)"CL_BGRA"};

    cl_uint types[]={
        CL_SNORM_INT8 , CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
        CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,CL_SIGNED_INT8,
        CL_SIGNED_INT16,  CL_SIGNED_INT32, CL_UNSIGNED_INT8, CL_UNSIGNED_INT16,
        CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT};

    char * typesstr[]={
        (char *)"CL_SNORM_INT8" ,(char *)"CL_SNORM_INT16",(char *)"CL_UNORM_INT8",(char *)"CL_UNORM_INT16",
        (char *)"CL_UNORM_SHORT_565",(char *)"CL_UNORM_SHORT_555",(char *)"CL_UNORM_INT_101010",
        (char *)"CL_SIGNED_INT8",(char *)"CL_SIGNED_INT16",(char *)"CL_SIGNED_INT32",(char *)"CL_UNSIGNED_INT8",
        (char *)"CL_UNSIGNED_INT16",(char *)"CL_UNSIGNED_INT32",(char *)"CL_HALF_FLOAT",(char *)"CL_FLOAT"};

    printf("Supported Formats:\n");
    for(int i = 0; i < (int)numFormats; i++) {
        printf("\tFormat %d: ", i);

        for(int j = 0; j < (int)(sizeof(orders)/sizeof(cl_int)); j++) {
            if(imageFormats[i].image_channel_order == orders[j]) {
                printf("%s, ", orderstr[j]);
            }
        }
        for(int j = 0; j < (int)(sizeof(types)/sizeof(cl_int)); j++) {
            if(imageFormats[i].image_channel_data_type == types[j]) {
                printf("%s, ", typesstr[j]);
            }
        }
        printf("\n");
    }

    free(imageFormats);
}


//-------------------------------------------------------
//          Platform and device information
//-------------------------------------------------------

//! Returns true if AMD is the device vendor
bool cl_deviceIsAMD(cl_device_id dev) {

    bool retval = false;

    char* vendor = cl_getDeviceVendor(dev);

    if(strncmp(vendor, "Advanced", 8) == 0) {
        retval = true;
    }

    free(vendor);

    return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_deviceIsNVIDIA(cl_device_id dev) {

    bool retval = false;

    char* vendor = cl_getDeviceVendor(dev);

    if(strncmp(vendor, "NVIDIA", 6) == 0) {
        retval = true;
    }

    free(vendor);

    return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_platformIsNVIDIA(cl_platform_id plat) {

    bool retval = false;

    char* vendor = cl_getPlatformVendor(plat);

    if(strncmp(vendor, "NVIDIA", 6) == 0) {
        retval = true;
    }

    free(vendor);

    return retval;
}

//! Get the name of the vendor for a device
char* cl_getDeviceDriverVersion(cl_device_id dev)
{
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DRIVER_VERSION, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getDeviceName(cl_device_id dev)
{
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the name
    status = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting device name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_NAME, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting device name", true);

    return(devInfoStr);
}

//! Get the name of the vendor for a device
char* cl_getDeviceVendor(cl_device_id dev)
{
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
}

//! Get the name of the vendor for a device
char* cl_getDeviceVersion(cl_device_id dev)
{
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_VERSION, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getPlatformName(cl_platform_id platform)
{
    cl_int status;
    size_t platformInfoSize;
    char* platformInfoStr = NULL;

    // Print the name
    status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0,
        NULL, &platformInfoSize);
    cl_errChk(status, "Getting platform name", true);

    platformInfoStr = (char*)alloc(platformInfoSize);

    status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformInfoSize,
        platformInfoStr, NULL);
    cl_errChk(status, "Getting platform name", true);

    return(platformInfoStr);
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getPlatformVendor(cl_platform_id platform)
{
    cl_int status;
    size_t platformInfoSize;
    char* platformInfoStr = NULL;

    // Print the name
    status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0,
        NULL, &platformInfoSize);
    cl_errChk(status, "Getting platform name", true);

    platformInfoStr = (char*)alloc(platformInfoSize);

    status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformInfoSize,
        platformInfoStr, NULL);
    cl_errChk(status, "Getting platform name", true);

    return(platformInfoStr);
}

//-------------------------------------------------------
//          Utility functions
//-------------------------------------------------------

//! Take a string and an int, and return a string
char* catStringWithInt(const char* string, int integer) {

    if(integer > 99999) {
        printf("Can't handle event identifiers with 6 digits\n");
        exit(-1);
    }

    // 5 characters for the identifier, 1 for the null terminator
    int strLen = strlen(string)+5+1;
    char* eventStr = (char*)alloc(sizeof(char)*strLen);

    char tmp[6];

    strcpy(eventStr, string);
    strncat(eventStr, itoa_portable(integer, tmp, 10), 5);

    return eventStr;
}

/**
 ** C++ version 0.4 char* style "itoa":
 ** Written by Lukás Chmela
 ** Released under GPLv3.
 **/
//portable itoa function
char* itoa_portable(int value, char* result, int base) {
    // check that the base if valid
    if (base < 2 || base > 36) { *result = '\0'; return result; }

    char* ptr = result, *ptr1 = result, tmp_char;
    int tmp_value;

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
    } while ( value );

    //Apply negative sign
    if (tmp_value < 0) *ptr++ = '-';
    *ptr-- = '\0';

    while(ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr--= *ptr1;
        *ptr1++ = tmp_char;
    }

    return result;
}