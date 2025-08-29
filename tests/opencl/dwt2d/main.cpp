#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <vector>
#include "common.h"
#include "components.h"
#include "dwt.h"
//using namespace std;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/opencl.h> 
#endif
 
#define THREADS 256

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};



cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_device_id cldevice = 0;
cl_kernel kernel = 0;
cl_kernel c_CopySrcToComponents = 0;
cl_kernel c_CopySrcToComponent = 0;
cl_kernel kl_fdwt53Kernel;
cl_mem memObjects[3] = { 0, 0, 0 };
cl_int errNum = 0;



///
// functions for preparing create opencl program, contains CreateContext, CreateProgram, CreateCommandQueue, CreateMemBuffer, and Cleanup
// Create an OpenCL context on the first available GPU platform. 
cl_context CreateContext()
{
    cl_context context = NULL;
    cl_uint platformIdCount = 0;
    cl_int errNum;

    // get number of platforms
    clGetPlatformIDs (0, NULL, &platformIdCount);
    
    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return NULL;
    }

    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data(), NULL);
	
    // Vortex: Use the first available platform instead of hardcoding platform index 1
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIds[0],
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }
    
    return context;

}

///
//  Create a command queue on the first device available on the context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *cldevice)
{
    cl_int errNum;
    cl_device_id *cldevices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    cldevices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, cldevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] cldevices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    commandQueue = clCreateCommandQueue(context, cldevices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] cldevices;
        std::cerr << "Failed to create commandQueue for device ";
        return NULL;
    }

    *cldevice = cldevices[0];
    delete [] cldevices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id cldevice, const char* fileName)
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
        clGetProgramBuildInfo(program, cldevice, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel)
{

    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

///
// Load the input image.
//
int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
    // printf("Loading ipnput: %s\n", srcFilename);
    // char path[] = "../../data/dwt2d/";
    char path[] = ""; // vortex

    char *newSrc = NULL;
    
    if((newSrc = (char *)malloc(strlen(srcFilename)+strlen(path)+1)) != NULL)
    {
        newSrc[0] = '\0';
        strcat(newSrc, path);
        strcat(newSrc, srcFilename);
        srcFilename= newSrc;
    }
    printf("Loading ipnput: %s\n", srcFilename);

    //read image
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) 
	{ 
        error(0,errno,"cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);

    return 0;
}

///
//Show user how to use this program
//
void usage() {
    printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
  -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
  -c, --components\t\tnumber of color components, default 3\n\
  -b, --depth\t\t\tbit depth, default 8\n\
  -l, --level\t\t\tDWT level, default 3\n\
  -D, --device\t\t\tcuda device\n\
  -f, --forward\t\t\tforward transform\n\
  -r, --reverse\t\t\treverse transform\n\
  -9, --97\t\t\t9/7 transform\n\
  -5, --53\t\t\t5/3 transform\n\
  -w  --write-visual\t\twrite output in visual (tiled) fashion instead of the linear\n");
}

///
// Check the type of error about opencl program
//
void fatal_CL(cl_int error, int line_no)
{

	printf("At line %d: ", line_no);

	switch(error) {

		case CL_SUCCESS: 									printf("CL_SUCCESS\n"); break;
		case CL_DEVICE_NOT_FOUND: 							printf("CL_DEVICE_NOT_FOUND\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: 						printf("CL_DEVICE_NOT_AVAILABLE\n"); break;
		case CL_COMPILER_NOT_AVAILABLE: 					printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: 				printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
		case CL_OUT_OF_RESOURCES: 							printf("CL_OUT_OF_RESOURCES\n"); break;
		case CL_OUT_OF_HOST_MEMORY: 						printf("CL_OUT_OF_HOST_MEMORY\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: 				printf("CL_PROFILING_INFO_NOT_AVAILABLE\n"); break;
		case CL_MEM_COPY_OVERLAP: 							printf("CL_MEM_COPY_OVERLAP\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: 						printf("CL_IMAGE_FORMAT_MISMATCH\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: 				printf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: 						printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
		case CL_MAP_FAILURE: 								printf("CL_MAP_FAILURE\n"); break;
		case CL_INVALID_VALUE: 								printf("CL_INVALID_VALUE\n"); break;
		case CL_INVALID_DEVICE_TYPE: 						printf("CL_INVALID_DEVICE_TYPE\n"); break;
		case CL_INVALID_PLATFORM: 							printf("CL_INVALID_PLATFORM\n"); break;
		case CL_INVALID_DEVICE: 							printf("CL_INVALID_DEVICE\n"); break;
		case CL_INVALID_CONTEXT: 							printf("CL_INVALID_CONTEXT\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: 					printf("CL_INVALID_QUEUE_PROPERTIES\n"); break;
		case CL_INVALID_COMMAND_QUEUE: 						printf("CL_INVALID_COMMAND_QUEUE\n"); break;
		case CL_INVALID_HOST_PTR: 							printf("CL_INVALID_HOST_PTR\n"); break;
		case CL_INVALID_MEM_OBJECT: 						printf("CL_INVALID_MEM_OBJECT\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n"); break;
		case CL_INVALID_IMAGE_SIZE: 						printf("CL_INVALID_IMAGE_SIZE\n"); break;
		case CL_INVALID_SAMPLER: 							printf("CL_INVALID_SAMPLER\n"); break;
		case CL_INVALID_BINARY: 							printf("CL_INVALID_BINARY\n"); break;
		case CL_INVALID_BUILD_OPTIONS: 						printf("CL_INVALID_BUILD_OPTIONS\n"); break;
		case CL_INVALID_PROGRAM: 							printf("CL_INVALID_PROGRAM\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: 				printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
		case CL_INVALID_KERNEL_NAME: 						printf("CL_INVALID_KERNEL_NAME\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: 					printf("CL_INVALID_KERNEL_DEFINITION\n"); break;
		case CL_INVALID_KERNEL: 							printf("CL_INVALID_KERNEL\n"); break;
		case CL_INVALID_ARG_INDEX: 							printf("CL_INVALID_ARG_INDEX\n"); break;
		case CL_INVALID_ARG_VALUE: 							printf("CL_INVALID_ARG_VALUE\n"); break;
		case CL_INVALID_ARG_SIZE: 							printf("CL_INVALID_ARG_SIZE\n"); break;
		case CL_INVALID_KERNEL_ARGS: 						printf("CL_INVALID_KERNEL_ARGS\n"); break;
		case CL_INVALID_WORK_DIMENSION: 					printf("CL_INVALID_WORK_DIMENSION\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: 					printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: 					printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: 						printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: 					printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
		case CL_INVALID_EVENT: 								printf("CL_INVALID_EVENT\n"); break;
		case CL_INVALID_OPERATION: 							printf("CL_INVALID_OPERATION\n"); break;
		case CL_INVALID_GL_OBJECT: 							printf("CL_INVALID_GL_OBJECT\n"); break;
		case CL_INVALID_BUFFER_SIZE: 						printf("CL_INVALID_BUFFER_SIZE\n"); break;
		case CL_INVALID_MIP_LEVEL: 							printf("CL_INVALID_MIP_LEVEL\n"); break;
		case CL_INVALID_GLOBAL_WORK_SIZE: 					printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;

		#ifdef CL_VERSION_1_1
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: 				printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
		#endif

		default:											printf("Invalid OpenCL error code\n");

	}
}



///
// Separate compoents of 8bit RGB source image
//in file components.cu
void rgbToComponents(cl_mem d_r, cl_mem d_g, cl_mem d_b, unsigned char * h_src, int width, int height)
{
    int pixels      = width * height;
    int alignedSize =  DIVANDRND(width*height, THREADS) * THREADS * 3; //aligned to thread block size -- THREADS
	
	cl_mem cl_d_src;
	cl_d_src = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixels*3, h_src, &errNum);
	// fatal_CL(errNum, __LINE__);

	size_t globalWorkSize[1] = { alignedSize/3};
    size_t localWorkSize[1] = { THREADS };

    printf("globalWorkSize[0]=%zu, localWorkSize[0]=%zu\n", globalWorkSize[0], localWorkSize[0]);

	errNum  = clSetKernelArg(c_CopySrcToComponents, 0, sizeof(cl_mem), &d_r);
	errNum |= clSetKernelArg(c_CopySrcToComponents, 1, sizeof(cl_mem), &d_g);
	errNum |= clSetKernelArg(c_CopySrcToComponents, 2, sizeof(cl_mem), &d_b);
	errNum |= clSetKernelArg(c_CopySrcToComponents, 3, sizeof(cl_mem), &cl_d_src);
	errNum |= clSetKernelArg(c_CopySrcToComponents, 4, sizeof(int), &pixels);
	// fatal_CL(errNum, __LINE__);	
	
	errNum = clEnqueueNDRangeKernel(commandQueue, c_CopySrcToComponents, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	// fatal_CL(errNum, __LINE__);
	
    // Free Memory 
	errNum = clReleaseMemObject(cl_d_src);  
	// fatal_CL(errNum, __LINE__);	
}



///
// Copy a 8bit source image data into a color compoment
//in file components.cu
void bwToComponent(cl_mem d_c, unsigned char * h_src, int width, int height)
{
	cl_mem cl_d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(pixels, THREADS) * THREADS;
	
	cl_d_src = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixels, h_src, NULL);
	// fatal_CL(errNum, __LINE__);
	
	size_t globalWorkSize[1] = { alignedSize/9};
    size_t localWorkSize[1] = { THREADS };
	assert(alignedSize%(THREADS*3) == 0);
	
	errNum  = clSetKernelArg(c_CopySrcToComponent, 0, sizeof(cl_mem), &d_c);
	errNum |= clSetKernelArg(c_CopySrcToComponent, 1, sizeof(cl_mem), &cl_d_src);
	errNum |= clSetKernelArg(c_CopySrcToComponent, 2, sizeof(int), &pixels);
	// fatal_CL(errNum, __LINE__);	
	
	errNum = clEnqueueNDRangeKernel(commandQueue, c_CopySrcToComponent, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	std::cout<<"in function bwToComponent errNum= "<<errNum<<"\n"; 
	// fatal_CL(errNum, __LINE__);
	std::cout<<"bwToComponent has finished\n";
	
    // Free Memory 
	errNum = clReleaseMemObject(cl_d_src);  
	// fatal_CL(errNum, __LINE__);	

}



/// Only computes optimal number of sliding window steps, number of threadblocks and then lanches the 5/3 FDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param in       input image
/// @param out      output buffer
/// @param sx       width of the input image 
/// @param sy       height of the input image
///launchFDWT53Kerneld is in file 
void launchFDWT53Kernel (int WIN_SX, int WIN_SY, cl_mem in, cl_mem out, int sx, int sy)
{
    printf("DEBUG: Entering launchFDWT53Kernel with WIN_SX=%d, WIN_SY=%d, sx=%d, sy=%d\n", WIN_SX, WIN_SY, sx, sy);
    
    // compute optimal number of steps of each sliding window
	// cuda_dwt called a function divRndUp from namespace cuda_gwt. this function takes n and d, "return (n / d) + ((n % d) ? 1 : 0);"
	//
	
    const int steps = ( sy/ (15 * WIN_SY)) + ((sy % (15 * WIN_SY)) ? 1 : 0);	
	
	int gx = ( sx/ WIN_SX) + ((sx %  WIN_SX) ? 1 : 0);  //use function divRndUp(n, d){return (n / d) + ((n % d) ? 1 : 0);}
	int gy = ( sy/ (WIN_SY*steps)) + ((sy %  (WIN_SY*steps)) ? 1 : 0);
	
	printf("sliding steps = %d , gx = %d , gy = %d \n", steps, gx, gy);
	
    // prepare grid size
	size_t globalWorkSize[2] = { gx*WIN_SX, gy*1};
    size_t localWorkSize[2]  = { WIN_SX , 1};
    // printf("\n globalx=%d, globaly=%d, blocksize=%d\n", gx, gy, WIN_SX);
	
    printf("DEBUG: Setting kernel arguments...\n");
	errNum  = clSetKernelArg(kl_fdwt53Kernel, 0, sizeof(cl_mem), &in);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 1, sizeof(cl_mem), &out);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 2, sizeof(int), &sx);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 3, sizeof(int), &sy);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 4, sizeof(int), &steps);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 5, sizeof(int), &WIN_SX);
	errNum |= clSetKernelArg(kl_fdwt53Kernel, 6, sizeof(int), &WIN_SY);
	if (errNum != CL_SUCCESS) {
        printf("DEBUG: Error setting kernel arguments: %d\n", errNum);
        fatal_CL(errNum, __LINE__);
    }
	
    printf("DEBUG: Enqueueing kernel execution...\n");
	errNum = clEnqueueNDRangeKernel(commandQueue, kl_fdwt53Kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS) {
        printf("DEBUG: Error enqueueing kernel: %d\n", errNum);
        fatal_CL(errNum, __LINE__);
    }
    
    printf("DEBUG: Waiting for kernel to finish...\n");
    clFinish(commandQueue);
    printf("DEBUG: Kernel execution completed successfully\n");
	printf("kl_fdwt53Kernel in launchFDW53Kernel has finished\n");
	

}


/// Simple cudaMemcpy wrapped in performance tester.
/// param dest  destination bufer
/// param src   source buffer
/// param sx    width of copied image
/// param sy    height of copied image
///from /cuda_gwt/common.h/namespace
void memCopy (cl_mem dest,  cl_mem src, const size_t sx, const size_t sy){

    printf("DEBUG: Entering memCopy with sx=%zu, sy=%zu, size=%zu bytes\n", sx, sy, sx*sy*sizeof(int));
	errNum = clEnqueueCopyBuffer (commandQueue, src, dest, 0, 0, sx*sy*sizeof(int), 0, NULL, NULL);
	if (errNum != CL_SUCCESS) {
        printf("DEBUG: Error in memCopy: %d\n", errNum);
        fatal_CL(errNum, __LINE__);
    }
    printf("DEBUG: memCopy completed successfully\n");

}



/// Forward 5/3 2D DWT. See common rules (above) for more details.
/// @param in      Expected to be normalized into range [-128, 127].
///                Will not be preserved (will be overwritten).
/// @param out     output buffer on GPU
/// @param sizeX   width of input image (in pixels)
/// @param sizeY   height of input image (in pixels)
/// @param levels  number of recursive DWT levels
/// @backup use to test time
//at the end of namespace dwt_cuda (line338)
void fdwt53(cl_mem in, cl_mem out, int sizeX, int sizeY, int levels)
{
    printf("DEBUG: Entering fdwt53 with sizeX=%d, sizeY=%d, levels=%d\n", sizeX, sizeY, levels);
    
    // select right width of kernel for the size of the image
	
    if(sizeX >= 960) 
	{
      launchFDWT53Kernel(192, 8, in, out, sizeX, sizeY);
    } 
	else if (sizeX >= 480) 
	{
      launchFDWT53Kernel(128, 8, in, out, sizeX, sizeY);
    } else 
	{
      launchFDWT53Kernel(64, 8, in, out, sizeX, sizeY);
	}
	
    printf("DEBUG: Kernel launch completed for level %d\n", levels);
		
	// if this was not the last level, continue recursively with other levels
	if (levels > 1)
	{
		// copy output's LL band back into input buffer 
		const int llSizeX = (sizeX / 2) + ((sizeX % 2) ? 1 :0);
		const int llSizeY = (sizeY / 2) + ((sizeY % 2) ? 1 :0);
        printf("DEBUG: About to call memCopy for recursive level, llSizeX=%d, llSizeY=%d\n", llSizeX, llSizeY);
		memCopy(in, out, llSizeX, llSizeY);
		
        printf("DEBUG: memCopy completed, about to call fdwt53 recursively\n");
		// run remaining levels of FDWT
		fdwt53(in, out, llSizeX, llSizeY, levels - 1);
        printf("DEBUG: Recursive fdwt53 call completed for level %d\n", levels);
	} else {
        printf("DEBUG: Final level reached, no more recursion\n");
    }
    printf("DEBUG: Exiting fdwt53 for level %d\n", levels);
}
	

///
// in dwt.cu
int nStage2dDWT(cl_mem in, cl_mem out, cl_mem backup, int pixWidth, int pixHeight, int stages, bool forward)
{
    printf("\n*** %d stages of 2D forward DWT:\n", stages);

    // create backup of input, because each test iteration overwrites it 
    const int size = pixHeight * pixWidth * sizeof(int);
	
	// Measure time of individual levels. 
	if (forward)
		fdwt53(in, out, pixWidth, pixHeight, stages );
	//else
	//	rdwt(in, out, pixWidth, pixHeight, stages);
	// rdwt means rdwt53(can be found in file rdwt53.cu) which has not been defined 
	
	
	return 0;
}



///
//in file dwt.cu
void samplesToChar(unsigned char * dst, int * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++)
	{
        int r = src[i]+128;
        if (r > 255) r = 255;
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}



///
//in file dwt.cu
/// Write output linear orderd
int writeLinear(cl_mem component, int pixWidth, int pixHeight, const char * filename, const char * suffix)
{
	unsigned char * result;
    int *gpu_output;
    int i;
    int size;
    int samplesNum = pixWidth*pixHeight;
	
	size = samplesNum*sizeof(int);
	gpu_output = (int *)malloc(size);
    memset(gpu_output, 0, size);
	result = (unsigned char *)malloc(samplesNum);
	
	errNum = clEnqueueReadBuffer(commandQueue, component, CL_TRUE, 0, size, gpu_output, 0, NULL, NULL);
	// fatal_CL(errNum, __LINE__);	
	
	// T to char 
	samplesToChar(result, gpu_output, samplesNum);
	
	// Write component 
	char outfile[strlen(filename)+strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile+strlen(filename), suffix);
    i = open(outfile, O_CREAT|O_WRONLY, 0644);
	if (i == -1) 
	{ 
        error(0,errno,"cannot access %s", outfile);
        return -1;
    }
	printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
    write(i, result, samplesNum);
    close(i);
	
	// Clean up 
    free(gpu_output);
    free(result);

    return 0;
}



///
// Write output visual ordered 
//in file dwt.cu
int writeNStage2DDWT(cl_mem component, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix)
{
	struct band {
        int dimX; 
        int dimY;
    };
    struct dimensions {
        struct band LL;
        struct band HL;
        struct band LH;
        struct band HH;
    };

    unsigned char * result;
    int *src;
	int	*dst;
    int i,s;
    int size;
    int offset;
    int yOffset;
    int samplesNum = pixWidth*pixHeight;
    struct dimensions * bandDims;
	
	bandDims = (struct dimensions *)malloc(stages * sizeof(struct dimensions));

    bandDims[0].LL.dimX = DIVANDRND(pixWidth,2);
    bandDims[0].LL.dimY = DIVANDRND(pixHeight,2);
    bandDims[0].HL.dimX = pixWidth - bandDims[0].LL.dimX;
    bandDims[0].HL.dimY = bandDims[0].LL.dimY;
    bandDims[0].LH.dimX = bandDims[0].LL.dimX;
    bandDims[0].LH.dimY = pixHeight - bandDims[0].LL.dimY;
    bandDims[0].HH.dimX = bandDims[0].HL.dimX;
    bandDims[0].HH.dimY = bandDims[0].LH.dimY;
	
	for (i = 1; i < stages; i++) 
	{
        bandDims[i].LL.dimX = DIVANDRND(bandDims[i-1].LL.dimX,2);
        bandDims[i].LL.dimY = DIVANDRND(bandDims[i-1].LL.dimY,2);
        bandDims[i].HL.dimX = bandDims[i-1].LL.dimX - bandDims[i].LL.dimX;
        bandDims[i].HL.dimY = bandDims[i].LL.dimY;
        bandDims[i].LH.dimX = bandDims[i].LL.dimX;
        bandDims[i].LH.dimY = bandDims[i-1].LL.dimY - bandDims[i].LL.dimY;
        bandDims[i].HH.dimX = bandDims[i].HL.dimX;
        bandDims[i].HH.dimY = bandDims[i].LH.dimY;
    }
	
#if 0
    printf("Original image pixWidth x pixHeight: %d x %d\n", pixWidth, pixHeight);
    for (i = 0; i < stages; i++) 
	{
        printf("Stage %d: LL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LL.dimX, bandDims[i].LL.dimY);
        printf("Stage %d: HL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HL.dimX, bandDims[i].HL.dimY);
        printf("Stage %d: LH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LH.dimX, bandDims[i].LH.dimY);
        printf("Stage %d: HH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HH.dimX, bandDims[i].HH.dimY);
    }
#endif

	size = samplesNum*sizeof(int);	
	
	src = (int *)malloc(size);
    memset(src, 0, size);
	dst = (int *)malloc(size);
    memset(dst, 0, size);
	result = (unsigned char *)malloc(samplesNum);

	errNum = clEnqueueReadBuffer(commandQueue, component, CL_TRUE, 0, size, src, 0, NULL, NULL);
	// fatal_CL(errNum, __LINE__);	
	

	// LL Band 	
	size = bandDims[stages-1].LL.dimX * sizeof(int);
	for (i = 0; i < bandDims[stages-1].LL.dimY; i++) 
	{
        memcpy(dst+i*pixWidth, src+i*bandDims[stages-1].LL.dimX, size);
	}
    
    for (s = stages - 1; s >= 0; s--) {
        // HL Band
        size = bandDims[s].HL.dimX * sizeof(int);
        offset = bandDims[s].LL.dimX * bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) 
		{
            memcpy(dst+i*pixWidth+bandDims[s].LL.dimX,
                src+offset+i*bandDims[s].HL.dimX, 
                size);
        }

        // LH band
		size = bandDims[s].LH.dimX * sizeof(int);
        offset += bandDims[s].HL.dimX * bandDims[s].HL.dimY;
        yOffset = bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) 
		{
            memcpy(dst+(yOffset+i)*pixWidth,
                src+offset+i*bandDims[s].LH.dimX, 
                size);
        }
	
		//HH band
        size = bandDims[s].HH.dimX * sizeof(int);
        offset += bandDims[s].LH.dimX * bandDims[s].LH.dimY;
        yOffset = bandDims[s].HL.dimY;
        for (i = 0; i < bandDims[s].HH.dimY; i++) 
		{
            memcpy(dst+(yOffset+i)*pixWidth+bandDims[s].LH.dimX,
                src+offset+i*bandDims[s].HH.dimX, 
                size);
        }
	}
	
    // Write component
	samplesToChar(result, dst, samplesNum);	
	
	char outfile[strlen(filename)+strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile+strlen(filename), suffix);
    i = open(outfile, O_CREAT|O_WRONLY, 0644);
	
    if (i == -1) 
	{
        error(0,errno,"cannot access %s", outfile);
        return -1;
    }
	
    printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
    write(i, result, samplesNum);
    close(i);
	
	free(src);
	free(dst);
    free(result);
    free(bandDims);
	
	return 0;
}




///
// Process of DWT algorithm
//
template <typename T>
void processDWT(struct dwt *d, int forward, int writeVisual)
{
	
	int componentSize = d->pixWidth * d->pixHeight * sizeof(T);
    
    T *c_r_out, *c_g_out, *c_b_out, *backup, *c_r, *c_g, *c_b;
	
	// initialize to zeros
	T *temp = (T *)malloc(componentSize);
	memset(temp, 0, componentSize);
	
	cl_mem cl_c_r_out;
	cl_c_r_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, componentSize, temp, &errNum);
	// fatal_CL(errNum, __LINE__);
	
	
	
	cl_mem cl_backup;
	cl_backup  = clCreateBuffer(context, CL_MEM_READ_WRITE |CL_MEM_COPY_HOST_PTR, componentSize, temp, &errNum);  
	// fatal_CL(errNum, __LINE__);
	
	if (d->components == 3) {
		// Alloc two more buffers for G and B 
		cl_mem cl_c_g_out;
		cl_c_g_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, componentSize, temp, &errNum);  
		// fatal_CL(errNum, __LINE__);
		
		cl_mem cl_c_b_out;
		cl_c_b_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, componentSize, temp, &errNum);
        // fatal_CL(errNum, __LINE__);	
        
        // Load components 
        cl_mem cl_c_r, cl_c_g, cl_c_b;
		cl_c_r = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,componentSize, temp, &errNum);
		// fatal_CL(errNum, __LINE__); 		
		cl_c_g = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,componentSize, temp, &errNum);
		// fatal_CL(errNum, __LINE__);		
		cl_c_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,componentSize, temp, &errNum); 
		// fatal_CL(errNum, __LINE__);
		
		
        rgbToComponents(cl_c_r, cl_c_g, cl_c_b, d->srcImg, d->pixWidth, d->pixHeight);
       
        
        //Compute DWT and always store int file
       
        nStage2dDWT(cl_c_r, cl_c_r_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(cl_c_g, cl_c_g_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(cl_c_b, cl_c_b_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        
        
        // ---------test----------
/*      T *h_r_out=(T*)malloc(componentSize);
		errNum = clEnqueueReadBuffer(commandQueue, cl_c_g_out, CL_TRUE, 0, componentSize, h_r_out, 0, NULL, NULL); 
		fatal_CL(errNum, __LINE__);
        int ii;
		for(ii=0;ii<componentSize/sizeof(T);ii++) {
			fprintf(stderr, "%d ", (int)h_r_out[ii]);
			if((ii+1) % (d->pixWidth) == 0) fprintf(stderr, "\n");
        }
*/        // ---------test----------

#ifdef OUTPUT        
        // Store DWT to file
        if(writeVisual){
            writeNStage2DDWT(cl_c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
            writeNStage2DDWT(cl_c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
            writeNStage2DDWT(cl_c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
        
        } else {
            writeLinear(cl_c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
            writeLinear(cl_c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
            writeLinear(cl_c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
        }
#endif		
		
		clReleaseMemObject(cl_c_r);
		clReleaseMemObject(cl_c_g);
		clReleaseMemObject(cl_c_b);
		clReleaseMemObject(cl_c_g_out);
		clReleaseMemObject(cl_c_b_out);

	} else if(d->components == 1) { 
        // Load components 
        cl_mem cl_c_r;
		cl_c_r = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,componentSize, temp, &errNum);
		// fatal_CL(errNum, __LINE__); 		
        
        bwToComponent(cl_c_r, d->srcImg, d->pixWidth, d->pixHeight);

        // Compute DWT
        nStage2dDWT(cl_c_r, cl_c_r_out, cl_backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
    
        //Store DWT to file
        if(writeVisual){
            writeNStage2DDWT(cl_c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
        } else {
            writeLinear(cl_c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
        }
		
		
		clReleaseMemObject(cl_c_r);

    } 

	free(temp);
	clReleaseMemObject(cl_c_r_out);
}



int main(int argc, char **argv) 
{
    int optindex = 0;
    char ch;
    struct option longopts[] = 
	{
        {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
        {"components",  required_argument, 0, 'c'}, //numger of components of src img
        {"depth",       required_argument, 0, 'b'}, //bit depth of src img
        {"level",       required_argument, 0, 'l'}, //level of dwt
        {"device",      required_argument, 0, 'D'}, //cuda device
        {"forward",     no_argument,       0, 'f'}, //forward transform
        {"reverse",     no_argument,       0, 'r'}, //forward transform
        {"97",          no_argument,       0, '9'}, //9/7 transform
        {"53",          no_argument,       0, '5' }, //5/3transform
        {"write-visual",no_argument,       0, 'w' }, //write output (subbands) in visual (tiled) order instead of linear
        {"help",        no_argument,       0, 'h'}  
    };
    
    int pixWidth    = 0; //<real pixWidth
    int pixHeight   = 0; //<real pixHeight
    int compCount   = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth    = 8; 
    int dwtLvls     = 3; //default numuber of DWT levels
    int device      = 0;
    int forward     = 1; //forward transform
    int dwt97       = 0; //1=dwt9/7, 0=dwt5/3 transform
    int writeVisual = 0; //write output (subbands) in visual (tiled) order instead of linear
    char * pos;
 
    while ((ch = getopt_long(argc, argv, "d:c:b:l:D:fr95wh", longopts, &optindex)) != -1) 
	{
        switch (ch) {
        case 'd':
            pixWidth = atoi(optarg);
            pos = strstr(optarg, "x");
            if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) 
			{
                usage();
                return -1;
            }
            pixHeight = atoi(pos+1);
            break;
        case 'c':
            compCount = atoi(optarg);
            break;
        case 'b':
            bitDepth = atoi(optarg);
            break;
        case 'l':
            dwtLvls = atoi(optarg);
            break;
        case 'D':
            device = atoi(optarg);
            break;
        case 'f':
            forward = 1;
            break;
        case 'r':
            forward = 0;
            break;
        case '9':
            dwt97 = 1;
            break;
        case '5':
            dwt97 = 0;
            break;
        case 'w':
            writeVisual = 1;
            break;
        case 'h':
            usage();
            return 0;
        case '?':
            return -1;
        default :
            usage();
            return -1;
        }
    }
	argc -= optind;
	argv += optind;

    if (argc == 0) 
	{ // at least one filename is expected
        printf("Please supply src file name\n");
        usage();
        return -1;
    }

    if (pixWidth <= 0 || pixHeight <=0) 
	{
        printf("Wrong or missing dimensions\n");
        usage();
        return -1;
    }

    if (forward == 0) 
	{
        writeVisual = 0; //do not write visual when RDWT
    }
	
	
	
	//
	// device init
	// Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &cldevice);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }
	
	// Create OpenCL program from com_dwt.cl kernel source
	program = CreateProgram(context, cldevice, "com_dwt.cl");
    if (program == NULL)
    {
        printf("fail to create program!!\n");
    }

	// Create OpenCL kernel
	c_CopySrcToComponents = clCreateKernel(program, "c_CopySrcToComponents", NULL); 
	if (c_CopySrcToComponents == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
    }

	c_CopySrcToComponent = clCreateKernel(program, "c_CopySrcToComponent", NULL); 
	if (c_CopySrcToComponent == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
    }
	
	kl_fdwt53Kernel = clCreateKernel(program, "cl_fdwt53Kernel", NULL); 
    if (kl_fdwt53Kernel == NULL)
	{
		std::cerr<<"Failed to create kernel\n";
	}
	

	
	//initialize struct dwt
	struct dwt *d;
    d = (struct dwt *)malloc(sizeof(struct dwt));
    d->srcImg = NULL;
    d->pixWidth = pixWidth;
    d->pixHeight = pixHeight;
    d->components = compCount;
    d->dwtLvls  = dwtLvls;
	
	// file names
    d->srcFilename = (char *)malloc(strlen(argv[0]));
    strcpy(d->srcFilename, argv[0]);
    if (argc == 1) 
	{ // only one filename supplyed
        d->outFilename = (char *)malloc(strlen(d->srcFilename)+4);
        strcpy(d->outFilename, d->srcFilename);
        strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");
    } else {
        d->outFilename = strdup(argv[1]);
    }

    //Input review
    printf("\nSource file:\t\t%s\n", d->srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
    printf(" Components count:\t%d\n", compCount);
    printf(" Bit depth:\t\t%d\n", bitDepth);
    printf(" DWT levels:\t\t%d\n", dwtLvls);
    printf(" Forward transform:\t%d\n", forward);
    printf(" 9/7 transform:\t\t%d\n", dwt97);
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

    //load img source image
	d->srcImg = (unsigned char *) malloc (inputSize);
	if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
        return -1;
		
	 // DWT
	// Create memory objects, Set arguments for kernel functions, Queue the kernel up for execution across the array, Read the output buffer back to the Host, Output the result buffer
	
    if (forward == 1) 
	{
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }
    else 
	{ // reverse
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }
	

	Cleanup(context, commandQueue, program, kernel);
	clReleaseKernel(c_CopySrcToComponents);
	clReleaseKernel(c_CopySrcToComponent);
	
    return 0;
	
}
