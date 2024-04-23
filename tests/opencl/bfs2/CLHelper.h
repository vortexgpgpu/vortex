//------------------------------------------
//--cambine:helper function for OpenCL
//--programmer:	Jianbin Fang
//--date:	27/12/2010
//------------------------------------------
#ifndef _CL_HELPER_
#define _CL_HELPER_

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#ifdef TIMING
#include "timing.h"
#endif

using std::string;
using std::ifstream;
using std::cerr;
using std::endl;
//using std::cout;
//#pragma OPENCL EXTENSION cl_nv_compiler_options:enable
#define WORK_DIM 2	//work-items dimensions

extern float init_time, mem_alloc_time, h2d_time, kernel_time,
      d2h_time, close_time, total_time;

struct oclHandleStruct {
    cl_context              context;
    cl_device_id            *devices;
    cl_command_queue        queue;
    cl_program              program;
    cl_int		cl_status;
    std::string error_str;
    std::vector<cl_kernel>  kernel;
};

struct oclHandleStruct oclHandles;

struct _clDeviceProp{
  char device_name[100];
};

char kernel_file[100]  = "Kernels.cl";
int total_kernels = 2;
string kernel_names[2] = {"BFS_1", "BFS_2"};
int work_group_size = 512;
int platform_id_inuse = 0; // platform id in use (default: 0)
int device_id_inuse = 0; //device id in use (default : 0)
cl_device_type device_type = CL_DEVICE_TYPE_GPU;

void _clGetDeviceProperties(int idx, _clDeviceProp *prop) throw(string){
   
   oclHandles.cl_status= clGetDeviceInfo(oclHandles.devices[idx], CL_DEVICE_NAME, 100, prop->device_name, NULL);
   
#ifdef ERRMSG
	if(oclHandles.cl_status != CL_SUCCESS){
	  oclHandles.error_str = "exception in _clGetDeviceProperties-> ";
	  switch(oclHandles.cl_status){
	  	case CL_INVALID_DEVICE:
	  		oclHandles.error_str += "CL_INVALID_DEVICE";
	  		break;
	  	case CL_INVALID_VALUE:
	  		oclHandles.error_str += "CL_INVALID_VALUE";
	  		break;	   
	  	default:
	  		oclHandles.error_str += "unknown reasons";
	  		break;	  		
	  }
	throw(oclHandles.error_str);
	}	
#endif
}

int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
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

/*
 * Converts the contents of a file into a string
 */
string FileToString(const string fileName)
{
    ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

    try {
        size_t size;
        char*  str;
        string s;

        if(f.is_open()) {
            size_t fileSize;
            f.seekg(0, ifstream::end);
            size = fileSize = f.tellg();
            f.seekg(0, ifstream::beg);

            str = new char[size+1];
            if (!str) throw(string("Could not allocate memory"));

            f.read(str, fileSize);
            f.close();
            str[size] = '\0';

            s = str;
            delete [] str;
            return s;
        }
    } catch(std::string msg) {
        printf("Exception caught in FileToString(): %s\n");
        if(f.is_open())
            f.close();
    } catch(...) {
        printf("Exception caught in FileToString()\n");
        if(f.is_open())
            f.close();
    }
    string errorMsg = "FileToString()::Error: Unable to open file "
                      + fileName;
    throw(errorMsg);
}
//---------------------------------------
//Read command line parameters
//
void _clCmdParams(int argc, char* argv[])
{
    for (int i =0; i < argc; ++i) {
        switch (argv[i][1]) {
        case 'g':	//--g stands for size of work group
            if (++i < argc) {
                sscanf(argv[i], "%u", &work_group_size);
            } else {
                printf("Could not read argument after option %d\n", argv[i-1]);
                throw;
            }
            break;
        case 'd':	 //--d stands for device id used in computaion
            if (++i < argc) {
                sscanf(argv[i], "%u", &device_id_inuse);
            } else {
                printf("Could not read argument after option %d\n", argv[i-1]);
                throw;
            }
            break;
        case 'p':   // --p stands for platform id used in computation
            if (++i < argc) {
                sscanf(argv[i], "%u", &platform_id_inuse);
            } else {
                printf("Could not read argument after option %d\n", argv[i-1]);
                throw;
            }
            break;
/*
        case 't':   // --t stands for device type, 0:GPU, 1:CPU
            if (++i < argc) {
                sscanf(argv[i], "%u", &device_type);
                device_type = (device_type == 0) ? CL_DEVICE_TYPE_GPU
                    : CL_DEVICE_TYPE_CPU;
            } else {
                std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
                throw;
            }
            break;
*/
        default:
            ;
        }
    }

}

//---------------------------------------
//Initlize CL objects
//--description: there are 5 steps to initialize all the OpenCL objects needed
//--revised on 04/01/2011: get the number of devices  and
//  devices have no relationship with context
void _clInit()
{
    cl_int resultCL;

    oclHandles.context = NULL;
    oclHandles.devices = NULL;
    oclHandles.queue = NULL;
    oclHandles.program = NULL;

    cl_uint deviceListSize;

    //-----------------------------------------------
    //--cambine-1: find the available platforms and select one

    cl_uint numPlatforms;
    cl_platform_id targetPlatform = NULL;

    resultCL = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting number of platforms (clGetPlatformIDs)"));
    printf("number of platforms:%d\n",numPlatforms);	//by cambine

    if (!(numPlatforms > 0))
        throw (string("InitCL()::Error: No platforms found (clGetPlatformIDs)"));

    cl_platform_id* allPlatforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

    resultCL = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
    if (resultCL != CL_SUCCESS)
        throw (string("InitCL()::Error: Getting platform ids (clGetPlatformIDs)"));

    for (int i = 0; i < numPlatforms; i++) {
        char pbuff[128];
        resultCL = clGetPlatformInfo( allPlatforms[i],
                                      CL_PLATFORM_VENDOR,
                                      sizeof(pbuff),
                                      pbuff,
                                      NULL);
        if (resultCL != CL_SUCCESS)
            throw (string("InitCL()::Error: Getting platform info (clGetPlatformInfo)"));

        printf("vendor is %s\n",pbuff);

    }
    /* Select the target platform. Default: first platform */
    targetPlatform = allPlatforms[platform_id_inuse];
    free(allPlatforms);

    //-----------------------------------------------
    //--cambine-3: detect OpenCL devices
    /* First, get the size of device list */
    oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceListSize);
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("exception in _clInit -> clGetDeviceIDs"));
    }
    if (deviceListSize == 0)
        throw(string("InitCL()::Error: No devices found."));
    printf("device number: %d\n");

    /* Now, allocate the device list */
    oclHandles.devices = (cl_device_id *)malloc(deviceListSize * sizeof(cl_device_id));

    if (oclHandles.devices == 0)
        throw(string("InitCL()::Error: Could not allocate memory."));

    /* Next, get the device list data */
    oclHandles.cl_status = clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_ALL, deviceListSize,
            oclHandles.devices, NULL);
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("exception in _clInit -> clGetDeviceIDs-2"));
    }

	  _clDeviceProp prop;
  	_clGetDeviceProperties(device_id_inuse, &prop);
	  printf("--cambine: device name=%s\n", prop.device_name);
	
    /* Then, get device type */
    /*
	oclHandles.cl_status = clGetDeviceInfo(oclHandles.devices[device_id_inuse],
            CL_DEVICE_TYPE, sizeof(cl_device_type),	(void *)&device_type,
            NULL);
	if (oclHandles.cl_status != CL_SUCCESS) {
        throw(string("error in Getting Device Info"));
    }
	if (device_type == CL_DEVICE_TYPE_GPU)
        printf("Creating GPU Context\n");
	else if (device_type == CL_DEVICE_TYPE_CPU)
        printf("Creating CPU Context\n");
	else
        throw(string("unsupported device type"));
      */
    //-----------------------------------------------
    //--cambine-2: create an OpenCL context
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)targetPlatform, 0 };
    /*
    oclHandles.context = clCreateContextFromType(cprops,
                         device_type,
                         NULL,
                         NULL,
                         &resultCL);
    */
    oclHandles.context = clCreateContext(0, 
                                        deviceListSize, 
                                        oclHandles.devices, 
                                        NULL,
										                    NULL,
                                        &resultCL);
 
    if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL))
        throw (string("InitCL()::Error: Creating Context (clCreateContextFromType)"));

    //-----------------------------------------------
    //--cambine-4: Create an OpenCL command queue
#ifdef TIMING
    oclHandles.queue = clCreateCommandQueue(oclHandles.context,
            oclHandles.devices[device_id_inuse], CL_QUEUE_PROFILING_ENABLE,
            &resultCL);
#else
    oclHandles.queue = clCreateCommandQueue(oclHandles.context,
            oclHandles.devices[device_id_inuse], 0, &resultCL);

#endif

    if ((resultCL != CL_SUCCESS) || (oclHandles.queue == NULL))
        throw(string("InitCL()::Creating Command Queue. (clCreateCommandQueue)"));

    //-----------------------------------------------
    //--cambine-5: Load CL file, build CL program object, create CL kernel object
    /* 
    std::string  source_str = FileToString(kernel_file);
    const char * source    = source_str.c_str();
    size_t sourceSize[]    = { source_str.length() };

    oclHandles.program = clCreateProgramWithSource(oclHandles.context,
                         1,
                         &source,
                         sourceSize,
                         &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
        throw(string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));
    */
   uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_int binary_status = 0;
    if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
      std::abort();

    oclHandles.program = clCreateProgramWithBinary(
          oclHandles.context, 1, &oclHandles.devices[device_id_inuse], &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &resultCL);
    free(kernel_bin);

    //insert debug information
    //std::string options= "-cl-nv-verbose"; //Doesn't work on AMD machines
    //options += " -cl-nv-opt-level=3";
    resultCL = clBuildProgram(oclHandles.program, deviceListSize, oclHandles.devices, NULL, NULL,NULL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)) {
        printf("InitCL()::Error: In clBuildProgram\n");

        size_t length;
        resultCL = clGetProgramBuildInfo(oclHandles.program,
                                         oclHandles.devices[device_id_inuse],
                                         CL_PROGRAM_BUILD_LOG,
                                         0,
                                         NULL,
                                         &length);
        if(resultCL != CL_SUCCESS)
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

        char* buffer = (char*)malloc(length);
        resultCL = clGetProgramBuildInfo(oclHandles.program,
                                         oclHandles.devices[device_id_inuse],
                                         CL_PROGRAM_BUILD_LOG,
                                         length,
                                         buffer,
                                         NULL);
        if(resultCL != CL_SUCCESS)
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

        printf("%s\n", buffer);
        free(buffer);

        throw(string("InitCL()::Error: Building Program (clBuildProgram)"));
    }

    //get program information in intermediate representation
#ifdef PTX_MSG
    size_t binary_sizes[deviceListSize];
    char * binaries[deviceListSize];
    //figure out number of devices and the sizes of the binary for each device.
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*deviceListSize, &binary_sizes, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-2"));
    }

    printf("--cambine: %d", binary_sizes);
    //copy over all of the generated binaries.
    for(int i=0; i<deviceListSize; i++)
        binaries[i] = (char *)malloc( sizeof(char)*(binary_sizes[i]+1));
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARIES, sizeof(char *)*deviceListSize, binaries, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-3"));
    }
    for(int i=0; i<deviceListSize; i++)
        binaries[i][binary_sizes[i]] = '\0';
    printf("--cambine:writing ptd information...\n");
    FILE * ptx_file = fopen("cl.ptx","w");
    if(ptx_file==NULL) {
        throw(string("exceptions in allocate ptx file."));
    }
    fprintf(ptx_file,"%s",binaries[DEVICE_ID_INUSE]);
    fclose(ptx_file);
    printf("--cambine:writing ptd information done.\n");
    for(int i=0; i<deviceListSize; i++)
        free(binaries[i]);
#endif

    for (int nKernel = 0; nKernel < total_kernels; nKernel++) {
        /* get a kernel object handle for a kernel with the given name */
        cl_kernel kernel = clCreateKernel(oclHandles.program,
                                          (kernel_names[nKernel]).c_str(),
                                          &resultCL);

        if ((resultCL != CL_SUCCESS) || (kernel == NULL)) {
            string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \"" + kernel_names[nKernel] + "\"";
            throw(errorMsg);
        }

        oclHandles.kernel.push_back(kernel);
    }
    //get resource alocation information
#ifdef RES_MSG
    char * build_log;
    size_t ret_val_size;
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSE], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("exceptions in _InitCL -> getting resource information"));
    }

    build_log = (char *)malloc(ret_val_size+1);
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSE], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(string("exceptions in _InitCL -> getting resources allocation information-2"));
    }
    build_log[ret_val_size] = '\0';
    printf("--cambine: %s\n", build_log);
    free(build_log);
#endif
}

//---------------------------------------
//release CL objects
void _clRelease()
{
    char errorFlag = false;

    for (int nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++) {
        if (oclHandles.kernel[nKernel] != NULL) {
            cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
            if (resultCL != CL_SUCCESS) {
                printf("ReleaseCL()::Error: In clReleaseKernel\n");
                errorFlag = true;
            }
            oclHandles.kernel[nKernel] = NULL;
        }
        oclHandles.kernel.clear();
    }

    if (oclHandles.program != NULL) {
        cl_int resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS) {
            printf("ReleaseCL()::Error: In clReleaseProgram\n");
            errorFlag = true;
        }
        oclHandles.program = NULL;
    }

    if (oclHandles.queue != NULL) {
        cl_int resultCL = clReleaseCommandQueue(oclHandles.queue);
        if (resultCL != CL_SUCCESS) {
            printf("ReleaseCL()::Error: In clReleaseCommandQueue\n");
            errorFlag = true;
        }
        oclHandles.queue = NULL;
    }

    free(oclHandles.devices);

    if (oclHandles.context != NULL) {
        cl_int resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS) {
            printf("ReleaseCL()::Error: In clReleaseContext\n");
            errorFlag = true;
        }
        oclHandles.context = NULL;
    }

    if (errorFlag) throw(string("ReleaseCL()::Error encountered."));
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device
cl_mem _clCreateAndCpyMem(int size, void * h_mem_source) throw(string)
{
    cl_mem d_mem;
    d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  \
                           size, h_mem_source, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateAndCpyMem()"));
#endif
    return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read only  buffer for devices
//--date:	17/01/2011
cl_mem _clMallocRW(int size, void * h_mem_ptr) throw(string)
{
    cl_mem d_mem;
    d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_WRITE, size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clMallocRW"));
#endif
    return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read and write buffer for devices
//--date:	17/01/2011
cl_mem _clMalloc(int size, void * h_mem_ptr) throw(string)
{
    cl_mem d_mem;
    d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY, size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clMalloc"));
#endif
    return d_mem;
}

//-------------------------------------------------------
//--cambine:	transfer data from host to device
//--date:	17/01/2011
void _clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(string)
{
    cl_event event;
    oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue, d_mem,
            CL_TRUE, 0, size, h_mem_ptr, 0, NULL, &event);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clMemcpyH2D"));
#endif
#ifdef TIMING
    h2d_time += probe_event_time(event, oclHandles.queue);
#endif
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device with pinned
// memory
cl_mem _clCreateAndCpyPinnedMem(int size, float* h_mem_source) throw(string)
{
    cl_mem d_mem, d_mem_pinned;
    float * h_mem_pinned = NULL;
    d_mem_pinned = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,  \
                                  size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateAndCpyMem()->d_mem_pinned"));
#endif
    //------------
    d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY,  \
                           size, NULL, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateAndCpyMem() -> d_mem "));
#endif
    //----------
    h_mem_pinned = (cl_float *)clEnqueueMapBuffer(oclHandles.queue, d_mem_pinned, CL_TRUE,  \
                   CL_MAP_WRITE, 0, size, 0, NULL,  \
                   NULL,  &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueMapBuffer"));
#endif
    int element_number = size/sizeof(float);
    #pragma omp parallel for
    for(int i=0; i<element_number; i++) {
        h_mem_pinned[i] = h_mem_source[i];
    }
    //----------
    oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue, d_mem, 	\
                           CL_TRUE, 0, size, h_mem_pinned,  \
                           0, NULL, NULL);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueWriteBuffer"));
#endif

    return d_mem;
}


//--------------------------------------------------------
//--cambine:create write only buffer on device
cl_mem _clMallocWO(int size) throw(string)
{
    cl_mem d_mem;
    d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY, size, 0, &oclHandles.cl_status);
#ifdef ERRMSG
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(string("excpetion in _clCreateMem()"));
#endif
    return d_mem;
}

//--------------------------------------------------------
//transfer data from device to host
void _clMemcpyD2H(cl_mem d_mem, int size, void * h_mem) throw(string)
{
    cl_event event;
    oclHandles.cl_status = clEnqueueReadBuffer(oclHandles.queue, d_mem, CL_TRUE, 0, size, h_mem, 0,0, &event);
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clCpyMemD2H -> ";
    switch(oclHandles.cl_status) {
    case CL_INVALID_COMMAND_QUEUE:
        oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
        break;
    case CL_INVALID_CONTEXT:
        oclHandles.error_str += "CL_INVALID_CONTEXT";
        break;
    case CL_INVALID_MEM_OBJECT:
        oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
        break;
    case CL_INVALID_VALUE:
        oclHandles.error_str += "CL_INVALID_VALUE";
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
        break;
    default:
        oclHandles.error_str += "Unknown reason";
        break;
    }
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(oclHandles.error_str);
#endif
#ifdef TIMING
    d2h_time += probe_event_time(event, oclHandles.queue);
#endif
}

//--------------------------------------------------------
//set kernel arguments
void _clSetArgs(int kernel_id, int arg_idx, void * d_mem, int size = 0) throw(string)
{
    if(!size) {
        oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, sizeof(d_mem), &d_mem);
#ifdef ERRMSG
        oclHandles.error_str = "excpetion in _clSetKernelArg() ";
        switch(oclHandles.cl_status) {
        case CL_INVALID_KERNEL:
            oclHandles.error_str += "CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            oclHandles.error_str += "CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            oclHandles.error_str += "CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_MEM_OBJECT:
            oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_SAMPLER:
            oclHandles.error_str += "CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_ARG_SIZE:
            oclHandles.error_str += "CL_INVALID_ARG_SIZE";
            break;
        case CL_OUT_OF_RESOURCES:
            oclHandles.error_str += "CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
            break;
        default:
            oclHandles.error_str += "Unknown reason";
            break;
        }
        if(oclHandles.cl_status != CL_SUCCESS)
            throw(oclHandles.error_str);
#endif
    } else {
        oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, size, d_mem);
#ifdef ERRMSG
        oclHandles.error_str = "excpetion in _clSetKernelArg() ";
        switch(oclHandles.cl_status) {
        case CL_INVALID_KERNEL:
            oclHandles.error_str += "CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            oclHandles.error_str += "CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            oclHandles.error_str += "CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_MEM_OBJECT:
            oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_SAMPLER:
            oclHandles.error_str += "CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_ARG_SIZE:
            oclHandles.error_str += "CL_INVALID_ARG_SIZE";
            break;
        case CL_OUT_OF_RESOURCES:
            oclHandles.error_str += "CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
            break;
        default:
            oclHandles.error_str += "Unknown reason";
            break;
        }
        if(oclHandles.cl_status != CL_SUCCESS)
            throw(oclHandles.error_str);
#endif
    }
}
void _clFinish() throw(string)
{
    oclHandles.cl_status = clFinish(oclHandles.queue);
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clFinish";
    switch(oclHandles.cl_status) {
    case CL_INVALID_COMMAND_QUEUE:
        oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
        break;
    case CL_OUT_OF_RESOURCES:
        oclHandles.error_str += "CL_OUT_OF_RESOURCES";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
        break;
    default:
        oclHandles.error_str += "Unknown reasons";
        break;

    }
    if(oclHandles.cl_status!=CL_SUCCESS) {
        throw(oclHandles.error_str);
    }
#endif
}
//--------------------------------------------------------
//--cambine:enqueue kernel
void _clInvokeKernel(int kernel_id, int work_items, int work_group_size) throw(string)
{
    cl_uint work_dim = WORK_DIM;
    cl_event e[1];
    if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
        work_items = work_items + (work_group_size-(work_items%work_group_size));
    size_t local_work_size[] = {work_group_size, 1};
    size_t global_work_size[] = {work_items, 1};
    oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, \
                           global_work_size, local_work_size, 0, 0, &(e[0]) );
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
    switch(oclHandles.cl_status) {
    case CL_INVALID_PROGRAM_EXECUTABLE:
        oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
        break;
    case CL_INVALID_COMMAND_QUEUE:
        oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
        break;
    case CL_INVALID_KERNEL:
        oclHandles.error_str += "CL_INVALID_KERNEL";
        break;
    case CL_INVALID_CONTEXT:
        oclHandles.error_str += "CL_INVALID_CONTEXT";
        break;
    case CL_INVALID_KERNEL_ARGS:
        oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
        break;
    case CL_INVALID_WORK_DIMENSION:
        oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
        break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
        break;
    case CL_INVALID_WORK_GROUP_SIZE:
        oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
        break;
    case CL_INVALID_WORK_ITEM_SIZE:
        oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
        break;
    case CL_INVALID_GLOBAL_OFFSET:
        oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
        break;
    case CL_OUT_OF_RESOURCES:
        oclHandles.error_str += "CL_OUT_OF_RESOURCES";
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
        break;
    default:
        oclHandles.error_str += "Unkown reseason";
        break;
    }
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(oclHandles.error_str);
#endif

#ifdef TIMING
    kernel_time += probe_event_time(e[0], oclHandles.queue);
#endif
    //_clFinish();
    // oclHandles.cl_status = clWaitForEvents(1, &e[0]);
    // #ifdef ERRMSG
    // if (oclHandles.cl_status!= CL_SUCCESS)
    //     throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));
    // #endif
}
void _clInvokeKernel2D(int kernel_id, int range_x, int range_y, int group_x, int group_y) throw(string)
{
    cl_uint work_dim = WORK_DIM;
    size_t local_work_size[] = {group_x, group_y};
    size_t global_work_size[] = {range_x, range_y};
    cl_event e[1];
    /*if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
      work_items = work_items + (work_group_size-(work_items%work_group_size));*/
    oclHandles.cl_status = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, \
                           global_work_size, local_work_size, 0, 0, &(e[0]) );
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
    switch(oclHandles.cl_status) {
    case CL_INVALID_PROGRAM_EXECUTABLE:
        oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
        break;
    case CL_INVALID_COMMAND_QUEUE:
        oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
        break;
    case CL_INVALID_KERNEL:
        oclHandles.error_str += "CL_INVALID_KERNEL";
        break;
    case CL_INVALID_CONTEXT:
        oclHandles.error_str += "CL_INVALID_CONTEXT";
        break;
    case CL_INVALID_KERNEL_ARGS:
        oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
        break;
    case CL_INVALID_WORK_DIMENSION:
        oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
        break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
        break;
    case CL_INVALID_WORK_GROUP_SIZE:
        oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
        break;
    case CL_INVALID_WORK_ITEM_SIZE:
        oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
        break;
    case CL_INVALID_GLOBAL_OFFSET:
        oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
        break;
    case CL_OUT_OF_RESOURCES:
        oclHandles.error_str += "CL_OUT_OF_RESOURCES";
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
        break;
    default:
        oclHandles.error_str += "Unkown reseason";
        break;
    }
    if(oclHandles.cl_status != CL_SUCCESS)
        throw(oclHandles.error_str);
#endif

#ifdef TIMING
    kernel_time += probe_event_time(e[0], oclHandles.queue);
#endif

    //_clFinish();
    /*oclHandles.cl_status = clWaitForEvents(1, &e[0]);
    #ifdef ERRMSG
        if (oclHandles.cl_status!= CL_SUCCESS)
            throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));
    #endif*/
}

//--------------------------------------------------------
//release OpenCL objects
void _clFree(cl_mem ob) throw(string)
{
    if(ob!=NULL)
        oclHandles.cl_status = clReleaseMemObject(ob);
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clFree() ->";
    switch(oclHandles.cl_status) {
    case CL_INVALID_MEM_OBJECT:
        oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
        break;
    case CL_OUT_OF_RESOURCES:
        oclHandles.error_str += "CL_OUT_OF_RESOURCES";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
        break;
    default:
        oclHandles.error_str += "Unkown reseason";
        break;
    }
    if (oclHandles.cl_status!= CL_SUCCESS)
        throw(oclHandles.error_str);
#endif
}
#endif //_CL_HELPER_
