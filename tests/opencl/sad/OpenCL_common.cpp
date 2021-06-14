

#include "OpenCL_common.h"
#include <stdlib.h>
#include <string.h>

// -1 for NO suitable device found, 0 if an appropriate device was found
int getOpenCLDevice(cl_platform_id *platform, cl_device_id *device, cl_device_type *reqDeviceType, int numRequests, ...) {
      
        // Supported Device Requests (anything that returns cl_bool)
        //   CL_DEVICE_IMAGE_SUPPORT
        //   CL_DEVICE_HOST_UNIFIED_MEMORY
        //   CL_DEVICE_ERROR_CORRECTION_SUPPORT
        //   CL_DEVICE_AVAILABLE
        //   CL_DEVICE_COMPILER_AVAILABLE
  
  cl_uint numEntries = 16;
  cl_platform_id clPlatforms[numEntries];
  cl_uint numPlatforms;
  
  cl_device_id clDevices[numEntries];
  cl_uint numDevices;

  OCL_ERRCK_RETVAL ( clGetPlatformIDs(numEntries, clPlatforms, &numPlatforms) );
  //fprintf(stderr, "Number of Platforms found: %d\n", numPlatforms);
  bool needDevice = true;
  
  for (int ip = 0; ip < numPlatforms && needDevice; ++ip) {

    cl_platform_id clPlatform = clPlatforms[ip];
    
    OCL_ERRCK_RETVAL ( clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, numEntries, clDevices, &numDevices) );
    //fprintf(stderr, "  Number of Devices found for Platform %d: %d\n", ip, numDevices);
    
    for (int id = 0; (id < numDevices) && needDevice ; ++id) {
      cl_device_id clDevice = clDevices[id];
      cl_device_type clDeviceType;

      bool canSatisfy = true;
      
      if (reqDeviceType != NULL) {
        OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &clDeviceType, NULL));
        if (*reqDeviceType != CL_DEVICE_TYPE_ALL) {
          if (*reqDeviceType != clDeviceType) {
            canSatisfy = false;
          }
        }
      }

      va_list paramList;
      va_start(paramList, numRequests);
      for (int i = 0; (i < numRequests) && canSatisfy ; ++i) {
      
        cl_device_info devReq = va_arg( paramList, cl_device_info );  
        cl_bool clInfoBool;
        size_t infoRetSize = sizeof(cl_bool);
        
        OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, devReq, infoRetSize, &clInfoBool, NULL));
        if (clInfoBool != true) {
          canSatisfy = false;
        }
      }
      
      va_end(paramList);
      if (canSatisfy) {
        *device = clDevice;
        *platform = clPlatform;
        needDevice = false;
        if (reqDeviceType != NULL && (*reqDeviceType == CL_DEVICE_TYPE_ALL)) {
          *reqDeviceType = clDeviceType;
        }
      }
    } // End checking all devices for a platform
  } // End checking all platforms

  int retVal = -1;
  if (needDevice) {
    retVal = -1;
  } else {
    retVal = 0;
  }
  
  return retVal;

}

const char* oclErrorString(cl_int error)
{
// From NVIDIA SDK
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;

	return (index >= 0 && index < errorCount) ? errorString[index] : "";
}


const char* oclDebugErrString(cl_int error, cl_device_id device)
{
// From NVIDIA SDK
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;
	
	if (index == 4) {
	cl_uint maxMemAlloc = 0;
	
	OCL_ERRCK_RETVAL ( clGetDeviceInfo(	device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAlloc, NULL) );

	
	  fprintf(stderr, "  Device Maximum block allocation size: %lu\n", maxMemAlloc);
	}

	return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    size_t szPreambleLength = strlen(cPreamble);
    szPreambleLength = 0;

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}
