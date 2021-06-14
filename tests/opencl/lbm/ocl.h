#ifndef __OCLH__
#define __OCLH__

typedef struct {
	cl_platform_id clPlatform;
	cl_context_properties clCps[3];
	cl_device_id clDevice;
	cl_context clContext;
	cl_command_queue clCommandQueue;
	cl_program clProgram;
	cl_kernel clKernel;
} OpenCL_Param;


#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

char* readFile(char*);

#endif
