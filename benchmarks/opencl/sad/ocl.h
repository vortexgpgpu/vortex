#ifndef __OCLH__
#define __OCLH__

typedef struct {
	cl_uint major;
	cl_uint minor;
	cl_uint multiProcessorCount;
} OpenCLDeviceProp;

void clMemSet(cl_command_queue, cl_mem, int, size_t);
char* readFile(const char*);

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

#endif
