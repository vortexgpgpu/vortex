#ifndef __OCLH__
#define __OCLH__

#include <stdlib.h>

typedef struct {
	cl_context clContext;
	cl_command_queue clCommandQueue;
	cl_kernel clKernel;
} clPrmtr;

void clMemSet(clPrmtr*, cl_mem, int, size_t);
char* readFile(const char*);

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

#endif
