#ifndef __OCLH__
#define __OCLH__

#include <stdlib.h>

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
