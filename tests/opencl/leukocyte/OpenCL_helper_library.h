#ifndef OPENCL_HELPER_LIBRARY_H
#define OPENCL_HELPER_LIBRARY_H

#include <CL/cl.h>

#include <stdio.h>
#include <sys/time.h>


// Function prototypes
char *load_kernel_source(const char *filename);
// long long get_time();
void fatal(const char *s);
void fatal_CL(cl_int error, char *file, int line);
void check_error(cl_int error, char *file, int line);


#endif
