// #ifdef __cplusplus
// extern "C" {
// #endif

#ifndef SOLVER_H
#define SOLVER_H

#include "common.h"
#include <CL/cl.h>

// Function declaration
int solver(fp **y,
           fp *x,
           int xmax,
           fp *params,
           fp *com,
           cl_mem d_initvalu,
           cl_mem d_finavalu,
           cl_mem d_params,
           cl_mem d_com,
           cl_command_queue command_queue,
           cl_kernel kernel,
           long long *timecopyin,
           long long *timecopykernel,
           long long *timecopyout);

#endif // SOLVER_H

// #ifdef __cplusplus
// }
// #endif
