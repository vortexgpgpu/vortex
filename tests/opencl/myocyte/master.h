// #ifdef __cplusplus
// extern "C" {
// #endif

#ifndef MASTER_H
#define MASTER_H

#include "common.h"
#include <CL/cl.h>

// Function declaration
void master(fp timeinst,
            fp *initvalu,
            fp *parameter,
            fp *finavalu,
            fp *com,
            cl_mem d_initvalu,
            cl_mem d_finavalu,
            cl_mem d_params,
            cl_mem d_com,
            cl_command_queue command_queue,
            cl_kernel kernel,
            long long *timecopyin,
            long long *timekernel,
            long long *timecopyout);

#endif // MASTER_H

// #ifdef __cplusplus
// }
// #endif
