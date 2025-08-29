// #ifdef __cplusplus
// extern "C" {
// #endif

#ifndef EMBEDDED_FEHLBERG_7_8_H
#define EMBEDDED_FEHLBERG_7_8_H

#include "common.h"
#include <CL/cl.h>

// Function declaration
void embedded_fehlberg_7_8(fp timeinst,
                           fp h,
                           fp *initvalu,
                           fp *finavalu,
                           fp *error,
                           fp *parameter,
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

#endif // EMBEDDED_FEHLBERG_7_8_H

// #ifdef __cplusplus
// }
// #endif
