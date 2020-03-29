/* pocl_device.h - global pocl declarations to be used in the device binaries in
                   case applicable by the target

   Copyright (c) 2012-2018 Pekka Jääskeläinen / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef POCL_DEVICE_H
#define POCL_DEVICE_H

#include "pocl_types.h"

#define MAX_KERNEL_ARGS 64
#define MAX_KERNEL_NAME_LENGTH 64

/* Metadata of a single kernel stored in the device.*/
typedef struct {
    const uchar name[MAX_KERNEL_NAME_LENGTH];
    ushort num_args;
    ushort num_locals;
    void *work_group_func;
} __kernel_metadata;

#ifdef _MSC_VER
  #define ALIGN4(x) __declspec(align(4)) x
  #define ALIGN8(x) __declspec(align(4)) x
#else
  #define ALIGN4(x) x __attribute__ ((aligned (4)))
  #define ALIGN8(x) x __attribute__ ((aligned (8)))
#endif

/* A kernel invocation command. */
typedef struct {
    /* The execution status of this queue slot. */
    ALIGN8(uint status);
    /* The kernel to execute. Points to the metadata in the device global
       memory. It will be casted to a __kernel_metadata* */
    ALIGN8(uint kernel);
    /* Pointers to the kernel arguments in the global memory. Will be
       casted to 32 bit void* */
    ALIGN8(uint args[MAX_KERNEL_ARGS]);
    /* Sizes of the dynamically allocated local buffers. */
/*    uint32_t dynamic_local_arg_sizes[MAX_KERNEL_ARGS] ALIGN4; */
    /* Number of dimensions in the work space. */
    ALIGN4(uint work_dim);
    ALIGN4(uint num_groups[3]);
    ALIGN4(uint global_offset[3]);
} __kernel_exec_cmd;

/* Kernel execution statuses. */

/* The invocation entry is free to use. */
#define POCL_KST_FREE     1
/* The kernel structure has been populated and is waiting to be
   executed. */
#define POCL_KST_READY    2
/* The kernel is currently running in the device. */
#define POCL_KST_RUNNING  3
/* The kernel has finished execution. The results can be collected and the
   execution entry be freed (by writing POCL_KST_FREE to the status). */
#define POCL_KST_FINISHED 4

#endif
