/* pocl.h - global pocl declarations for the host side runtime.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen

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

/**
 * @file pocl.h
 *
 * The declarations in this file are such that are used both in the
 * libpocl implementation CL and the kernel compiler. Others should be
 * moved to pocl_cl.h of lib/CL or under the kernel compiler dir.
 * @todo Check if there are extra declarations here that could be moved.
 */
#ifndef POCL_H
#define POCL_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 220
#endif
#include <CL/opencl.h>

#include "config.h"

#include "pocl_context.h"

/* detects restrict, variadic macros etc */
#include "pocl_compiler_features.h"

#define POCL_FILENAME_LENGTH 1024

#define WORKGROUP_STRING_LENGTH 1024

typedef struct _mem_mapping mem_mapping_t;
/* represents a single buffer to host memory mapping */
struct _mem_mapping {
  void *host_ptr; /* the location of the mapped buffer chunk in the host memory */
  size_t offset; /* offset to the beginning of the buffer */
  size_t size;
  mem_mapping_t *prev, *next;
  /* This is required, because two clEnqueueMap() with the same buffer+size+offset,
     will create two identical mappings in the buffer->mappings LL.
     Without this flag, both corresponding clEnqUnmap()s will find
     the same mapping (the first one in mappings LL), which will lead
     to memory double-free corruption later. */
  long unmap_requested;
  cl_map_flags map_flags;
  /* image mapping data */
  size_t origin[3];
  size_t region[3];
  size_t row_pitch;
  size_t slice_pitch;
};

/* memory identifier: id to point the global memory where memory resides
                      + pointer to actual data */
typedef struct _pocl_mem_identifier
{
  int available; /* ... in this mem objs context */
  int global_mem_id;
  void *mem_ptr;
  void *image_data;
} pocl_mem_identifier;

typedef struct _mem_destructor_callback mem_destructor_callback_t;
/* represents a memory object destructor callback */
struct _mem_destructor_callback
{
  void (CL_CALLBACK * pfn_notify) (cl_mem, void*); /* callback function */
  void *user_data; /* user supplied data passed to callback function */
  mem_destructor_callback_t *next;
};

typedef struct _build_program_callback build_program_callback_t;
struct _build_program_callback
{
    void (CL_CALLBACK * callback_function) (cl_program, void*); /* callback function */
    void *user_data; /* user supplied data passed to callback function */
};

// Command Queue datatypes

#define POCL_KERNEL_DIGEST_SIZE 16
typedef uint8_t pocl_kernel_hash_t[POCL_KERNEL_DIGEST_SIZE];

// clEnqueueNDRangeKernel
typedef struct
{
  void *hash;
  void *wg; /* The work group function ptr. Device specific. */
  cl_kernel kernel;
  /* The launch data that can be passed to the kernel execution environment. */
  struct pocl_context pc;
  struct pocl_argument *arguments;
  /* Can be used to store/cache arbitrary device-specific data. */
  void *device_data;
  /* If set to 1, disallow any work-group function specialization. */
  int force_generic_wg_func;
  /* If set to 1, disallow "small grid" WG function specialization. */
  int force_large_grid_wg_func;
  unsigned device_i;
} _cl_command_run;

// clEnqueueNativeKernel
typedef struct
{
  void *args;
  size_t cb_args;
  void (*user_func)(void *);
} _cl_command_native;

// clEnqueueReadBuffer
typedef struct
{
  void *__restrict__ dst_host_ptr;
  pocl_mem_identifier *src_mem_id;
  size_t offset;
  size_t size;
} _cl_command_read;

// clEnqueueWriteBuffer
typedef struct
{
  const void *__restrict__ src_host_ptr;
  pocl_mem_identifier *dst_mem_id;
  size_t offset;
  size_t size;
} _cl_command_write;

// clEnqueueCopyBuffer
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  size_t src_offset;
  size_t dst_offset;
  size_t size;
} _cl_command_copy;

// clEnqueueReadBufferRect
typedef struct
{
  void *__restrict__ dst_host_ptr;
  pocl_mem_identifier *src_mem_id;
  size_t buffer_origin[3];
  size_t host_origin[3];
  size_t region[3];
  size_t buffer_row_pitch;
  size_t buffer_slice_pitch;
  size_t host_row_pitch;
  size_t host_slice_pitch;
} _cl_command_read_rect;

// clEnqueueWriteBufferRect
typedef struct
{
  const void *__restrict__ src_host_ptr;
  pocl_mem_identifier *dst_mem_id;
  size_t buffer_origin[3];
  size_t host_origin[3];
  size_t region[3];
  size_t buffer_row_pitch;
  size_t buffer_slice_pitch;
  size_t host_row_pitch;
  size_t host_slice_pitch;
} _cl_command_write_rect;

// clEnqueueCopyBufferRect
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  size_t dst_origin[3];
  size_t src_origin[3];
  size_t region[3];
  size_t src_row_pitch;
  size_t src_slice_pitch;
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} _cl_command_copy_rect;

// clEnqueueMapBuffer
typedef struct
{
  pocl_mem_identifier *mem_id;
  mem_mapping_t *mapping;
} _cl_command_map;

/* clEnqueueUnMapMemObject */
typedef struct
{
  pocl_mem_identifier *mem_id;
  mem_mapping_t *mapping;
} _cl_command_unmap;

/* clEnqueueFillBuffer */
typedef struct
{
  pocl_mem_identifier *dst_mem_id;
  size_t size;
  size_t offset;
  void *__restrict__ pattern;
  size_t pattern_size;
} _cl_command_fill_mem;

/* clEnqueue(Write/Read)Image */
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  void *__restrict__ dst_host_ptr;
  pocl_mem_identifier *dst_mem_id;
  size_t dst_offset;
  size_t origin[3];
  size_t region[3];
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} _cl_command_read_image;

typedef struct
{
  pocl_mem_identifier *dst_mem_id;
  const void *__restrict__ src_host_ptr;
  pocl_mem_identifier *src_mem_id;
  size_t src_offset;
  size_t origin[3];
  size_t region[3];
  size_t src_row_pitch;
  size_t src_slice_pitch;
} _cl_command_write_image;

typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  size_t dst_origin[3];
  size_t src_origin[3];
  size_t region[3];
} _cl_command_copy_image;

/* clEnqueueFillImage */
typedef struct
{
  pocl_mem_identifier *mem_id;
  size_t origin[3];
  size_t region[3];
  void *__restrict__ fill_pixel;
  size_t pixel_size;
} _cl_command_fill_image;

/* clEnqueueMarkerWithWaitlist */
typedef struct
{
  void *data;
  int has_wait_list;
} _cl_command_marker;

/* clEnqueueBarrierWithWaitlist */
typedef _cl_command_marker _cl_command_barrier;

/* clEnqueueMigrateMemObjects */
typedef struct
{
  void *data;
  size_t num_mem_objects;
  cl_mem *mem_objects;
  cl_device_id *source_devices;
} _cl_command_migrate;

typedef struct
{
  void* data;
  void* queue;
  unsigned  num_svm_pointers;
  void  **svm_pointers;
  void (CL_CALLBACK  *pfn_free_func) ( cl_command_queue queue,
                                       cl_uint num_svm_pointers,
                                       void *svm_pointers[],
                                       void  *user_data);
} _cl_command_svm_free;

typedef struct
{
  void* svm_ptr;
  size_t size;
  cl_map_flags flags;
} _cl_command_svm_map;

typedef struct
{
  void* svm_ptr;
} _cl_command_svm_unmap;

typedef struct
{
  const void *__restrict__ src;
  void *__restrict__ dst;
  size_t size;
} _cl_command_svm_cpy;

typedef struct
{
  void *__restrict__ svm_ptr;
  size_t size;
  void *__restrict__ pattern;
  size_t pattern_size;
} _cl_command_svm_fill;

typedef union
{
  _cl_command_run run;
  _cl_command_native native;

  _cl_command_read read;
  _cl_command_write write;
  _cl_command_copy copy;
  _cl_command_read_rect read_rect;
  _cl_command_write_rect write_rect;
  _cl_command_copy_rect copy_rect;
  _cl_command_fill_mem memfill;

  _cl_command_read_image read_image;
  _cl_command_write_image write_image;
  _cl_command_copy_image copy_image;
  _cl_command_fill_image fill_image;

  _cl_command_map map;
  _cl_command_unmap unmap;

  _cl_command_marker marker;
  _cl_command_barrier barrier;
  _cl_command_migrate migrate;

  _cl_command_svm_free svm_free;
  _cl_command_svm_map svm_map;
  _cl_command_svm_unmap svm_unmap;
  _cl_command_svm_cpy svm_memcpy;
  _cl_command_svm_fill svm_fill;
} _cl_command_t;

// one item in the command queue
typedef struct _cl_command_node _cl_command_node;
struct _cl_command_node
{
  _cl_command_t command;
  cl_command_type type;
  _cl_command_node *next; // for linked-list storage
  _cl_command_node *prev;
  cl_event event;
  const cl_event *event_wait_list;
  cl_device_id device;
  /* The index of the targeted device in the platform's device list. */
  unsigned device_i;
  cl_int ready;
};

#ifndef LLVM_10_0
#define LLVM_OLDER_THAN_10_0 1

#ifndef LLVM_9_0
#define LLVM_OLDER_THAN_9_0 1

#ifndef LLVM_8_0
#define LLVM_OLDER_THAN_8_0 1

#ifndef LLVM_7_0
#define LLVM_OLDER_THAN_7_0 1

#ifndef LLVM_6_0
#define LLVM_OLDER_THAN_6_0 1

#endif
#endif
#endif
#endif
#endif

#endif /* POCL_H */
