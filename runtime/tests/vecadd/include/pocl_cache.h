/* pocl_cache.h: global declarations of caching functions used mostly in runtime

   Copyright (c) 2015 pocl developers

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

#ifndef POCL_CACHE_H
#define POCL_CACHE_H

#include "pocl_cl.h"

/* The filename in which the work group (parallelizable) kernel LLVM bc is stored in
   the kernel's temp dir. */
#define POCL_PARALLEL_BC_FILENAME   "/parallel.bc"

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

int pocl_cache_init_topdir ();

int
pocl_cache_create_program_cachedir(cl_program program, unsigned device_i,
                                   const char* preprocessed_source, size_t source_len,
                                   char *program_bc_path);

void pocl_cache_cleanup_cachedir(cl_program program);

int pocl_cl_device_to_index(cl_program   program,
                            cl_device_id device);

int pocl_cache_tempname (char *path_template, const char *suffix, int *fd);

int pocl_cache_create_tempdir(char* path);

int pocl_cache_write_program_source(char *program_cl_path,
                                    cl_program program);

int pocl_cache_write_kernel_objfile (char *objfile_path,
                                     const char *objfile_content,
                                     uint64_t objfile_size);

int pocl_cache_write_spirv (char *spirv_path,
                            const char *spirv_content,
                            uint64_t file_size);

int pocl_cache_update_program_last_access(cl_program program,
                                          unsigned device_i);


char* pocl_cache_read_buildlog(cl_program program, unsigned device_i);

int pocl_cache_append_to_buildlog(cl_program  program,
                                  unsigned    device_i,
                                  const char *content,
                                  size_t      size);


int pocl_cache_device_cachedir_exists(cl_program   program,
                                      unsigned device_i);

int pocl_cache_write_descriptor(cl_program   program,
                                unsigned     device_i,
                                const char*  kernel_name,
                                const char*  content,
                                size_t       size);

void pocl_cache_kernel_cachedir_path (char *kernel_cachedir_path,
                                      cl_program program, unsigned device_i,
                                      cl_kernel kernel, const char *append_str,
                                      _cl_command_node *command,
                                      int specialize);

int pocl_cache_write_kernel_parallel_bc (void *bc, cl_program program,
                                         int device_i, cl_kernel kernel,
                                         _cl_command_node *command,
                                         int specialize);

// required by pocl_binary.c

void pocl_cache_program_path (char *path, cl_program program,
                              unsigned device_i);

void pocl_cache_kernel_cachedir (char *kernel_cachedir_path,
                                 cl_program program, unsigned device_i,
                                 const char *kernel_name);

// these two required by llvm API

void pocl_cache_program_bc_path(char*       program_bc_path,
                               cl_program   program,
                               unsigned     device_i);

void pocl_cache_work_group_function_path (char *parallel_bc_path,
                                          cl_program program,
                                          unsigned device_i, cl_kernel kernel,
                                          _cl_command_node *command,
                                          int specialize);

void pocl_cache_final_binary_path (char *final_binary_path, cl_program program,
                                   unsigned device_i, cl_kernel kernel,
                                   _cl_command_node *command, int specialize);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}
#endif


#endif
