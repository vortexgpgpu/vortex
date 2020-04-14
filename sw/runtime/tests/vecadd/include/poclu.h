/* OpenCL runtime library: poclu - useful utility functions for OpenCL programs

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCLU_H
#define POCLU_H

#include <CL/opencl.h>

#ifdef _MSC_VER
#define POCLU_CALL __cdecl
#define POCLU_API __declspec(dllexport)
#else
#define POCLU_CALL
#define POCLU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 */
POCLU_API cl_int POCLU_CALL
poclu_bswap_cl_int(cl_device_id device, cl_int original);

POCLU_API cl_half POCLU_CALL
poclu_bswap_cl_half(cl_device_id device, cl_half original);

POCLU_API cl_float POCLU_CALL
poclu_bswap_cl_float(cl_device_id device, cl_float original);

POCLU_API cl_float2 POCLU_CALL
poclu_bswap_cl_float2(cl_device_id device, cl_float2 original);

/* In-place swapping of arrays. */
POCLU_API void POCLU_CALL
poclu_bswap_cl_int_array(cl_device_id device, cl_int* array, size_t num_elements);

POCLU_API void POCLU_CALL
poclu_bswap_cl_half_array(cl_device_id device, cl_half* array, size_t num_elements);

POCLU_API void POCLU_CALL
poclu_bswap_cl_float_array(cl_device_id device, cl_float* array, size_t num_elements);

POCLU_API void POCLU_CALL
poclu_bswap_cl_float2_array(cl_device_id device, cl_float2* array, size_t num_elements);

/**
 * Misc. helper functions for streamlining OpenCL API usage. 
 */

/* Create a context in the first platform found. */
POCLU_API cl_context POCLU_CALL
poclu_create_any_context();

/* Set up a context, device and queue for platform 0, device 0.
 * All input parameters must be allocated by caller!
 * Returns CL_SUCCESS on success, or a descriptive OpenCL error code upon failure.
 */
POCLU_API cl_int POCLU_CALL poclu_get_any_device2 (cl_context *context,
                                                   cl_device_id *device,
                                                   cl_command_queue *queue,
                                                   cl_platform_id *platform);

POCLU_API cl_int POCLU_CALL poclu_get_any_device (cl_context *context,
                                                  cl_device_id *device,
                                                  cl_command_queue *queue);

POCLU_API cl_int POCLU_CALL poclu_get_multiple_devices (
    cl_platform_id *platform, cl_context *context, cl_uint *num_devices,
    cl_device_id **devices, cl_command_queue **queues);
/**
 * cl_half related helpers.
 */
POCLU_API cl_half POCLU_CALL
poclu_float_to_cl_half(float value);

POCLU_API float POCLU_CALL
poclu_cl_half_to_float(cl_half value);

/* Read content of file to a malloc'd buffer, which is returned.
 * Return NULL on errors */
POCLU_API char *POCLU_CALL poclu_read_file (const char *filemane);

POCLU_API char *POCLU_CALL poclu_read_binfile (const char *filename,
                                               size_t *len);

POCLU_API int POCLU_CALL poclu_write_file (const char *filemane, char *content,
                                           size_t size);

int poclu_load_program (cl_context context, cl_device_id device,
                        const char *basename, int spir, int spirv, int poclbin,
                        const char *explicit_binary,
                        const char *extra_build_opts, cl_program *p);

int poclu_load_program_multidev (cl_context context, cl_device_id *devices,
                                 cl_uint num_devices, const char *basename,
                                 int spir, int spirv, int poclbin,
                                 const char *explicit_binary,
                                 const char *extra_build_opts, cl_program *p);

/* In case cl_err != CL_SUCCESS, prints out the error +
 * function : line to stderr, and returns 1, otherwise
 * returns 0
 */
POCLU_API int POCLU_CALL check_cl_error (cl_int cl_err, int line,
                                         const char *func_name);

#define _POCLU_CHECK_CL_ERROR_INNER(cond, func, line)                         \
  do                                                                          \
    {                                                                         \
      if (check_cl_error (cond, line, func))                                  \
        return (EXIT_FAILURE);                                                \
    }                                                                         \
  while (0)

#define CHECK_CL_ERROR(cond) _POCLU_CHECK_CL_ERROR_INNER(cond, __PRETTY_FUNCTION__, __LINE__)

#define CHECK_OPENCL_ERROR_IN(message) _POCLU_CHECK_CL_ERROR_INNER(err, message, __LINE__)


#define TEST_ASSERT(EXP)                                                \
do {                                                                    \
  if (!(EXP)) {                                                         \
    fprintf(stderr, "Assertion: \n" #EXP "\nfailed on %s:%i\n",         \
        __FILE__, __LINE__);                                            \
    return EXIT_FAILURE;                                                \
  }                                                                     \
} while (0)

#define CHECK_CL_ERROR2(err)                                                  \
  if (check_cl_error (err, __LINE__, __PRETTY_FUNCTION__))                    \
  goto ERROR

#ifdef __cplusplus
}
#endif

#endif
