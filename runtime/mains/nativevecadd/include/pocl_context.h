/* pocl_context.h - The 32b and 64b versions of the "context struct" that can be
   passed as a hidden kernel argument for kernels to fetch their WG/WI ID and
   dimension data.

   Copyright (c) 2018 Pekka Jääskeläinen / Tampere University of Technology

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

/* This header can be included both from device and host sources. */

#ifndef POCL_CONTEXT_H
#define POCL_CONTEXT_H

#include "pocl_types.h"

struct pocl_context {
#if __INTPTR_WIDTH__ == 64
  ulong num_groups[3];
  ulong global_offset[3];
  ulong local_size[3];
#elif __INTPTR_WIDTH__ == 32
  uint num_groups[3];
  uint global_offset[3];
  uint local_size[3];
#else 
  #error unsupported architecture
#endif
  uchar *printf_buffer;
  uint *printf_buffer_position;
  uint printf_buffer_capacity;
  uint work_dim;
};

struct pocl_context32 {
  uint num_groups[3];
  uint global_offset[3];
  uint local_size[3];
  uchar *printf_buffer;
  uint *printf_buffer_position;
  uint printf_buffer_capacity;
  uint work_dim;
};

/* Copy a 64b context struct to a 32b one. */
#define POCL_CONTEXT_COPY64TO32(__DST, __SRC)				\
  do {									\
    struct pocl_context *__src = (struct pocl_context *)__SRC;		\
    struct pocl_context32 *__dst = (struct pocl_context32 *)__DST;	\
    __dst->work_dim = __src->work_dim;					\
    __dst->num_groups[0] = __src->num_groups[0];			\
    __dst->num_groups[1] = __src->num_groups[1];			\
    __dst->num_groups[2] = __src->num_groups[2];			\
    __dst->global_offset[0] = __src->global_offset[0];			\
    __dst->global_offset[1] = __src->global_offset[1];			\
    __dst->global_offset[2] = __src->global_offset[2];			\
    __dst->local_size[0] = __src->local_size[0];			\
    __dst->local_size[1] = __src->local_size[1];			\
    __dst->local_size[2] = __src->local_size[2];			\
    __dst->printf_buffer = __src->printf_buffer;			\
    __dst->printf_buffer_position = __src->printf_buffer_position;	\
    __dst->printf_buffer_capacity = __src->printf_buffer_capacity;	\
  } while (0)

#define POCL_CONTEXT_SIZE(__BITNESS)					\
  (__BITNESS == 64 ?							\
   sizeof (struct pocl_context) :					\
   sizeof (struct pocl_context32))

#endif
