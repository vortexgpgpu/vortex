/* pocl/_kernel_c.h - C compatible OpenCL types and runtime library
   functions declarations for kernel builtin implementations using C.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2011-2017 Pekka Jääskeläinen / TUT
   Copyright (c) 2011-2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                           Perimeter Institute for Theoretical Physics

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
 * Header that can be implemented in C compiled implementations of
 * built-in functions to introduce the OpenCL C compatible types etc.
 */
#ifndef _KERNEL_C_H
#define _KERNEL_C_H

#include "pocl_types.h"

#include "_kernel_constants.h"

/* Function/type attributes supported by Clang/SPIR */
#if __has_attribute(__always_inline__)
#  define _CL_ALWAYSINLINE __attribute__((__always_inline__))
#else
#  define _CL_ALWAYSINLINE
#endif
#if __has_attribute(__noinline__)
#  define _CL_NOINLINE __attribute__((__noinline__))
#else
#  define _CL_NOINLINE
#endif
#if __has_attribute(__overloadable__)
#  define _CL_OVERLOADABLE __attribute__((__overloadable__))
#else
#  define _CL_OVERLOADABLE
#endif
#if __has_attribute(__pure__)
#  define _CL_READONLY __attribute__((__pure__))
#else
#  define _CL_READONLY
#endif
#if __has_attribute(__const__)
#  define _CL_READNONE __attribute__((__const__))
#else
#  define _CL_READNONE
#endif
#if __has_attribute(convergent)
#  define _CL_CONVERGENT __attribute__((convergent))
#else
#  define _CL_CONVERGENT
#endif


typedef char char2  __attribute__((__ext_vector_type__(2)));
typedef char char3  __attribute__((__ext_vector_type__(3)));
typedef char char4  __attribute__((__ext_vector_type__(4)));
typedef char char8  __attribute__((__ext_vector_type__(8)));
typedef char char16 __attribute__((__ext_vector_type__(16)));

typedef uchar uchar2  __attribute__((__ext_vector_type__(2)));
typedef uchar uchar3  __attribute__((__ext_vector_type__(3)));
typedef uchar uchar4  __attribute__((__ext_vector_type__(4)));
typedef uchar uchar8  __attribute__((__ext_vector_type__(8)));
typedef uchar uchar16 __attribute__((__ext_vector_type__(16)));

typedef short short2  __attribute__((__ext_vector_type__(2)));
typedef short short3  __attribute__((__ext_vector_type__(3)));
typedef short short4  __attribute__((__ext_vector_type__(4)));
typedef short short8  __attribute__((__ext_vector_type__(8)));
typedef short short16 __attribute__((__ext_vector_type__(16)));

typedef ushort ushort2  __attribute__((__ext_vector_type__(2)));
typedef ushort ushort3  __attribute__((__ext_vector_type__(3)));
typedef ushort ushort4  __attribute__((__ext_vector_type__(4)));
typedef ushort ushort8  __attribute__((__ext_vector_type__(8)));
typedef ushort ushort16 __attribute__((__ext_vector_type__(16)));

typedef int int2  __attribute__((__ext_vector_type__(2)));
typedef int int3  __attribute__((__ext_vector_type__(3)));
typedef int int4  __attribute__((__ext_vector_type__(4)));
typedef int int8  __attribute__((__ext_vector_type__(8)));
typedef int int16 __attribute__((__ext_vector_type__(16)));

typedef uint uint2  __attribute__((__ext_vector_type__(2)));
typedef uint uint3  __attribute__((__ext_vector_type__(3)));
typedef uint uint4  __attribute__((__ext_vector_type__(4)));
typedef uint uint8  __attribute__((__ext_vector_type__(8)));
typedef uint uint16 __attribute__((__ext_vector_type__(16)));

#if defined(__CBUILD__) && defined(cl_khr_fp16)
/* NOTE: the Clang's __fp16 does not work robustly in C mode,
   it might produce invalid code at least with half vectors.
   Using the native 'half' type in OpenCL C mode works better. */
typedef __fp16 half;
#endif

typedef half half2  __attribute__((__ext_vector_type__(2)));
typedef half half3  __attribute__((__ext_vector_type__(3)));
typedef half half4  __attribute__((__ext_vector_type__(4)));
typedef half half8  __attribute__((__ext_vector_type__(8)));
typedef half half16 __attribute__((__ext_vector_type__(16)));

typedef float float2  __attribute__((__ext_vector_type__(2)));
typedef float float3  __attribute__((__ext_vector_type__(3)));
typedef float float4  __attribute__((__ext_vector_type__(4)));
typedef float float8  __attribute__((__ext_vector_type__(8)));
typedef float float16 __attribute__((__ext_vector_type__(16)));

#ifdef cl_khr_fp64
#  ifndef __CBUILD__
#    pragma OPENCL EXTENSION cl_khr_fp64 : enable
#  endif
typedef double double2  __attribute__((__ext_vector_type__(2)));
typedef double double3  __attribute__((__ext_vector_type__(3)));
typedef double double4  __attribute__((__ext_vector_type__(4)));
typedef double double8  __attribute__((__ext_vector_type__(8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));
#endif

#ifdef cl_khr_int64
typedef long long2  __attribute__((__ext_vector_type__(2)));
typedef long long3  __attribute__((__ext_vector_type__(3)));
typedef long long4  __attribute__((__ext_vector_type__(4)));
typedef long long8  __attribute__((__ext_vector_type__(8)));
typedef long long16 __attribute__((__ext_vector_type__(16)));

typedef ulong ulong2  __attribute__((__ext_vector_type__(2)));
typedef ulong ulong3  __attribute__((__ext_vector_type__(3)));
typedef ulong ulong4  __attribute__((__ext_vector_type__(4)));
typedef ulong ulong8  __attribute__((__ext_vector_type__(8)));
typedef ulong ulong16 __attribute__((__ext_vector_type__(16)));
#endif

#if defined(__TCE__)

#define POCL_ADDRESS_SPACE_PRIVATE 0
#define POCL_ADDRESS_SPACE_GLOBAL 1
#define POCL_ADDRESS_SPACE_LOCAL 3
#define POCL_ADDRESS_SPACE_CONSTANT 2
#define POCL_ADDRESS_SPACE_GENERIC 6

#endif

typedef uint cl_mem_fence_flags;

/* Integer Constants */

#if defined(__CBUILD__)

#define CHAR_BIT  8
#define CHAR_MAX  SCHAR_MAX
#define CHAR_MIN  SCHAR_MIN
#define INT_MAX   2147483647
#define INT_MIN   (-2147483647 - 1)
#ifdef cl_khr_int64
#define LONG_MAX  0x7fffffffffffffffL
#define LONG_MIN  (-0x7fffffffffffffffL - 1)
#endif
#define SCHAR_MAX 127
#define SCHAR_MIN (-127 - 1)
#define SHRT_MAX  32767
#define SHRT_MIN  (-32767 - 1)
#define UCHAR_MAX 255
#define USHRT_MAX 65535
#define UINT_MAX  0xffffffff
#ifdef cl_khr_int64
#define ULONG_MAX 0xffffffffffffffffUL
#endif

#endif /* __CBUILD__ */

#endif
