/* pocl/_kernel.h - OpenCL types and runtime library
   functions declarations. This should be included only from OpenCL C files.

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

/* If the -cl-std build option is not specified, the highest OpenCL C 1.x
 * language version supported by each device is used as the version of
 * OpenCL C when compiling the program for each device.
 */
#ifndef __OPENCL_C_VERSION__
#define __OPENCL_C_VERSION__ 120
#endif

#if (__OPENCL_C_VERSION__ > 99)
#define CL_VERSION_1_0 100
#endif

#if (__OPENCL_C_VERSION__ > 109)
#define CL_VERSION_1_1 110
#endif

#if (__OPENCL_C_VERSION__ > 119)
#define CL_VERSION_1_2 120
#endif

#if (__OPENCL_C_VERSION__ > 199)
#define CL_VERSION_2_0 200
#endif

#include "_enable_all_exts.h"

#include "_builtin_renames.h"

/* Define some feature test macros to help write generic code. These are used
 * mostly in _pocl_opencl.h header + some .cl files in kernel library */

#ifdef cl_khr_int64
#  define __IF_INT64(x) x
#else
#  define __IF_INT64(x)
#endif
#ifdef cl_khr_fp16
#  define __IF_FP16(x) x
#else
#  define __IF_FP16(x)
#endif
#ifdef cl_khr_fp64
#  define __IF_FP64(x) x
#else
#  define __IF_FP64(x)
#endif
#ifdef cl_khr_int64_base_atomics
#define __IF_BA64(x) x
#else
#define __IF_BA64(x)
#endif
#ifdef cl_khr_int64_extended_atomics
#define __IF_EA64(x) x
#else
#define __IF_EA64(x)
#endif

/****************************************************************************/

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

/************************ setup Clang version macros ******************/

#if (__clang_major__ == 6)

# undef LLVM_6_0
# define LLVM_6_0

#elif (__clang_major__ == 7)

# undef LLVM_7_0
# define LLVM_7_0

#elif (__clang_major__ == 8)

# undef LLVM_8_0
# define LLVM_8_0

#elif (__clang_major__ == 9)

# undef LLVM_9_0
# define LLVM_9_0

#elif (__clang_major__ == 10)

# undef LLVM_10_0
# define LLVM_10_0

#else

#error Unsupported Clang/LLVM version.

#endif

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

/****************************************************************************/

/* A static assert statement to catch inconsistencies at build time */
#if __has_extension(__c_static_assert__)
#  define _CL_STATIC_ASSERT(_t, _x) _Static_assert(_x, #_t)
#else
#  define _CL_STATIC_ASSERT(_t, _x) typedef int __cl_ai##_t[(x) ? 1 : -1];
#endif

/****************************************************************************/

#define IMG_RO_AQ __read_only
#define IMG_WO_AQ __write_only

#if (__OPENCL_C_VERSION__ > 199)
#define CLANG_HAS_RW_IMAGES
#define IMG_RW_AQ __read_write
#else
#undef CLANG_HAS_RW_IMAGES
#define IMG_RW_AQ __RW_IMAGES_UNSUPPORTED_BEFORE_CL_20
#endif

/****************************************************************************/
/* use Clang opencl header for definitions. */

#ifdef POCL_DEVICE_ADDRESS_BITS

/* If we wish to override the Clang set __SIZE_TYPE__ for this target,
   let's do it here so the opencl-c.h sets size_t to the wanted type. */

#ifdef __SIZE_TYPE__
#undef __SIZE_TYPE__
#endif

#if POCL_DEVICE_ADDRESS_BITS == 32
#define __SIZE_TYPE__ uint
#elif POCL_DEVICE_ADDRESS_BITS == 64
#define __SIZE_TYPE__ ulong
#else
#error Unsupported POCL_DEVICE_ADDRESS_BITS value.
#endif

#endif

#include "_clang_opencl.h"

/****************************************************************************/

/* GNU's libm seems to use INT_MIN here while the Clang's header uses
   INT_MAX. Both are allowed by the OpenCL specs, but we want them to
   be unified to avoid failing tests. */
#undef FP_ILOGBNAN
#undef FP_ILOGB0
#define FP_ILOGBNAN INT_MIN
#define FP_ILOGB0 INT_MIN

/****************************************************************************/

#include "pocl_image_types.h"

#pragma OPENCL EXTENSION all : disable
