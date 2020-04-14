/* pocl_types.h - The basic OpenCL C device side scalar data types.

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

/* This header is designed to be included both from the device and the host.
   In case compiling OpenCL C sources, __OPENCL_VERSION__ should be set.
   In case compiling in the host, all but the device-specific types are
   defined (size_t and others). Devices should avoid including the C
   stdint.h instead of this one as OpenCL C size_t et al. is allowed to
   be of different width than when targeting C.

   TODO: replace this header (partially) with Clang's opencl-c.h
*/

#ifndef POCL_DEVICE_TYPES_H
#define POCL_DEVICE_TYPES_H

#ifdef __OPENCL_VERSION__

#ifdef __USE_CLANG_OPENCL_C_H

/* Minimal definitions, only the target specific macro overrides,
   just in case Clang export the C ones which might differ for
   OpenCL C. */

#ifdef __INTPTR_TYPE__
#undef __INTPTR_TYPE__
#endif

#ifdef __UINTPTR_TYPE__
#undef __UINTPTR_TYPE__
#endif

#ifdef __SIZE_TYPE__
#undef __SIZE_TYPE__
#endif

#ifdef __SIZE_MAX__
#undef __SIZE_MAX__
#endif

#if defined(POCL_DEVICE_ADDRESS_BITS) && POCL_DEVICE_ADDRESS_BITS == 32
#define __SIZE_TYPE__ uint
#define __SIZE_MAX__ UINT_MAX
#else
#define __SIZE_TYPE__ ulong
#define __SIZE_MAX__ ULONG_MAX
#endif

#define __INTPTR_TYPE__ __SIZE_TYPE__
#define __UINTPTR_TYPE__ __INTPTR_TYPE__

#else

/* Compiling Device-specific OpenCL C or builtin library C. */

#if defined cl_khr_fp64 && !defined cl_khr_int64
#error "cl_khr_fp64 requires cl_khr_int64"
#endif

/* TODO FIXME We should not use these in OpenCL library's C code at all.
 * The problem is that 1) these are predefined by glibc, 2) while we can
 * re-define "ulong", we cannot control the size of "long" at all.
 * which can lead to "ulong" being 64bit and "long" 32bit, resulting in
 * mysterious errors and bugs. Therefore OpenCL library's C code should
 * use the fixed size C types where integer size matters. */

#ifdef __CBUILD__

/* Builtin library C code definitions. */

#define size_t csize_t
#define uintptr_t cuintptr_t

#include <stdint.h>

#undef size_t
#undef uintptr_t

typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;

#ifdef cl_khr_int64
typedef uint64_t ulong;
#else
typedef uint32_t ulong;
#endif

#ifndef cl_khr_fp16
typedef short half;
#endif

#endif

/* The definitions below intentionally lead to errors if these types
   are used when they are not available in the language. This prevents
   accidentally using them if the compiler does not disable these
   types, but only e.g. defines them with an incorrect size.*/

#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#define double error_undefined_type_double
#endif

#ifdef __SIZE_TYPE__
#undef __SIZE_TYPE__
#endif

#ifdef __SIZE_MAX__
#undef __SIZE_MAX__
#endif

#if defined(POCL_DEVICE_ADDRESS_BITS) && POCL_DEVICE_ADDRESS_BITS == 32
#define __SIZE_TYPE__ uint
#define __SIZE_MAX__ UINT_MAX
#else
#define __SIZE_TYPE__ ulong
#define __SIZE_MAX__ ULONG_MAX
#endif

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef ptrdiff_t intptr_t;
typedef size_t uintptr_t;

#endif /* #ifdef __USE_CLANG_OPENCL_C_H */

#else /* #ifdef __OPENCL_VERSION__ */

/* Including from a host source (runtime API implementation). Introduce
   the fixed width datatypes, but do not override C's size_t and other
   target specific datatypes. */

typedef unsigned char uchar;

/* FIXME see the above TODO about these types. */

#if !(defined(_SYS_TYPES_H) && defined(__USE_MISC))
/* glibc, when including sys/types.h, typedefs these. */

typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;

#endif

#include <stdint.h>

#endif

#endif
