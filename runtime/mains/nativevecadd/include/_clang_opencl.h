/* This file includes opencl-c.h from Clang and fixes a few pocl extras.

   Copyright (c) 2011-2017 Pekka Jääskeläinen / TUT
   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#ifndef _OPENCL_H_
/* Use the declarations shipped with Clang. */
/* Check for _OPENCL_H already here because the kernel compiler loads the
   header beforehand, but cannot find the file due to include paths not
   set up. */
#include <opencl-c.h>

/* Missing declarations from opencl-c.h. Some of the geometric builtins are
   defined only up to 4 vectors, but we implement them all: */
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half _CL_OVERLOADABLE _CL_READNONE length (half8 p);
half _CL_OVERLOADABLE _CL_READNONE length (half16 p);

half _CL_OVERLOADABLE _CL_READNONE fast_length (half8 p);
half _CL_OVERLOADABLE _CL_READNONE fast_length (half16 p);

half8 _CL_OVERLOADABLE _CL_READNONE normalize (half8 p);
half16 _CL_OVERLOADABLE _CL_READNONE normalize (half16 p);

half8 _CL_OVERLOADABLE _CL_READNONE fast_normalize (half8 p);
half16 _CL_OVERLOADABLE _CL_READNONE fast_normalize (half16 p);

half _CL_OVERLOADABLE _CL_READNONE dot (half8 p0, half8 p1);
half _CL_OVERLOADABLE _CL_READNONE dot (half16 p0, half16 p1);
#endif

float _CL_OVERLOADABLE _CL_READNONE length (float8 p);
float _CL_OVERLOADABLE _CL_READNONE length (float16 p);

float _CL_OVERLOADABLE _CL_READNONE fast_length (float8 p);
float _CL_OVERLOADABLE _CL_READNONE fast_length (float16 p);

float8 _CL_OVERLOADABLE _CL_READNONE normalize (float8 p);
float16 _CL_OVERLOADABLE _CL_READNONE normalize (float16 p);

float8 _CL_OVERLOADABLE _CL_READNONE fast_normalize (float8 p);
float16 _CL_OVERLOADABLE _CL_READNONE fast_normalize (float16 p);

float _CL_OVERLOADABLE _CL_READNONE dot (float8 p0, float8 p1);
float _CL_OVERLOADABLE _CL_READNONE dot (float16 p0, float16 p1);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double _CL_OVERLOADABLE _CL_READNONE length (double8 p);
double _CL_OVERLOADABLE _CL_READNONE length (double16 p);

double _CL_OVERLOADABLE _CL_READNONE fast_length (double p);
double _CL_OVERLOADABLE _CL_READNONE fast_length (double2 p);
double _CL_OVERLOADABLE _CL_READNONE fast_length (double3 p);
double _CL_OVERLOADABLE _CL_READNONE fast_length (double4 p);
double _CL_OVERLOADABLE _CL_READNONE fast_length (double8 p);
double _CL_OVERLOADABLE _CL_READNONE fast_length (double16 p);

double8 _CL_OVERLOADABLE _CL_READNONE normalize (double8 p);
double16 _CL_OVERLOADABLE _CL_READNONE normalize (double16 p);

double8 _CL_OVERLOADABLE _CL_READNONE fast_normalize (double8 p);
double16 _CL_OVERLOADABLE _CL_READNONE fast_normalize (double16 p);

double _CL_OVERLOADABLE _CL_READNONE dot (double8 p0, double8 p1);
double _CL_OVERLOADABLE _CL_READNONE dot (double16 p0, double16 p1);

#endif

#endif