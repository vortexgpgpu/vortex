/* pocl/_kernel_constants.h - C compatible OpenCL types and runtime library
   constants declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2011-2013 Pekka Jääskeläinen / TUT
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
 * built-in functions to introduce the OpenCL C compatible constants.
 */
#ifndef _KERNEL_CONSTANTS_H
#define _KERNEL_CONSTANTS_H

/* clang's header defines these */
#ifndef _OPENCL_H_

/* cl_channel_order */
#define CLK_R                                        0x10B0
#define CLK_A                                        0x10B1
#define CLK_RG                                       0x10B2
#define CLK_RA                                       0x10B3
#define CLK_RGB                                      0x10B4
#define CLK_RGBA                                     0x10B5
#define CLK_BGRA                                     0x10B6
#define CLK_ARGB                                     0x10B7
#define CLK_INTENSITY                                0x10B8
#define CLK_LUMINANCE                                0x10B9
#define CLK_Rx                                       0x10BA
#define CLK_RGx                                      0x10BB
#define CLK_RGBx                                     0x10BC
#define CLK_DEPTH                                    0x10BD
#define CLK_DEPTH_STENCIL                            0x10BE

/* cl_channel_type */
#define CLK_SNORM_INT8                               0x10D0
#define CLK_SNORM_INT16                              0x10D1
#define CLK_UNORM_INT8                               0x10D2
#define CLK_UNORM_INT16                              0x10D3
#define CLK_UNORM_SHORT_565                          0x10D4
#define CLK_UNORM_SHORT_555                          0x10D5
#define CLK_UNORM_INT_101010                         0x10D6
#define CLK_SIGNED_INT8                              0x10D7
#define CLK_SIGNED_INT16                             0x10D8
#define CLK_SIGNED_INT32                             0x10D9
#define CLK_UNSIGNED_INT8                            0x10DA
#define CLK_UNSIGNED_INT16                           0x10DB
#define CLK_UNSIGNED_INT32                           0x10DC
#define CLK_HALF_FLOAT                               0x10DD
#define CLK_FLOAT                                    0x10DE
#define CLK_UNORM_INT24                              0x10DF

/* cl_addressing _mode */
#define CLK_ADDRESS_NONE                            0x00
#define CLK_ADDRESS_CLAMP_TO_EDGE                   0x02
#define CLK_ADDRESS_CLAMP                           0x04
#define CLK_ADDRESS_REPEAT                          0x06
#define CLK_ADDRESS_MIRRORED_REPEAT                 0x08

/* cl_sampler_info */
#define CLK_NORMALIZED_COORDS_FALSE                 0x00
#define CLK_NORMALIZED_COORDS_TRUE                  0x01

/* filter_mode */
#define CLK_FILTER_NEAREST                          0x10
#define CLK_FILTER_LINEAR                           0x20

/* barrier() flags */
#define CLK_LOCAL_MEM_FENCE                         0x01
#define CLK_GLOBAL_MEM_FENCE                        0x02

#endif

#endif
