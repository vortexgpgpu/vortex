/* pocl_image_types.h - image data structure used by device implementations

   Copyright (c) 2013 Ville Korhonen
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

#ifndef __X86_IMAGE_H__
#define __X86_IMAGE_H__

#ifdef __CBUILD__
#define INTTYPE cl_int
#else
#define INTTYPE int
#endif

typedef uintptr_t dev_sampler_t;

typedef struct dev_image_t {
  void *_data;
  INTTYPE _width;
  INTTYPE _height;
  INTTYPE _depth;
  INTTYPE _image_array_size;
  INTTYPE _row_pitch;
  INTTYPE _slice_pitch;
  INTTYPE _num_mip_levels; /* maybe not needed */
  INTTYPE _num_samples;    /* maybe not needed */
  INTTYPE _order;
  INTTYPE _data_type;
  INTTYPE _num_channels;
  INTTYPE _elem_size;
} dev_image_t;

#endif
