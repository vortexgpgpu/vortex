#ifndef _COMMON_H_
#define _COMMON_H_

// Shared by kernel.cl and main.cc. Integer-only fields, so host and device
// lay the structs out identically.

#ifndef __OPENCL_VERSION__
#include <stdint.h>
typedef int8_t   cl_char_t;
typedef int16_t  cl_short_t;
typedef int32_t  cl_int_t;
typedef int64_t  cl_long_t;
#else
typedef char  cl_char_t;
typedef short cl_short_t;
typedef int   cl_int_t;
typedef long  cl_long_t;
#endif

// By-value kernel argument shapes: sizes 1, 12, 16 and 24, alignments 1, 4, 8.
typedef struct { cl_char_t c; } s1_t;
typedef struct { cl_int_t a; cl_short_t b; cl_char_t c; cl_int_t d; } s12_t;
typedef struct { cl_long_t q; cl_int_t i; } s16_t;
typedef struct { cl_int_t v[5]; cl_char_t tag; } s24_t;

#define BYVAL_OUTS 8   // ints written per work-item by k_byval
#define VEC4_RUN 4    // float4 elements per sorted run merged by k_vec4

#endif
