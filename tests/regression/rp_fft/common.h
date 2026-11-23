#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef RADIX
#define RADIX 4
#endif

#define FFT_LEN 4096

typedef struct {
  uint32_t num_ffts;
  uint32_t padding;
  uint64_t data_addr;   // [FFT_LEN][num_ffts] complex (re, im) input; holds the result
  uint64_t tmp_addr;    // [FFT_LEN][num_ffts] complex scratch
  uint64_t tw_addr;     // [FFT_LEN] complex twiddles exp(-2 pi i k / FFT_LEN)
} kernel_arg_t;

#endif
