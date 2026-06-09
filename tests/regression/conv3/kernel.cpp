#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto I     = reinterpret_cast<TYPE*>(arg->I_addr);
  auto O     = reinterpret_cast<TYPE*>(arg->O_addr);
  auto width = arg->width;

  // Each CTA loads W into its private lmem slice if use_lmem
  TYPE* W;
  if (arg->use_lmem) {
    auto L    = reinterpret_cast<TYPE*>(__local_mem());
    auto Wsrc = reinterpret_cast<TYPE*>(arg->W_addr);
    if (threadIdx.x == 0) {
      for (int i = 0; i < 9; ++i) L[i] = Wsrc[i];
    }
    W = L;
  } else {
    W = reinterpret_cast<TYPE*>(arg->W_addr);
  }

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;

  if (col >= (int)width)
    return;

  int paddedWidth = width + 2;
  int paddedX     = col + 1;
  int paddedY     = row + 1;

  float sum = 0.0f;
  sum += I[(paddedY-1)*paddedWidth + (paddedX-1)] * W[0];
  sum += I[(paddedY-1)*paddedWidth +  paddedX    ] * W[1];
  sum += I[(paddedY-1)*paddedWidth + (paddedX+1)] * W[2];
  sum += I[ paddedY   *paddedWidth + (paddedX-1)] * W[3];
  sum += I[ paddedY   *paddedWidth +  paddedX    ] * W[4];
  sum += I[ paddedY   *paddedWidth + (paddedX+1)] * W[5];
  sum += I[(paddedY+1)*paddedWidth + (paddedX-1)] * W[6];
  sum += I[(paddedY+1)*paddedWidth +  paddedX    ] * W[7];
  sum += I[(paddedY+1)*paddedWidth + (paddedX+1)] * W[8];

  O[row * width + col] = sum;
}
