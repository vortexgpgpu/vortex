#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Kernel-argument block, shared by main.cpp (host) and kernel.cpp
// (device). The runtime stages it into a scratch slot and the KMU
// hands its address to the kernel in a0.
typedef struct {
  uint32_t count;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
