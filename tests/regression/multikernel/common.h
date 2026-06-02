#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Kernel-argument block shared between host and device.
// The runtime stages it into a scratch slot; the KMU passes its address in a0.
typedef struct {
  uint32_t count;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
