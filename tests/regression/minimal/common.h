#ifndef _COMMON_H_
#define _COMMON_H_

// Deliberately distinctive so a wrong value in the readback tells you whether
// the buffer was never written (0), or written by a kernel that ran with a
// corrupted binary (anything else).
#define MINIMAL_MAGIC 0x5A5A0000u

typedef struct {
  uint32_t num_points;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
