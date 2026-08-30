#ifndef _COMMON_H_
#define _COMMON_H_

#define FAULT_MODE_UNMAPPED 0   // dereference an address with no mapping
#define FAULT_MODE_READONLY 1   // store to a page mapped without write access

typedef struct {
  uint64_t bad_addr;   // virtual address the kernel dereferences
  uint64_t dst_addr;
  uint32_t mode;
} kernel_arg_t;

#endif
