#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

struct kernel_arg_t {
  uint32_t num_tasks;
  uint32_t nblocks;
  uint32_t key_ptr;
  uint32_t indec_ptr;
  uint32_t inenc_ptr;
  uint32_t outdec_ptr;
  uint32_t outenc_ptr;
};

#endif
