#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

struct kernel_arg_t {
  uint32_t num_tasks;
  uint32_t msgsize;
  uint32_t nmsg;
  uint32_t msg_ptr;
  uint32_t digest_ptr;  
};

#endif
