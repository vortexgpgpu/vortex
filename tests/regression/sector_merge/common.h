#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_lines;
  uint32_t line_words;
  uint32_t group;
  uint32_t reserved;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
