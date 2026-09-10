#ifndef _COMMON_H_
#define _COMMON_H_

#define WORDS_PER_PAGE (4096 / 4)

typedef struct {
  uint32_t num_tasks;
  uint32_t pages_per_task;
  uint32_t total_pages;
  uint32_t stride_pages;
  uint32_t phys_words;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t phys_addr;
} kernel_arg_t;

#endif
