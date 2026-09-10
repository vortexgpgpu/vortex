#ifndef _COMMON_H_
#define _COMMON_H_

// Test modes, selected by kernel_arg_t::mode.
#define VM_MODE_STRIDE     0  // page-strided write/readback sweep
#define VM_MODE_FENCE      1  // cold-page store, fence, barrier, neighbor read
#define VM_MODE_DRAIN      2  // trailing cold-page store burst at kernel end
#define VM_MODE_AMO        3  // atomics contended on cold pages
#define VM_MODE_SUPERPAGE  4  // repeated reads through a superpage-mapped region

typedef struct {
  uint32_t mode;
  uint32_t num_tasks;
  uint32_t pages_per_task;
  uint64_t buf_addr;      // page-granular working buffer
  uint64_t dst_addr;      // per-task result word
  uint64_t aux_addr;      // mode-specific: AMO counters / superpage base
} kernel_arg_t;

#endif
