#ifndef _COMMON_H_
#define _COMMON_H_

// Test modes, selected by kernel_arg_t::mode.
#define VM_MODE_STRIDE     0  // page-strided write/readback sweep
#define VM_MODE_FENCE      1  // cold-page store, fence, barrier, neighbor read
#define VM_MODE_DRAIN      2  // trailing cold-page store burst at kernel end
#define VM_MODE_AMO        3  // atomics contended on cold pages
#define VM_MODE_SUPERPAGE  4  // repeated reads through a superpage-mapped region
#define VM_MODE_RELEASE    5  // cold-page store, fence, atomic guard; the partner acquires
#define VM_MODE_REUSE      6  // freed pages reused by a larger buffer, then a new one

// Release mode: block 0 writes and block 1 reads. The writer's lanes send
// RELEASE_PAD_BURSTS bursts of stores to cold pages ahead of the page store,
// one page per page-table region, so the buffer must span
// RELEASE_REGIONS regions.
#define RELEASE_BLOCK_SIZE 4u
#define RELEASE_PAD_BURSTS 3u
#define RELEASE_REGIONS    ((RELEASE_PAD_BURSTS + 1) * RELEASE_BLOCK_SIZE)

// Pages mapped by one leaf page-table page.
#define VM_REGION_PAGES    (VX_VM_PT_SIZE / VX_VM_PTE_SIZE)

typedef struct {
  uint32_t mode;
  uint32_t num_tasks;
  uint32_t pages_per_task;
  uint64_t buf_addr;      // page-granular working buffer
  uint64_t dst_addr;      // per-task result word
  uint64_t aux_addr;      // mode-specific: AMO counters / superpage base
} kernel_arg_t;

#endif
