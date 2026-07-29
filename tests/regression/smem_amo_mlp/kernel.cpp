#include <stdint.h>
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include "common.h"

static inline uint32_t amoadd_word(volatile uint32_t* addr, uint32_t value) {
  uint32_t old;
  asm volatile ("amoadd.w %0, %2, (%1)"
                : "=r"(old)
                : "r"(addr), "r"(value)
                : "memory");
  return old;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto old_sums = reinterpret_cast<uint32_t*>(arg->old_sums_addr);
  auto final_values = reinterpret_cast<uint32_t*>(arg->final_values_addr);
  auto bank_words = lmem;
  auto private_words = lmem + arg->num_banks;

  if (tid < arg->num_banks)
    bank_words[tid] = 0;
  private_words[tid] = arg->mode == SMEM_AMO_DIRECTED ? 7 : 0;

  vortex::barrier sync(0);
  const bool needs_sync = get_num_sub_groups() > 1;
  if (needs_sync)
    sync.arrive_and_wait();

  uint32_t old_sum = 0;
  if (arg->mode == SMEM_AMO_DIRECTED) {
    auto target = &private_words[tid];
    old_sums[tid] = amoadd_word(target, 0);
    old_sums[arg->num_harts + tid] = amoadd_word(target, 1);
    old_sums[2 * arg->num_harts + tid] = amoadd_word(target, uint32_t(-1));
  } else if (arg->mode == SMEM_AMO_SAME_BANK) {
    for (uint32_t i = 0; i < arg->iters; ++i)
      old_sum += amoadd_word(&bank_words[0], 1);
  } else if (arg->mode == SMEM_AMO_ALL_BANKS) {
    auto target = &bank_words[tid % arg->num_banks];
    for (uint32_t i = 0; i < arg->iters; ++i)
      old_sum += amoadd_word(target, 1);
  } else {
    auto target = &private_words[tid];
    for (uint32_t i = 0; i < arg->iters; ++i) {
      uint32_t old = *target;
      *target = old + 1;
      old_sum += old;
    }
  }
  if (arg->mode != SMEM_AMO_DIRECTED)
    old_sums[tid] = old_sum;

  if (needs_sync)
    sync.arrive_and_wait();
  if (arg->mode == SMEM_PRIVATE_RMW || arg->mode == SMEM_AMO_DIRECTED) {
    final_values[tid] = private_words[tid];
  } else if (tid < arg->num_banks) {
    final_values[tid] = bank_words[tid];
  }
}

// Current master consumes __vx_kentry_* symbols in vxbin.py, while the
// installed Vortex LLVM predates automatic emission of those aliases.
extern "C" void __vx_kentry_smem_amo_mlp(kernel_arg_t*)
  __attribute__((alias("kernel_main"), used));
