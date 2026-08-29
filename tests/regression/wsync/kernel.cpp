#include <vx_intrinsics.h>
#include <vx_spawn2.h>
#include "common.h"

// Minimum extra cycles that sync_cycle must exceed raw_cycle beyond the
// idle baseline.  L2 latency is ~30-50 cycles; DRAM is ~100-200 cycles.
// Either is well above this threshold.
static constexpr uint32_t kMinDrainExtra = 16;
static constexpr uint32_t kLoadsPerIter  = 8;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid        = threadIdx.x;
  uint32_t iterations = arg->iterations;

  // Shared read-only source buffer.  Declared volatile so the compiler
  // emits every load and does not hoist/CSE them across the memory barrier.
  volatile uint32_t* src =
    reinterpret_cast<volatile uint32_t*>(arg->src_addr);

  auto results = reinterpret_cast<lane_result_t*>(arg->results_addr);
  lane_result_t result = {};

  // --- Baseline: drain an idle pipeline and measure the inherent gap ---
  vx_wsync();
  uint32_t baseline_raw = csr_read(VX_CSR_MCYCLE);
  __asm__ volatile("" : : "r"(baseline_raw) : "memory");
  uint32_t baseline_sync = static_cast<uint32_t>(vx_rdcycle_sync());
  __asm__ volatile("" : : "r"(baseline_sync) : "memory");
  result.baseline_gap = baseline_sync - baseline_raw;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Issue kLoadsPerIter loads to distinct cache lines.
    // base cycles through WSYNC_BUF_LINES lines so the working set
    // (8 lines × ceil(iters/64) passes) always exceeds the 256-line L1.
    uint32_t line = ((iter * kLoadsPerIter) % WSYNC_BUF_LINES) * WSYNC_LINE_WORDS;

    // Each lN targets a different cache line (stride = WSYNC_LINE_WORDS words
    // = 64 bytes).  These dispatch to the LSU and proceed toward DRAM.
    // CSRR has no register dependency on lN, so the scoreboard does not
    // stall it; it dispatches to the SFU while the loads are still in-flight.
    uint32_t l0 = src[line + 0*WSYNC_LINE_WORDS];
    uint32_t l1 = src[line + 1*WSYNC_LINE_WORDS];
    uint32_t l2 = src[line + 2*WSYNC_LINE_WORDS];
    uint32_t l3 = src[line + 3*WSYNC_LINE_WORDS];
    uint32_t l4 = src[line + 4*WSYNC_LINE_WORDS];
    uint32_t l5 = src[line + 5*WSYNC_LINE_WORDS];
    uint32_t l6 = src[line + 6*WSYNC_LINE_WORDS];
    uint32_t l7 = src[line + 7*WSYNC_LINE_WORDS];

    // Compiler barrier: all loads must be emitted before the CSRR.
    __asm__ volatile("" : : : "memory");

    // CSRR dispatches to SFU while loads are in-flight.
    uint32_t raw_cycle = csr_read(VX_CSR_MCYCLE);
    __asm__ volatile("" : : "r"(raw_cycle) : "memory");

    // WSYNC drains all in-flight loads before sampling the cycle counter.
    uint32_t sync_cycle = static_cast<uint32_t>(vx_rdcycle_sync());
    uint32_t gap = sync_cycle - raw_cycle;
    __asm__ volatile("" : : "r"(sync_cycle) : "memory");

    // Include load results in the checksum to prevent dead-code elimination.
    result.checksum ^= (l0^l1^l2^l3^l4^l5^l6^l7) ^ raw_cycle ^ sync_cycle ^ gap;

    if (gap <= result.baseline_gap + kMinDrainExtra) {
      if (0 == result.failures) {
        result.first_iteration = iter;
        result.raw_cycle       = raw_cycle;
        result.sync_cycle      = sync_cycle;
        result.gap             = gap;
      }
      ++result.failures;
    }
  }

  results[tid] = result;
}
