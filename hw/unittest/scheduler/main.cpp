// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vl_simulator.h"
#include "VVX_scheduler_top.h"

#include <cstdint>
#include <cstdio>

#include <VX_config.h>

using Device = VVX_scheduler_top;

static uint64_t timestamp = 0;
static uint32_t checks = 0;
static uint32_t failures = 0;

double sc_time_stamp() { return timestamp; }

#define CHECKX(cond, ...)                                      \
  do {                                                         \
    ++checks;                                                  \
    if (!(cond)) {                                             \
      std::printf("FAILED (t=%llu): %s -- ",                 \
                  (unsigned long long)timestamp, #cond);       \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                     \
      ++failures;                                              \
    }                                                          \
  } while (0)

struct Bench {
  vl_simulator<Device> sim;

  void eval() { sim->eval(); }
  void tick() { timestamp = sim.step(timestamp, 2); }

  void reset() {
    sim->task_valid = 0;
    sim->task_block_size = 0;
    sim->task_ctx_id = 0;
    sim->ctl_tmc_valid = 0;
    sim->ctl_wid = 0;
    sim->ctl_tmask = 0;
    sim->group_drained_mask = (1u << VX_CFG_NUM_WARPS) - 1;
    timestamp = sim.reset(timestamp);
  }

  void submit_cta(uint32_t block_size, uint32_t ctx_id) {
    sim->task_block_size = block_size;
    sim->task_ctx_id = ctx_id;
    sim->task_valid = 1;
    for (uint32_t guard = 0; guard < 1000; ++guard) {
      eval();
      const bool fire = sim->task_ready;
      tick();
      if (fire) {
        sim->task_valid = 0;
        return;
      }
    }
    CHECKX(false, "CTA request was never accepted");
  }

  uint32_t collect_dispatches(uint32_t count) {
    uint32_t mask = 0;
    for (uint32_t guard = 0; guard < 1000 && count != 0; ++guard) {
      eval();
      if (sim->owner_start_valid) {
        const uint32_t wid = sim->owner_start_wid;
        CHECKX((mask & (1u << wid)) == 0,
               "wid %u dispatched more than once", wid);
        mask |= 1u << wid;
        --count;
      }
      tick();
    }
    CHECKX(count == 0, "timed out waiting for CTA warp dispatches");
    return mask;
  }

  void tmc(uint32_t wid, uint32_t tmask, bool expect_owner_close = false) {
    sim->ctl_wid = wid;
    sim->ctl_tmask = tmask;
    sim->ctl_tmc_valid = 1;
    eval();
    CHECKX(bool(sim->owner_close_valid) == expect_owner_close,
           "owner_close valid mismatch for wid%u tmask=0x%x", wid, tmask);
    if (expect_owner_close)
      CHECKX(sim->owner_close_wid == wid, "owner_close tagged wrong wid");
    tick();
    sim->ctl_tmc_valid = 0;
  }

  void verify_exit_hold() {
    const uint32_t all_warps = (1u << VX_CFG_NUM_WARPS) - 1;
    const uint32_t all_threads = (1u << VX_CFG_NUM_THREADS) - 1;

    submit_cta(VX_CFG_NUM_THREADS * VX_CFG_NUM_WARPS, 1);
    const uint32_t dispatched = collect_dispatches(VX_CFG_NUM_WARPS);
    CHECKX(dispatched == all_warps,
           "initial dispatch mask=0x%x want=0x%x", dispatched, all_warps);

    sim->group_drained_mask = all_warps & ~1u;
    tmc(0, 0, true);
    eval();
    CHECKX((sim->active_warps & 1u) == 0,
           "retiring wid0 remained active while waiting for drain");

    for (uint32_t wid = 1; wid < VX_CFG_NUM_WARPS; ++wid)
      tmc(wid, 0, true);
    eval();
    CHECKX(sim->active_warps == 0, "active mask did not clear on exit");
    CHECKX(sim->scheduler_busy, "busy dropped while wid0 was exit-held");

    // Occupy every non-held wid, leaving wid0 as the only apparent free slot.
    for (uint32_t wid = 1; wid < VX_CFG_NUM_WARPS; ++wid)
      tmc(wid, all_threads, false);
    submit_cta(VX_CFG_NUM_THREADS, 2);
    for (uint32_t cycle = 0; cycle < 16; ++cycle) {
      eval();
      CHECKX(!sim->owner_start_valid,
             "CTA reused a wid while its prior owner had not drained");
      tick();
    }

    sim->group_drained_mask = all_warps;
    bool reused = false;
    for (uint32_t guard = 0; guard < 64 && !reused; ++guard) {
      eval();
      if (sim->owner_start_valid) {
        CHECKX(sim->owner_start_wid == 0,
               "post-drain dispatch reused wid%u instead of wid0",
               unsigned(sim->owner_start_wid));
        reused = true;
      }
      tick();
    }
    CHECKX(reused, "wid0 was not reused after its tracker drained");
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Bench bench;
  bench.reset();
  bench.verify_exit_hold();
  std::printf("checks=%u failed=%u\n", checks, failures);
  if (failures != 0)
    return 1;
  std::printf("PASSED\n");
  return 0;
}
