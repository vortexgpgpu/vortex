// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_cache_mshr at a non-power-of-two MSHR_SIZE.
//
// 1. Out of reset, with no fill yet, the dequeue pointer must name a valid
//    entry: the RAM read port and the tables are indexed by it every cycle.
//    Registers power up random (--x-initial unique); the test sweeps seeds,
//    so a pointer that is not reset lands on an id >= MSHR_SIZE for some of
//    them (and trips the module's candidate-select assertion).
// 2. A miss allocated, then filled, dequeues its own entry.

#include "VVX_cache_mshr.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <memory>

#ifndef MSHR_SIZE
#error "MSHR_SIZE"
#endif

#define NUM_SEEDS 32

double sc_time_stamp() { return 0; }

static int failures = 0;

#define CHECK(x, ...)                                  \
  do {                                                 \
    if (!(x)) {                                        \
      printf("FAILED: %s: ", #x); printf(__VA_ARGS__); \
      printf("\n");                                    \
      ++failures;                                      \
    }                                                  \
  } while (false)

static void clear_inputs(VVX_cache_mshr* m) {
  m->fill_valid = 0; m->fill_id = 0;
  m->probe_addr = 0;
  m->dequeue_ready = 0; m->dequeue_park = 0; m->dequeue_park_retry = 0; m->dequeue_block = 0;
  m->allocate_valid = 0; m->allocate_addr = 0; m->allocate_sector = 0; m->allocate_word_idx = 0;
  m->allocate_rw = 0; m->allocate_is_amo = 0; m->allocate_refetch = 0; m->allocate_data = 0;
  m->finalize_valid = 0; m->finalize_is_release = 0; m->finalize_is_pending = 0;
  m->finalize_previd = 0; m->finalize_id = 0;
  m->deq_req_uuid = 0; m->alc_req_uuid = 0; m->fin_req_uuid = 0;
}

static void cycle(VerilatedContext* ctx, VVX_cache_mshr* m) {
  m->clk = 0; m->eval(); ctx->timeInc(1);
  m->clk = 1; m->eval(); ctx->timeInc(1);
}

static void run_seed(int seed) {
  auto ctx = std::make_unique<VerilatedContext>();
  ctx->randReset(2);
  ctx->randSeed(seed);
  ctx->fatalOnError(false);
  auto m = std::make_unique<VVX_cache_mshr>(ctx.get());
  clear_inputs(m.get());
  m->reset = 1;
  for (int i = 0; i < 4; ++i) cycle(ctx.get(), m.get());
  m->reset = 0;

  // 1. idle out of reset
  for (int i = 0; i < 8; ++i) {
    cycle(ctx.get(), m.get());
    CHECK(m->dequeue_id < MSHR_SIZE, "seed %d cycle %d: dequeue_id=%u", seed, i, (unsigned)m->dequeue_id);
    CHECK(!m->dequeue_valid, "seed %d cycle %d: dequeue_valid without a fill", seed, i);
  }
  CHECK(!ctx->gotError(), "seed %d: assertion fired out of reset", seed);
  CHECK(m->empty, "seed %d: not empty out of reset", seed);

  // 2. allocate one miss, fill it, dequeue it
  const uint32_t addr = 0x123u + seed;
  m->allocate_valid = 1; m->allocate_addr = addr; m->allocate_data = 0xBEEF;
  m->eval();
  CHECK(m->allocate_ready, "seed %d: allocate not ready", seed);
  const unsigned id = m->allocate_id;
  CHECK(id < MSHR_SIZE, "seed %d: allocate_id=%u", seed, id);
  cycle(ctx.get(), m.get());
  m->allocate_valid = 0;
  cycle(ctx.get(), m.get());
  m->fill_valid = 1; m->fill_id = id;
  cycle(ctx.get(), m.get());
  m->fill_valid = 0;
  m->eval();
  CHECK(m->dequeue_valid, "seed %d: no dequeue after the fill", seed);
  CHECK(m->dequeue_id == id, "seed %d: dequeue_id=%u, filled %u", seed, (unsigned)m->dequeue_id, id);
  CHECK(m->dequeue_addr == addr, "seed %d: dequeue_addr=0x%x, allocated 0x%x", seed, (unsigned)m->dequeue_addr, addr);
  CHECK(m->dequeue_data == 0xBEEF, "seed %d: dequeue_data=0x%x", seed, (unsigned)m->dequeue_data);
  m->dequeue_ready = 1;
  cycle(ctx.get(), m.get());
  m->dequeue_ready = 0;
  cycle(ctx.get(), m.get());
  CHECK(!m->dequeue_valid, "seed %d: dequeue still valid after the last entry", seed);
  CHECK(!ctx->gotError(), "seed %d: assertion fired", seed);
  m->final();
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  for (int seed = 1; seed <= NUM_SEEDS; ++seed)
    run_seed(seed);
  if (failures) {
    printf("MSHR_SIZE=%d: %d check(s) FAILED\n", MSHR_SIZE, failures);
    return 1;
  }
  printf("MSHR_SIZE=%d: PASSED (%d seeds)\n", MSHR_SIZE, NUM_SEEDS);
  return 0;
}
