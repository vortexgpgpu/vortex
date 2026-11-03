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

// VX_core out of reset with no kernel launched: it must stay off the D-cache
// port. Registers and undriven nets power up random (--x-initial unique), so
// the test sweeps power-on seeds; any state that is neither reset nor driven
// and that reaches a request valid shows up as a request for some of them.

#include "VVX_core_top.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <memory>

#define NUM_SEEDS   32
#define RESET_DELAY 8
#define IDLE_CYCLES 1000

bool sim_trace_enabled() {
  return true;
}

double sc_time_stamp() { return 0; }

static void cycle(VerilatedContext* ctx, VVX_core_top* m) {
  m->clk = 0; m->eval(); ctx->timeInc(1);
  m->clk = 1; m->eval(); ctx->timeInc(1);
}

// Returns true when the core stayed idle for this power-on seed.
static bool run_seed(int seed) {
  auto ctx = std::make_unique<VerilatedContext>();
  ctx->randReset(2);
  ctx->randSeed(seed);
  ctx->fatalOnError(false);
  ctx->assertOn(false);
  auto m = std::make_unique<VVX_core_top>(ctx.get());

  m->dcache_req_ready = 0;
  m->dcache_rsp_valid = 0;
  m->icache_req_ready = 0;
  m->icache_rsp_valid = 0;
  m->gbar_req_ready   = 0;
  m->gbar_rsp_valid   = 0;
  m->dcr_req_valid    = 0;
  m->dcr_req_rw       = 0;
  m->dcr_req_addr     = 0;
  m->dcr_req_data     = 0;

  m->reset = 1;
  for (int i = 0; i < RESET_DELAY; ++i) cycle(ctx.get(), m.get());
  m->reset = 0;
  ctx->assertOn(true);

  bool ok = true;
  for (int i = 0; i < IDLE_CYCLES && !ctx->gotError(); ++i) {
    cycle(ctx.get(), m.get());
    if ((uint64_t)m->dcache_req_valid != 0) {
      printf("seed %d cycle %d: D-cache request from an idle core, valid=0x%llx\n",
             seed, i, (unsigned long long)m->dcache_req_valid);
      ok = false;
      break;
    }
  }
  if (ctx->gotError()) {
    printf("seed %d: assertion fired in an idle core\n", seed);
    ok = false;
  }
  m->final();
  return ok;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  int failed = 0;
  for (int seed = 1; seed <= NUM_SEEDS; ++seed) {
    if (!run_seed(seed))
      ++failed;
  }
  if (failed) {
    printf("FAILED: %d of %d power-on seeds\n", failed, NUM_SEEDS);
    return 1;
  }
  printf("PASSED (%d power-on seeds, %d idle cycles each)\n", NUM_SEEDS, IDLE_CYCLES);
  return 0;
}
