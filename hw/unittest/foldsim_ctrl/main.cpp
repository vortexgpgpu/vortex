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

// foldsim's control endpoint (hw/rtl/afu/foldsim/foldsim_ctrl.sv) as the
// runtime sees it, through sw/common/foldsim_regs.h over AXI4-Lite: the id and
// version, the DRAM timings read back, every memory port's counters at its
// place in the stats window, every class's finishes, zero past the end of each
// window, and the idle-slot rate: kept with BUBBLES, 0 without.

#include <cstdint>
#include <cstdio>
#include <memory>

#include <foldsim_regs.h>

#include "Vfoldsim_ctrl.h"
#include "verilated.h"

namespace {

std::unique_ptr<Vfoldsim_ctrl> dut;
int failures = 0;

void tick() {
  dut->clk = 0;
  dut->eval();
  dut->clk = 1;
  dut->eval();
}

void check(bool ok, const char* what, uint32_t addr, uint32_t got, uint32_t want) {
  if (!ok) {
    printf("  %s at 0x%03x: 0x%08x, want 0x%08x\n", what, addr, got, want);
    ++failures;
  }
}

void reg_write(uint32_t addr, uint32_t data) {
  dut->s_awvalid = 1;
  dut->s_awaddr = addr;
  dut->s_wvalid = 1;
  dut->s_wdata = data;
  dut->s_bready = 1;
  for (int i = 0; i < 16 && (dut->s_awvalid || dut->s_wvalid); ++i) {
    dut->eval();
    const bool aw = dut->s_awready, w = dut->s_wready;
    tick();
    if (aw) dut->s_awvalid = 0;
    if (w) dut->s_wvalid = 0;
  }
  for (int i = 0; i < 16 && !dut->s_bvalid; ++i)
    tick();
  tick();
  dut->s_bready = 0;
}

uint32_t reg_read(uint32_t addr) {
  dut->s_arvalid = 1;
  dut->s_araddr = addr;
  for (int i = 0; i < 16 && dut->s_arvalid; ++i) {
    dut->eval();
    const bool ar = dut->s_arready;
    tick();
    if (ar) dut->s_arvalid = 0;
  }
  for (int i = 0; i < 16 && !dut->s_rvalid; ++i)
    tick();
  const uint32_t data = dut->s_rdata;
  dut->s_rready = 1;
  tick();
  dut->s_rready = 0;
  return data;
}

// Port p's 32-bit word k of its 512 counter bits.
uint32_t stat_word(int p, int k) { return 0xA0000000u | (uint32_t(p) << 16) | uint32_t(k); }

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = std::make_unique<Vfoldsim_ctrl>();

  for (int p = 0; p < MEM_PORTS; ++p)
    for (int k = 0; k < 16; ++k)
      dut->mem_stats[p * 16 + k] = stat_word(p, k);
  dut->reset = 1;
  for (int i = 0; i < 8; ++i)
    tick();
  dut->reset = 0;
  // Every class finishes on c + 1 host cycles, the last for longest.
  for (int i = 0; i < NUM_CLASSES; ++i) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
      const bool on = c >= i;
#if NUM_CLASSES > 64
      dut->profile_finish[c / 32] = (dut->profile_finish[c / 32] & ~(1u << (c % 32))) | (uint32_t(on) << (c % 32));
#else
      const uint64_t v = dut->profile_finish;
      dut->profile_finish = (v & ~(uint64_t(1) << c)) | (uint64_t(on) << c);
#endif
    }
    tick();
  }
#if NUM_CLASSES > 64
  for (int w = 0; w < (NUM_CLASSES + 31) / 32; ++w)
    dut->profile_finish[w] = 0;
#else
  dut->profile_finish = 0;
#endif
  tick();

  uint32_t got;
  got = reg_read(FOLDSIM_REG_ID);
  check(got == FOLDSIM_ID, "id", FOLDSIM_REG_ID, got, FOLDSIM_ID);
  got = reg_read(FOLDSIM_REG_VERSION);
  check(got == FOLDSIM_VERSION, "version", FOLDSIM_REG_VERSION, got, FOLDSIM_VERSION);
  got = reg_read(FOLDSIM_REG_NUM_CLASSES);
  check(got == NUM_CLASSES, "classes", FOLDSIM_REG_NUM_CLASSES, got, NUM_CLASSES);

  for (uint32_t i = 0; i < FOLDSIM_MEM_NUM_TIMINGS; ++i)
    reg_write(FOLDSIM_REG_MEM_TIMING(i), 0x100u + i);
  for (uint32_t i = 0; i < FOLDSIM_MEM_NUM_TIMINGS; ++i) {
    got = reg_read(FOLDSIM_REG_MEM_TIMING(i));
    check(got == 0x100u + i, "timing", FOLDSIM_REG_MEM_TIMING(i), got, 0x100u + i);
  }
  got = reg_read(FOLDSIM_REG_BUBBLE);
  check(got == 0, "idle slots at reset", FOLDSIM_REG_BUBBLE, got, 0);
  reg_write(FOLDSIM_REG_BUBBLE, 0x1C0);
  got = reg_read(FOLDSIM_REG_BUBBLE);
  check(got == (BUBBLES ? 0xC0u : 0u), "idle slots", FOLDSIM_REG_BUBBLE, got, BUBBLES ? 0xC0u : 0u);
  check(dut->bubble == (BUBBLES ? 0xC0 : 0), "idle-slot output", FOLDSIM_REG_BUBBLE, dut->bubble, BUBBLES ? 0xC0u : 0u);

  got = reg_read(FOLDSIM_REG_MEM_TIMING(FOLDSIM_MEM_NUM_TIMINGS));
  check(got == 0, "past the timings", FOLDSIM_REG_MEM_TIMING(FOLDSIM_MEM_NUM_TIMINGS), got, 0);

  for (int p = 0; p < MEM_PORTS; ++p) {
    for (uint32_t i = 0; i < FOLDSIM_MEM_NUM_STATS; ++i) {
      const uint32_t a = FOLDSIM_REG_MEM_STATS(p, i);
      got = reg_read(a);
      check(got == stat_word(p, 2 * i), "counter, low", a, got, stat_word(p, 2 * i));
      got = reg_read(a + 4);
      check(got == stat_word(p, 2 * i + 1), "counter, high", a + 4, got, stat_word(p, 2 * i + 1));
    }
  }
  if (MEM_PORTS < 32) {
    const uint32_t a = FOLDSIM_REG_MEM_STATS(MEM_PORTS, 0);
    got = reg_read(a);
    check(got == 0, "past the ports", a, got, 0);
  }

  for (int c = 0; c < NUM_CLASSES; ++c) {
    got = reg_read(FOLDSIM_REG_FINISH(c));
    check(got == uint32_t(c + 1), "finishes", FOLDSIM_REG_FINISH(c), got, c + 1);
    got = reg_read(FOLDSIM_REG_FINISH(c) + 4);
    check(got == 0, "finishes, high", FOLDSIM_REG_FINISH(c) + 4, got, 0);
  }
  // At 96 classes the window is full: the timings follow.
  if (NUM_CLASSES < 96) {
    got = reg_read(FOLDSIM_REG_FINISH(NUM_CLASSES));
    check(got == 0, "past the classes", FOLDSIM_REG_FINISH(NUM_CLASSES), got, 0);
  }

  dut->final();
  dut.reset();
  if (failures != 0) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
