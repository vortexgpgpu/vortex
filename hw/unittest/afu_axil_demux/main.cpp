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

// Regression for the AFU AXI-Lite demux.
//
// The bug this exists to prevent: a write to the legacy window (addr[12]=0)
// whose predecessor went to the CP window (addr[12]=1) had its AW routed to
// AFU_ctrl and its W routed to the CP regfile. AFU_ctrl then waited forever
// for a W beat the CP had already swallowed, so no BRESP was ever produced.
// On a V80 that killed every subsequent MMIO access with a PCIe completion
// timeout, and the ap_reset the write was asking for never even fired.
//
// The master modelled here presents AW and W in the same cycle, which AXI4
// explicitly permits and which Xilinx AXI-Lite masters do.

#include <cstdio>
#include <cstdint>
#include "VVX_afu_axil_demux_top.h"
#include "verilated.h"

namespace {

VVX_afu_axil_demux_top* dut = nullptr;
bool ap_reset_seen = false;
int  failures = 0;

constexpr uint16_t CP_BASE       = 0x1000;
constexpr uint16_t CP_REG_CTRL   = 0x000;
constexpr uint16_t CP_REG_STATUS = 0x004;
constexpr uint16_t CP_Q_CONTROL  = 0x11C;
constexpr uint16_t MMIO_CTL_ADDR = 0x0000;
constexpr uint32_t CTL_AP_RESET  = 0x10;
constexpr uint32_t CTL_AP_IDLE   = 0x04;

constexpr int TIMEOUT = 256;

void tick() {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
  if (dut->ap_reset_out) ap_reset_seen = true;
}

void do_reset() {
  dut->reset = 1;
  dut->s_awvalid = 0;
  dut->s_wvalid = 0;
  dut->s_arvalid = 0;
  dut->s_rready = 1;
  dut->s_bready = 1;
  dut->s_wstrb = 0xF;
  for (int i = 0; i < 8; ++i) tick();
  dut->reset = 0;
  for (int i = 0; i < 4; ++i) tick();
  ap_reset_seen = false;
}

// AW and W presented together. Returns true if BRESP came back.
bool axil_write(uint16_t addr, uint32_t data) {
  dut->s_awaddr  = addr;
  dut->s_wdata   = data;
  dut->s_awvalid = 1;
  dut->s_wvalid  = 1;
  dut->s_bready  = 1;
  for (int i = 0; i < TIMEOUT; ++i) {
    dut->eval();
    bool aw_fire = dut->s_awvalid && dut->s_awready;
    bool w_fire  = dut->s_wvalid  && dut->s_wready;
    bool b_fire  = dut->s_bvalid  && dut->s_bready;
    tick();
    if (aw_fire) dut->s_awvalid = 0;
    if (w_fire)  dut->s_wvalid  = 0;
    if (b_fire)  return true;
  }
  dut->s_awvalid = 0;
  dut->s_wvalid  = 0;
  return false;
}

// W asserted first, AW only after the master has been waiting a few cycles.
bool axil_write_w_first(uint16_t addr, uint32_t data) {
  dut->s_wdata  = data;
  dut->s_wvalid = 1;
  dut->s_bready = 1;
  for (int i = 0; i < 4; ++i) tick();
  dut->s_awaddr  = addr;
  dut->s_awvalid = 1;
  for (int i = 0; i < TIMEOUT; ++i) {
    dut->eval();
    bool aw_fire = dut->s_awvalid && dut->s_awready;
    bool w_fire  = dut->s_wvalid  && dut->s_wready;
    bool b_fire  = dut->s_bvalid  && dut->s_bready;
    tick();
    if (aw_fire) dut->s_awvalid = 0;
    if (w_fire)  dut->s_wvalid  = 0;
    if (b_fire)  return true;
  }
  dut->s_awvalid = 0;
  dut->s_wvalid  = 0;
  return false;
}

// AW first, W only after AW has been accepted.
bool axil_write_aw_first(uint16_t addr, uint32_t data) {
  dut->s_awaddr  = addr;
  dut->s_awvalid = 1;
  dut->s_wvalid  = 0;
  dut->s_bready  = 1;
  bool aw = false;
  for (int i = 0; i < TIMEOUT && !aw; ++i) {
    dut->eval();
    aw = dut->s_awvalid && dut->s_awready;
    tick();
  }
  dut->s_awvalid = 0;
  if (!aw) return false;
  dut->s_wdata  = data;
  dut->s_wvalid = 1;
  for (int i = 0; i < TIMEOUT; ++i) {
    dut->eval();
    bool w_fire = dut->s_wvalid && dut->s_wready;
    bool b_fire = dut->s_bvalid && dut->s_bready;
    tick();
    if (w_fire) dut->s_wvalid = 0;
    if (b_fire) return true;
  }
  dut->s_wvalid = 0;
  return false;
}

bool axil_read(uint16_t addr, uint32_t* out) {
  dut->s_araddr  = addr;
  dut->s_arvalid = 1;
  dut->s_rready  = 1;
  for (int i = 0; i < TIMEOUT; ++i) {
    dut->eval();
    bool ar_fire = dut->s_arvalid && dut->s_arready;
    bool r_fire  = dut->s_rvalid  && dut->s_rready;
    uint32_t rd  = dut->s_rdata;
    tick();
    if (ar_fire) dut->s_arvalid = 0;
    if (r_fire) { *out = rd; return true; }
  }
  dut->s_arvalid = 0;
  return false;
}

void check(bool ok, const char* what) {
  printf("  %-58s %s\n", what, ok ? "PASS" : "*** FAIL ***");
  if (!ok) ++failures;
}

} // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new VVX_afu_axil_demux_top;

  // ------------------------------------------------------------------
  printf("1. legacy write is the first write after reset (XRT ordering)\n");
  do_reset();
  check(axil_write(MMIO_CTL_ADDR, CTL_AP_RESET), "write 0x0000 returns BRESP");
  for (int i = 0; i < 8; ++i) tick();
  check(ap_reset_seen, "ap_reset pulsed");

  // ------------------------------------------------------------------
  printf("2. legacy write follows CP-window writes (AVED ordering)\n");
  printf("   -- this is the sequence that took the V80 off the bus\n");
  do_reset();
  check(axil_write(CP_BASE + CP_Q_CONTROL, 0), "write 0x111c (CP) returns BRESP");
  check(axil_write(CP_BASE + CP_REG_CTRL, 0), "write 0x1000 (CP) returns BRESP");
  uint32_t st = 0;
  check(axil_read(CP_BASE + CP_REG_STATUS, &st), "read 0x1004 (CP) returns RRESP");
  ap_reset_seen = false;
  check(axil_write(MMIO_CTL_ADDR, CTL_AP_RESET), "write 0x0000 returns BRESP");
  for (int i = 0; i < 8; ++i) tick();
  check(ap_reset_seen, "ap_reset pulsed");
  check(!dut->dbg_cp_wr_data_buf_valid, "no orphaned W beat left in the CP slave");

  // ------------------------------------------------------------------
  printf("3. the reverse direction: CP write follows a legacy write\n");
  do_reset();
  check(axil_write(MMIO_CTL_ADDR, 0), "write 0x0000 returns BRESP");
  check(axil_write(CP_BASE + CP_REG_CTRL, 0), "write 0x1000 (CP) returns BRESP");
  check(!dut->dbg_cp_wr_addr_buf_valid, "CP write committed, address buffer drained");

  // ------------------------------------------------------------------
  printf("4. W presented before AW (AXI4 permits this)\n");
  do_reset();
  check(axil_write(CP_BASE + CP_REG_CTRL, 0), "write 0x1000 (CP) returns BRESP");
  check(axil_write_w_first(MMIO_CTL_ADDR, CTL_AP_RESET), "W-then-AW to 0x0000 returns BRESP");

  // ------------------------------------------------------------------
  printf("5. W presented after AW\n");
  do_reset();
  check(axil_write(CP_BASE + CP_REG_CTRL, 0), "write 0x1000 (CP) returns BRESP");
  check(axil_write_aw_first(MMIO_CTL_ADDR, CTL_AP_RESET), "AW-then-W to 0x0000 returns BRESP");

  // ------------------------------------------------------------------
  printf("6. alternating windows, 16 writes\n");
  do_reset();
  bool all_ok = true;
  for (int i = 0; i < 16; ++i) {
    uint16_t addr = (i & 1) ? uint16_t(CP_BASE + CP_REG_CTRL) : MMIO_CTL_ADDR;
    if (!axil_write(addr, 0)) { all_ok = false; break; }
  }
  check(all_ok, "every write returns BRESP");

  // ------------------------------------------------------------------
  printf("7. alternating reads route to the right slave\n");
  do_reset();
  uint32_t v_cp = 0, v_lg = 0;
  bool reads_ok = axil_read(CP_BASE + CP_REG_STATUS, &v_cp)
               && axil_read(MMIO_CTL_ADDR, &v_lg);
  check(reads_ok, "both reads return RRESP");
  check(v_cp == 0xC0DE0000u, "CP-window read returns the CP slave's data");
  check((v_lg & CTL_AP_IDLE) != 0, "legacy read returns ap_idle from AFU_ctrl");

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
         failures, failures == 1 ? "" : "s");
  delete dut;
  return failures ? 1 : 0;
}
