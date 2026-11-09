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

// The property under test: the AFU never asserts its internal reset while an
// AXI master still owes the interconnect a response, and when a master will
// not drain it refuses and says so rather than resetting anyway.

#include <cstdio>
#include <cstdint>
#include "VVX_afu_reset_seq_top.h"
#include "verilated.h"

namespace {

VVX_afu_reset_seq_top* dut = nullptr;
int  failures = 0;
bool saw_assert = false;

constexpr int TIMEOUT_CYCLES = 64;   // must match the top's parameter

void tick() {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
  if (dut->rst_assert) saw_assert = true;
}

void clear_fires() {
  dut->aw_fire = 0; dut->w_fire = 0; dut->w_fire_last = 0; dut->b_fire = 0;
  dut->ar_fire = 0; dut->r_fire_last = 0;
  dut->gate_in_valid = 0; dut->gate_out_ready = 0;
  dut->wg_in_awvalid = 0; dut->wg_out_awready = 0;
  dut->wg_in_wvalid = 0; dut->wg_in_wlast = 0; dut->wg_out_wready = 0;
}

void do_reset() {
  dut->reset = 1;
  dut->ap_reset_req = 0;
  clear_fires();
  for (int i = 0; i < 8; ++i) tick();
  dut->reset = 0;
  for (int i = 0; i < 8; ++i) tick();
  saw_assert = false;
}

void pulse_request() {
  dut->ap_reset_req = 1;
  tick();
  dut->ap_reset_req = 0;
}

// Run n cycles with a one-cycle pulse on the given member at cycle `at`.
void run(int n) { for (int i = 0; i < n; ++i) tick(); }

void check(bool ok, const char* what) {
  printf("  %-58s %s\n", what, ok ? "PASS" : "*** FAIL ***");
  if (!ok) ++failures;
}

} // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new VVX_afu_reset_seq_top;

  // ------------------------------------------------------------------
  printf("1. idle device: reset is asserted promptly\n");
  do_reset();
  check(dut->masters_idle, "masters report idle before the request");
  pulse_request();
  run(8);
  check(saw_assert, "rst_assert fired");
  check(!dut->timeout_error, "no timeout error");
  run(16);
  check(!dut->busy, "sequence finished (busy low)");
  check(!dut->stop_req, "request gate released");

  // ------------------------------------------------------------------
  printf("2. outstanding read: reset is withheld until it completes\n");
  do_reset();
  clear_fires();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;   // one read launched
  tick();
  check(!dut->masters_idle, "masters not idle with a read outstanding");

  pulse_request();
  run(16);
  check(dut->stop_req, "stop_req asserted while quiescing");
  check(!saw_assert, "rst_assert withheld while the read is outstanding");
  check(dut->busy, "still busy");

  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;   // response returns
  run(8);
  check(dut->masters_idle, "masters idle once the response returns");
  check(saw_assert, "rst_assert fired after the drain");
  check(!dut->timeout_error, "no timeout error");

  // ------------------------------------------------------------------
  printf("3. outstanding write burst: AW needs both W-last and B\n");
  do_reset();
  dut->aw_fire = 1; tick(); dut->aw_fire = 0;
  tick();
  check(!dut->masters_idle, "not idle after AW");
  dut->w_fire_last = 1; tick(); dut->w_fire_last = 0;
  tick();
  check(!dut->masters_idle, "still not idle after W-last (B outstanding)");
  pulse_request();
  run(16);
  check(!saw_assert, "rst_assert withheld while BRESP is outstanding");
  dut->b_fire = 1; tick(); dut->b_fire = 0;
  run(8);
  check(saw_assert, "rst_assert fired once BRESP returned");

  // ------------------------------------------------------------------
  printf("4. a master that never drains: refuse, do not reset\n");
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;   // never answered
  tick();
  pulse_request();
  run(TIMEOUT_CYCLES + 32);
  check(!saw_assert, "rst_assert never fired");
  check(dut->timeout_error, "timeout_error raised");
  check(!dut->busy, "sequencer returned to idle");
  check(!dut->stop_req, "request gate released so the AFU keeps running");

  // ------------------------------------------------------------------
  printf("5. a later successful request clears the sticky error\n");
  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;   // the read finally lands
  run(4);
  check(dut->masters_idle, "masters idle again");
  saw_assert = false;
  pulse_request();
  run(16);
  check(saw_assert, "rst_assert fired");
  check(!dut->timeout_error, "timeout_error cleared");

  // ------------------------------------------------------------------
  printf("6. platform reset always works, sequencer or not\n");
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;   // leave a read outstanding
  run(4);
  dut->reset = 1;
  run(4);
  check(dut->vx_reset, "vx_reset asserted by the platform reset");
  dut->reset = 0;
  run(8);
  check(!dut->busy, "sequencer idle after a platform reset");
  check(dut->masters_idle, "drain counters cleared by the platform reset");

  // ------------------------------------------------------------------
  // AXI4 §A3.2.1: a master that has asserted VALID must hold it until the
  // edge where VALID and READY are both high. Quiescing must therefore take
  // effect between transactions, never inside one. A gate written as the
  // obvious `out_valid = in_valid && !stop_req` fails this: it withdraws an
  // offer the shell has already latched, and the port never recovers.
  printf("7. quiescing never withdraws an offer already made\n");
  do_reset();
  // An earlier read keeps the drain busy, so QUIESCE lasts long enough to
  // observe the gate rather than completing in two cycles.
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;

  dut->gate_in_valid = 1;      // the AFU offers a request...
  dut->gate_out_ready = 0;     // ...and the shell is not ready for it yet
  tick();
  check(dut->gate_out_valid, "offer reaches the shell before quiescing");

  pulse_request();             // stop_req rises mid-offer
  bool held = true;
  for (int i = 0; i < 16; ++i) {
    tick();
    if (!dut->gate_out_valid) held = false;
  }
  check(dut->stop_req, "sequencer is quiescing");
  check(held, "VALID held asserted across stop_req (AXI4 A3.2.1)");
  check(!saw_assert, "reset withheld: the reads are still outstanding");

  // The shell finally accepts. Only now may the gate close, and the next
  // request must be held off.
  dut->gate_out_ready = 1;
  dut->ar_fire = 1;            // the accepted AR is now outstanding too
  tick();
  dut->ar_fire = 0;
  dut->gate_out_ready = 0;
  tick();
  check(!dut->gate_out_valid, "the next request is withheld once the gate closes");
  check(!dut->gate_in_ready, "and the AFU is not told it was accepted");

  // Both reads return; only then may the reset proceed.
  dut->r_fire_last = 1; tick();
  check(!dut->masters_idle, "one read still outstanding");
  tick(); dut->r_fire_last = 0;
  run(16);
  check(dut->masters_idle, "both reads returned");
  check(saw_assert, "reset proceeds once the masters drained");
  dut->gate_in_valid = 0;
  run(8);
  check(!dut->stop_req, "gate reopens after the sequence");

  // ------------------------------------------------------------------
  // AXI4 lets write data precede its address. Quiescing must neither strand
  // data whose AW it then withholds nor withhold the AW of data the shell
  // already took: the shell must see whole writes only.
  printf("8. quiescing never splits a write\n");
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;   // keep QUIESCE long enough to observe

  // (a) data first: a single-beat W is taken before its AW is presented.
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1; dut->wg_out_wready = 1;
  tick();
  dut->wg_in_wvalid = 0; dut->wg_in_wlast = 0;
  pulse_request();
  run(4);
  check(dut->stop_req, "sequencer is quiescing");
  dut->wg_in_awvalid = 1; dut->wg_out_awready = 0;
  dut->eval();
  check(dut->wg_out_awvalid, "the AW owed to accepted data still goes out");
  tick();
  dut->wg_out_awready = 1;
  dut->eval();
  check(dut->wg_out_awvalid && dut->wg_in_awready, "and is accepted");
  tick();
  dut->wg_out_awready = 0;
  dut->eval();
  check(!dut->wg_out_awvalid, "the next AW is withheld");
  dut->wg_out_awready = 1;
  dut->eval();
  check(!dut->wg_in_awready, "and not reported accepted");
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1;
  dut->eval();
  check(!dut->wg_out_wvalid, "its data is withheld too");
  check(!dut->wg_in_wready, "and not reported accepted");
  run(4);
  check(!dut->wg_out_awvalid && !dut->wg_out_wvalid, "and both stay withheld");
  clear_fires();
  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;
  run(16);
  check(saw_assert, "reset proceeds");
  run(8);

  // (b) address first, then a two-beat burst: the data owed to an accepted AW
  // goes out in full, and nothing after it.
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;
  dut->wg_in_awvalid = 1; dut->wg_out_awready = 1;
  tick();
  dut->wg_in_awvalid = 0;
  pulse_request();
  run(4);
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 0; dut->wg_out_wready = 1;
  dut->eval();
  check(dut->wg_out_wvalid, "the first beat owed to the AW goes out");
  tick();
  dut->wg_in_wlast = 1;
  dut->eval();
  check(dut->wg_out_wvalid, "the burst runs to its last beat");
  tick();
  dut->wg_in_wlast = 0;
  dut->eval();
  check(!dut->wg_out_wvalid, "data with no accepted AW is withheld");
  dut->wg_in_awvalid = 1;
  dut->eval();
  check(!dut->wg_out_awvalid, "a new AW is withheld");
  clear_fires();
  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;
  run(16);
  check(saw_assert, "reset proceeds");
  run(8);
  check(!dut->stop_req, "gate reopens after the sequence");
  dut->wg_in_awvalid = 1; dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1;
  dut->wg_out_awready = 1; dut->wg_out_wready = 1;
  dut->eval();
  check(dut->wg_out_awvalid && dut->wg_out_wvalid, "writes flow again once reopened");
  clear_fires();

  // (c) a slave may hold AWREADY until it has seen the data: an address
  // already offered when quiescing begins obliges its data.
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;
  dut->wg_in_awvalid = 1; dut->wg_out_awready = 0;
  tick();
  pulse_request();
  run(4);
  check(dut->stop_req, "sequencer is quiescing");
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1; dut->wg_out_wready = 1;
  dut->eval();
  check(dut->wg_out_wvalid, "data owed to an offered AW goes out");
  tick();
  dut->wg_in_wvalid = 0; dut->wg_in_wlast = 0;
  dut->wg_out_awready = 1;
  dut->eval();
  check(dut->wg_out_awvalid, "and the slave then takes the AW");
  tick();
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1;
  dut->eval();
  check(!dut->wg_out_wvalid && !dut->wg_out_awvalid, "nothing further goes out");
  clear_fires();
  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;
  run(16);
  check(saw_assert, "reset proceeds");
  run(8);

  // (d) a slave may hold WREADY until it has seen the address: data already
  // offered when quiescing begins obliges its AW.
  do_reset();
  dut->ar_fire = 1; tick(); dut->ar_fire = 0;
  dut->wg_in_wvalid = 1; dut->wg_in_wlast = 1; dut->wg_out_wready = 0;
  tick();
  pulse_request();
  run(4);
  dut->wg_in_awvalid = 1; dut->wg_out_awready = 1;
  dut->eval();
  check(dut->wg_out_awvalid, "the AW owed to offered data goes out");
  tick();
  dut->wg_in_awvalid = 0;
  dut->wg_out_wready = 1;
  dut->eval();
  check(dut->wg_out_wvalid, "and the slave then takes the data");
  tick();
  dut->wg_in_awvalid = 1;
  dut->eval();
  check(!dut->wg_out_wvalid && !dut->wg_out_awvalid, "nothing further goes out");
  clear_fires();
  dut->r_fire_last = 1; tick(); dut->r_fire_last = 0;
  run(16);
  check(saw_assert, "reset proceeds");
  run(8);

  // ------------------------------------------------------------------
  // The drain counters see last beats only. A burst whose first beats went
  // out ahead of its AW still owes the interconnect the rest of the burst
  // and the AW, so the port is not idle and reset must wait.
  printf("9. a write burst in progress keeps the port busy\n");
  do_reset();
  dut->w_fire = 1; tick(); dut->w_fire = 0;     // beat 1 of 2, no AW yet
  tick();
  check(!dut->masters_idle, "not idle with a burst under way");
  pulse_request();
  run(16);
  check(!saw_assert, "rst_assert withheld mid-burst");
  dut->w_fire = 1; dut->w_fire_last = 1; tick(); dut->w_fire = 0; dut->w_fire_last = 0;
  dut->aw_fire = 1; tick(); dut->aw_fire = 0;
  dut->b_fire = 1; tick(); dut->b_fire = 0;
  run(8);
  check(dut->masters_idle, "idle once the write completes");
  check(saw_assert, "reset proceeds");

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
         failures, failures == 1 ? "" : "s");
  delete dut;
  return failures ? 1 : 0;
}
