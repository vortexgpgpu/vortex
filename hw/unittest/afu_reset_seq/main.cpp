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
  dut->aw_fire = 0; dut->w_fire_last = 0; dut->b_fire = 0;
  dut->ar_fire = 0; dut->r_fire_last = 0;
  dut->gate_in_valid = 0; dut->gate_out_ready = 0;
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

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
         failures, failures == 1 ? "" : "s");
  delete dut;
  return failures ? 1 : 0;
}
