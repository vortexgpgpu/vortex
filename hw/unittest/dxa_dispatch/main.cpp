#include <cstdint>
#include <iostream>
#include <verilated.h>
#include <VX_config.h>
#include "VVX_dxa_dispatch_top.h"

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  VVX_dxa_dispatch_top dut;
  constexpr uint32_t workers = VX_CFG_NUM_DXA_CORES;
  constexpr uint32_t cores = VX_CFG_SOCKET_SIZE;
  static_assert(workers <= 16, "test masks fit in uint32_t");
  uint32_t checks = 0, failed = 0;
  auto check = [&](bool ok) { ++checks; failed += !ok; };
  auto tick = [&]() { dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); };
  dut.reset = 1;
  dut.valid = 0;
  dut.core_id = 0;
  dut.worker_ready = 0;
  tick();
  dut.reset = 0;
  for (uint32_t core = 0; core < VX_CFG_NUM_CORES; ++core) {
    const uint32_t worker = (core % cores) * workers / cores;
    dut.core_id = core;
    dut.valid = 1;
    for (uint32_t mask = 0; mask < (1u << workers); ++mask) {
      dut.worker_ready = mask;
      for (unsigned stall = 0; stall < 3; ++stall) {
        dut.eval();
        check(dut.worker_valid == (1u << worker));
        check(bool(dut.ready) == bool(mask & (1u << worker)));
        check(!dut.payload_error);
        tick();
      }
    }
    std::cout << "core=" << core << " worker=" << worker << '\n';
  }
  dut.valid = 0;
  dut.eval();
  check(dut.worker_valid == 0);
  std::cout << "checks=" << checks << " failed=" << failed << '\n';
  return failed ? 1 : 0;
}
