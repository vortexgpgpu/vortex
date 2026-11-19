#include "vl_simulator.h"
#include "VVX_mmu_park_top.h"
#include <algorithm>
#include <cstdio>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  vl_simulator<VVX_mmu_park_top> sim;
  sim->walk_enable = 0;
  auto tick = sim.reset(0);
  bool passed = true;
  // Saturate the two miss slots, then let walks complete one at a time.
  for (unsigned cycle = 0; cycle < 256; ++cycle) {
    sim->walk_enable = cycle >= 16 && cycle % 8 == 0;
    tick = sim.step(tick, 2);
    const auto& counts = sim->accepted;
    auto bounds = std::minmax_element(counts.data(), counts.data() + 4);
    if (passed && *bounds.second - *bounds.first > 1) {
      std::printf("unbalanced grants at cycle %u: %u %u %u %u\n",
                  cycle, counts[0], counts[1], counts[2], counts[3]);
      passed = false;
    }
  }
  for (unsigned p = 0; p < 4; ++p) {
    std::printf("port %u accepted %u misses\n", p, sim->accepted[p]);
    passed &= sim->accepted[p] >= 2;
  }
  return passed ? 0 : 1;
}
