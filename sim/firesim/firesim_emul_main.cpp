// The metasimulator's main loop, as a function the runtime calls on its
// simulator thread rather than as a process entry point.
//
// This is Verilator's --main loop for a --timing model. The emul top's initial
// block calls simulator_entry() through the DPI, which reaches entry() and
// spawns the driver thread, exactly as the standalone metasim does. The loop
// ends when the top $finishes, which the DPI tick triggers once the driver has
// returned -- that is, once the runtime has shut the simulation down.

#include <memory>

#include "verilated.h"
#include "Vemul.h"

int firesim_emul_run(int argc, char **argv) {
  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
  contextp->commandArgs(argc, argv);
  const std::unique_ptr<Vemul> topp{new Vemul{contextp.get()}};

  while (!contextp->gotFinish()) {
    topp->eval();
    if (!topp->eventsPending()) {
      break;
    }
    contextp->time(topp->nextTimeSlot());
  }

  topp->final();
  return 0;
}
