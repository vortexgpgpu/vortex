// The metasimulator's main loop, as a function the runtime calls on its
// simulator thread rather than as a process entry point.
//
// This is Verilator's --main loop for a --timing model. The emul top's initial
// block calls simulator_entry() through the DPI, which reaches entry() and
// spawns the driver thread, exactly as the standalone metasim does. The loop
// ends when the top $finishes, which the DPI tick triggers once the driver has
// returned -- that is, once the runtime has shut the simulation down.

#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "verilated.h"
#include "Vemul.h"

int firesim_emul_run(int argc, char **argv) {
  // VORTEX_FIRESIM_EMUL_ARGS appends Verilator runtime arguments, e.g.
  // "+verilator+rand+reset+2 +verilator+seed+7" to start every register that
  // has no reset from a random value instead of zero -- the closest a 2-state
  // simulator gets to a 4-state one's X on an uninitialised flop.
  std::vector<std::string> extra;
  if (const char *env = getenv("VORTEX_FIRESIM_EMUL_ARGS")) {
    std::istringstream in(env);
    for (std::string tok; in >> tok;) {
      extra.push_back(tok);
    }
  }
  std::vector<const char *> args(argv, argv + argc);
  for (const auto &tok : extra) {
    args.push_back(tok.c_str());
  }

  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
  // Required before time 0 for an EMUL_DEBUG build's $dumpvars to arm; a
  // no-op in a build without tracing. (Verilator's own --main emits it.)
  contextp->traceEverOn(true);
  contextp->commandArgs(static_cast<int>(args.size()), args.data());
  if (!extra.empty()) {
    fprintf(stderr, "[emul] extra args:%s -> randReset=%d seed=%d\n",
            [&] { std::string s; for (auto &t : extra) s += " " + t; return s; }().c_str(),
            contextp->randReset(), contextp->randSeed());
  }
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
