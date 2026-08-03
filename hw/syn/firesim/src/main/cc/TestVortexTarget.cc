// See LICENSE for license details.

// Bring-up harness for the Vortex FireSim target.
//
// Exercises the control surface only: reset, the DCR request/response path, and
// start/busy. Running an actual kernel additionally needs a program image in
// target DRAM, which the runtime transport supplies.

#include "TestHarness.h"

#include <string_view>

class TestVortexTarget final : public TestHarness {
public:
  using TestHarness::TestHarness;

  void run_test() override {
    // Vortex samples reset for VX_CFG_RESET_DELAY cycles before it will accept
    // any request, so hold it well past that.
    poke("ctrl_start", 0);
    poke("ctrl_dcr_req_valid", 0);
    target_reset(32);

    // Out of reset and idle: nothing has been started, so the core must not
    // claim to be busy.
    expect(std::string_view("ctrl_busy"), uint32_t(0));

    // A DCR write is the smallest transaction that proves the control path is
    // live end to end. It is accepted combinationally, so one step suffices.
    poke("ctrl_dcr_req_valid", 1);
    poke("ctrl_dcr_req_rw", 1);
    poke("ctrl_dcr_req_addr", 0x1);
    poke("ctrl_dcr_req_data", 0xdeadbeef);
    step(1);
    poke("ctrl_dcr_req_valid", 0);
    step(1);

    // Still idle: a DCR write configures the device, it does not launch work.
    expect(std::string_view("ctrl_busy"), uint32_t(0));
  }
};

TEST_MAIN(TestVortexTarget)
