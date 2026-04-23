#pragma once
#include <cstdint>
#include "debug_module.h"

enum jtag_state_t {
    TEST_LOGIC_RESET,
    RUN_TEST_IDLE,
    SELECT_DR_SCAN,
    CAPTURE_DR,
    SHIFT_DR,
    EXIT1_DR,
    PAUSE_DR,
    EXIT2_DR,
    UPDATE_DR,
    SELECT_IR_SCAN,
    CAPTURE_IR,
    SHIFT_IR,
    EXIT1_IR,
    PAUSE_IR,
    EXIT2_IR,
    UPDATE_IR
};

class jtag_dtm_t {
public:
    // Constructor: Initializes the JTAG Debug Transport Module (DTM) with a reference to the Debug Module.
    // Use case: Creates a DTM instance that bridges JTAG protocol to RISC-V debug module operations.
    jtag_dtm_t(DebugModule* dm);

    // Resets the DTM to its initial state (TEST_LOGIC_RESET).
    // Use case: Called when JTAG reset is detected or when the debugger needs to reinitialize the DTM.
    void reset();
    
    // Updates JTAG pin states and advances the TAP state machine based on TCK/TMS/TDI transitions.
    // Use case: Called for each JTAG clock cycle to simulate the TAP controller state machine.
    void set_pins(bool tck, bool tms, bool tdi);
    
    // Returns the current TDO (Test Data Out) pin value.
    // Use case: Used by the remote bitbang protocol to read data being shifted out of the DTM.
    bool tdo() const { return _tdo; }
    
    // Returns the current JTAG TAP state machine state.
    // Use case: Used to check if the DTM is in a specific state (e.g., RUN_TEST_IDLE) for protocol handling.
    jtag_state_t state() const { return _state; }
    
    // Forwards run_test_idle() call to the debug module.
    // Use case: Called periodically when JTAG is in Run-Test-Idle state to allow debug module to process state updates.
    void run_test_idle() { dm->run_test_idle(); }

private:
    DebugModule* dm;

    bool _tck, _tms, _tdi, _tdo;
    jtag_state_t _state;
    uint32_t ir;
    uint64_t dr;
    unsigned dr_length;
    const unsigned abits;
    bool busy_stuck;
    uint64_t dmi;

    static constexpr uint32_t IR_IDCODE     = 0x01;
    static constexpr uint32_t IR_DTMCONTROL = 0x10;
    static constexpr uint32_t IR_DBUS       = 0x11;
    static constexpr uint32_t IR_BYPASS     = 0x1F;

    // Captures data from the selected register into the DR shift register based on current IR instruction.
    // Use case: Called during CAPTURE_DR state to prepare data for shifting out via TDO.
    void capture_dr();
    
    // Updates the selected register with data from the DR shift register after shifting is complete.
    // Use case: Called during UPDATE_DR state to execute DMI read/write operations based on shifted data.
    void update_dr();
};
