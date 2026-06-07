#include "jtag_dtm.h"
#include <cstdio>

// Constructor: Initializes the JTAG Debug Transport Module (DTM) with a reference to the Debug Module.
// Use case: Creates a DTM instance that bridges JTAG protocol to RISC-V debug module operations.
jtag_dtm_t::jtag_dtm_t(DebugModule* dm)
    : dm(dm),
      _tck(0), _tms(0), _tdi(0), _tdo(0),
      _state(TEST_LOGIC_RESET),
      ir(IR_IDCODE), dr(0), dr_length(1),
      abits(7), busy_stuck(false), dmi(0) {}

// Resets the DTM to its initial state (TEST_LOGIC_RESET).
// Use case: Called when JTAG reset is detected or when the debugger needs to reinitialize the DTM.
void jtag_dtm_t::reset() {
    _state = TEST_LOGIC_RESET;
    ir = IR_IDCODE;
    busy_stuck = false;
    dmi = 0;
}

// Updates JTAG pin states and advances the TAP state machine based on TCK/TMS/TDI transitions.
// Use case: Called for each JTAG clock cycle to simulate the TAP controller state machine.
// The state transition table implements the standard JTAG TAP state machine where each state has
// two possible next states based on the TMS value (0 or 1).
void jtag_dtm_t::set_pins(bool tck, bool tms, bool tdi) {
    static const jtag_state_t next[16][2] = {
        {RUN_TEST_IDLE, TEST_LOGIC_RESET},
        {RUN_TEST_IDLE, SELECT_DR_SCAN},
        {CAPTURE_DR, SELECT_IR_SCAN},
        {SHIFT_DR, EXIT1_DR},
        {SHIFT_DR, EXIT1_DR},
        {PAUSE_DR, UPDATE_DR},
        {PAUSE_DR, EXIT2_DR},
        {SHIFT_DR, UPDATE_DR},
        {RUN_TEST_IDLE, SELECT_DR_SCAN},
        {CAPTURE_IR, TEST_LOGIC_RESET},
        {SHIFT_IR, EXIT1_IR},
        {SHIFT_IR, EXIT1_IR},
        {PAUSE_IR, UPDATE_IR},
        {PAUSE_IR, EXIT2_IR},
        {SHIFT_IR, UPDATE_IR},
        {RUN_TEST_IDLE, SELECT_DR_SCAN}
    };

    // Rising edge of TCK: sample TDI and shift data/instruction registers
    if (!_tck && tck) {
        switch (_state) {
            case SHIFT_DR: dr >>= 1; dr |= (uint64_t)_tdi << (dr_length - 1); break;
            case SHIFT_IR: ir >>= 1; ir |= _tdi << 4; break;
            default: break;
        }
        _state = next[_state][_tms];
    }
    // Falling edge of TCK: update TDO and trigger register operations
    else if (_tck && !tck) {
        switch (_state) {
            case CAPTURE_DR: capture_dr(); break;
            case UPDATE_DR:  update_dr();  break;
            case SHIFT_DR:   _tdo = dr & 1; break;
            case SHIFT_IR:   _tdo = ir & 1; break;
            default: break;
        }
    }

    _tck = tck;
    _tms = tms;
    _tdi = tdi;
}

// Captures data from the selected register into the DR shift register based on current IR instruction.
// Use case: Called during CAPTURE_DR state to prepare data for shifting out via TDO.
// The captured data depends on the instruction register (IR) value:
// - IR_IDCODE: Returns a dummy ID code
// - IR_DTMCONTROL: Returns DTM control register with version, address bits, and status
// - IR_DBUS: Returns the result of the previous DMI operation (read data or write status)
// - IR_BYPASS: Returns 0 for bypass mode
void jtag_dtm_t::capture_dr() {
    switch (ir) {
        case IR_IDCODE:     dr = 0xdeadbeef; dr_length = 32; break;
        case IR_DTMCONTROL: {
            uint32_t dmistat = busy_stuck ? 1 : 0;
            dr = (dmistat << 18) | (abits << 4) | 1;
            dr_length = 32;
            break;
        }
        case IR_DBUS:
            dr = dmi;
            dr_length = abits + 34;
            break;
        case IR_BYPASS:     dr = 0; dr_length = 1; break;
        default:            dr = 0; dr_length = 1; break;
    }
}

// Updates the selected register with data from the DR shift register after shifting is complete.
// Use case: Called during UPDATE_DR state to execute DMI read/write operations based on shifted data.
// For IR_DBUS, the DR contains: [addr][data][op] where op=1 (read), op=2 (write), op=0 (nop).
// After the operation, the result is stored in 'dmi' for the next capture_dr() call.
void jtag_dtm_t::update_dr() {
    if (ir == IR_DBUS) {
        uint32_t op   = dr & 0x3;
        uint32_t data = (dr >> 2) & 0xFFFFFFFF;
        uint32_t addr = (dr >> 34) & ((1 << abits) - 1);

        bool success = true;
        if (op == 1) {
            // DMI read operation: read from debug module and store result with status bits [1:0]
            uint32_t val = 0;
            success = dm->dmi_read(addr, &val);
            // Status codes: 0=success, 2=not supported, 3=failed
            // Unimplemented addresses return false, which means "not supported" (status=2)
            uint32_t status = success ? 0 : 2;
            dmi = ((uint64_t)val << 2) | status;
        } else if (op == 2) {
            // DMI write operation: write to debug module and store only status bits [1:0]
            success = dm->dmi_write(addr, data);
            // Status codes: 0=success, 2=not supported, 3=failed
            // Unimplemented addresses return false, which means "not supported" (status=2)
            uint32_t status = success ? 0 : 2;
            dmi = status;
        } else {
            // No-op: clear the result
            dmi = 0;
        }

        busy_stuck = !success;
    }
}
