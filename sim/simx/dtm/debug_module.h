#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <set>
#include <map>
#include <VX_config.h>
#include "types.h"

namespace vortex {
    class Core;
    class RAM;
}


#define DM_DATA0           0x04
#define DM_DMCONTROL       0x10
#define DM_DMSTATUS        0x11
#define DM_HARTINFO        0x12
#define DM_ABSTRACTCS      0x16
#define DM_COMMAND         0x17
#define DM_ABSTRACTAUTO    0x18
#define DM_DMCONTROL2      0x1a
#define DM_AUTHDATA        0x30
#define DM_SBCS            0x38
#define DM_SBADDRESS0      0x39
#define DM_SBDATA0         0x3c


template<typename T>
static inline T set_field(T reg, T mask, T val) {
    return (reg & ~mask) | (val & mask);
}

template<typename T>
static inline T get_field(T reg, T mask) {
    return (reg & mask);
}

template<typename T>
static inline T set_field_pos(T reg, T mask, unsigned pos, unsigned val) {
    return set_field(reg, mask, static_cast<T>(val) << pos);
}

template<typename T>
static inline T get_field_pos(T reg, T mask, unsigned pos) {
    return (reg & mask) >> pos;
}


struct dmcontrol_t {
    bool dmactive;
    bool ndmreset;
    bool clrresethaltreq;
    bool setresethaltreq;
    bool hartreset;
    bool ackhavereset;
    bool resumereq;
    bool haltreq;
    unsigned hartsel;
    bool hasel;

    dmcontrol_t() : dmactive(true), ndmreset(false), clrresethaltreq(false),
                    setresethaltreq(false), hartreset(false), ackhavereset(false),
                    resumereq(false), haltreq(false), hartsel(0), hasel(false) {}
};


struct dmstatus_t {
    unsigned version;
    bool confstrptrvalid;
    bool hasresethaltreq;
    bool authbusy;
    bool authenticated;
    bool anyhalted;
    bool allhalted;
    bool anyrunning;
    bool allrunning;
    bool anyunavail;
    bool allunavail;
    bool anynonexistent;
    bool allnonexistent;
    bool anyresumeack;
    bool allresumeack;
    bool anyhavereset;
    bool allhavereset;

    bool impebreak;
    bool sr32;  // hart supports 32-bit abstract register access
    bool sr64;  // hart supports 64-bit abstract register access
    bool sr128; // hart supports 128-bit abstract register access


    dmstatus_t() : version(2), confstrptrvalid(false), hasresethaltreq(false),
                   authbusy(false), authenticated(true),
                   anyhalted(false), allhalted(false),
                   anyrunning(false), allrunning(false),
                   anyunavail(false), allunavail(false),
                   anynonexistent(false), allnonexistent(false),
                   anyresumeack(false), allresumeack(true),
                   anyhavereset(false), allhavereset(false),
                   impebreak(false),
                   sr32(false), sr64(false), sr128(false) {}
};


struct abstractcs_t {
    unsigned datacount;
    unsigned progbufsize;
    bool busy;
    unsigned cmderr;

    // datacount: number of data registers (1 for RV32, 2 for RV64)
    // OpenOCD uses this to determine XLEN
    abstractcs_t() : datacount((XLEN == 64) ? 2 : 1), progbufsize(0), busy(false), cmderr(0) {}
};

class DebugModule {
public:
    // Constructor: Initializes the RISC-V Debug Module with a simulated memory space.
    // Use case: Creates a debug module instance that implements the RISC-V Debug Specification 0.13.
    DebugModule(vortex::Core* core = nullptr, vortex::RAM* ram = nullptr, size_t mem_size = 4096);

    // Reads a value from a DMI (Debug Module Interface) register by address.
    // Use case: Called by JTAG DTM to read debug module registers (dmcontrol, dmstatus, abstractcs, etc.).
    bool dmi_read(unsigned address, uint32_t *value);
    
    // Writes a value to a DMI (Debug Module Interface) register by address.
    // Use case: Called by JTAG DTM to write debug module registers (dmcontrol, command, data0, etc.).
    bool dmi_write(unsigned address, uint32_t value);


    static void set_verbose_logging(bool enable);
    static bool verbose_logging();


    vortex::Word direct_read_register(uint16_t regaddr);
    void direct_write_register(uint16_t regaddr, vortex::Word value);
    bool read_memory_block(uint64_t addr, uint8_t* dest, size_t len) const;
    bool write_memory_block(uint64_t addr, const uint8_t* src, size_t len);
    // Halts the warp (SIMD thread group) and enters debug mode with the specified cause.
    // Use case: Called when debugger requests a halt or a breakpoint is hit.
    void halt_hart(uint8_t cause);
    
    // Resumes the warp execution, optionally in single-step mode.
    // Use case: Called when debugger requests resume or step execution.
    void resume_hart(bool single_step);
    
    // Returns true if the warp is currently halted.
    // Use case: Used to check warp state for status reporting.
    bool hart_is_halted() const;

    // Called periodically when JTAG is in Run-Test-Idle state.
    // Use case: Allows the debug module to process state updates during idle periods.
    void run_test_idle();

    // Debug flag query methods (read-only)
    bool is_halt_requested() const;
    bool is_single_step_active() const;
    bool is_debug_mode_enabled() const;
    
    // Debug flag control methods (set flags)
    void set_halt_requested(bool halt);
    void set_single_step_active(bool step);
    void set_debug_mode_enabled(bool enabled);
    
    // Software breakpoint management
    bool has_breakpoint(uint32_t addr) const;
    void add_breakpoint(uint32_t addr);
    void remove_breakpoint(uint32_t addr);
    
    // Notification from emulator when program completes
    void notify_program_completed(vortex::Word final_pc);

private:

    dmcontrol_t dmcontrol;
    dmstatus_t dmstatus;
    abstractcs_t abstractcs;

    vortex::Core* core_;
    vortex::RAM*  ram_;

    // Debug state flags
    bool halt_requested_;
    bool single_step_active_;
    bool debug_mode_enabled_;
    
    // Debug Control and Status Register (DCSR)
    struct DCSR {
        uint32_t prv       : 2;
        uint32_t step      : 1;
        uint32_t ebreakm   : 1;
        uint32_t ebreaks   : 1;
        uint32_t ebreaku   : 1;
        uint32_t stopcount : 1;
        uint32_t stoptime  : 1;
        uint32_t cause     : 4;
        uint32_t mprven    : 1;
        uint32_t nmip      : 1;
        uint32_t reserved  : 14;
        uint32_t xdebugver : 4;

        DCSR() : prv(3), step(0), ebreakm(0), ebreaks(0), ebreaku(0),
                 stopcount(0), stoptime(0), cause(0), mprven(0),
                 nmip(0), reserved(0), xdebugver(4) {}

        uint32_t to_u32() const {
            uint32_t value = 0;
            value |= (prv & 0x3);
            value |= (step & 0x1) << 2;
            value |= (ebreakm & 0x1) << 3;
            value |= (ebreaks & 0x1) << 4;
            value |= (ebreaku & 0x1) << 5;
            value |= (stopcount & 0x1) << 6;
            value |= (stoptime & 0x1) << 7;
            value |= (cause & 0xF) << 8;
            value |= (mprven & 0x1) << 12;
            value |= (nmip & 0x1) << 13;
            value |= (xdebugver & 0xF) << 28;
            return value;
        }

        void from_u32(uint32_t value) {
            prv       = value & 0x3;
            step      = (value >> 2) & 0x1;
            ebreakm   = (value >> 3) & 0x1;
            ebreaks   = (value >> 4) & 0x1;
            ebreaku   = (value >> 5) & 0x1;
            stopcount = (value >> 6) & 0x1;
            stoptime  = (value >> 7) & 0x1;
            cause     = (value >> 8) & 0xF;
            mprven    = (value >> 12) & 0x1;
            nmip      = (value >> 13) & 0x1;
            xdebugver = 4;
            reserved  = 0;
        }
    } dcsr_;
    
    // Debug Program Counter (DPC) - PC value when entering debug mode
    vortex::Word dpc_;
    
    // Debug state tracking
    bool resumeack_;
    bool havereset_;
    bool is_halted_;
    
    // Software breakpoint storage: address -> original instruction
    std::map<uint32_t, uint32_t> software_breakpoints_;

    // datacount: number of data registers (1 for RV32, 2 for RV64)
    // OpenOCD uses this to determine XLEN
    static constexpr unsigned datacount = (XLEN == 64) ? 2 : 1;
    uint32_t dmdata[datacount];
    uint32_t data1;  // DATA1 register (address 0x5)
    uint32_t data2;  // DATA2 register (address 0x6)
    uint32_t data3;  // DATA3 register (address 0x7)

    uint32_t& data0() { return dmdata[0]; }


    static constexpr unsigned progbufsize = 0;


    uint32_t command;


    bool resumereq_prev;


    std::vector<uint8_t> memory;
    
    // Temporary storage for Access Memory command address
    // OpenOCD sets address in DATA0, then data, then executes command
    vortex::Word access_mem_addr;
    bool access_mem_addr_valid;


    void reset();
    void update_dmstatus();


    bool perform_abstract_command();
    void execute_command(uint32_t cmd);


    vortex::Word read_register(uint16_t regaddr);
    void write_register(uint16_t regaddr, vortex::Word val);


    vortex::Word read_mem(vortex::Word addr, size_t size = sizeof(vortex::Word));
    void write_mem(vortex::Word addr, vortex::Word val, size_t size = sizeof(vortex::Word));
    
    // Program memory access (via emulator)
    vortex::Word read_program_memory(vortex::Word addr, size_t size = sizeof(uint32_t)) const;
    void write_program_memory(vortex::Word addr, vortex::Word value, size_t size = sizeof(uint32_t));


    uint32_t read_dmcontrol();
    uint32_t read_dmstatus();
    uint32_t read_abstractcs();
    uint32_t read_data0();
    uint32_t read_authdata();

    bool write_dmcontrol(uint32_t value);
    bool write_command(uint32_t value);
    bool write_data0(uint32_t value);
    bool write_authdata(uint32_t value);
};
