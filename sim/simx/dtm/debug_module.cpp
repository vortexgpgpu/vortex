#include "debug_module.h"
#include <cstdarg>
#include <atomic>
#include <cstring>
#include "emulator.h"
#include <VX_config.h>
#include <bitmanip.h>

namespace {

std::atomic<bool> g_debug_module_verbose{false};

void dm_log(const char* fmt, ...) {
    if (!g_debug_module_verbose.load(std::memory_order_relaxed)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}

// Enables or disables verbose logging for debug module operations.
// Use case: Used to control debug output during development and troubleshooting.
void DebugModule::set_verbose_logging(bool enable) {
    g_debug_module_verbose.store(enable, std::memory_order_relaxed);
}

bool DebugModule::verbose_logging() {
    return g_debug_module_verbose.load(std::memory_order_relaxed);
}

// Constructor: Initializes the RISC-V Debug Module with a simulated memory space.
// Use case: Creates a debug module instance that implements the RISC-V Debug Specification 0.13.
DebugModule::DebugModule(vortex::Emulator* emulator, size_t mem_size)
    : emulator_(emulator),
      command(0),
      resumereq_prev(false),
      data1(0),
      data2(0),
      data3(0),
      memory(mem_size, 0),
      access_mem_addr(0),
      halt_requested_(false),
      single_step_active_(false),
      debug_mode_enabled_(false)
{
    for (unsigned i = 0; i < datacount; i++) {
        dmdata[i] = 0;
    }

    reset();
}

// Resets the debug module to its initial state: clears all registers and resets the hart.
// Use case: Called when dmactive is set from 0 to 1, or during initialization.
void DebugModule::reset()
{
    dmcontrol = dmcontrol_t();
    dmstatus = dmstatus_t();
    abstractcs = abstractcs_t();

    // Initialize debug state
    dcsr_ = DCSR();
    dpc_ = 0;
    resumeack_ = false;
    havereset_ = false;
    is_halted_ = false;

    dmcontrol.dmactive = true;
    dmstatus.authenticated = true;
    dmstatus.authbusy = false;
    dmstatus.version = 2;
    // Set XLEN support bits: sr32, sr64, sr128 (bits 20, 21, 22)
    // OpenOCD checks these FIRST to determine XLEN
    dmstatus.sr32 = (XLEN == 32);
    dmstatus.sr64 = (XLEN == 64);
    dmstatus.sr128 = false; // No 128-bit support

    dmstatus.allnonexistent = false;
    dmstatus.anynonexistent = false;
    dmstatus.allunavail = false;
    dmstatus.anyunavail = false;

    access_mem_addr = 0;
    access_mem_addr_valid = false;

    update_dmstatus();

    dmstatus.authenticated = true;
    dmstatus.authbusy = false;
}

// Updates the dmstatus register fields based on current hart state.
// Use case: Called before reading dmstatus to ensure it reflects current hart state (halted/running/etc.).
// Preserves authentication state which must remain true for OpenOCD compatibility.
void DebugModule::update_dmstatus()
{
    bool saved_authenticated = dmstatus.authenticated;
    bool saved_authbusy = dmstatus.authbusy;
    unsigned saved_version = dmstatus.version;

    // Check running state - emulator notifies us when program completes, so we just check our flags
    if (is_halted_ || halt_requested_) {
        dmstatus.allhalted = true;
        dmstatus.anyhalted = true;
        dmstatus.allrunning = false;
        dmstatus.anyrunning = false;
    } else {
        dmstatus.allhalted = false;
        dmstatus.anyhalted = false;
        dmstatus.allrunning = true;
        dmstatus.anyrunning = true;
    }

    dmstatus.allresumeack = resumeack_;
    dmstatus.anyresumeack = resumeack_;
    dmstatus.allhavereset = havereset_;
    dmstatus.anyhavereset = havereset_;

    // Check if selected hartsel (thread) is valid (0-31)
    // In our implementation, we have 32 threads, so hartsel must be 0-31
    unsigned thread_id = dmcontrol.hartsel & 0x1F;
    bool hart_exists = (thread_id < 32);  // We support threads 0-31
    
    dmstatus.allnonexistent = !hart_exists;
    dmstatus.anynonexistent = !hart_exists;
    dmstatus.allunavail = false;
    dmstatus.anyunavail = false;


    dmstatus.authenticated = saved_authenticated;
    dmstatus.authbusy = saved_authbusy;
    dmstatus.version = saved_version;
}

// Reads a value from a DMI (Debug Module Interface) register by address.
// Use case: Called by JTAG DTM to read debug module registers (dmcontrol, dmstatus, abstractcs, etc.).
// Returns true on success, false for unimplemented addresses.
bool DebugModule::dmi_read(unsigned address, uint32_t *value)
{
    switch (address) {
        case DM_DMCONTROL:
            *value = read_dmcontrol();
            break;
        case DM_DMSTATUS: {
            update_dmstatus();
            *value = read_dmstatus();
            // Log DMSTATUS reads to help debug thread discovery
            unsigned current_thread = dmcontrol.hartsel & 0x1F;
            bool exists = (current_thread < 32);
            // dm_log("[DM] DMSTATUS read: hartsel=0x%x (thread=%u), anynonexistent=%d, value=0x%08x\n", 
            //       dmcontrol.hartsel, current_thread, dmstatus.anynonexistent ? 1 : 0, *value);


            if (exists) {
                *value &= ~((1U << 14) | (1U << 15));  // Clear anynonexistent and allnonexistent bits
            }

            // Ensure authenticated bit (bit 7) is always set - critical for OpenOCD compatibility
            if ((*value & (1U << 7)) == 0) {
        dm_log("[DM] ERROR: authenticated bit (bit 7) not set! value=0x%x\n", *value);
                *value |= (1U << 7);
            }
            break;
        }
        case DM_HARTINFO:
            // Hart info: nscratch=1, dataaccess=1, datasize=datacount, dataaddr=0x380
            *value = (1 << 20) | (1 << 19) | (datacount << 16) | 0x380;
            break;
        case DM_ABSTRACTCS:
            *value = read_abstractcs();
            break;
        case DM_COMMAND:
            *value = 0;
            break;
        case DM_ABSTRACTAUTO:
            *value = 0;
            break;
        case DM_DATA0:
            *value = read_data0();
            break;
        case 0x5:  // DATA1
            *value = data1;
            dm_log("[DM] DATA1 read: 0x%08x\n", data1);
            break;
        case 0x6:  // DATA2
            *value = data2;
            break;
        case 0x7:  // DATA3
            *value = data3;
            break;
        case DM_AUTHDATA:
            *value = read_authdata();
            break;
        case DM_SBCS:
            // System Bus Control and Status: return 0 to indicate no system bus access available
            // This is optional functionality, so returning 0 is acceptable
            *value = 0;
            break;
        default:
            *value = 0;
            dm_log("[DM] DMI READ  addr=0x%x -> 0x%x (unimplemented)\n", address, *value);
            return false;
    }

    // dm_log("[DM] DMI READ  addr=0x%x -> 0x%x\n", address, *value);
    return true;
}

// Writes a value to a DMI (Debug Module Interface) register by address.
// Use case: Called by JTAG DTM to write debug module registers (dmcontrol, command, data0, etc.).
// Returns true on success, false for unimplemented addresses.
bool DebugModule::dmi_write(unsigned address, uint32_t value)
{
    dm_log("[DM] DMI WRITE addr=0x%x data=0x%x\n", address, value);

    switch (address) {
        case DM_DMCONTROL:
            return write_dmcontrol(value);
        case DM_COMMAND:
            return write_command(value);
        case DM_DATA0:
            return write_data0(value);
        case 0x5:  // DATA1
            data1 = value;
            dm_log("[DM] DATA1 written: 0x%08x\n", value);
            return true;
        case 0x6:  // DATA2
            data2 = value;
            dm_log("[DM] DATA2 written: 0x%08x\n", value);
            return true;
        case 0x7:  // DATA3
            data3 = value;
            dm_log("[DM] DATA3 written: 0x%08x\n", value);
            return true;
        case DM_AUTHDATA:
            return write_authdata(value);
        case DM_ABSTRACTAUTO:
            // Auto-execute not implemented in stub
            return true;
        case DM_ABSTRACTCS:
            // Clear command error if writing 1 to error bits (bits [10:8])
            if (value & (7 << 8)) {
                abstractcs.cmderr = 0;
            }
            return true;
        case DM_SBCS:
            // System Bus Control and Status: accept writes but do nothing (no system bus access)
            return true;
        default:
            dm_log("[DM] DMI WRITE addr=0x%x unimplemented\n", address);
            return false;
    }
}

// Reads the dmcontrol register, encoding all control fields into a 32-bit value.
// Use case: Returns the current debug module control state (dmactive, haltreq, resumereq, hartsel, etc.).
uint32_t DebugModule::read_dmcontrol()
{
    uint32_t result = 0;
    result = set_field_pos<uint32_t>(result, 0x1U, 0, dmcontrol.dmactive ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 1, dmcontrol.ndmreset ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 2, dmcontrol.clrresethaltreq ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 3, dmcontrol.setresethaltreq ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 16, dmcontrol.hartreset ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 28, dmcontrol.ackhavereset ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 30, dmcontrol.resumereq ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x1U, 31, dmcontrol.haltreq ? 1U : 0U);


    result = set_field_pos<uint32_t>(result, 0x3ffU << 6, 6, dmcontrol.hartsel);
    result = set_field_pos<uint32_t>(result, 0x1U, 26, dmcontrol.hasel ? 1U : 0U);


    result |= 1;

    return result;
}

// Reads the dmstatus register, encoding all status fields into a 32-bit value per RISC-V Debug Spec 0.13.2.
// Use case: Returns the current debug module status (version, authenticated, halted/running state, etc.).
// Always ensures authenticated bit (bit 7) is set - critical for OpenOCD compatibility.
uint32_t DebugModule::read_dmstatus()
{

    dmstatus.authenticated = true;
    dmstatus.authbusy = false;

    uint32_t result = 0;



    result |= (dmstatus.version & 0xf);


    if (dmstatus.confstrptrvalid) result |= (1U << 4);


    if (dmstatus.hasresethaltreq) result |= (1U << 5);





    result |= (1U << 7);


    if (dmstatus.anyhalted) result |= (1U << 8);


    if (dmstatus.allhalted) result |= (1U << 9);


    if (dmstatus.anyrunning) result |= (1U << 10);


    if (dmstatus.allrunning) result |= (1U << 11);


    if (dmstatus.anyunavail) result |= (1U << 12);


    if (dmstatus.allunavail) result |= (1U << 13);


    if (dmstatus.anynonexistent) result |= (1U << 14);


    if (dmstatus.allnonexistent) result |= (1U << 15);


    if (dmstatus.anyresumeack) result |= (1U << 16);


    if (dmstatus.allresumeack) result |= (1U << 17);


    if (dmstatus.anyhavereset) result |= (1U << 18);


    if (dmstatus.allhavereset) result |= (1U << 19);




    if (dmstatus.impebreak) result |= (1U << 22);




    if ((result & (1U << 7)) == 0) {
        dm_log("[DM] ERROR: authenticated bit (bit 7) not set! result=0x%x\n", result);
        result |= (1U << 7);
    }

    return result;
}

// Reads the abstractcs register, encoding abstract command status fields.
// Use case: Returns abstract command status (datacount, progbufsize, busy flag, cmderr).
uint32_t DebugModule::read_abstractcs()
{
    uint32_t result = 0;
    result = set_field_pos<uint32_t>(result, 0x1fU << 8, 8, abstractcs.datacount);
    result = set_field_pos<uint32_t>(result, 0xffU << 16, 16, abstractcs.progbufsize);
    result = set_field_pos<uint32_t>(result, 0x1U, 28, abstractcs.busy ? 1U : 0U);
    result = set_field_pos<uint32_t>(result, 0x7U << 8, 8, abstractcs.cmderr);
    result = set_field_pos<uint32_t>(result, 0xfU << 24, 24, 1);
    return result;
}

// Reads the DATA0 register value (used for abstract command data transfer).
// Use case: Returns the value stored in DATA0, typically used to read register/memory values.
uint32_t DebugModule::read_data0()
{
    uint32_t value = dmdata[0];
    dm_log("[DM] DATA0 read: 0x%08x\n", value);
    return value;
}

// Reads the authdata register (authentication not implemented in stub).
// Use case: Returns 0 since authentication protocol is not implemented.
uint32_t DebugModule::read_authdata()
{
    return 0;
}

// Writes the dmcontrol register, updating control fields and processing requests (halt/resume).
// Use case: Called to control the debug module (halt/resume hart, select hart, reset, etc.).
// Processes haltreq and resumereq immediately, and resets module if dmactive transitions from 0 to 1.
bool DebugModule::write_dmcontrol(uint32_t value)
{
    // If setting dmactive from 0 to 1, reset the module
    if (!dmcontrol.dmactive && (value & 1)) {
        reset();
    }


    dmcontrol.dmactive = (value & 0x1) != 0;
    dmcontrol.ndmreset = (value & (0x1 << 1)) != 0;
    dmcontrol.clrresethaltreq = (value & (0x1 << 2)) != 0;
    dmcontrol.setresethaltreq = (value & (0x1 << 3)) != 0;
    dmcontrol.hartreset = (value & (0x1 << 16)) != 0;
    dmcontrol.ackhavereset = (value & (0x1 << 28)) != 0;
    dmcontrol.resumereq = (value & (0x1 << 30)) != 0;
    dmcontrol.haltreq = (value & (0x1 << 31)) != 0;

    // Extract hartsel (hart selection) and hasel (hart array select) fields
    dmcontrol.hartsel = (value >> 6) & 0x3ff;
    dmcontrol.hasel = (value & (0x1 << 26)) != 0;

    // Use lower 5 bits of hartsel to select thread within warp (0-31)
    // Note: We always use thread 0 from emulator's warp 0
    unsigned thread_id = dmcontrol.hartsel & 0x1F;
    if (thread_id < 32) {
        dm_log("[DM] Thread selection: hartsel=0x%x, selected thread=%u (using thread 0)\n", dmcontrol.hartsel, thread_id);
    } else {
        dm_log("[DM] Invalid thread selection: hartsel=0x%x, thread_id=%u (max 31)\n", dmcontrol.hartsel, thread_id);
    }

    // Always keep dmactive set for stub (always active)
    dmcontrol.dmactive = true;

    // Handle halt request immediately (cause = 3 per spec)
    if (dmcontrol.haltreq) {
        halt_hart(3);
        dmcontrol.haltreq = false;
    }

    if (dmcontrol.resumereq && !resumereq_prev) {
        resumeack_ = false;
        resume_hart(false);
    }

    resumereq_prev = dmcontrol.resumereq;

    if (dmcontrol.ackhavereset) {
        havereset_ = false;
    }

    update_dmstatus();
    return true;
}

// Writes the abstract command register and executes the command if not busy.
// Use case: Called to execute abstract commands (e.g., access register, quick access).
// Returns false if busy, otherwise executes command and returns true.
bool DebugModule::write_command(uint32_t value)
{
    command = value;
    dm_log("[DM] COMMAND written: 0x%08x\n", value);


    // Execute command immediately if not busy (stub implementation)
    if (!abstractcs.busy) {
        return perform_abstract_command();
    } else {
        abstractcs.cmderr = 1;  // BUSY error
        dm_log("[DM] COMMAND error: BUSY (cmderr=1)\n");
        return false;
    }
}

// Writes the DATA0 register value (used for abstract command data transfer).
// Use case: Called to set data for abstract commands (e.g., register value to write).
bool DebugModule::write_data0(uint32_t value)
{
    dmdata[0] = value;
    dm_log("[DM] DATA0 written: 0x%08x\n", value);
    return true;
}

// Writes the authdata register (authentication not implemented in stub).
// Use case: Accepts any authdata write and marks as authenticated (stub always authenticates).
bool DebugModule::write_authdata(uint32_t)
{
    dmstatus.authenticated = true;
    dmstatus.authbusy = false;
    return true;
}

// Performs the abstract command stored in the command register.
// Use case: Executes abstract commands (supports Access Register cmdtype=0 and Access Memory cmdtype=0x02).
// Returns false if busy or unsupported command type, sets cmderr accordingly.
bool DebugModule::perform_abstract_command()
{
    if (abstractcs.busy) {
        abstractcs.cmderr = 1;
        dm_log("[DM] COMMAND error: BUSY (cmderr=1)\n");
        return false;
    }


    unsigned cmdtype = (command >> 24) & 0xff;

    if (cmdtype == 0 || cmdtype == 0x02) {
        // Access Register (cmdtype=0) or Access Memory (cmdtype=0x02)
        abstractcs.busy = true;
        execute_command(command);
        abstractcs.busy = false;
        return true;
    } else {
        abstractcs.cmderr = 2;
        dm_log("[DM] COMMAND error: NOTSUP (cmderr=2), cmdtype=0x%02x\n", cmdtype);
        return false;
    }
}

// Executes an abstract command (supports Access Register and Access Memory commands).
// Use case: Processes abstract commands to read/write hart registers or memory, optionally with postexec step.
// Command format: [cmdtype][aarsize/aamsize][postexec][transfer][write][regaddr/aamaddress]
void DebugModule::execute_command(uint32_t value)
{
    uint8_t cmdtype = (value >> 24) & 0xFF;
    if (cmdtype == 0) {
        // Access Register command
        uint8_t aarsize = (value >> 20) & 0x7;
        bool postexec   = value & (1 << 18);
        bool transfer   = value & (1 << 17);
        bool write      = value & (1 << 16);
        uint16_t regaddr  = value & 0xFFFF;

        dm_log("[DM] EXECUTE COMMAND: Access Register, regaddr=0x%04x, write=%d, transfer=%d, postexec=%d, aarsize=%d\n",
               regaddr, write ? 1 : 0, transfer ? 1 : 0, postexec ? 1 : 0, aarsize);

        if (transfer) {
            if (write) {
                // For 64-bit systems, combine data0 (low) and data1 (high) if available
                vortex::Word val;
                if (XLEN == 64 && abstractcs.datacount >= 2) {
                    val = static_cast<vortex::Word>(data0()) | (static_cast<vortex::Word>(data1) << 32);
                } else {
                    val = static_cast<vortex::Word>(data0());
                }
                write_register(regaddr, val);
            } else {
                vortex::Word val = read_register(regaddr);
                // For 64-bit systems, split into data0 (low) and data1 (high) if available
                if (XLEN == 64 && abstractcs.datacount >= 2) {
                    data0() = static_cast<uint32_t>(val);
                    data1 = static_cast<uint32_t>(val >> 32);
                } else {
                    data0() = static_cast<uint32_t>(val);
                }
            }
        }

        if (postexec) {
            // Get PC from emulator
            vortex::Word pc = 0;
            if (emulator_ != nullptr) {
                auto& warp0 = emulator_->get_warp(0);
                pc = warp0.PC;
            }
            
            // Check for software breakpoint: if instruction at PC is EBREAK, halt
            vortex::Word instruction = read_mem(pc, sizeof(uint32_t));  // Read 4-byte instruction
            if ((instruction & 0xFFFFFFFF) == 0x00100073) {
                // EBREAK instruction - software breakpoint
                dm_log("[DM] Software breakpoint hit at 0x%0*llx (EBREAK), halting hart\n", 
                       (XLEN == 64) ? 16 : 8, (unsigned long long)pc);
                halt_hart(1);  // Cause 1 = ebreak instruction
                return;  // Don't execute the instruction
            }
            
            // Note: Step is handled by emulator, not here
            dm_log("[DM] STEP: PC=0x%08x (step handled by emulator)\n", pc);
        }
    } else if (cmdtype == 0x02) {
        // Access Memory command (per updated RISC-V Debug Spec)
        // Fields:
        // [31:24] cmdtype (0x02)
        // [23]    aamvirtual
        // [22:20] aamsize (0=8-bit, 1=16-bit, 2=32-bit, 3=64-bit)
        // [19]    aampostincrement
        // [18:17] 0
        // [16]    write (1=write, 0=read)
        // [15:14] target-specific-info
        // [13:0]  0

        uint8_t aamvirtual       = (value >> 23) & 0x1;   // currently unused
        uint8_t aamsize          = (value >> 20) & 0x7;
        bool    aampostincrement = ((value >> 19) & 0x1) != 0;
        bool    write            = ((value >> 16) & 0x1) != 0;
        (void)aamvirtual; // suppress unused warning for now

        size_t access_size = (aamsize == 0) ? 1 :
                             (aamsize == 1) ? 2 :
                             (aamsize == 2) ? 4 : 8;

        // Decide base address and remember where it came from so we can apply postincrement correctly.
        enum AddrSource {
            ADDR_NONE,
            ADDR_DATA2,
            ADDR_DATA1,
            ADDR_DATA0,
            ADDR_PREV
        } addr_src = ADDR_NONE;

        vortex::Word mem_addr = 0;

        // If this looks like a continuation of a postincrement sequence (no explicit address
        // in DATA[0-3]), reuse the last address.
        if (aampostincrement &&
            access_mem_addr_valid &&
            data3 == 0 && data2 == 0 && data1 == 0 && data0() == 0) {
            mem_addr = access_mem_addr;
            addr_src = ADDR_PREV;
        } else if (data3 != 0 || data2 != 0) {
            // For 64-bit addresses, combine DATA3 (high) and DATA2 (low)
            if (XLEN == 64 && data3 != 0) {
                mem_addr = (static_cast<vortex::Word>(data3) << 32) | static_cast<vortex::Word>(data2);
                addr_src = ADDR_DATA2;
                dm_log("[DM] Access Memory: Combining DATA3 (0x%08x) and DATA2 (0x%08x) -> addr=0x%016llx\n",
                       data3, data2, (unsigned long long)mem_addr);
            } else {
                mem_addr = static_cast<vortex::Word>(data2);
                addr_src = ADDR_DATA2;
            }
        } else if (data1 != 0 || data0() != 0) {
            // For 64-bit addresses, combine DATA1 (high) and DATA0 (low)
            if (XLEN == 64 && data1 != 0) {
                mem_addr = (static_cast<vortex::Word>(data1) << 32) | static_cast<vortex::Word>(data0());
                addr_src = ADDR_DATA0;
                dm_log("[DM] Access Memory: Combining DATA1 (0x%08x) and DATA0 (0x%08x) -> addr=0x%016llx\n",
                       data1, data0(), (unsigned long long)mem_addr);
            } else if (data1 != 0) {
                mem_addr = static_cast<vortex::Word>(data1);
                addr_src = ADDR_DATA1;
            } else {
                mem_addr = static_cast<vortex::Word>(data0());
                addr_src = ADDR_DATA0;
            }
        } else if (access_mem_addr_valid) {
            // Fallback to previous address if we have one.
            mem_addr = access_mem_addr;
            addr_src = ADDR_PREV;
        } else {
            mem_addr = 0;
            addr_src = ADDR_NONE;
        }
        
        dm_log("[DM] EXECUTE COMMAND: Access Memory, addr=0x%0*llx, write=%d, aamsize=%u, postinc=%d\n",
               (XLEN == 64) ? 16 : 8, (unsigned long long)mem_addr, 
               write ? 1 : 0, aamsize, aampostincrement ? 1 : 0);

        // Always perform one memory access per command.
            if (write) {
                // Write memory: For 64-bit, combine data0 (low) and data1 (high) if available
                vortex::Word write_data;
                if (access_size == 8 && XLEN == 64 && abstractcs.datacount >= 2) {
                    write_data = static_cast<vortex::Word>(data0()) | (static_cast<vortex::Word>(data1) << 32);
                } else {
                    write_data = static_cast<vortex::Word>(data0());
                }
                
            dm_log("[DM] Access Memory WRITE: addr=0x%llx, data=0x%0*llx, size=%zu\n",
                   (unsigned long long)mem_addr, (access_size == 8) ? 16 : 8, 
                   (unsigned long long)write_data, access_size);
                
                if (access_size == 1) {
                    vortex::Word old_val = read_mem(mem_addr, access_size);
                    write_mem(mem_addr, (old_val & ~0xFF) | (write_data & 0xFF), access_size);
                } else if (access_size == 2) {
                // Detect compressed EBREAK (0x9002) being written - save original instruction
                if ((write_data & 0xFFFF) == 0x9002 && !has_breakpoint(mem_addr)) {
                    add_breakpoint(mem_addr);
                }
                    vortex::Word old_val = read_mem(mem_addr, access_size);
                    write_mem(mem_addr, (old_val & ~0xFFFF) | (write_data & 0xFFFF), access_size);
                } else if (access_size == 4) {
                // Detect EBREAK instruction (32-bit: 0x00100073 or compressed: 0x00009002) being written
                bool is_ebreak = (write_data == 0x00100073) || ((write_data & 0xFFFF) == 0x9002);
                if (is_ebreak && !has_breakpoint(mem_addr)) {
                    add_breakpoint(mem_addr);
                }
                    write_mem(mem_addr, write_data, access_size);
                } else if (access_size == 8) {
                    // 64-bit write
                    write_mem(mem_addr, write_data, access_size);
                } else {
                    dm_log("[DM] Access Memory: unsupported write size %zu\n", access_size);
                }
        } else {
                // Read memory: result goes into DATA0 (and DATA1 for 64-bit if available)
                vortex::Word read_val = read_mem(mem_addr, access_size);
                
                if (access_size == 1) {
                    data0() = static_cast<uint32_t>(read_val & 0xFF);
                } else if (access_size == 2) {
                    data0() = static_cast<uint32_t>(read_val & 0xFFFF);
                } else if (access_size == 4) {
                    data0() = static_cast<uint32_t>(read_val);
                } else if (access_size == 8) {
                    // 64-bit read: split into data0 (low) and data1 (high) if available
                    if (XLEN == 64 && abstractcs.datacount >= 2) {
                        data0() = static_cast<uint32_t>(read_val);
                        data1 = static_cast<uint32_t>(read_val >> 32);
                    } else {
                        // Fallback: only return low 32 bits
                        data0() = static_cast<uint32_t>(read_val);
                    }
                } else {
                    dm_log("[DM] Access Memory: unsupported read size %zu\n", access_size);
                    data0() = 0;
                }
            }
            
        // Implement aampostincrement: advance the address and write it back to the same source.
        vortex::Word new_addr = mem_addr;
            if (aampostincrement) {
            new_addr = mem_addr + access_size;
            switch (addr_src) {
                case ADDR_DATA2:
                    // If address came from DATA3+DATA2 (64-bit), write back both parts
                    if (XLEN == 64 && data3 != 0) {
                        data3 = static_cast<uint32_t>(new_addr >> 32);
                        data2 = static_cast<uint32_t>(new_addr);
                    } else {
                        data2 = static_cast<uint32_t>(new_addr);
                    }
                    break;
                case ADDR_DATA1:
                    data1 = static_cast<uint32_t>(new_addr);
                    break;
                case ADDR_DATA0:
                    // If address came from DATA1+DATA0 (64-bit), write back both parts
                    if (XLEN == 64 && data1 != 0) {
                        data1 = static_cast<uint32_t>(new_addr >> 32);
                        data0() = static_cast<uint32_t>(new_addr);
                    } else {
                        data0() = static_cast<uint32_t>(new_addr);
                    }
                    break;
                case ADDR_PREV:
                    // When address came from a previous implicit address sequence (access_mem_addr),
                    // DO NOT write back to data registers - only update internal state.
                    // This prevents overwriting data registers that OpenOCD might use for other purposes.
                    // The incremented address is stored in access_mem_addr for the next postincrement operation.
                    dm_log("[DM] Access Memory postincrement: address from access_mem_addr, NOT writing back to data registers (prev=0x%0*llx, new=0x%0*llx)\n",
                           (XLEN == 64) ? 16 : 8, (unsigned long long)mem_addr,
                           (XLEN == 64) ? 16 : 8, (unsigned long long)new_addr);
                    break;
                case ADDR_NONE:
                default:
                    break;
            }
            access_mem_addr = new_addr;
            access_mem_addr_valid = true;
            } else {
                access_mem_addr = mem_addr;
            access_mem_addr_valid = true;
        }
    } else {
        abstractcs.cmderr = 2;  // NOTSUP
        dm_log("[DM] COMMAND error: NOTSUP (cmderr=2), cmdtype=0x%02x\n", cmdtype);
    }
}

// Reads a hart register by abstract register address (used by access register commands).
// Use case: Called during abstract command execution to read GPRs, PC, DCSR, DPC, or CSRs.
// Register address mapping: 0x1000-0x101F (GPRs), 0x1020 (PC), 0x7B0 (DCSR), 0x7B1 (DPC), 0x0000-0x0FFF/0xC000-0xFFFF (CSRs).
vortex::Word DebugModule::read_register(uint16_t regaddr)
{
    // General purpose registers (x0–x31) at addresses 0x1000–0x101F
    if (regaddr >= 0x1000 && regaddr <= 0x101F) {
        int gpr_index = regaddr - 0x1000;
        vortex::Word value;
        if (emulator_ != nullptr) {
            // Use emulator's warp 0, thread 0 register
            auto& warp0 = emulator_->get_warp(0);
            value = warp0.ireg_file.at(gpr_index).at(0);  // Direct assignment, no cast needed
        } else {
            // No emulator available
            value = 0;
        }
        dm_log("[DM] READ REG  x%d (0x%04x) -> 0x%0*llx\n", gpr_index, regaddr, 
               (XLEN == 64) ? 16 : 8, (unsigned long long)value);
        return value;
    }

    if (regaddr == 0x1020) {
        vortex::Word value;
        if (emulator_ != nullptr) {
            // Use emulator's warp 0 PC
            auto& warp0 = emulator_->get_warp(0);
            value = warp0.PC;  // PC is already Word type
        } else {
            // No emulator available
            value = 0;
        }
        dm_log("[DM] READ REG  pc (0x1020) -> 0x%0*llx\n", (XLEN == 64) ? 16 : 8, (unsigned long long)value);
        return value;
    }

    if (regaddr == 0x07b0 || regaddr == 0x7B0) {
        vortex::Word value = dcsr_.to_u32();  // DCSR is always 32-bit
        dm_log("[DM] READ REG  dcsr (0x7B0) -> 0x%08x\n", (uint32_t)value);
        return value;
    }

    if (regaddr == 0x07b1 || regaddr == 0x7B1) {
        vortex::Word value = dpc_;
        dm_log("[DM] READ REG  dpc (0x7B1) -> 0x%0*llx\n", (XLEN == 64) ? 16 : 8, (unsigned long long)value);
        return value;
    }

    // Helper function to read CSR by number
    // Note: CSRs are always 32-bit per RISC-V spec, but we return Word for consistency
    auto read_csr = [this](uint16_t csr_num, uint16_t regaddr) -> vortex::Word {
        if (csr_num == 0x0301) {
            // Calculate MISA based on configured extensions
            // MXL field (bits 31:30): 1=RV32, 2=RV64, 3=RV128
            uint32_t mxl = (vortex::log2floor(XLEN) - 4);
            uint32_t value = (mxl << 30) | MISA_STD;
            dm_log("[DM] READ REG  misa (0x%03x via 0x%04x) -> 0x%08x (RV%d, MXL=%d, MISA_STD=0x%08x)\n", 
                   csr_num, regaddr, value, XLEN, mxl, MISA_STD);
            return value;
        }

        if (csr_num == 0x0c22) {
            uint32_t value = 0;
            dm_log("[DM] READ REG  vlenb (0x%03x via 0x%04x) -> 0x%08x (no vector support)\n", 
                   csr_num, regaddr, value);
            return value;
        }

        dm_log("[DM] READ REG  csr[0x%03x] (0x%04x) -> 0x00000000\n", csr_num, regaddr);
        return 0;
    };

    // Direct CSR access: 0x0000-0x0FFF (CSR number = regaddr)
    if (regaddr >= 0x0000 && regaddr <= 0x0FFF) {
        return read_csr(static_cast<uint16_t>(regaddr), regaddr);
    }

    dm_log("[DM] READ REG unknown regaddr=0x%04x -> 0x00000000\n", regaddr);
    return vortex::Word(0);
}

void DebugModule::write_register(uint16_t regaddr, vortex::Word val)
{
    if (regaddr >= 0x1000 && regaddr <= 0x101F) {
        int gpr_index = regaddr - 0x1000;
        if (gpr_index == 0) {
            dm_log("[DM] WRITE REG x0 (0x%04x) <- 0x%0*llx (ignored, x0 is read-only)\n", 
                   regaddr, (XLEN == 64) ? 16 : 8, (unsigned long long)val);
            return;
        }
        if (emulator_ != nullptr) {
            auto& warp0 = emulator_->get_warp(0);
            warp0.ireg_file.at(gpr_index).at(0) = val;  // Direct assignment
        }
        dm_log("[DM] WRITE REG x%d (0x%04x) <- 0x%0*llx\n", 
               gpr_index, regaddr, (XLEN == 64) ? 16 : 8, (unsigned long long)val);
        return;
    }

    if (regaddr == 0x1020) {
        if (emulator_ != nullptr) {
            auto& warp0 = emulator_->get_warp(0);
            warp0.PC = val;  // Direct assignment, PC is Word type
        }
        dm_log("[DM] WRITE REG pc (0x1020) <- 0x%0*llx\n", 
               (XLEN == 64) ? 16 : 8, (unsigned long long)val);
        return;
    }

    if (regaddr == 0x07b0 || regaddr == 0x7B0) {
        dcsr_.from_u32(static_cast<uint32_t>(val));  // DCSR is always 32-bit
        dm_log("[DM] WRITE REG dcsr (0x7B0) <- 0x%08x\n", (uint32_t)val);
        return;
    }

    if (regaddr == 0x07b1 || regaddr == 0x7B1) {
        dpc_ = val;
        dm_log("[DM] WRITE REG dpc (0x7B1) <- 0x%0*llx\n", 
               (XLEN == 64) ? 16 : 8, (unsigned long long)val);
        return;
    }

    if (regaddr >= 0xC000 && regaddr <= 0xFFFF) {
        dm_log("[DM] WRITE REG csr[0x%04x] (0x%04x) <- 0x%0*llx (ignored)\n", 
               regaddr - 0xC000, regaddr, (XLEN == 64) ? 16 : 8, (unsigned long long)val);
        return;
    }

    dm_log("[DM] WRITE REG unknown regaddr=0x%04x <- 0x%0*llx (ignored)\n", 
           regaddr, (XLEN == 64) ? 16 : 8, (unsigned long long)val);
}

vortex::Word DebugModule::read_mem(vortex::Word addr, size_t size)
{
    vortex::Word val = read_program_memory(addr, size);
    dm_log("[DM] READ MEM  addr=0x%0*llx -> 0x%0*llx (size=%zu)\n", 
           (XLEN == 64) ? 16 : 8, (unsigned long long)addr,
           (XLEN == 64) ? 16 : 8, (unsigned long long)val, size);
    return val;
}

void DebugModule::write_mem(vortex::Word addr, vortex::Word val, size_t size)
{
    write_program_memory(addr, val, size);
    dm_log("[DM] WRITE MEM addr=0x%0*llx <- 0x%0*llx (size=%zu)\n", 
           (XLEN == 64) ? 16 : 8, (unsigned long long)addr,
           (XLEN == 64) ? 16 : 8, (unsigned long long)val, size);
}

vortex::Word DebugModule::read_program_memory(vortex::Word addr, size_t size) const
{
    if (!emulator_) {
        return 0;
    }
    // Read the specified number of bytes
    uint8_t buffer[8] = {0};  // Max 8 bytes for 64-bit access
    emulator_->dcache_read(buffer, static_cast<uint64_t>(addr), size);
    
    // Convert to Word based on size
    vortex::Word value = 0;
    for (size_t i = 0; i < size && i < sizeof(vortex::Word); ++i) {
        value |= static_cast<vortex::Word>(buffer[i]) << (i * 8);
    }
    return value;
}

void DebugModule::write_program_memory(vortex::Word addr, vortex::Word value, size_t size)
{
    if (!emulator_) {
        return;
    }
    
    // Write the specified number of bytes
    uint8_t buffer[8];  // Max 8 bytes for 64-bit access
    for (size_t i = 0; i < size && i < sizeof(vortex::Word); ++i) {
        buffer[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    }
    emulator_->dcache_write(buffer, static_cast<uint64_t>(addr), size);
}

vortex::Word DebugModule::direct_read_register(uint16_t regaddr)
{
    return read_register(regaddr);
}

void DebugModule::direct_write_register(uint16_t regaddr, vortex::Word value)
{
    write_register(regaddr, value);
}

bool DebugModule::read_memory_block(uint64_t addr, uint8_t* dest, size_t len) const
{
    if (addr + len > memory.size()) {
        return false;
    }
    std::memcpy(dest, memory.data() + addr, len);
    return true;
}

bool DebugModule::write_memory_block(uint64_t addr, const uint8_t* src, size_t len)
{
    if (addr + len > memory.size()) {
        return false;
    }
    std::memcpy(memory.data() + addr, src, len);
    return true;
}


// Halts the hart (CPU core) and enters debug mode with the specified cause.
// Use case: Called when debugger requests a halt or a breakpoint is hit.
// Cause values: 0=reserved, 1=ebreak, 2=trigger, 3=haltreq, 4=step, 5=resume after step, etc.
void DebugModule::halt_hart(uint8_t cause)
{
    dm_log("[DM] Halt requested - hart halted (cause=%u)\n", cause);
    // Enter debug mode: update DCSR (DPC will be updated by emulator when it actually halts)
    dcsr_.cause = cause & 0xF;
    is_halted_ = true;
    // Set halt flag so emulator will stop execution and update DPC
    set_halt_requested(true);
    update_dmstatus();
    // Log DCSR value after setting cause to verify encoding
    uint32_t dcsr_val = dcsr_.to_u32();
    uint8_t cause_field = (dcsr_val >> 8) & 0xF;
    dm_log("[DM] DCSR after halt: 0x%08x, cause field: 0x%x (should be 0x%x)\n", dcsr_val, cause_field, cause);
}

// Resumes the hart execution, optionally in single-step mode.
// Use case: Called when debugger requests resume or step execution.
// If single_step is true or hart is in step mode, executes one instruction then halts again.
void DebugModule::resume_hart(bool single_step)
{
    dm_log("[DM] Resume requested (single_step=%d)\n", single_step ? 1 : 0);
    is_halted_ = false;

    // Log current program state before resuming
    if (emulator_ != nullptr) {
        auto& warp0 = emulator_->get_warp(0);
        vortex::Word current_pc = warp0.PC;
        vortex::Word dpc = dpc_;
        dm_log("[DM] Resume state: PC=0x%0*llx, DPC=0x%0*llx, halt_requested=%d\n", 
               (XLEN == 64) ? 16 : 8, (unsigned long long)current_pc,
               (XLEN == 64) ? 16 : 8, (unsigned long long)dpc,
               halt_requested_ ? 1 : 0);
    }

    bool do_step = single_step || dcsr_.step;
    if (do_step) {
        // Set single-step flag so emulator will execute one instruction then halt
        set_single_step_active(true);
        set_halt_requested(false);  // Clear halt to allow execution
        // No need to resume - just clearing halt_requested_ is enough
        dm_log("[DM] Single-step mode: halt_requested cleared, will execute one instruction\n");
        resumeack_ = true;
    } else {
        // Clear halt flag to allow continuous execution
        set_halt_requested(false);
        set_single_step_active(false);
        // No need to resume - just clearing halt_requested_ is enough
        dm_log("[DM] Continuous execution resumed: halt_requested=%d, single_step_active=%d\n", 
               halt_requested_ ? 1 : 0, single_step_active_ ? 1 : 0);
        resumeack_ = true;
    }
    update_dmstatus();
}

bool DebugModule::hart_is_halted() const
{
    return is_halted_ || halt_requested_;
}

bool DebugModule::is_halt_requested() const
{
    return halt_requested_;
}

bool DebugModule::is_single_step_active() const
{
    return single_step_active_;
}

bool DebugModule::is_debug_mode_enabled() const
{
    return debug_mode_enabled_;
}

void DebugModule::set_halt_requested(bool halt)
{
    halt_requested_ = halt;
}

void DebugModule::set_single_step_active(bool step)
{
    single_step_active_ = step;
}

void DebugModule::set_debug_mode_enabled(bool enabled)
{
    debug_mode_enabled_ = enabled;
}

bool DebugModule::has_breakpoint(uint32_t addr) const
{
    return software_breakpoints_.find(addr) != software_breakpoints_.end();
}

void DebugModule::add_breakpoint(uint32_t addr)
{
    if (has_breakpoint(addr)) {
        return; // Already has breakpoint
    }
    // Read and store the original instruction (should be called before EBREAK is written)
    uint32_t original = static_cast<uint32_t>(read_program_memory(addr, sizeof(uint32_t)));  // Read 4-byte instruction
    software_breakpoints_[addr] = original;
}

void DebugModule::remove_breakpoint(uint32_t addr)
{
    auto it = software_breakpoints_.find(addr);
    if (it == software_breakpoints_.end()) {
        return; // No breakpoint at this address
    }
    // Restore the original instruction
    write_program_memory(addr, it->second, sizeof(uint32_t));  // Write 4-byte instruction
    software_breakpoints_.erase(it);
}

// Notification from emulator when program completes naturally
void DebugModule::notify_program_completed(vortex::Word final_pc)
{
    // Only process if we weren't already explicitly halted
    if (!is_halted_ && !halt_requested_) {
        dm_log("[DM] Program completed naturally at PC=0x%0*llx, halting hart\n", 
               (XLEN == 64) ? 16 : 8, (unsigned long long)final_pc);
        
        // Update DPC to final PC
        direct_write_register(0x7B1, final_pc);
        
        // Mark as halted (cause 0 = reserved, but we use it for natural completion)
        is_halted_ = true;
        set_halt_requested(true);
        dcsr_.cause = 0;  // Natural completion
    }
}


void DebugModule::run_test_idle()
{
    static uint64_t log_counter = 0;
    if (!is_halted_ && !halt_requested_) {
        if ((log_counter++ % 1000) == 0 && emulator_ != nullptr) {
            auto& warp0 = emulator_->get_warp(0);
            vortex::Word pc = warp0.PC;
            dm_log("[DM] run_test_idle: hart running, PC=0x%0*llx\n", 
                   (XLEN == 64) ? 16 : 8, (unsigned long long)pc);
        }
    } else {
        // Only log occasionally when halted too
        if ((log_counter++ % 1000) == 0) {
        dm_log("[DM] run_test_idle: hart is halted, nothing to do\n");
        }
    }
}

