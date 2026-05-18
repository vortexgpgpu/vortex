// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

#include "CommandProcessor.h"

#include <cstring>
#include <cassert>

namespace vortex {

CommandProcessor::CommandProcessor(const Hooks& hooks)
    : hooks_(hooks) {}

bool CommandProcessor::enabled() const {
    return (cp_ctrl_ & 0x1) && (q0_.control & 0x1);
}

bool CommandProcessor::busy() const {
    return enabled() && (q0_.head < q0_.tail
                         || cl_loaded_
                         || eng_state_ != EngState::Idle
                         || launch_state_ != LaunchState::Idle);
}

// ============================================================================
// MMIO surface
// ============================================================================

void CommandProcessor::mmio_write(uint32_t off, uint32_t value) {
    // Globals
    switch (off) {
        case 0x000: cp_ctrl_ = value; return;
        // STATUS / DEV_CAPS / CYCLE are RO; ignore writes.
        case 0x004: case 0x008: case 0x010: case 0x014: return;
    }
    // Queue 0 (offsets 0x100..0x12F)
    if (off >= 0x100 && off < 0x140) {
        switch (off - 0x100) {
            case 0x00: q0_.ring_base   = (q0_.ring_base & 0xFFFFFFFF00000000ULL) | uint64_t(value);            return;
            case 0x04: q0_.ring_base   = (q0_.ring_base & 0x00000000FFFFFFFFULL) | (uint64_t(value) << 32);   return;
            case 0x08: q0_.head_addr   = (q0_.head_addr & 0xFFFFFFFF00000000ULL) | uint64_t(value);           return;
            case 0x0C: q0_.head_addr   = (q0_.head_addr & 0x00000000FFFFFFFFULL) | (uint64_t(value) << 32);   return;
            case 0x10: q0_.cmpl_addr   = (q0_.cmpl_addr & 0xFFFFFFFF00000000ULL) | uint64_t(value);           return;
            case 0x14: q0_.cmpl_addr   = (q0_.cmpl_addr & 0x00000000FFFFFFFFULL) | (uint64_t(value) << 32);   return;
            case 0x18: q0_.ring_log2   = uint8_t(value & 0xFF);                                                 return;
            case 0x1C: q0_.control     = value;                                                                 return;
            case 0x20: q0_.tail_lo_staging = value;                                                             return;
            case 0x24: {
                // Atomic tail commit (matches the hardware's "write HI to commit" rule).
                q0_.tail = (uint64_t(value) << 32) | uint64_t(q0_.tail_lo_staging);
                return;
            }
            // SEQNUM / ERROR are RO; ignore.
            case 0x28: case 0x2C: return;
        }
    }
    // Unknown offset — silently ignored (mirrors hardware DECERR behavior
    // from the host's perspective is via the MMIO bus, not this object).
}

uint32_t CommandProcessor::mmio_read(uint32_t off) const {
    switch (off) {
        case 0x000: return cp_ctrl_;
        case 0x004: return uint32_t(busy() ? 1 : 0);    // CP_STATUS bit0
        case 0x008: {
            // CP_DEV_CAPS: matches VX_cp_axil_regfile §17.4.
            // {AXI_TID_W:8 | RING_LOG2:8 | NUM_QUEUES:8}
            // We use the same defaults as the hardware (TID=6, RING=16, N=1).
            return (uint32_t(6) << 16) | (uint32_t(16) << 8) | uint32_t(1);
        }
        case 0x010: return uint32_t(cycle_counter_ & 0xFFFFFFFF);
        case 0x014: return uint32_t(cycle_counter_ >> 32);
    }
    if (off >= 0x100 && off < 0x140) {
        switch (off - 0x100) {
            case 0x00: return uint32_t(q0_.ring_base & 0xFFFFFFFF);
            case 0x04: return uint32_t(q0_.ring_base >> 32);
            case 0x08: return uint32_t(q0_.head_addr & 0xFFFFFFFF);
            case 0x0C: return uint32_t(q0_.head_addr >> 32);
            case 0x10: return uint32_t(q0_.cmpl_addr & 0xFFFFFFFF);
            case 0x14: return uint32_t(q0_.cmpl_addr >> 32);
            case 0x18: return uint32_t(q0_.ring_log2);
            case 0x1C: return q0_.control;
            case 0x20: return q0_.tail_lo_staging;
            case 0x24: return uint32_t(q0_.tail >> 32);
            case 0x28: return uint32_t(q0_.seqnum & 0xFFFFFFFF);
            case 0x2C: return q0_.error;
        }
    }
    return 0xDEADBEEF;
}

// ============================================================================
// Fetch + unpack
// ============================================================================

void CommandProcessor::fetch_if_needed() {
    if (cl_loaded_) return;
    if (q0_.head >= q0_.tail) return;
    const uint64_t mask = (uint64_t(1) << q0_.ring_log2) - 1;
    const uint64_t off  = q0_.head & mask;
    if (!hooks_.dram_read) return;
    hooks_.dram_read(q0_.ring_base + off, cl_buf_.data(), CL_BYTES);
    cl_loaded_   = true;
    cl_cmd_slot_ = 0;
    unpack_cl();
}

int CommandProcessor::decode_cmd(int off, Cmd& out) {
    auto rd8 = [&](int o) -> uint8_t {
        return (o >= 0 && o < int(CL_BYTES)) ? cl_buf_[o] : 0;
    };
    auto rd64 = [&](int o) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= uint64_t(rd8(o + i)) << (8 * i);
        return v;
    };
    out.opcode   = rd8(off + 0);
    out.flags    = rd8(off + 1);
    out.reserved = uint16_t(rd8(off + 2)) | (uint16_t(rd8(off + 3)) << 8);
    out.arg0     = rd64(off + 4);
    out.arg1     = rd64(off + 12);
    out.arg2     = rd64(off + 20);
    // Size table mirrors cmd_size_bytes() in VX_cp_pkg.sv.
    switch (out.opcode) {
        case OP_NOP:        return 4;
        case OP_LAUNCH:     return 12;
        case OP_FENCE:      return 8;
        case OP_DCR_WRITE:  return 20;
        case OP_DCR_READ:   return 20;
        case OP_EVENT_SIG:  return 20;
        case OP_EVENT_WAIT: return 28;
        case OP_MEM_WRITE:
        case OP_MEM_READ:
        case OP_MEM_COPY:   return 28;
        default:            return 4;
    }
}

void CommandProcessor::unpack_cl() {
    cl_cmd_count_ = 0;
    cl_cmd_slot_  = 0;
    int offset = 0;
    for (int slot = 0; slot < MAX_CMDS_PER_CL; ++slot) {
        if (offset + 4 > int(CL_BYTES)) break;
        const uint8_t opcode = cl_buf_[offset];
        const uint8_t flags  = cl_buf_[offset + 1];
        // Zero header = padding sentinel; stop.
        if (opcode == 0 && flags == 0) break;
        Cmd c;
        const int sz = decode_cmd(offset, c);
        if (offset + sz > int(CL_BYTES)) break;
        ++cl_cmd_count_;
        offset += sz;
    }
}

// ============================================================================
// Engine FSM
// ============================================================================

void CommandProcessor::publish_completion() {
    if (!hooks_.dram_write || q0_.cmpl_addr == 0) return;
    uint64_t seq = q0_.seqnum;
    hooks_.dram_write(q0_.cmpl_addr, &seq, sizeof(seq));
}

void CommandProcessor::tick_launch() {
    switch (launch_state_) {
        case LaunchState::Idle:        return;
        case LaunchState::PulseStart:
            if (hooks_.vortex_start) hooks_.vortex_start();
            launch_state_ = LaunchState::WaitBusy;
            return;
        case LaunchState::WaitBusy:
            // Wait for Vortex to actually start. Matches VX_cp_launch.sv.
            if (hooks_.vortex_busy && hooks_.vortex_busy())
                launch_state_ = LaunchState::WaitDrain;
            return;
        case LaunchState::WaitDrain:
            if (!hooks_.vortex_busy || !hooks_.vortex_busy())
                launch_state_ = LaunchState::Idle;
            return;
    }
}

void CommandProcessor::tick_engine() {
    // Decode a single cmd at the current slot and walk it through the FSM.
    auto load_next_cmd = [this]() -> bool {
        if (!cl_loaded_) return false;
        if (cl_cmd_slot_ >= cl_cmd_count_) {
            // All commands in this CL consumed (or it was pure padding);
            // advance head and drop the CL.
            q0_.head   += CL_BYTES;
            cl_loaded_ = false;
            return false;
        }
        int off = 0;
        for (int s = 0; s < cl_cmd_slot_; ++s) {
            Cmd skip;
            off += decode_cmd(off, skip);
        }
        decode_cmd(off, cur_cmd_);
        cur_is_launch_ = (cur_cmd_.opcode == OP_LAUNCH);
        switch (cur_cmd_.opcode) {
            case OP_NOP: case OP_FENCE:
            case OP_EVENT_SIG: case OP_EVENT_WAIT:
                // No resource — retire as NOP (matches engine Phase 2b
                // skip_flag path for unimplemented opcodes).
                cur_is_no_resource_ = true;
                break;
            default:
                cur_is_no_resource_ = false;
                break;
        }
        return true;
    };

    switch (eng_state_) {
        case EngState::Idle:
            fetch_if_needed();
            if (load_next_cmd())
                eng_state_ = EngState::Decode;
            return;

        case EngState::Decode:
            if (cur_is_no_resource_) {
                eng_state_ = EngState::Retire;
            } else {
                eng_state_ = EngState::Bid;
            }
            return;

        case EngState::Bid:
            // Dispatch to the resource. Single-queue means we always win
            // the arbiter, so transition immediately to WaitDone.
            if (cur_is_launch_) {
                launch_state_ = LaunchState::PulseStart;
                eng_state_    = EngState::WaitDone;
            } else if (cur_cmd_.opcode == OP_DCR_WRITE) {
                // Issue the DCR write through the hook immediately;
                // the "proxy" is functionally instantaneous in C++.
                if (hooks_.vortex_dcr_write) {
                    uint32_t addr = uint32_t(cur_cmd_.arg0 & 0xFFF); // VX_DCR_ADDR_BITS=12
                    uint32_t val  = uint32_t(cur_cmd_.arg1 & 0xFFFFFFFF);
                    hooks_.vortex_dcr_write(addr, val);
                }
                eng_state_ = EngState::Retire;
            } else {
                // DCR_READ / MEM_* not yet implemented in this functional
                // model — retire as NOP (matches the engine's Phase 2b
                // behavior for unimplemented opcodes).
                eng_state_ = EngState::Retire;
            }
            return;

        case EngState::WaitDone:
            // For LAUNCH: wait until the launch FSM is back in Idle.
            if (cur_is_launch_ && launch_state_ != LaunchState::Idle)
                return;
            eng_state_ = EngState::Retire;
            return;

        case EngState::Retire:
            q0_.seqnum += 1;
            publish_completion();
            ++cl_cmd_slot_;
            eng_state_ = EngState::Idle;
            return;
    }
}

void CommandProcessor::tick() {
    ++cycle_counter_;
    if (!enabled()) return;
    tick_engine();
    tick_launch();
}

} // namespace vortex
