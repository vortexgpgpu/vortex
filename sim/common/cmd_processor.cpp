// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

#include "cmd_processor.h"

#include <VX_config.h>     // VX_CFG_* device configuration
#include <VX_types.h>      // VX_ISA_IMPL_ID

#include <cstring>
#include <cassert>
#include <vector>

namespace vortex {

// ============================================================================
// Static GPU capability words
//
// Packed identically to the canonical RTL in
// hw/rtl/cp/VX_cp_axil_regfile.sv (gpu_dev_caps / gpu_isa_caps). The
// CommandProcessor is the functional twin of that regfile, so simx /
// rtlsim / gem5 expose the same RO GPU_DEV_CAPS / GPU_ISA_CAPS registers.
// sw/runtime/common/vx_caps.h is the single matching decoder.
// ============================================================================
namespace {
constexpr unsigned cp_clog2(uint64_t n) {
    unsigned r = 0;
    while ((uint64_t(1) << r) < n) ++r;
    return r;
}
uint64_t gpu_dev_caps() {
    const unsigned cluster_size = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;
    const unsigned bank_addr_w  = VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH
                                - cp_clog2(VX_CFG_PLATFORM_MEMORY_NUM_BANKS);
    return  (uint64_t(VX_ISA_IMPL_ID) & 0xFF)
         | ((uint64_t(cp_clog2(VX_CFG_NUM_THREADS))  & 0x7)  << 8)
         | ((uint64_t(cp_clog2(VX_CFG_NUM_WARPS))    & 0x7)  << 11)
         | ((uint64_t(cp_clog2(VX_CFG_SOCKET_SIZE))  & 0x7)  << 14)
         | ((uint64_t(cp_clog2(cluster_size))        & 0x7)  << 17)
         | ((uint64_t(cp_clog2(VX_CFG_NUM_CLUSTERS)) & 0x7)  << 20)
         | ((uint64_t(cp_clog2(VX_CFG_ISSUE_WIDTH))  & 0x7)  << 23)
         | ((uint64_t(VX_CFG_LMEM_ENABLED ? VX_CFG_LMEM_LOG_SIZE : 0) & 0xFF) << 26)
         | ((uint64_t(cp_clog2(VX_CFG_PLATFORM_MEMORY_NUM_BANKS)) & 0x7) << 34)
         | ((uint64_t(bank_addr_w - 20) & 0x1F) << 37);
}
uint64_t gpu_isa_caps() {
    return  (uint64_t(VX_CFG_MISA_EXT) << 32)
         | ((uint64_t(cp_clog2(VX_CFG_XLEN) - 4) & 0x3) << 30)
         |  uint64_t(VX_CFG_MISA_STD);
}
} // namespace

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
        // CP_SATP — page-table root for the CP DMA's MMU.
        case 0x028: satp_ = (satp_ & 0xFFFFFFFF00000000ULL) |  uint64_t(value);         return;
        case 0x02C: satp_ = (satp_ & 0x00000000FFFFFFFFULL) | (uint64_t(value) << 32);  return;
        // STATUS / DEV_CAPS / CYCLE / GPU caps are RO; ignore writes.
        case 0x004: case 0x008: case 0x010: case 0x014:
        case 0x018: case 0x01C: case 0x020: case 0x024: return;
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
    // Unknown offset — silently ignored. The hardware would respond with
    // DECERR on the MMIO bus; this functional model presents no failure
    // surface for it.
}

uint32_t CommandProcessor::mmio_read(uint32_t off) const {
    switch (off) {
        case 0x000: return cp_ctrl_;
        case 0x004: return uint32_t(busy() ? 1 : 0);    // CP_STATUS bit0
        case 0x008: {
            // CP_DEV_CAPS: {VM_ENABLED:1 @bit24 | AXI_TID_W:8 | RING_LOG2:8
            // | NUM_QUEUES:8}. Defaults match the hardware (TID=6,
            // RING_LOG2=16, NUM_QUEUES=1). VM_ENABLED is published from this
            // sim's build config so the config-agnostic libvortex.so can
            // discover VM at vx_device_open instead of #ifdef-ing on it.
            uint32_t vm_enabled = 0;
#ifdef VX_CFG_VM_ENABLE
            vm_enabled = 1u << 24;
#endif
            return vm_enabled | (uint32_t(6) << 16)
                 | (uint32_t(16) << 8) | uint32_t(1);
        }
        case 0x010: return uint32_t(cycle_counter_ & 0xFFFFFFFF);
        case 0x014: return uint32_t(cycle_counter_ >> 32);
        case 0x018: return uint32_t(gpu_dev_caps() & 0xFFFFFFFF);
        case 0x01C: return uint32_t(gpu_dev_caps() >> 32);
        case 0x020: return uint32_t(gpu_isa_caps() & 0xFFFFFFFF);
        case 0x024: return uint32_t(gpu_isa_caps() >> 32);
        case 0x028: return uint32_t(satp_ & 0xFFFFFFFF);
        case 0x02C: return uint32_t(satp_ >> 32);
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
            case 0x30: return last_dcr_rsp_;  // last CMD_DCR_READ response
        }
    }
    return 0xDEADBEEF;
}

// ============================================================================
// VM — page-table walk (the CP DMA is an MMU-aware copy engine)
// ============================================================================

uint64_t CommandProcessor::cp_translate(uint64_t vaddr, bool physical) const {
#ifdef VX_CFG_VM_ENABLE
    if (physical || satp_ == 0)
        return vaddr;
    SATP_t satp(satp_);
    if (satp.get_mode() == BARE)
        return vaddr;
    // Sv32/Sv39 walk — mirrors VMManager::page_table_walk so the CP DMA and
    // the host driver resolve addresses identically.
    int i = VX_VM_PT_LEVEL - 1;
    vAddr_t va(vaddr);
    uint64_t cur_base_ppn = satp.get_base_ppn();
    for (;;) {
        uint64_t pte_addr  = cur_base_ppn * VX_VM_PT_SIZE
                           + va.vpn[i] * VX_VM_PTE_SIZE;
        uint64_t pte_bytes = 0;
        if (hooks_.dram_read)
            hooks_.dram_read(pte_addr, &pte_bytes, VX_VM_PTE_SIZE);
        PTE_t pte(pte_bytes);
        if (pte.v == 0)
            return vaddr;            // unmapped — pass through (defensive)
        if (pte.r == 0 && pte.w == 0 && pte.x == 0) {
            if (--i < 0)
                return vaddr;        // no leaf — pass through
            cur_base_ppn = pte.ppn;
            continue;
        }
        cur_base_ppn = pte.ppn;      // leaf found at level i
        break;
    }
    // Reconstruct the physical address. For a leaf found at level i > 0 (a
    // mega/gigapage) the low VX_VM_PAGE_LOG2_SIZE + i*VPN_BITS address bits
    // are the offset *within* the superpage and must come from the VA, not
    // from the (superpage-aligned) leaf PPN. For a 4 KB leaf (i == 0) this
    // reduces to the ordinary ppn<<12 | page-offset.
    constexpr unsigned VPN_BITS = cp_clog2(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
    const uint64_t off_mask =
        (uint64_t(1) << (VX_VM_PAGE_LOG2_SIZE + i * VPN_BITS)) - 1;
    return ((cur_base_ppn << VX_VM_PAGE_LOG2_SIZE) & ~off_mask)
         | (vaddr & off_mask);
#else
    (void)physical;
    return vaddr;
#endif
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
    // Size table matches cmd_size_bytes() in VX_cp_pkg.sv.
    switch (out.opcode) {
        case OP_NOP:        return 4;
        case OP_LAUNCH:     return 12;
        case OP_FENCE:      return 8;
        case OP_CACHE_FLUSH: return 12;
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

// Mirrors hw/rtl/cp/VX_cp_pkg.sv: wait_op_e encoded in arg2[1:0].
bool CommandProcessor::event_wait_satisfied_() {
    if (!hooks_.dram_read) return true;   // no DRAM hook -> retire as NOP
    uint64_t cur = 0;
    hooks_.dram_read(cur_cmd_.arg0, &cur, sizeof(cur));
    const uint64_t target = cur_cmd_.arg1;
    const uint32_t op = uint32_t(cur_cmd_.arg2) & 0x3;
    switch (op) {
        case 0: return cur == target;   // WAIT_OP_EQ
        case 1: return cur >= target;   // WAIT_OP_GE
        case 2: return cur >  target;   // WAIT_OP_GT
        case 3: return cur != target;   // WAIT_OP_NE
        default: return true;
    }
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
            case OP_NOP: case OP_FENCE: case OP_CACHE_FLUSH:
                // No resource bid for these opcodes; retire as NOP. The
                // functional model has no caches, so CMD_CACHE_FLUSH is a
                // pure no-op here — memory is already coherent.
                cur_is_no_resource_ = true;
                break;
            default:
                // LAUNCH, DCR_*, MEM_*, EVENT_SIG, EVENT_WAIT all bid a
                // resource.
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
            } else if (cur_cmd_.opcode == OP_DCR_READ) {
                // Issue the DCR read; latch the response into the regfile
                // slot so the host can grab it after polling Q_SEQNUM.
                if (hooks_.vortex_dcr_read) {
                    uint32_t addr = uint32_t(cur_cmd_.arg0 & 0xFFF);
                    uint32_t tag  = uint32_t(cur_cmd_.arg1 & 0xFFFFFFFF);
                    last_dcr_rsp_ = hooks_.vortex_dcr_read(addr, tag);
                }
                eng_state_ = EngState::Retire;
            } else if (cur_cmd_.opcode == OP_EVENT_SIG) {
                // CMD_EVENT_SIGNAL: write arg1 (8-byte value) to arg0
                // (device counter slot). Matches VX_cp_event_unit's
                // SIGNAL path (AW + W + B).
                if (hooks_.dram_write) {
                    uint64_t v = cur_cmd_.arg1;
                    hooks_.dram_write(cur_cmd_.arg0, &v, sizeof(v));
                }
                eng_state_ = EngState::Retire;
            } else if (cur_cmd_.opcode == OP_EVENT_WAIT) {
                // CMD_EVENT_WAIT: spin reading arg0 until the value
                // matches arg1 under the wait_op encoded in arg2[1:0].
                // Functional model: do the compare in this single tick;
                // if it doesn't match, transition to WaitDone and keep
                // re-checking on subsequent ticks (so multiple events
                // can interleave).
                if (event_wait_satisfied_()) {
                    eng_state_ = EngState::Retire;
                } else {
                    // Reuse WaitDone state via a dedicated launch-shaped
                    // flag isn't needed — just spin in Bid by NOT
                    // transitioning. The cp_mmio_read host loop will
                    // tick the simulator further, on each tick we'll
                    // re-check.
                    // Stay in Bid (no state change).
                }
            } else if (cur_cmd_.opcode == OP_MEM_WRITE ||
                       cur_cmd_.opcode == OP_MEM_READ  ||
                       cur_cmd_.opcode == OP_MEM_COPY) {
                // CMD_MEM_*: copy arg2 bytes from src (arg1) to dst (arg0).
                // The CP DMA is an MMU-aware copy engine: the device-side
                // operand is a virtual address, translated here by a
                // page-table walk. MEM_WRITE -> arg0 is the device dst;
                // MEM_READ -> arg1 is the device src; MEM_COPY -> both.
                // Host-side operands and physical-flagged commands pass
                // through untranslated. A buffer is one contiguous PA
                // allocation, so translating the base covers the transfer.
                if (hooks_.dram_read && hooks_.dram_write
                 && cur_cmd_.arg2 != 0) {
                    const bool physical =
                        (cur_cmd_.flags & MEM_FLAG_PHYSICAL) != 0;
                    uint64_t dst = cur_cmd_.arg0;
                    uint64_t src = cur_cmd_.arg1;
                    if (cur_cmd_.opcode == OP_MEM_WRITE) {
                        dst = cp_translate(dst, physical);
                    } else if (cur_cmd_.opcode == OP_MEM_READ) {
                        src = cp_translate(src, physical);
                    } else { // OP_MEM_COPY — both operands are device
                        dst = cp_translate(dst, physical);
                        src = cp_translate(src, physical);
                    }
                    const uint64_t total = cur_cmd_.arg2;
                    constexpr uint64_t CHUNK = 64 * 1024;
                    std::vector<uint8_t> buf(
                        std::size_t(total < CHUNK ? total : CHUNK));
                    for (uint64_t done = 0; done < total; ) {
                        uint64_t n = total - done;
                        if (n > CHUNK) n = CHUNK;
                        hooks_.dram_read (src + done, buf.data(), n);
                        hooks_.dram_write(dst + done, buf.data(), n);
                        done += n;
                    }
                }
                eng_state_ = EngState::Retire;
            } else {
                // Unknown opcode — retire as NOP.
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
