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
// GPU_DEV_CAPS / GPU_ISA_CAPS register values exposed as read-only MMIO.
// sw/runtime/common/vx_caps.h is the matching decoder.
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
                // Atomic tail commit: writing the HI word completes the 64-bit update.
                q0_.tail = (uint64_t(value) << 32) | uint64_t(q0_.tail_lo_staging);
                return;
            }
            // SEQNUM / ERROR are RO; ignore.
            case 0x28: case 0x2C: return;
        }
    }
    // Unknown offset — silently ignored.
}

uint32_t CommandProcessor::mmio_read(uint32_t off) const {
    switch (off) {
        case 0x000: return cp_ctrl_;
        case 0x004: return uint32_t(busy() ? 1 : 0);    // CP_STATUS bit0
        case 0x008: {
            // CP_DEV_CAPS: {SUPPORTS_QMD:1 @bit26 | SUPPORTS_DRAW:1 @bit25 |
            // VM_ENABLED:1 @bit24 | AXI_TID_W:8 | RING_LOG2:8 | NUM_QUEUES:8}.
            // Defaults: TID=6, RING_LOG2=16, NUM_QUEUES=1. VM_ENABLED reflects
            // the build-config so the config-agnostic libvortex.so can
            // discover VM at open. SUPPORTS_DRAW / SUPPORTS_QMD = 1: this
            // (Emulation) CP decodes CMD_DRAW (OP_DRAW) and CMD_LAUNCH_QMD;
            // the RTL CP advertises 0 until its mirrors are synth-validated,
            // so the runtime falls back to plain ring commands there.
            uint32_t vm_enabled = 0;
#ifdef VX_CFG_VM_ENABLE
            vm_enabled = 1u << 24;
#endif
            const uint32_t supports_draw = 1u << 25;
            const uint32_t supports_qmd  = 1u << 26;
            return supports_qmd | supports_draw | vm_enabled
                 | (uint32_t(6) << 16) | (uint32_t(16) << 8) | uint32_t(1);
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
    return decode_cmd_bytes(cl_buf_.data(), int(CL_BYTES), off, out);
}

int CommandProcessor::decode_cmd_bytes(const uint8_t* buf, int len,
                                       int off, Cmd& out) {
    auto rd8 = [&](int o) -> uint8_t {
        return (o >= 0 && o < len) ? buf[o] : 0;
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
    // Command size in bytes by opcode.
    switch (out.opcode) {
        case OP_NOP:        return 4;
        case OP_LAUNCH:     return 12;
        case OP_LAUNCH_QMD: return 12;   // arg0 = QMD descriptor address
        case OP_DRAW:       return 12;   // arg0 = draw descriptor address
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

// wait_op_e encoded in arg2[1:0].
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

// CMD_LAUNCH_QMD: read the KMU descriptor from device memory and replay it
// through the DCR-write hook. The descriptor is a {uint32 count, then count ×
// (uint32 dcr_addr, uint32 value)} list the host staged before submit (like a
// kernel-args blob). cp_translate matches the address the host's CMD_MEM_WRITE
// staged it at (VM walk when active, passthrough otherwise).
void CommandProcessor::apply_qmd_(uint64_t qmd_addr) {
    if (!hooks_.dram_read || !hooks_.vortex_dcr_write) return;
    uint64_t addr = cp_translate(qmd_addr, /*physical=*/false);
    uint32_t count = 0;
    hooks_.dram_read(addr, &count, sizeof(count));
    addr += sizeof(count);
    constexpr uint32_t MAX_QMD_DCRS = 64;   // backstop against a corrupt count
    if (count > MAX_QMD_DCRS) count = MAX_QMD_DCRS;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t pair[2] = {0, 0};          // {dcr_addr, value}
        hooks_.dram_read(addr, pair, sizeof(pair));
        addr += sizeof(pair);
        hooks_.vortex_dcr_write(pair[0] & 0xFFF, pair[1]);  // VX_DCR_ADDR_BITS=12
    }
}

// Execute one draw-bundle step. Inline ops (DCR_WRITE/DCR_READ/CACHE_FLUSH)
// complete immediately and return false; a launch step (LAUNCH/LAUNCH_QMD)
// kicks the launch sub-FSM and returns true so the caller waits for the drain
// (the inter-stage barrier). Mirrors the per-opcode logic of the ring Bid path.
bool CommandProcessor::exec_inline_cmd_(const Cmd& c) {
    switch (c.opcode) {
        case OP_LAUNCH:
        case OP_LAUNCH_QMD:
            if (c.opcode == OP_LAUNCH_QMD)
                apply_qmd_(c.arg0);
            launch_state_ = LaunchState::PulseStart;
            return true;
        case OP_DCR_WRITE:
            if (hooks_.vortex_dcr_write)
                hooks_.vortex_dcr_write(uint32_t(c.arg0 & 0xFFF),
                                        uint32_t(c.arg1 & 0xFFFFFFFF));
            return false;
        case OP_DCR_READ:
            if (hooks_.vortex_dcr_read)
                last_dcr_rsp_ = hooks_.vortex_dcr_read(
                    uint32_t(c.arg0 & 0xFFF), uint32_t(c.arg1 & 0xFFFFFFFF));
            return false;
        case OP_CACHE_FLUSH:
            if (hooks_.vortex_dcr_read) {
                uint32_t n = uint32_t(c.arg0 & 0xFFFFFFFF);
                for (uint32_t cid = 0; cid < n; ++cid)
                    (void)hooks_.vortex_dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid);
            }
            return false;
        default:
            // NOP / FENCE / unknown step — no-op (draws don't use MEM_*/EVENT_*).
            return false;
    }
}

// Read draw_step_'s 28-byte cmd record from the descriptor into draw_cmd_.
void CommandProcessor::draw_load_step_() {
    draw_cmd_ = Cmd{};
    if (!hooks_.dram_read) return;
    uint8_t buf[DRAW_STEP_BYTES] = {0};
    const uint64_t at = draw_phys_ + 4
                      + uint64_t(draw_step_) * DRAW_STEP_BYTES;
    hooks_.dram_read(at, buf, DRAW_STEP_BYTES);
    decode_cmd_bytes(buf, DRAW_STEP_BYTES, 0, draw_cmd_);
}

void CommandProcessor::tick_launch() {
    switch (launch_state_) {
        case LaunchState::Idle:        return;
        case LaunchState::PulseStart:
            if (hooks_.vortex_start) hooks_.vortex_start();
            launch_state_ = LaunchState::WaitBusy;
            return;
        case LaunchState::WaitBusy:
            // Wait for Vortex to actually start.
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
        cur_is_launch_ = (cur_cmd_.opcode == OP_LAUNCH ||
                          cur_cmd_.opcode == OP_LAUNCH_QMD);
        switch (cur_cmd_.opcode) {
            case OP_NOP: case OP_FENCE:
                // No resource bid for these opcodes; retire as NOP.
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
                // QMD launch: the KMU descriptor lives in memory as a
                // {count, (dcr_addr,value)...} list (NVIDIA QMD model). Apply
                // it through the DCR-write hook, then pulse start exactly like
                // a plain CMD_LAUNCH — collapsing ~18 ring DCR writes to one.
                if (cur_cmd_.opcode == OP_LAUNCH_QMD)
                    apply_qmd_(cur_cmd_.arg0);
                launch_state_ = LaunchState::PulseStart;
                eng_state_    = EngState::WaitDone;
            } else if (cur_cmd_.opcode == OP_DRAW) {
                // Device-orchestrated draw: arg0 → resident draw descriptor
                // {uint32 num_steps, steps[28 B]...}. Walk the embedded command
                // bundle (DrawStep), retiring the one OP_DRAW when drained.
                draw_phys_ = cp_translate(cur_cmd_.arg0, /*physical=*/false);
                draw_num_steps_ = 0;
                if (hooks_.dram_read)
                    hooks_.dram_read(draw_phys_, &draw_num_steps_,
                                     sizeof(draw_num_steps_));
                if (draw_num_steps_ > MAX_DRAW_STEPS)
                    draw_num_steps_ = MAX_DRAW_STEPS;
                draw_step_ = 0;
                eng_state_ = EngState::DrawStep;
            } else if (cur_cmd_.opcode == OP_DCR_WRITE) {
                // Issue the DCR write through the hook and retire immediately.
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
            } else if (cur_cmd_.opcode == OP_CACHE_FLUSH) {
                // Sweep a per-core DCR-read of VX_DCR_BASE_CACHE_FLUSH across
                // [0, num_cores). arg0 carries num_cores; dcr_read routes each
                // call to flush_caches() (drains dcache/L2/L3 to memsim).
                if (hooks_.vortex_dcr_read) {
                    uint32_t n = uint32_t(cur_cmd_.arg0 & 0xFFFFFFFF);
                    for (uint32_t cid = 0; cid < n; ++cid) {
                        (void)hooks_.vortex_dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid);
                    }
                }
                eng_state_ = EngState::Retire;
            } else if (cur_cmd_.opcode == OP_EVENT_SIG) {
                // CMD_EVENT_SIGNAL: write arg1 (8-byte value) to arg0
                // (device counter slot).
                if (hooks_.dram_write) {
                    uint64_t v = cur_cmd_.arg1;
                    hooks_.dram_write(cur_cmd_.arg0, &v, sizeof(v));
                }
                eng_state_ = EngState::Retire;
            } else if (cur_cmd_.opcode == OP_EVENT_WAIT) {
                // CMD_EVENT_WAIT: poll arg0 until it satisfies the wait_op
                // (arg2[1:0]) comparison against arg1; retry each tick.
                if (event_wait_satisfied_()) {
                    eng_state_ = EngState::Retire;
                } else {
                    // Spin in Bid (no state change); re-checked each tick.
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

        case EngState::DrawStep:
            // Walk the draw descriptor one step per tick. Inline ops apply now;
            // a launch step transitions to DrawLaunchWait until its kernel
            // drains (the inter-stage barrier).
            if (draw_step_ >= draw_num_steps_) {
                eng_state_ = EngState::Retire;
                return;
            }
            draw_load_step_();
            if (exec_inline_cmd_(draw_cmd_)) {
                eng_state_ = EngState::DrawLaunchWait;
            } else {
                ++draw_step_;
            }
            return;

        case EngState::DrawLaunchWait:
            // tick_launch advances launch_state_; resume the walk on drain.
            if (launch_state_ != LaunchState::Idle)
                return;
            ++draw_step_;
            eng_state_ = EngState::DrawStep;
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
