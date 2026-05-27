// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"
#include "dispatcher.h"  // dispatcher_get_callbacks — load the backend selected by $VORTEX_DRIVER
#include <vortex.h>  // vx_dump_perf — legacy MPM dumper wrapped by vx_device_dump_perf
#include <VX_types.h>  // VX_MEM_IO_COUT_* (console buffer layout)
#include "common.h"    // ALLOC_BASE_ADDR / GLOBAL_MEM_SIZE / *_SIZE constants
#include "caps.h"      // vortex::load_caps / decode_caps
#ifdef SCOPE
#include "scope.h"   // vx_scope_drain — lossless SCOPE tap-ring drainer
#endif

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace vx {

// Resolve the pinned-region size: compile-time default
// VX_CFG_VM_PINNED_REGION_SIZE, optionally overridden by the
// VORTEX_VM_PINNED_SIZE env var (decimal bytes). Returns 0 when VM is
// disabled at build time (the pinned-region carve-out is a no-op then —
// VX_MEM_PHYS has no effect without VM).
static uint64_t resolve_pinned_size() {
#if VX_CFG_VM_ENABLED
    uint64_t size = (uint64_t)VX_CFG_VM_PINNED_REGION_SIZE;
    if (const char* s = std::getenv("VORTEX_VM_PINNED_SIZE")) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 0);
        if (end != s) size = (uint64_t)v;
    }
    // Bound at GLOBAL_MEM_SIZE/2 so the paged pool always has room.
    const uint64_t user_size = GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR;
    if (size > user_size / 2) size = user_size / 2;
    // Round down to a page so the slab boundary is page-aligned (the
    // VM page-table walker installs leaf PTEs at page granularity).
    size &= ~uint64_t(RAM_PAGE_SIZE - 1);
    return size;
#else
    return 0;
#endif
}

Device::Device(std::unique_ptr<Platform> plat)
    : platform_(std::move(plat)), cycle_freq_hz_(0),
      global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                  RAM_PAGE_SIZE, CACHE_BLOCK_SIZE) {
    // cycle_freq_hz_=0 tells the ns conversion path to use the wall clock.
    // global_mem_ is the device-memory allocator — pure host-side address
    // bookkeeping; the CP DMAs to whatever addresses it hands out.

    // Pinned slab: under VM, carve a fixed low-address region for
    // VX_MEM_PHYS allocations. Reserved out of global_mem_ so the
    // paged-pool side never hands out an address that collides with
    // an identity-mapped pinned buffer. See
    // docs/proposals/gfx_vm_pinned_buffers_proposal.md.
    pinned_size_ = resolve_pinned_size();
    if (pinned_size_ > 0) {
        pinned_base_ = ALLOC_BASE_ADDR;
        // Reserve the slab from global_mem_; the slab is owned by pinned_mem_.
        if (global_mem_.reserve(pinned_base_, pinned_size_) != 0) {
            // Should not fail at ctor time — global_mem_ was just constructed
            // and nothing else has touched it.
            pinned_size_ = 0;
            pinned_base_ = 0;
        } else {
            pinned_mem_.reset(new vortex::MemoryAllocator(
                pinned_base_, pinned_size_,
                RAM_PAGE_SIZE, CACHE_BLOCK_SIZE));
        }
    }
}

Device::~Device() {
    // Release whatever default-queue / last-event the legacy wrapper holds.
    if (legacy_last_)   { legacy_last_->release();   legacy_last_   = nullptr; }
    if (legacy_q_)      { legacy_q_->release();      legacy_q_      = nullptr; }
    // Drain the kernel-args scratch pool.
    {
        std::lock_guard<std::mutex> g(args_pool_mu_);
        for (uint64_t addr : args_pool_free_)
            this->mem_free(addr);
        args_pool_free_.clear();
    }
    // Release the CP ring / head / completion host buffers.
    if (cp_ring_.cp_addr) host_free(cp_ring_.cp_addr);
    if (cp_head_.cp_addr) host_free(cp_head_.cp_addr);
    if (cp_cmpl_.cp_addr) host_free(cp_cmpl_.cp_addr);
    // Queues / buffers are torn down by their own refcount path; this
    // just detaches the device backlinks.
    std::lock_guard<std::mutex> g(mu_);
    queues_.clear();
    buffers_.clear();
}

// ============================================================================
// Phase 2 — kernel-args scratch pool
// ============================================================================

vx_result_t Device::args_slot_acquire(uint64_t size, uint64_t* out_addr,
                                      bool* out_pooled) {
    if (!out_addr || !out_pooled) return VX_ERR_INVALID_VALUE;
    if (size > ARGS_SLOT_SIZE) {
        // Oversized args block — one-off allocation, not pooled.
        *out_pooled = false;
        return this->mem_alloc(size, /*VX_MEM_READ*/ 0x1, out_addr);
    }
    *out_pooled = true;
    {
        std::lock_guard<std::mutex> g(args_pool_mu_);
        if (!args_pool_free_.empty()) {
            *out_addr = args_pool_free_.back();
            args_pool_free_.pop_back();
            return VX_SUCCESS;
        }
    }
    // Pool empty — allocate a fresh pooled slot (recycled on release).
    return this->mem_alloc(ARGS_SLOT_SIZE, /*VX_MEM_READ*/ 0x1, out_addr);
}

void Device::args_slot_release(uint64_t addr, bool pooled) {
    if (!pooled) {
        this->mem_free(addr);
        return;
    }
    std::lock_guard<std::mutex> g(args_pool_mu_);
    args_pool_free_.push_back(addr);
}

vx_result_t Device::open(uint32_t index, Device** out) {
    if (!out) return VX_ERR_INVALID_VALUE;
    if (index != 0) return VX_ERR_INVALID_VALUE;   // one device per backend

    const callbacks_t* cb = nullptr;
    auto r = dispatcher_get_callbacks(&cb);
    if (r != VX_SUCCESS) return r;

    void* dev_ctx = nullptr;
    if (cb->dev_open(&dev_ctx) != 0)
        return VX_ERR_DEVICE_LOST;

    std::unique_ptr<Platform> plat(new CallbacksAdapter(*cb, dev_ctx));
    Device* d = new Device(std::move(plat));
    auto cr = d->cp_init();
    if (cr != VX_SUCCESS) {
        d->release();
        return cr;
    }
    *out = d;
    return VX_SUCCESS;
}

// ============================================================================
// Command Processor submission path. One source of truth for the CP wire
// protocol — every backend goes through this code via platform()->cp_reg_*
// (the register channel) and host_alloc/host_free (CP-visible host memory).
// The runtime writes commands straight through the ring's host pointer; the
// CP fetches and executes them.
// ============================================================================

namespace {
// CP regfile offsets (CP-internal; backends translate to physical addrs).
// Matches VX_cp_axil_regfile.
constexpr uint32_t CP_REG_CTRL          = 0x000;
constexpr uint32_t CP_DEV_CAPS          = 0x008;  // {VM_ENABLED@24|TID|RING|NQ}
constexpr uint32_t CP_Q_RING_BASE_LO    = 0x100;
constexpr uint32_t CP_Q_RING_BASE_HI    = 0x104;
constexpr uint32_t CP_Q_HEAD_ADDR_LO    = 0x108;
constexpr uint32_t CP_Q_HEAD_ADDR_HI    = 0x10C;
constexpr uint32_t CP_Q_CMPL_ADDR_LO    = 0x110;
constexpr uint32_t CP_Q_CMPL_ADDR_HI    = 0x114;
constexpr uint32_t CP_Q_RING_SIZE_LOG2  = 0x118;
constexpr uint32_t CP_Q_CONTROL         = 0x11C;
constexpr uint32_t CP_Q_TAIL_LO         = 0x120;
constexpr uint32_t CP_Q_TAIL_HI         = 0x124;
constexpr uint32_t CP_Q_SEQNUM          = 0x128;
constexpr uint32_t CP_Q_LAST_DCR_RSP    = 0x130;
constexpr uint32_t CP_SATP_LO           = 0x028;  // CP DMA MMU page-table root
constexpr uint32_t CP_SATP_HI           = 0x02C;

// CMD_MEM_* header flag (cmd_t.flags bit2 = F_MEM_PHYSICAL): device operand
// is physical — the MMU-aware CP DMA skips translation. Mirrors
// cmd_processor.h MEM_FLAG_PHYSICAL and VX_cp_pkg.sv F_MEM_PHYSICAL.
constexpr uint8_t  CP_MEM_FLAG_PHYSICAL = 0x04;

constexpr uint32_t CP_RING_SIZE_LOG2 = 16;       // 64 KiB
constexpr uint32_t CP_RING_SIZE      = 1u << CP_RING_SIZE_LOG2;
constexpr uint8_t  CP_OPCODE_MEM_WRITE  = 0x01;
constexpr uint8_t  CP_OPCODE_MEM_READ   = 0x02;
constexpr uint8_t  CP_OPCODE_MEM_COPY   = 0x03;
constexpr uint8_t  CP_OPCODE_DCR_WR     = 0x04;
constexpr uint8_t  CP_OPCODE_DCR_RD     = 0x05;
constexpr uint8_t  CP_OPCODE_LAUNCH     = 0x06;
constexpr uint8_t  CP_OPCODE_EVT_SIG    = 0x08;
constexpr uint8_t  CP_OPCODE_EVT_WAIT   = 0x09;
constexpr uint8_t  CP_OPCODE_CACHE_FLUSH= 0x0A;
constexpr std::size_t CP_CL_BYTES    = 64;

// CMD_EVENT_WAIT comparison operations (encoded in arg2[1:0]).
// Mirrors hw/rtl/cp/VX_cp_pkg.sv:wait_op_e.
constexpr uint8_t  CP_WAIT_OP_EQ = 0;
constexpr uint8_t  CP_WAIT_OP_GE = 1;
constexpr uint8_t  CP_WAIT_OP_GT = 2;
constexpr uint8_t  CP_WAIT_OP_NE = 3;

} // namespace

// VMManager's device-memory port: PA-direct page-table I/O through the CP
// DMA. The `physical` flag bypasses the CP DMA's VA translation, so the
// page-table region itself is written/read at its true physical address.
class Device::CpMemIO : public vortex::DeviceMemIO {
public:
    explicit CpMemIO(Device* dev) : dev_(dev) {}
    void read(void* dst, uint64_t addr, size_t size) override {
        dev_->cp_submit_mem_read(dst, addr, size, /*physical=*/true);
    }
    void write(const void* src, uint64_t addr, size_t size) override {
        dev_->cp_submit_mem_write(addr, src, size, /*physical=*/true);
    }
private:
    Device* dev_;
};

vx_result_t Device::cp_init() {
    // Ring + head + completion live in CP-visible host memory: the runtime
    // appends commands straight through the ring's host pointer and the CP
    // fetches them over its m_axi_host master — no per-command DMA.
    auto* p = platform();
    auto r = host_alloc(CP_RING_SIZE, &cp_ring_);
    if (r != VX_SUCCESS) return r;
    r = host_alloc(CP_CL_BYTES, &cp_head_);
    if (r != VX_SUCCESS) return r;
    r = host_alloc(CP_CL_BYTES, &cp_cmpl_);
    if (r != VX_SUCCESS) return r;

    // Zero them so the CP doesn't read stale data on first fetch.
    std::memset(cp_ring_.host_ptr, 0, CP_RING_SIZE);
    std::memset(cp_head_.host_ptr, 0, CP_CL_BYTES);
    std::memset(cp_cmpl_.host_ptr, 0, CP_CL_BYTES);

    // Program CP queue 0. Any failure here is fatal — the CP regfile is
    // the sole control path, so a botched setup means cp_enabled_=true
    // would lie about a working device. Macro keeps the call list legible.
    #define CP_WR(_off, _val) do {                                          \
        auto _r = p->cp_reg_write((_off), (_val));                          \
        if (_r != VX_SUCCESS) return _r;                                    \
    } while (0)
    CP_WR(CP_Q_RING_BASE_LO,   uint32_t(cp_ring_.cp_addr & 0xFFFFFFFFu));
    CP_WR(CP_Q_RING_BASE_HI,   uint32_t(cp_ring_.cp_addr >> 32));
    CP_WR(CP_Q_HEAD_ADDR_LO,   uint32_t(cp_head_.cp_addr & 0xFFFFFFFFu));
    CP_WR(CP_Q_HEAD_ADDR_HI,   uint32_t(cp_head_.cp_addr >> 32));
    CP_WR(CP_Q_CMPL_ADDR_LO,   uint32_t(cp_cmpl_.cp_addr & 0xFFFFFFFFu));
    CP_WR(CP_Q_CMPL_ADDR_HI,   uint32_t(cp_cmpl_.cp_addr >> 32));
    CP_WR(CP_Q_RING_SIZE_LOG2, CP_RING_SIZE_LOG2);
    CP_WR(CP_Q_CONTROL,        0x1);
    CP_WR(CP_REG_CTRL,         0x1);

    cp_enabled_ = true;

    // Discover virtual memory from the device, never from a compile-time
    // #ifdef: the CP publishes a VM_ENABLED bit in DEV_CAPS (bit 24). The
    // generic libvortex.so dispatcher reads it once here — exactly as a
    // real GPU driver queries "does this device have an MMU?".
    {
        uint32_t dev_caps = 0;
        if (p->cp_reg_read(CP_DEV_CAPS, &dev_caps) != VX_SUCCESS)
            return VX_ERR_DEVICE_LOST;
        vm_enabled_ = (dev_caps & (1u << 24)) != 0;
    }

    if (vm_enabled_) {
        // Virtual memory: build the page tables (the host driver's job) and
        // program the CP DMA's MMU with the page-table root. After this,
        // mem_alloc mints VAs and the CP DMA translates VA->PA per CMD_MEM_*.
        //
        // First carve the page-table region out of the device-memory
        // allocator. VMManager keeps its PT pages in a separate allocator
        // over [VX_MEM_PAGE_TABLE_BASE_ADDR, +VX_VM_PT_SIZE_LIMIT); that
        // range lies inside global_mem_, so without this reserve a later
        // mem_alloc could hand back a buffer PA that overlaps the page
        // tables. Done before any mem_alloc is reachable (still in open()).
        {
            std::lock_guard<std::mutex> g(mu_);
            if (global_mem_.reserve(VX_MEM_PAGE_TABLE_BASE_ADDR,
                                    VX_VM_PT_SIZE_LIMIT) != 0)
                return VX_ERR_DEVICE_LOST;
        }
        vm_io_  = std::unique_ptr<CpMemIO>(new CpMemIO(this));
        vm_mgr_ = std::unique_ptr<vortex::VMManager>(
                      new vortex::VMManager(vm_io_.get()));
        if (vm_mgr_->init() != 0)
            return VX_ERR_DEVICE_LOST;
        const uint64_t satp = vm_mgr_->satp();
        CP_WR(CP_SATP_LO, uint32_t(satp & 0xFFFFFFFFu));
        CP_WR(CP_SATP_HI, uint32_t(satp >> 32));
    }
    #undef CP_WR

    // Zero the COUT stream-ring metadata (wr[]/rd[]/lost[]) so the first
    // drain sees empty rings and a zero overflow baseline. data[] is
    // overwritten by vx_putchar before being read by drain_cout, so we
    // skip it: zero only wr[] (offset 0), rd[] (offset SLOTS*4), and
    // lost[] (offset SLOTS*8 + SLOTS*RING). Routed via dev_write — on a
    // CP-only-DMA backend this is a CP transfer, so it must follow CP enable.
    {
        constexpr uint32_t SLOTS = VX_MEM_IO_COUT_SLOTS;
        constexpr uint32_t RING  = VX_MEM_IO_COUT_RING;
        std::vector<uint8_t> zeros_meta(SLOTS * 4, 0);
        // wr[] + rd[] are contiguous at the start of the region.
        r = dev_write(VX_MEM_IO_COUT_ADDR,
                      std::vector<uint8_t>(SLOTS * 8, 0).data(),
                      SLOTS * 8);
        if (r != VX_SUCCESS) return r;
        // lost[] sits past data[].
        const uint64_t LOST_BASE = VX_MEM_IO_COUT_ADDR
                                 + uint64_t(SLOTS) * 8
                                 + uint64_t(SLOTS) * RING;
        r = dev_write(LOST_BASE, zeros_meta.data(), zeros_meta.size());
        if (r != VX_SUCCESS) return r;
    }

    return VX_SUCCESS;
}

vx_result_t Device::cp_submit_cl_(const void* cl) {
    auto* p = platform();
    uint64_t target;
    {
        // Phase 1c: holding cp_mu_ across the SEQNUM poll deadlocks when a
        // CP-side CMD_EVENT_WAIT (which spins until its slot is signaled)
        // is at the head of the ring — concurrent submitters can't post the
        // SIGNAL that would unblock the WAIT. Release the mutex after the
        // ring/state mutation + TAIL commit, then poll without it.
        std::lock_guard<std::mutex> g(cp_mu_);

        // 1) Write one CL into the ring at the current tail — a plain
        //    memcpy through the ring's CP-visible host pointer.
        const uint64_t ring_off = cp_tail_ & (CP_RING_SIZE - 1);
        if (ring_off + CP_CL_BYTES > CP_RING_SIZE)
            return VX_ERR_INVALID_VALUE;  // mid-CL ring wrap not yet supported
        std::memcpy(static_cast<uint8_t*>(cp_ring_.host_ptr) + ring_off,
                    cl, CP_CL_BYTES);

        // 2) Bump tail + reserve our seqnum slot atomically, capture target.
        cp_tail_           += CP_CL_BYTES;
        cp_expected_seqnum_ += 1;
        target = cp_expected_seqnum_;

        // Release fence between the ring memcpy and the doorbell MMIO so
        // the CP cannot read a stale ring entry. The MMIO write is a
        // serializing UC store on x86 (sfence-equivalent), but on ARM /
        // RISC-V and on shells that map host_only BOs WB this fence is
        // required for correctness. Cheap on x86; matters everywhere else.
        std::atomic_thread_fence(std::memory_order_release);

        // 3) Commit the new tail. Atomic-pair: LO stages, HI commits both.
        auto r = p->cp_reg_write(CP_Q_TAIL_LO, uint32_t(cp_tail_ & 0xFFFFFFFFu));
        if (r != VX_SUCCESS) return r;
        r = p->cp_reg_write(CP_Q_TAIL_HI, uint32_t(cp_tail_ >> 32));
        if (r != VX_SUCCESS) return r;
    }   // release cp_mu_ — another submitter can now post its own command

    // 4) Poll Q_SEQNUM. Reacquire cp_mu_ around each individual MMIO read
    // so simx's tick() (which mutates simulator state) and concurrent
    // posts from other queues don't race; this still leaves a window
    // between iterations for other submitters to come in.
    for (;;) {
        uint32_t seqnum32 = 0;
        vx_result_t r;
        {
            std::lock_guard<std::mutex> g(cp_mu_);
            r = p->cp_reg_read(CP_Q_SEQNUM, &seqnum32);
        }
        if (r != VX_SUCCESS) return r;
        if (uint64_t(seqnum32) >= target) return VX_SUCCESS;
        // COUT is drained post-launch only (see cp_submit_launch). The CP
        // ring is serial — a COUT CMD_MEM_READ posted here would queue
        // behind the very command being waited on, so mid-launch draining
        // is impossible until the CP grows a second queue (Track 2). A
        // kernel that overruns its lossless COUT ring within one launch
        // therefore back-pressures until the launch ends.
    #ifdef SCOPE
        // Same discipline for the SCOPE tap rings: drain continuously so
        // the on-chip ring pauses capture only briefly. Best-effort.
        (void)vx_scope_drain();
    #endif
        // No host sleep: each MMIO read already ticks sim cycles.
    }
}

vx_result_t Device::cp_submit_dcr_write(uint32_t addr, uint32_t value) {
    // VM safety check (R1 in gfx_vm_pinned_buffers_proposal.md): when VM
    // is active and the pinned slab is configured, every HW-addr DCR
    // must reference a buffer that lives inside the slab — the HW
    // master at the other end bypasses the per-core MMU and would
    // otherwise dereference a stale VA.
    //
    // The TEX / RASTER / OM address DCRs share a single encoding:
    // value is the cache-block index, i.e. pa = (value << 6). DXA's
    // BASE_LO/HI pair is split across two writes — validating that
    // pairing belongs in the DXA helper, not here, so it is excluded
    // from this check for now (tracked as future work).
    if (vm_enabled_ && pinned_mem_) {
        switch (addr) {
        case VX_DCR_TEX_ADDR:
        case VX_DCR_RASTER_TBUF_ADDR:
        case VX_DCR_RASTER_PBUF_ADDR:
        case VX_DCR_OM_CBUF_ADDR:
        case VX_DCR_OM_ZBUF_ADDR: {
            const uint64_t pa = uint64_t(value) << 6;
            if (pa < pinned_base_ ||
                pa >= pinned_base_ + pinned_size_) {
                std::cerr << "[VXDRV] dcr 0x" << std::hex << addr
                          << " value 0x" << value << " (pa 0x" << pa
                          << ") not in pinned slab [0x" << pinned_base_
                          << ", 0x" << (pinned_base_ + pinned_size_)
                          << ") — buffer needs VX_MEM_PHYS"
                          << std::dec << std::endl;
                return VX_ERR_INVALID_VALUE;
            }
            break;
        }
        default:
            break;
        }
    }
    // CMD_DCR_WRITE on-wire layout (cmd_size=20):
    //   bytes 0..3   header  { opcode=0x04, flags=0, reserved=0 }
    //   bytes 4..11  arg0    DCR addr
    //   bytes 12..19 arg1    DCR value
    // Rest of CL is padded with zeros (NOP sentinel for the unpacker).
    uint8_t cl[CP_CL_BYTES] = {0};
    uint32_t* p32 = reinterpret_cast<uint32_t*>(cl);
    p32[0] = CP_OPCODE_DCR_WR;
    p32[1] = addr;
    p32[3] = value;
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_launch() {
    // CMD_LAUNCH on-wire layout (cmd_size=12):
    //   bytes 0..3   header  { opcode=0x06, flags=0, reserved=0 }
    //   bytes 4..11  arg0    unused by VX_cp_launch
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = CP_OPCODE_LAUNCH;
    auto r = cp_submit_cl_(cl);
    if (r != VX_SUCCESS) return r;
    // Cache coherence: post an explicit cache flush right after the launch
    // (AMD ACQUIRE_MEM model) so the host observes coherent kernel results.
    r = cp_submit_cache_flush();
    if (r != VX_SUCCESS) return r;
    // Final COUT drain: the flush has made the kernel's writes coherent, so
    // the tail-end console output left in the rings is now safe to read.
    return drain_cout();
}

vx_result_t Device::cp_submit_cache_flush() {
    // CMD_CACHE_FLUSH on-wire layout (cmd_size=12):
    //   bytes 0..3   header  { opcode=0x0A, flags=0, reserved=0 }
    //   bytes 4..11  arg0    number of cores to flush
    // The CP sweeps a per-core flush DCR-read across [0, num_cores) and
    // retires the command only when the last core's flush completes.
    // No-op on write-through cache configs (the Vortex default).
    if (cp_num_cores_ == 0) {
        auto r = this->query_caps(VX_CAPS_NUM_CORES, &cp_num_cores_);
        if (r != VX_SUCCESS) return r;
    }
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = CP_OPCODE_CACHE_FLUSH;
    std::memcpy(cl + 4, &cp_num_cores_, sizeof(cp_num_cores_));
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_dcr_read(uint32_t addr, uint32_t tag,
                                       uint32_t* out_value) {
    if (!out_value) return VX_ERR_INVALID_VALUE;
    // CMD_DCR_READ on-wire layout (cmd_size=20):
    //   bytes 0..3   header  { opcode=0x05, flags=0, reserved=0 }
    //   bytes 4..11  arg0    DCR addr (low 12 bits used)
    //   bytes 12..19 arg1    tag (data on the DCR bus; e.g. core index
    //                        for VX_DCR_BASE_CACHE_FLUSH)
    uint8_t cl[CP_CL_BYTES] = {0};
    uint32_t* p32 = reinterpret_cast<uint32_t*>(cl);
    p32[0] = CP_OPCODE_DCR_RD;
    p32[1] = addr;
    p32[3] = tag;
    auto r = cp_submit_cl_(cl);
    if (r != VX_SUCCESS) return r;
    // Pick up the response from the CP regfile: VX_cp_dcr_proxy latches
    // it on Q_LAST_DCR_RSP at the same offset as the engine's retire.
    return platform()->cp_reg_read(CP_Q_LAST_DCR_RSP, out_value);
}

vx_result_t Device::cp_submit_event_signal(uint64_t event_dev_addr,
                                           uint64_t value) {
    // CMD_EVENT_SIGNAL on-wire layout (cmd_size=20):
    //   bytes 0..3   header  { opcode=0x08, flags=0, reserved=0 }
    //   bytes 4..11  arg0    device byte address of 8-byte counter slot
    //   bytes 12..19 arg1    64-bit value to write
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = CP_OPCODE_EVT_SIG;
    std::memcpy(cl + 4,  &event_dev_addr, sizeof(event_dev_addr));
    std::memcpy(cl + 12, &value,          sizeof(value));
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_event_wait(uint64_t event_dev_addr,
                                         uint64_t value) {
    // CMD_EVENT_WAIT on-wire layout (cmd_size=28):
    //   bytes 0..3   header  { opcode=0x09, flags=0, reserved=0 }
    //   bytes 4..11  arg0    device byte address of 8-byte counter slot
    //   bytes 12..19 arg1    target value
    //   bytes 20..27 arg2    wait_op (low 2 bits) — we always submit GE
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = CP_OPCODE_EVT_WAIT;
    std::memcpy(cl + 4,  &event_dev_addr, sizeof(event_dev_addr));
    std::memcpy(cl + 12, &value,          sizeof(value));
    uint64_t op = CP_WAIT_OP_GE;
    std::memcpy(cl + 20, &op, sizeof(op));
    return cp_submit_cl_(cl);
}

// ============================================================================
// CP-driven host<->device DMA (CMD_MEM_*)
// ============================================================================

vx_result_t Device::cp_submit_mem_(uint8_t opcode, uint64_t arg0,
                                   uint64_t arg1, uint64_t arg2,
                                   bool physical) {
    // CMD_MEM_* on-wire layout (cmd_size=28):
    //   bytes 0..3   header  { opcode, flags, reserved=0 }
    //   bytes 4..11  arg0    dst address
    //   bytes 12..19 arg1    src address
    //   bytes 20..27 arg2    size in bytes
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = opcode;
    cl[1] = physical ? CP_MEM_FLAG_PHYSICAL : 0;   // skip CP-DMA VM translation
    std::memcpy(cl + 4,  &arg0, sizeof(arg0));
    std::memcpy(cl + 12, &arg1, sizeof(arg1));
    std::memcpy(cl + 20, &arg2, sizeof(arg2));
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_mem_copy(uint64_t dst, uint64_t src,
                                       uint64_t size) {
    if (size == 0 || dst == src) return VX_SUCCESS;
    return cp_submit_mem_(CP_OPCODE_MEM_COPY, dst, src, size);
}

vx_result_t Device::cp_submit_mem_write(uint64_t dev_dst, const void* host_src,
                                        uint64_t size, bool physical) {
    if (size == 0)  return VX_SUCCESS;
    if (!host_src)  return VX_ERR_INVALID_VALUE;
    // Stage the payload into CP-visible host memory (a plain memcpy through
    // the host pointer), then have the CP DMA it to device memory. `physical`
    // (set for page-table writes) tells the CP DMA to skip VM translation.
    HostMem staging;
    auto r = host_alloc(size, &staging);
    if (r != VX_SUCCESS) return r;
    std::memcpy(staging.host_ptr, host_src, size);
    r = cp_submit_mem_(CP_OPCODE_MEM_WRITE, dev_dst, staging.cp_addr, size,
                       physical);
    host_free(staging.cp_addr);
    return r;
}

vx_result_t Device::cp_submit_mem_read(void* host_dst, uint64_t dev_src,
                                       uint64_t size, bool physical) {
    if (size == 0)  return VX_SUCCESS;
    if (!host_dst)  return VX_ERR_INVALID_VALUE;
    // Have the CP DMA device->host into a CP-visible host staging buffer,
    // then memcpy it back to the caller's pointer.
    HostMem staging;
    auto r = host_alloc(size, &staging);
    if (r != VX_SUCCESS) return r;
    r = cp_submit_mem_(CP_OPCODE_MEM_READ, staging.cp_addr, dev_src, size,
                       physical);
    if (r == VX_SUCCESS)
        std::memcpy(host_dst, staging.host_ptr, size);
    host_free(staging.cp_addr);
    return r;
}

// ============================================================================
// CP-visible host memory + device-memory allocation + caps (common core).
// ============================================================================

vx_result_t Device::host_alloc(uint64_t size, HostMem* out) {
    void* hp = nullptr;
    uint64_t ca = 0;
    auto r = platform_->host_mem_alloc(size, &hp, &ca);
    if (r != VX_SUCCESS) return r;
    HostMem hm{hp, ca, size};
    {
        std::lock_guard<std::mutex> g(mu_);
        host_mems_[ca] = hm;
    }
    *out = hm;
    return VX_SUCCESS;
}

void Device::host_free(uint64_t cp_addr) {
    {
        std::lock_guard<std::mutex> g(mu_);
        host_mems_.erase(cp_addr);
    }
    platform_->host_mem_free(cp_addr);
}

vx_result_t Device::mem_alloc(uint64_t size, uint32_t flags,
                              uint64_t* out_addr) {
    if (!out_addr || size == 0) return VX_ERR_INVALID_VALUE;
    if (flags & VX_MEM_HOST) {
        // CP-visible host memory — the cp_addr doubles as the handle.
        HostMem hm;
        auto r = host_alloc(size, &hm);
        if (r != VX_SUCCESS) return r;
        *out_addr = hm.cp_addr;
        return VX_SUCCESS;
    }
    const uint64_t asize =
        (size + CACHE_BLOCK_SIZE - 1) & ~uint64_t(CACHE_BLOCK_SIZE - 1);
    // Route PHYS allocations to the pinned slab when configured; everything
    // else (and PHYS allocations when the slab is disabled) comes from
    // global_mem_. PHYS exhaustion is a hard fail (VX_ERR_OUT_OF_DEVICE_MEMORY)
    // — silent fallback to the paged pool would resurrect the fragmentation
    // problem the slab is meant to solve. See
    // docs/proposals/gfx_vm_pinned_buffers_proposal.md §"Pre-allocated …".
    const bool from_pinned = (flags & VX_MEM_PHYS) && pinned_mem_;
    {
        std::lock_guard<std::mutex> g(mu_);
        auto& alloc = from_pinned ? *pinned_mem_ : global_mem_;
        if (alloc.allocate(asize, out_addr) != 0)
            return VX_ERR_OUT_OF_DEVICE_MEMORY;
    }
    // VM: *out_addr is a PA. VX_MEM_PHYS keeps it (identity-mapped so the
    // kernel reaches it at VA==PA); otherwise mint a fresh VA + install
    // PTEs so the kernel's MMU and the CP DMA both translate it. vm_mgr_ is
    // non-null iff the device reported an MMU at vx_device_open.
    if (vm_mgr_) {
        std::lock_guard<std::mutex> g(vm_mu_);
        int rc = (flags & VX_MEM_PHYS)
                   ? vm_mgr_->install_identity_map(*out_addr, asize)
                   : vm_mgr_->phy_to_virt_map(asize, out_addr, flags);
        if (rc != 0) return VX_ERR_INVALID_VALUE;
    }
    return VX_SUCCESS;
}

vx_result_t Device::mem_reserve(uint64_t addr, uint64_t size, uint32_t flags) {
    (void)flags;
    if (size == 0) return VX_ERR_INVALID_VALUE;
    const uint64_t asize =
        (size + CACHE_BLOCK_SIZE - 1) & ~uint64_t(CACHE_BLOCK_SIZE - 1);
    // Caller-chosen PA: dispatch by address to whichever allocator owns
    // the range. The pinned slab sits at [pinned_base_, pinned_base_ +
    // pinned_size_); everything else is in global_mem_. reserve() on the
    // wrong pool would unconditionally fail, so picking the right one
    // here lets reservations into the pinned slab succeed.
    const bool in_pinned = pinned_mem_
        && addr >= pinned_base_
        && addr + asize <= pinned_base_ + pinned_size_;
    {
        std::lock_guard<std::mutex> g(mu_);
        auto& alloc = in_pinned ? *pinned_mem_ : global_mem_;
        if (alloc.reserve(addr, asize) != 0)
            return VX_ERR_INVALID_VALUE;
    }
    // VM: a reserved region sits at a caller-chosen PA — identity-map it
    // (VA == PA) so the kernel reaches it at the same address.
    if (vm_mgr_) {
        std::lock_guard<std::mutex> g(vm_mu_);
        if (vm_mgr_->install_identity_map(addr, asize) != 0)
            return VX_ERR_INVALID_VALUE;
    }
    return VX_SUCCESS;
}

vx_result_t Device::mem_free(uint64_t addr) {
    bool is_host;
    {
        std::lock_guard<std::mutex> g(mu_);
        is_host = host_mems_.count(addr) != 0;
    }
    if (is_host) {
        host_free(addr);
        return VX_SUCCESS;
    }
    // VM: `addr` is a VA — resolve to the PA the allocator handed out.
    if (vm_mgr_) {
        std::lock_guard<std::mutex> g(vm_mu_);
        try { addr = vm_mgr_->page_table_walk(addr); }
        catch (...) { /* already unmapped — fall through to release */ }
    }
    // Dispatch by resolved PA to whichever pool owns the range. PHYS
    // buffers are identity-mapped so their PA falls inside the slab;
    // non-PHYS buffers live in global_mem_.
    std::lock_guard<std::mutex> g(mu_);
    const bool in_pinned = pinned_mem_
        && addr >= pinned_base_
        && addr < pinned_base_ + pinned_size_;
    auto& alloc = in_pinned ? *pinned_mem_ : global_mem_;
    return (alloc.release(addr) == 0)
               ? VX_SUCCESS : VX_ERR_INVALID_VALUE;
}

vx_result_t Device::memory_info(uint64_t* out_free, uint64_t* out_used) {
    std::lock_guard<std::mutex> g(mu_);
    // Report combined paged + pinned counters — callers (mesa, hipcc)
    // typically want the device-wide total. Per-pool inspection is
    // available via VX_CAPS_VM_PINNED_SIZE / _FREE.
    const uint64_t pinned_free = pinned_mem_ ? pinned_mem_->free()      : 0;
    const uint64_t pinned_used = pinned_mem_ ? pinned_mem_->allocated() : 0;
    if (out_free) *out_free = global_mem_.free() + pinned_free;
    if (out_used) *out_used = global_mem_.allocated() + pinned_used;
    return VX_SUCCESS;
}

vx_result_t Device::query_caps(uint32_t caps_id, uint64_t* out_value) {
    if (!out_value) return VX_ERR_INVALID_VALUE;
    // The two static caps words are read once from the CP regfile caps
    // window; serialize the load against concurrent cp_reg_* traffic.
    if (!caps_loaded_) {
        std::lock_guard<std::mutex> g(cp_mu_);
        if (!caps_loaded_) {
            auto rd = [this](uint32_t off, uint32_t* v) -> int {
                return platform_->cp_reg_read(off, v) == VX_SUCCESS ? 0 : -1;
            };
            if (vortex::load_caps(rd, &dev_caps_, &isa_caps_) != 0)
                return VX_ERR_DEVICE_LOST;
            caps_loaded_ = true;
        }
    }
    if (vortex::decode_caps(dev_caps_, isa_caps_, caps_id, out_value))
        return VX_SUCCESS;
    // Caps not encoded in the CP words — platform/runtime constants.
    switch (caps_id) {
    case VX_CAPS_CACHE_LINE_SIZE: *out_value = CACHE_BLOCK_SIZE;             break;
    case VX_CAPS_GLOBAL_MEM_SIZE: *out_value = GLOBAL_MEM_SIZE;              break;
    case VX_CAPS_CLOCK_RATE:      *out_value = VX_CFG_PLATFORM_CLOCK_RATE;   break;
    case VX_CAPS_PEAK_MEM_BW:     *out_value = VX_CFG_PLATFORM_MEMORY_PEAK_BW; break;
    case VX_CAPS_VM_SUPPORT:      *out_value = vm_enabled_ ? 1 : 0;          break;
    case VX_CAPS_VM_PINNED_SIZE:  *out_value = pinned_size_;                 break;
    case VX_CAPS_VM_PINNED_FREE: {
        std::lock_guard<std::mutex> g(mu_);
        *out_value = pinned_mem_ ? pinned_mem_->free() : 0;
        break;
    }
    default:                      return VX_ERR_INVALID_VALUE;
    }
    return VX_SUCCESS;
}

// ============================================================================
// Device-memory transfer router — see vortex2_internal.h.
// ============================================================================

vx_result_t Device::dev_write(uint64_t dev_addr, const void* src,
                              uint64_t size) {
    return cp_submit_mem_write(dev_addr, src, size);
}

vx_result_t Device::dev_read(void* dst, uint64_t dev_addr, uint64_t size) {
    return cp_submit_mem_read(dst, dev_addr, size);
}

vx_result_t Device::dev_copy(uint64_t dst, uint64_t src, uint64_t size) {
    return cp_submit_mem_copy(dst, src, size);
}

vx_result_t Device::drain_cout() {
    // Lossy COUT (proposal §10, O-1): drain each hart's ring — read wr[]
    // and lost[], copy out [rd,wr) bytes, emit "#slot: <line>", surface any
    // new lost-byte deltas, and publish the advanced rd[]. The kernel-side
    // vx_putchar is non-blocking (drops + bumps lost[slot] on full ring),
    // so this drain has no host/kernel deadlock concern — see the legacy
    // back-pressure spin in vx_print.S which O-1 replaced.
    constexpr uint32_t SLOTS = VX_MEM_IO_COUT_SLOTS;
    constexpr uint32_t RING  = VX_MEM_IO_COUT_RING;
    const uint64_t WR_BASE   = VX_MEM_IO_COUT_ADDR;
    const uint64_t RD_BASE   = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 4;
    const uint64_t DATA_BASE = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 8;
    const uint64_t LOST_BASE = DATA_BASE + uint64_t(SLOTS) * RING;

    // dev_read/dev_write route through cp_submit_* (which take cp_mu_
    // themselves), so this must not hold cp_mu_ — and need not: drain_cout
    // is only ever called post-launch, when the CP ring is otherwise idle.
    uint32_t wr  [SLOTS] = {};
    uint32_t lost[SLOTS] = {};
    auto r = dev_read(wr,   WR_BASE,   sizeof(wr));
    if (r != VX_SUCCESS) return r;
    r = dev_read(lost, LOST_BASE, sizeof(lost));
    if (r != VX_SUCCESS) return r;

    bool advanced = false;
    for (uint32_t s = 0; s < SLOTS; ++s) {
        const uint32_t rd = cout_rd_[s];
        // Surface a per-slot overflow delta: kernel atomically bumps
        // lost[slot] on a full-ring drop, host reports the delta and
        // remembers the latest value so each byte is counted exactly once.
        if (lost[s] != cout_lost_seen_[s]) {
            uint32_t delta = lost[s] - cout_lost_seen_[s];   // wrap-safe (32-bit)
            std::cout << "[#" << s << ": lost " << delta << " bytes]"
                      << std::endl;
            cout_lost_seen_[s] = lost[s];
        }
        if (wr[s] == rd) continue;
        uint32_t n = wr[s] - rd;
        // Defensive cap: the ring is lossy, so wr-rd should never exceed
        // RING. If it ever does (corrupt slot, host missed many polls),
        // drain only the most recent RING bytes and let the producer keep
        // bumping `lost`.
        if (n > RING) n = RING;
        char data[RING];
        r = dev_read(data, DATA_BASE + uint64_t(s) * RING, RING);
        if (r != VX_SUCCESS) return r;
        for (uint32_t i = 0; i < n; ++i) {
            const char c = data[(rd + i) & (RING - 1)];
            cout_line_[s].push_back(c);
            if (c == '\n') {
                std::cout << "#" << s << ": " << cout_line_[s] << std::flush;
                cout_line_[s].clear();
            }
        }
        cout_rd_[s] = wr[s];
        advanced = true;
    }
    if (advanced) {
        r = dev_write(RD_BASE, cout_rd_, sizeof(cout_rd_));
        if (r != VX_SUCCESS) return r;
    }
    return VX_SUCCESS;
}

void Device::register_queue(Queue* q) {
    std::lock_guard<std::mutex> g(mu_);
    queues_.insert(q);
}

void Device::unregister_queue(Queue* q) {
    std::lock_guard<std::mutex> g(mu_);
    queues_.erase(q);
}

void Device::register_buffer(Buffer* b) {
    std::lock_guard<std::mutex> g(mu_);
    buffers_.insert(b);
}

void Device::unregister_buffer(Buffer* b) {
    std::lock_guard<std::mutex> g(mu_);
    buffers_.erase(b);
}

Queue* Device::legacy_default_queue() {
    // Fast path: already created.
    {
        std::lock_guard<std::mutex> g(mu_);
        if (legacy_q_) return legacy_q_;
    }
    // Slow path: create OUTSIDE the lock. Queue::create takes this same
    // mutex via register_queue, so holding it here would block.
    vx_queue_info_t info = {};
    info.struct_size = sizeof(info);
    info.priority    = VX_QUEUE_PRIORITY_NORMAL;
    info.flags       = 0;
    Queue* q = nullptr;
    if (Queue::create(this, &info, &q) != VX_SUCCESS) return nullptr;
    // Publish (and handle race where two threads created queues
    // concurrently — keep one, release the other).
    {
        std::lock_guard<std::mutex> g(mu_);
        if (legacy_q_) {
            q->release();
            return legacy_q_;
        }
        legacy_q_ = q;
    }
    return q;
}

Event* Device::legacy_take_last_event() {
    std::lock_guard<std::mutex> g(mu_);
    Event* ev = legacy_last_;
    legacy_last_ = nullptr;
    return ev;
}

void Device::legacy_remember_last_event(Event* ev) {
    std::lock_guard<std::mutex> g(mu_);
    if (legacy_last_) legacy_last_->release();
    legacy_last_ = ev;   // takes ownership
}

} // namespace vx

// ============================================================================
// C entry points
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_device_count(uint32_t* out_count) {
    if (!out_count) return VX_ERR_INVALID_VALUE;
    *out_count = 1;   // each backend exposes a single device
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_device_open(uint32_t index, vx_device_h* out) {
    VX_C_ENTRY_TRY
    if (!out) return VX_ERR_INVALID_VALUE;
    Device* d = nullptr;
    auto r = Device::open(index, &d);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(d);
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_device_retain(vx_device_h dev) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    to_device(dev)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_device_release(vx_device_h dev) {
    VX_C_ENTRY_TRY
    if (!dev) return VX_ERR_INVALID_HANDLE;
    // ~Device tears down queues/buffers/modules; any of their dtors going
    // off-rails (e.g. an XRT bo cleanup throwing on a dead device) must not
    // propagate across the C boundary.
    to_device(dev)->release();
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_device_query(vx_device_h dev, uint32_t caps_id,
                                       uint64_t* out_value) {
    if (!dev)       return VX_ERR_INVALID_HANDLE;
    if (!out_value) return VX_ERR_INVALID_VALUE;
    return to_device(dev)->query_caps(caps_id, out_value);
}

extern "C" vx_result_t vx_device_memory_info(vx_device_h dev,
                                             uint64_t* free,
                                             uint64_t* used) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    return to_device(dev)->memory_info(free, used);
}

// Formatted MPM performance-counter dump (per core / cluster / cache). The
// counter walk + report formatting already lives in legacy_perf.cpp's
// vx_dump_perf; this is the vortex2.h-shaped wrapper so callers need not
// reach into the legacy surface.
extern "C" vx_result_t vx_device_dump_perf(vx_device_h dev, FILE* stream) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    return (vx_dump_perf(dev, stream) == 0) ? VX_SUCCESS
                                            : VX_ERR_INVALID_VALUE;
}
