// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"
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
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Per-process handle on the dlopened backend library (libvortex-<NAME>.so).
// One backend per process; reused across vx_device_open calls.
void*       g_backend_lib = nullptr;
callbacks_t g_backend_cb  {};

vx_result_t load_backend_once() {
    if (g_backend_lib != nullptr) return VX_SUCCESS;   // already loaded

    const char* drv = std::getenv("VORTEX_DRIVER");
    if (drv == nullptr) drv = "simx";   // default backend
    std::string lib = std::string("libvortex-") + drv + ".so";

    void* h = dlopen(lib.c_str(), RTLD_LAZY);
    if (h == nullptr) {
        std::cerr << "vortex: cannot open backend library '" << lib
                  << "': " << dlerror() << std::endl;
        return VX_ERR_DEVICE_LOST;
    }

    using vx_dev_init_t = int (*)(callbacks_t*);
    auto init = reinterpret_cast<vx_dev_init_t>(dlsym(h, "vx_dev_init"));
    if (init == nullptr) {
        std::cerr << "vortex: backend library '" << lib
                  << "' is missing vx_dev_init: " << dlerror() << std::endl;
        dlclose(h);
        return VX_ERR_DEVICE_LOST;
    }

    if (init(&g_backend_cb) != 0) {
        std::cerr << "vortex: vx_dev_init failed in '" << lib << "'"
                  << std::endl;
        dlclose(h);
        return VX_ERR_DEVICE_LOST;
    }

    g_backend_lib = h;
    return VX_SUCCESS;
}

} // anonymous namespace

namespace vx {

Device::Device(std::unique_ptr<Platform> plat)
    : platform_(std::move(plat)), cycle_freq_hz_(0),
      global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                  RAM_PAGE_SIZE, CACHE_BLOCK_SIZE) {
    // cycle_freq_hz_=0 tells the ns conversion path to use the wall clock.
    // global_mem_ is the device-memory allocator — pure host-side address
    // bookkeeping; the CP DMAs to whatever addresses it hands out.
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

    auto r = load_backend_once();
    if (r != VX_SUCCESS) return r;

    void* dev_ctx = nullptr;
    if (g_backend_cb.dev_open(&dev_ctx) != 0)
        return VX_ERR_DEVICE_LOST;

    std::unique_ptr<Platform> plat(new CallbacksAdapter(g_backend_cb, dev_ctx));
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

    // Program CP queue 0.
    p->cp_reg_write(CP_Q_RING_BASE_LO,   uint32_t(cp_ring_.cp_addr & 0xFFFFFFFFu));
    p->cp_reg_write(CP_Q_RING_BASE_HI,   uint32_t(cp_ring_.cp_addr >> 32));
    p->cp_reg_write(CP_Q_HEAD_ADDR_LO,   uint32_t(cp_head_.cp_addr & 0xFFFFFFFFu));
    p->cp_reg_write(CP_Q_HEAD_ADDR_HI,   uint32_t(cp_head_.cp_addr >> 32));
    p->cp_reg_write(CP_Q_CMPL_ADDR_LO,   uint32_t(cp_cmpl_.cp_addr & 0xFFFFFFFFu));
    p->cp_reg_write(CP_Q_CMPL_ADDR_HI,   uint32_t(cp_cmpl_.cp_addr >> 32));
    p->cp_reg_write(CP_Q_RING_SIZE_LOG2, CP_RING_SIZE_LOG2);
    p->cp_reg_write(CP_Q_CONTROL,        0x1);
    p->cp_reg_write(CP_REG_CTRL,         0x1);

    cp_enabled_ = true;

    // Zero the COUT stream-ring wr[]/rd[] pointers (proposal §10) so the
    // first drain sees empty rings. Routed via dev_write — on a CP-only-DMA
    // backend (opae) this is a CP transfer, so it must follow CP enable.
    std::vector<uint8_t> zeros_cout(VX_MEM_IO_COUT_SLOTS * 8, 0);
    r = dev_write(VX_MEM_IO_COUT_ADDR, zeros_cout.data(), zeros_cout.size());
    if (r != VX_SUCCESS) return r;

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
                                   uint64_t arg1, uint64_t arg2) {
    // CMD_MEM_* on-wire layout (cmd_size=28):
    //   bytes 0..3   header  { opcode, flags=0, reserved=0 }
    //   bytes 4..11  arg0    dst address
    //   bytes 12..19 arg1    src address
    //   bytes 20..27 arg2    size in bytes
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = opcode;
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
                                        uint64_t size) {
    if (size == 0)  return VX_SUCCESS;
    if (!host_src)  return VX_ERR_INVALID_VALUE;
    // Stage the payload into CP-visible host memory (a plain memcpy through
    // the host pointer), then have the CP DMA it to device memory.
    HostMem staging;
    auto r = host_alloc(size, &staging);
    if (r != VX_SUCCESS) return r;
    std::memcpy(staging.host_ptr, host_src, size);
    r = cp_submit_mem_(CP_OPCODE_MEM_WRITE, dev_dst, staging.cp_addr, size);
    host_free(staging.cp_addr);
    return r;
}

vx_result_t Device::cp_submit_mem_read(void* host_dst, uint64_t dev_src,
                                       uint64_t size) {
    if (size == 0)  return VX_SUCCESS;
    if (!host_dst)  return VX_ERR_INVALID_VALUE;
    // Have the CP DMA device->host into a CP-visible host staging buffer,
    // then memcpy it back to the caller's pointer.
    HostMem staging;
    auto r = host_alloc(size, &staging);
    if (r != VX_SUCCESS) return r;
    r = cp_submit_mem_(CP_OPCODE_MEM_READ, staging.cp_addr, dev_src, size);
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
    std::lock_guard<std::mutex> g(mu_);
    return (global_mem_.allocate(asize, out_addr) == 0)
               ? VX_SUCCESS : VX_ERR_OUT_OF_DEVICE_MEMORY;
}

vx_result_t Device::mem_reserve(uint64_t addr, uint64_t size, uint32_t flags) {
    (void)flags;
    if (size == 0) return VX_ERR_INVALID_VALUE;
    const uint64_t asize =
        (size + CACHE_BLOCK_SIZE - 1) & ~uint64_t(CACHE_BLOCK_SIZE - 1);
    std::lock_guard<std::mutex> g(mu_);
    return (global_mem_.reserve(addr, asize) == 0)
               ? VX_SUCCESS : VX_ERR_INVALID_VALUE;
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
    std::lock_guard<std::mutex> g(mu_);
    return (global_mem_.release(addr) == 0)
               ? VX_SUCCESS : VX_ERR_INVALID_VALUE;
}

vx_result_t Device::memory_info(uint64_t* out_free, uint64_t* out_used) {
    std::lock_guard<std::mutex> g(mu_);
    if (out_free) *out_free = global_mem_.free();
    if (out_used) *out_used = global_mem_.allocated();
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
    // Lossless COUT (proposal §10): drain each hart's back-pressured ring
    // — read wr[], copy out [rd,wr), print "#slot: <line>", and publish
    // the advanced rd[] so the kernel sees the freed space. Called every
    // CP launch-wait poll iteration, concurrently with the producing
    // kernel, so the kernel's back-pressure spin never deadlocks.
    constexpr uint32_t SLOTS = VX_MEM_IO_COUT_SLOTS;
    constexpr uint32_t RING  = VX_MEM_IO_COUT_RING;
    const uint64_t WR_BASE   = VX_MEM_IO_COUT_ADDR;
    const uint64_t RD_BASE   = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 4;
    const uint64_t DATA_BASE = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 8;

    // dev_read/dev_write route through cp_submit_* (which take cp_mu_
    // themselves), so this must not hold cp_mu_ — and need not: drain_cout
    // is only ever called post-launch, when the CP ring is otherwise idle.
    uint32_t wr[SLOTS] = {};
    auto r = dev_read(wr, WR_BASE, sizeof(wr));
    if (r != VX_SUCCESS) return r;

    bool advanced = false;
    for (uint32_t s = 0; s < SLOTS; ++s) {
        const uint32_t rd = cout_rd_[s];
        if (wr[s] == rd) continue;
        uint32_t n = wr[s] - rd;
        if (n > RING) n = RING;          // defensive — a lossless ring never overruns
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
    if (advanced)
        dev_write(RD_BASE, cout_rd_, sizeof(cout_rd_));
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
    if (!out) return VX_ERR_INVALID_VALUE;
    Device* d = nullptr;
    auto r = Device::open(index, &d);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(d);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_device_retain(vx_device_h dev) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    to_device(dev)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_device_release(vx_device_h dev) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    to_device(dev)->release();
    return VX_SUCCESS;
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
