// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

#include <cassert>
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
    : platform_(std::move(plat)), cycle_freq_hz_(0) {
    // Future CP-aware backends will report a real cycle frequency; v1 uses 0
    // and the legacy ns conversion path treats 0 as "use wall clock".
}

Device::~Device() {
    // Drop any outstanding default-queue / last-event the legacy wrapper
    // accumulated.
    if (legacy_last_)   { legacy_last_->release();   legacy_last_   = nullptr; }
    if (legacy_q_)      { legacy_q_->release();      legacy_q_      = nullptr; }
    // Queues / buffers are torn down by their own refcount path; this just
    // detaches the device backlinks.
    std::lock_guard<std::mutex> g(mu_);
    queues_.clear();
    buffers_.clear();
}

vx_result_t Device::open(uint32_t index, Device** out) {
    if (!out) return VX_ERR_INVALID_VALUE;
    if (index != 0) return VX_ERR_INVALID_VALUE;   // v1: one device per backend

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
// Command Processor submission path (Phase C of cp_pure_v2_callbacks_proposal).
// One source of truth for the CP wire protocol — every backend goes through
// this code via platform()->cp_mmio_*  +  platform()->mem_upload.
// ============================================================================

namespace {
// CP regfile offsets (CP-internal; backends translate to physical addrs).
// Mirrors VX_cp_axil_regfile §17.4.
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
constexpr uint8_t  CP_OPCODE_DCR_WR  = 0x04;
constexpr uint8_t  CP_OPCODE_DCR_RD  = 0x05;
constexpr uint8_t  CP_OPCODE_LAUNCH  = 0x06;
constexpr std::size_t CP_CL_BYTES    = 64;

} // namespace

vx_result_t Device::cp_init() {
    // Allocate ring + head + completion slots in device memory.
    // VX_MEM_READ flag for ring (CP reads from it), VX_MEM_WRITE for
    // head + cmpl (CP writes seqnum/head pointers there).
    auto* p = platform();
    auto r = p->mem_alloc(CP_RING_SIZE, /*VX_MEM_READ*/ 0x1, &cp_ring_dev_addr_);
    if (r != VX_SUCCESS) return r;
    r = p->mem_alloc(CP_CL_BYTES, /*VX_MEM_WRITE*/ 0x2, &cp_head_dev_addr_);
    if (r != VX_SUCCESS) return r;
    r = p->mem_alloc(CP_CL_BYTES, /*VX_MEM_WRITE*/ 0x2, &cp_cmpl_dev_addr_);
    if (r != VX_SUCCESS) return r;

    // Zero them so CP doesn't read stale data on first fetch.
    std::vector<uint8_t> zeros_cl(CP_CL_BYTES, 0);
    std::vector<uint8_t> zeros_ring(CP_RING_SIZE, 0);
    p->mem_upload(cp_ring_dev_addr_, zeros_ring.data(), CP_RING_SIZE);
    p->mem_upload(cp_head_dev_addr_, zeros_cl.data(), CP_CL_BYTES);
    p->mem_upload(cp_cmpl_dev_addr_, zeros_cl.data(), CP_CL_BYTES);

    // Program CP queue 0.
    p->cp_mmio_write(CP_Q_RING_BASE_LO,   uint32_t(cp_ring_dev_addr_ & 0xFFFFFFFFu));
    p->cp_mmio_write(CP_Q_RING_BASE_HI,   uint32_t(cp_ring_dev_addr_ >> 32));
    p->cp_mmio_write(CP_Q_HEAD_ADDR_LO,   uint32_t(cp_head_dev_addr_ & 0xFFFFFFFFu));
    p->cp_mmio_write(CP_Q_HEAD_ADDR_HI,   uint32_t(cp_head_dev_addr_ >> 32));
    p->cp_mmio_write(CP_Q_CMPL_ADDR_LO,   uint32_t(cp_cmpl_dev_addr_ & 0xFFFFFFFFu));
    p->cp_mmio_write(CP_Q_CMPL_ADDR_HI,   uint32_t(cp_cmpl_dev_addr_ >> 32));
    p->cp_mmio_write(CP_Q_RING_SIZE_LOG2, CP_RING_SIZE_LOG2);
    p->cp_mmio_write(CP_Q_CONTROL,        0x1);
    p->cp_mmio_write(CP_REG_CTRL,         0x1);

    cp_enabled_ = true;
    return VX_SUCCESS;
}

vx_result_t Device::cp_submit_cl_(const void* cl) {
    std::lock_guard<std::mutex> g(cp_mu_);
    auto* p = platform();

    // 1) Upload one CL into the ring at the current tail.
    const uint64_t ring_off = cp_tail_ & (CP_RING_SIZE - 1);
    if (ring_off + CP_CL_BYTES > CP_RING_SIZE)
        return VX_ERR_INVALID_VALUE;  // mid-CL ring wrap not yet supported
    auto r = p->mem_upload(cp_ring_dev_addr_ + ring_off, cl, CP_CL_BYTES);
    if (r != VX_SUCCESS) return r;

    // 2) Commit the new tail. Atomic-pair: LO stages, HI commits both.
    cp_tail_           += CP_CL_BYTES;
    cp_expected_seqnum_ += 1;
    r = p->cp_mmio_write(CP_Q_TAIL_LO, uint32_t(cp_tail_ & 0xFFFFFFFFu));
    if (r != VX_SUCCESS) return r;
    r = p->cp_mmio_write(CP_Q_TAIL_HI, uint32_t(cp_tail_ >> 32));
    if (r != VX_SUCCESS) return r;

    // 3) Poll Q_SEQNUM until it catches up to this command's slot.
    //    Each MMIO read drives the simulator one or more cycles; on
    //    real hardware this is a cheap PCIe read.
    const uint64_t target = cp_expected_seqnum_;
    for (;;) {
        uint32_t seqnum32 = 0;
        r = p->cp_mmio_read(CP_Q_SEQNUM, &seqnum32);
        if (r != VX_SUCCESS) return r;
        if (uint64_t(seqnum32) >= target) return VX_SUCCESS;
        // No host sleep: each MMIO read already ticks sim cycles.
    }
}

vx_result_t Device::cp_submit_dcr_write(uint32_t addr, uint32_t value) {
    // CMD_DCR_WRITE on-wire layout (per VX_cp_pkg.sv cmd_t + cmd_size=20):
    //   bytes 0..3  header  { opcode=0x04, flags=0, reserved=0 }
    //   bytes 4..11 arg0    DCR addr
    //   bytes 12..19 arg1   DCR value
    // Pad rest of CL to 0 (NOP sentinel for unpack).
    uint8_t cl[CP_CL_BYTES] = {0};
    uint32_t* p32 = reinterpret_cast<uint32_t*>(cl);
    p32[0] = CP_OPCODE_DCR_WR;
    p32[1] = addr;
    p32[3] = value;
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_launch() {
    // CMD_LAUNCH on-wire layout (cmd_size=12):
    //   bytes 0..3  header  { opcode=0x06, flags=0, reserved=0 }
    //   bytes 4..11 arg0    unused by VX_cp_launch in v1
    uint8_t cl[CP_CL_BYTES] = {0};
    cl[0] = CP_OPCODE_LAUNCH;
    return cp_submit_cl_(cl);
}

vx_result_t Device::cp_submit_dcr_read(uint32_t addr, uint32_t tag,
                                       uint32_t* out_value) {
    if (!out_value) return VX_ERR_INVALID_VALUE;
    // CMD_DCR_READ on-wire layout (cmd_size=20):
    //   bytes 0..3  header  { opcode=0x05, flags=0, reserved=0 }
    //   bytes 4..11 arg0    DCR addr (low 12 bits used)
    //   bytes 12..19 arg1   tag (data on the DCR bus; e.g. core index
    //                       for VX_DCR_BASE_CACHE_FLUSH)
    uint8_t cl[CP_CL_BYTES] = {0};
    uint32_t* p32 = reinterpret_cast<uint32_t*>(cl);
    p32[0] = CP_OPCODE_DCR_RD;
    p32[1] = addr;
    p32[3] = tag;
    auto r = cp_submit_cl_(cl);
    if (r != VX_SUCCESS) return r;
    // Pick up the response from the CP regfile (latched by
    // VX_cp_dcr_proxy.last_rsp_data and exposed at offset 0x130).
    return platform()->cp_mmio_read(CP_Q_LAST_DCR_RSP, out_value);
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
    // Slow path: create OUTSIDE the lock (Queue::create acquires this
    // same mutex via register_queue — holding it here would deadlock).
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
    *out_count = 1;   // v1: each backend exposes a single device
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
    return to_device(dev)->platform()->query_caps(caps_id, out_value);
}

extern "C" vx_result_t vx_device_memory_info(vx_device_h dev,
                                             uint64_t* free,
                                             uint64_t* used) {
    if (!dev) return VX_ERR_INVALID_HANDLE;
    return to_device(dev)->platform()->memory_info(free, used);
}
