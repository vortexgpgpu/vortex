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
    *out = new Device(std::move(plat));
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
