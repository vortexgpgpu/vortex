// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

namespace vx {

Event::Event(Device* dev) : device_(dev) {}

Event::~Event() {
    if (cp_slot_addr_ != 0 && device_) {
        // Best-effort free of the device-resident counter slot.
        device_->platform()->mem_free(cp_slot_addr_);
    }
}

vx_result_t Event::create(Device* dev, Event** out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    *out = new Event(dev);
    return VX_SUCCESS;
}

uint64_t Event::cp_slot() {
    std::lock_guard<std::mutex> g(mu_);
    if (cp_slot_addr_ != 0) return cp_slot_addr_;
    if (!device_) return 0;
    // 8 bytes — VX_cp_event_unit operates with awsize=3 (single 8 B beat).
    // VX_MEM_READ_WRITE so both the CP and the host can mmio-read/write it.
    uint64_t addr = 0;
    auto r = device_->platform()->mem_alloc(/*size=*/8,
                                            /*flags=*/0x3 /*RW*/,
                                            &addr);
    if (r != VX_SUCCESS) return 0;
    // Zero-initialize so the CP's first poll sees the canonical "not yet
    // signaled" value.
    uint64_t zero = 0;
    device_->platform()->mem_upload(addr, &zero, sizeof(zero));
    cp_slot_addr_ = addr;
    return cp_slot_addr_;
}

void Event::signal(uint64_t value) {
    uint64_t mirror_addr = 0;
    {
        std::lock_guard<std::mutex> g(mu_);
        if (value <= counter_) return;   // monotonic — no decrement, no-op
        counter_ = value;
        mirror_addr = cp_slot_addr_;
    }
    // Mirror to the device-side counter slot if one's allocated. The CP's
    // VX_cp_event_unit might be polling this address inside an in-flight
    // CMD_EVENT_WAIT, so the host-side signal needs to be visible to it.
    if (mirror_addr != 0 && device_) {
        device_->platform()->mem_upload(mirror_addr, &value, sizeof(value));
    }
    cv_.notify_all();
}

void Event::complete(vx_result_t status) {
    uint64_t mirror_addr = 0;
    bool mirror_needed = false;
    {
        std::lock_guard<std::mutex> g(mu_);
        if (status != VX_SUCCESS && error_ == VX_SUCCESS) {
            error_ = status;
        }
        if (counter_ >= 1) {
            // Already signaled — but if a later error came in, broadcast so
            // any waiters that woke on counter==1 with error==SUCCESS can
            // re-observe the error. In practice complete() is called once
            // per command so this is a defensive path.
            if (status != VX_SUCCESS) cv_.notify_all();
            return;
        }
        counter_ = 1;
        mirror_addr = cp_slot_addr_;
        mirror_needed = true;
    }
    if (mirror_needed && mirror_addr != 0 && device_) {
        uint64_t one = 1;
        device_->platform()->mem_upload(mirror_addr, &one, sizeof(one));
    }
    cv_.notify_all();
}

vx_result_t Event::wait_value(uint64_t value, uint64_t timeout_ns) {
    std::unique_lock<std::mutex> g(mu_);
    const auto pred = [&] {
        return counter_ >= value || error_ != VX_SUCCESS;
    };
    if (!pred()) {
        if (timeout_ns == VX_TIMEOUT_INFINITE) {
            cv_.wait(g, pred);
        } else {
            if (!cv_.wait_for(g, std::chrono::nanoseconds(timeout_ns), pred))
                return VX_ERR_TIMEOUT;
        }
    }
    return (error_ != VX_SUCCESS) ? error_ : VX_SUCCESS;
}

uint64_t Event::get_value() const {
    std::lock_guard<std::mutex> g(mu_);
    return counter_;
}

void Event::set_profile(uint64_t queued_ns, uint64_t submit_ns,
                        uint64_t start_ns, uint64_t end_ns) {
    std::lock_guard<std::mutex> g(mu_);
    profile_.queued_ns = queued_ns;
    profile_.submit_ns = submit_ns;
    profile_.start_ns  = start_ns;
    profile_.end_ns    = end_ns;
    has_profile_ = true;
}

vx_result_t Event::get_profile(vx_profile_info_t* out) {
    if (!out) return VX_ERR_INVALID_VALUE;
    std::lock_guard<std::mutex> g(mu_);
    if (!has_profile_) return VX_ERR_NOT_SUPPORTED;
    *out = profile_;
    return VX_SUCCESS;
}

} // namespace vx

// ============================================================================
// C entry points — timeline API
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_event_create(vx_device_h dev, vx_event_h* out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Event* ev = nullptr;
    auto r = Event::create(to_device(dev), &ev);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(ev);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_signal(vx_event_h ev, uint64_t value) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    to_event(ev)->signal(value);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_get_value(vx_event_h ev, uint64_t* out_value) {
    if (!ev)        return VX_ERR_INVALID_HANDLE;
    if (!out_value) return VX_ERR_INVALID_VALUE;
    *out_value = to_event(ev)->get_value();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_wait_value(vx_event_h ev, uint64_t value,
                                           uint64_t timeout_ns) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    return to_event(ev)->wait_value(value, timeout_ns);
}

extern "C" vx_result_t vx_event_wait_values(uint32_t n,
                                            const vx_event_h* evs,
                                            const uint64_t*   values,
                                            uint64_t timeout_ns) {
    if (n != 0 && (!evs || !values)) return VX_ERR_INVALID_VALUE;
    // For each event, wait until its counter reaches the requested value.
    // Single-event fast path is the common case; multi-event loops are fine
    // for small n (the OpenCL clients always pass <= 16).
    for (uint32_t i = 0; i < n; ++i) {
        if (!evs[i]) return VX_ERR_INVALID_HANDLE;
        auto r = to_event(evs[i])->wait_value(values[i], timeout_ns);
        if (r != VX_SUCCESS) return r;
    }
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_retain(vx_event_h ev) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    to_event(ev)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_release(vx_event_h ev) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    to_event(ev)->release();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_get_profiling(vx_event_h ev,
                                              vx_profile_info_t* out) {
    if (!ev)  return VX_ERR_INVALID_HANDLE;
    if (!out) return VX_ERR_INVALID_VALUE;
    return to_event(ev)->get_profile(out);
}
