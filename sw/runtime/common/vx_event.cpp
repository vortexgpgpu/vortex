// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

namespace vx {

Event::Event(Device* dev, bool is_user)
    : device_(dev), is_user_(is_user) {
    // Both user events and runtime-managed events are created in the
    // QUEUED state; user events transition only on vx_user_event_signal,
    // runtime-managed events transition when the dispatcher's worker
    // calls complete().
    status_ = VX_EVENT_STATUS_QUEUED;
}

vx_result_t Event::create(Device* dev, Event** out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    *out = new Event(dev, /*is_user=*/false);
    return VX_SUCCESS;
}

vx_result_t Event::create_user(Device* dev, Event** out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    *out = new Event(dev, /*is_user=*/true);
    return VX_SUCCESS;
}

void Event::complete(vx_result_t status) {
    {
        std::lock_guard<std::mutex> g(mu_);
        if (status_ == VX_EVENT_STATUS_COMPLETE ||
            status_ == VX_EVENT_STATUS_ERROR) {
            return;   // already signaled — idempotent
        }
        status_ = (status == VX_SUCCESS)
                    ? VX_EVENT_STATUS_COMPLETE
                    : VX_EVENT_STATUS_ERROR;
        error_ = status;
    }
    cv_.notify_all();
}

vx_result_t Event::signal_user(vx_result_t status) {
    if (!is_user_) return VX_ERR_NOT_SUPPORTED;
    complete(status);
    return VX_SUCCESS;
}

vx_result_t Event::status(vx_event_status_e* out) {
    if (!out) return VX_ERR_INVALID_VALUE;
    std::lock_guard<std::mutex> g(mu_);
    *out = status_;
    return VX_SUCCESS;
}

vx_result_t Event::wait(uint64_t timeout_ns) {
    std::unique_lock<std::mutex> g(mu_);
    if (status_ == VX_EVENT_STATUS_COMPLETE) return VX_SUCCESS;
    if (status_ == VX_EVENT_STATUS_ERROR)    return error_;
    if (timeout_ns == VX_TIMEOUT_INFINITE) {
        cv_.wait(g, [&] {
            return status_ == VX_EVENT_STATUS_COMPLETE ||
                   status_ == VX_EVENT_STATUS_ERROR;
        });
    } else {
        const auto pred = [&] {
            return status_ == VX_EVENT_STATUS_COMPLETE ||
                   status_ == VX_EVENT_STATUS_ERROR;
        };
        if (!cv_.wait_for(g, std::chrono::nanoseconds(timeout_ns), pred))
            return VX_ERR_TIMEOUT;
    }
    return (status_ == VX_EVENT_STATUS_COMPLETE) ? VX_SUCCESS : error_;
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
// C entry points
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_user_event_create(vx_device_h dev, vx_event_h* out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Event* ev = nullptr;
    auto r = Event::create_user(to_device(dev), &ev);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(ev);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_user_event_signal(vx_event_h ev, vx_result_t status) {
    if (!ev) return VX_ERR_INVALID_HANDLE;
    return to_event(ev)->signal_user(status);
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

extern "C" vx_result_t vx_event_status(vx_event_h ev, vx_event_status_e* out) {
    if (!ev)  return VX_ERR_INVALID_HANDLE;
    if (!out) return VX_ERR_INVALID_VALUE;
    return to_event(ev)->status(out);
}

extern "C" vx_result_t vx_event_wait_all(uint32_t n, const vx_event_h* evs,
                                         uint64_t timeout_ns) {
    if (n != 0 && !evs) return VX_ERR_INVALID_VALUE;
    for (uint32_t i = 0; i < n; ++i) {
        if (!evs[i]) return VX_ERR_INVALID_HANDLE;
        auto r = to_event(evs[i])->wait(timeout_ns);
        if (r != VX_SUCCESS) return r;
    }
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_event_get_profiling(vx_event_h ev,
                                              vx_profile_info_t* out) {
    if (!ev)  return VX_ERR_INVALID_HANDLE;
    if (!out) return VX_ERR_INVALID_VALUE;
    return to_event(ev)->get_profile(out);
}
