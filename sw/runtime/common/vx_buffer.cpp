// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "vortex2_internal.h"

#include <cstdlib>

namespace vx {

Buffer::Buffer(Device* dev, uint64_t dev_addr, uint64_t size, uint32_t flags)
    : device_(dev), dev_addr_(dev_addr), size_(size), flags_(flags) {
    device_->retain();
    device_->register_buffer(this);
}

Buffer::~Buffer() {
    if (mapped_ && host_mirror_) {
        std::free(host_mirror_);
        host_mirror_ = nullptr;
    }
    if (device_) {
        // Best-effort free on the device. Ignore errors at destruction.
        device_->platform()->mem_free(dev_addr_);
        device_->unregister_buffer(this);
        device_->release();
    }
}

vx_result_t Buffer::create(Device* dev, uint64_t size, uint32_t flags,
                           Buffer** out) {
    if (!dev || !out || size == 0) return VX_ERR_INVALID_VALUE;
    uint64_t dev_addr = 0;
    auto r = dev->platform()->mem_alloc(size, flags, &dev_addr);
    if (r != VX_SUCCESS) return r;
    *out = new Buffer(dev, dev_addr, size, flags);
    return VX_SUCCESS;
}

vx_result_t Buffer::reserve(Device* dev, uint64_t address, uint64_t size,
                            uint32_t flags, Buffer** out) {
    if (!dev || !out || size == 0) return VX_ERR_INVALID_VALUE;
    auto r = dev->platform()->mem_reserve(address, size, flags);
    if (r != VX_SUCCESS) return r;
    *out = new Buffer(dev, address, size, flags);
    return VX_SUCCESS;
}

vx_result_t Buffer::access(uint64_t off, uint64_t size, uint32_t flags) {
    if (off + size > size_) return VX_ERR_INVALID_VALUE;
    return device_->platform()->mem_access(dev_addr_ + off, size, flags);
}

vx_result_t Buffer::map_reserve(uint64_t off, uint64_t size, uint32_t flags,
                                void** out) {
    if (!out)                return VX_ERR_INVALID_VALUE;
    if (off + size > size_)  return VX_ERR_INVALID_VALUE;

    std::lock_guard<std::mutex> g(map_mu_);
    if (mapped_) return VX_ERR_NOT_SUPPORTED;   // single mapping at a time

    // Allocate a host mirror. map_commit prefills it from the device for
    // READ maps; unmap uploads it back for WRITE maps. Correct (no
    // use-after-free) but loses the zero-copy benefit pinned memory
    // would provide on real hardware.
    host_mirror_ = std::malloc(size);
    if (!host_mirror_) return VX_ERR_OUT_OF_HOST_MEMORY;

    mapped_off_   = off;
    mapped_size_  = size;
    mapped_flags_ = flags;
    mapped_       = true;
    *out = host_mirror_;
    return VX_SUCCESS;
}

vx_result_t Buffer::map_commit() {
    std::lock_guard<std::mutex> g(map_mu_);
    if (!mapped_) return VX_ERR_INVALID_VALUE;
    if ((mapped_flags_ & VX_MEM_READ) && mapped_size_ != 0) {
        return device_->platform()->mem_download(host_mirror_,
                                                 dev_addr_ + mapped_off_,
                                                 mapped_size_);
    }
    return VX_SUCCESS;
}

void Buffer::map_cancel() {
    std::lock_guard<std::mutex> g(map_mu_);
    if (mapped_) {
        std::free(host_mirror_);
        host_mirror_ = nullptr;
        mapped_      = false;
    }
}

vx_result_t Buffer::map(uint64_t off, uint64_t size, uint32_t flags,
                        void** out) {
    auto r = this->map_reserve(off, size, flags, out);
    if (r != VX_SUCCESS) return r;
    r = this->map_commit();
    if (r != VX_SUCCESS) this->map_cancel();
    return r;
}

vx_result_t Buffer::unmap(void* host_ptr) {
    std::lock_guard<std::mutex> g(map_mu_);
    if (!mapped_ || host_ptr != host_mirror_)
        return VX_ERR_INVALID_VALUE;
    vx_result_t r = VX_SUCCESS;
    if (mapped_flags_ & VX_MEM_WRITE) {
        r = device_->platform()->mem_upload(dev_addr_ + mapped_off_,
                                            host_mirror_, mapped_size_);
    }
    std::free(host_mirror_);
    host_mirror_ = nullptr;
    mapped_      = false;
    return r;
}

} // namespace vx

// ============================================================================
// C entry points
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_buffer_create(vx_device_h dev, uint64_t size,
                                        uint32_t flags, vx_buffer_h* out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Buffer* b = nullptr;
    auto r = Buffer::create(to_device(dev), size, flags, &b);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(b);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_reserve(vx_device_h dev, uint64_t address,
                                         uint64_t size, uint32_t flags,
                                         vx_buffer_h* out) {
    if (!dev || !out) return VX_ERR_INVALID_VALUE;
    Buffer* b = nullptr;
    auto r = Buffer::reserve(to_device(dev), address, size, flags, &b);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(b);
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_retain(vx_buffer_h buf) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    to_buffer(buf)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_release(vx_buffer_h buf) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    to_buffer(buf)->release();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_address(vx_buffer_h buf, uint64_t* out) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    if (!out) return VX_ERR_INVALID_VALUE;
    *out = to_buffer(buf)->dev_address();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_access(vx_buffer_h buf, uint64_t offset,
                                        uint64_t size, uint32_t flags) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    return to_buffer(buf)->access(offset, size, flags);
}

extern "C" vx_result_t vx_buffer_map(vx_buffer_h buf, uint64_t offset,
                                     uint64_t size, uint32_t flags,
                                     void** out_host_ptr) {
    if (!buf)          return VX_ERR_INVALID_HANDLE;
    if (!out_host_ptr) return VX_ERR_INVALID_VALUE;
    return to_buffer(buf)->map(offset, size, flags, out_host_ptr);
}

extern "C" vx_result_t vx_buffer_unmap(vx_buffer_h buf, void* host_ptr) {
    if (!buf) return VX_ERR_INVALID_HANDLE;
    return to_buffer(buf)->unmap(host_ptr);
}
