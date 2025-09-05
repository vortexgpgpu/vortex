#pragma once
#include <cstdint>

namespace vortex {
struct IMemBackend {
    virtual ~IMemBackend() = default;
    virtual void reset() = 0;
    virtual void tick() = 0;
    virtual void send_request(uint64_t addr, bool write,
                              uint32_t size, uint32_t tag,
                              uint32_t cid, uint64_t uuid) = 0;
};
} // namespace vortex
