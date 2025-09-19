// mem_backend_sst.h
#pragma once
#include "mem_backend.h"
#include <unordered_map>
#include <functional>
#include "types.h"

extern "C" {
  // Function pointer type for SST to call
  typedef void (*vx_submit_fn)(uint64_t addr, bool write, uint32_t size, uint64_t tag);
  // SST calls this to register its submit function
  void vx_register_submit(vx_submit_fn fn);
  // SST calls this when a memory response completes
  void vx_on_mem_complete(uint64_t tag);
}

namespace vortex {

class MemBackendSST : public IMemBackend {
public:
    static MemBackendSST* instance() { return inst_; }
    static vx_submit_fn get_vx_submit_fn() { return submit_fn_; }
    static void set_vx_submit_fn(vx_submit_fn fn) { submit_fn_ = fn; }

    MemBackendSST();
    void reset() override;
    void tick() override {}
    void send_request(uint64_t addr, bool write,
                      uint32_t size, uint32_t tag,
                      uint32_t cid, uint64_t uuid) override;

    // Called from vx_on_mem_complete
    void complete(uint64_t tag);


private:
    struct Info { uint32_t cid; uint64_t uuid; bool write; uint32_t bank;};
    std::unordered_map<uint64_t,Info> inflight_;
    static MemBackendSST* inst_;
    static vx_submit_fn submit_fn_;
};

} // namespace vortex
