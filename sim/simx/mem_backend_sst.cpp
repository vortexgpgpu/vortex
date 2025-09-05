// mem_backend_sst.cpp
// Implementation of the SST-backed memory backend.  This backend forwards
// all off-chip memory requests to the SST StandardMem interface via a
// registered callback (vx_submit_fn).  It maintains a table of inflight
// transactions keyed by the original request tag so that completions can
// be correlated back to the correct cluster and request.  When a read
// completion is observed via vx_on_mem_complete(), the backend pushes a
// MemRsp back into the crossbar using the stored cid/uuid.  Writes
// complete silently.  Bank routing is currently fixed to bank 0; this
// preserves correctness but may underutilize bank-level parallelism.

#include "mem_backend_sst.h"

extern "C" {

// Register a submit function provided by the SST component.  The
// MemBackendSST stores it in a static member so that calls to
// send_request() can forward requests into SST.
void vx_register_submit(vx_submit_fn fn) {
    vortex::MemBackendSST::set_vx_submit_fn(fn);
}

// Notify MemBackendSST that the SST memory system has completed a
// request identified by 'tag'.  The backend will produce a MemRsp for
// reads and erase the entry from its inflight table.
void vx_on_mem_complete(uint64_t tag) {
    if (auto inst = vortex::MemBackendSST::instance())
        inst->complete(tag);
}

} // extern "C"

using namespace vortex;

// Initialise static pointers
MemBackendSST* MemBackendSST::inst_ = nullptr;
vx_submit_fn   MemBackendSST::submit_fn_ = nullptr;

MemBackendSST::MemBackendSST() {
    // Record this instance so the C wrapper can find us
    inst_ = this;
}

void MemBackendSST::reset() {
    // Drop all inflight transactions; pending responses are ignored
    inflight_.clear();
}

void MemBackendSST::send_request(uint64_t addr, bool write,
                                 uint32_t size, uint32_t tag,
                                 uint32_t cid, uint64_t uuid) {
    // Save request metadata so we can form a response on completion
    inflight_.emplace(tag, Info{cid, uuid, write});
    // Forward the request into SST.  The SST wrapper will create a
    // StandardMem::Read or ::Write using this address, size and tag.
    if (submit_fn_) {
        submit_fn_(addr, write, size, tag);
    }
}

void MemBackendSST::complete(uint64_t tag) {
    auto it = inflight_.find(tag);
    if (it == inflight_.end())
        return;
    const Info &info = it->second;
    // Only produce a MemRsp for reads; writes complete silently
    if (!info.write) {
        MemRsp rsp{tag, info.cid, info.uuid};
        // Always route completions to bank 0; adjust if you need per-bank
        // completion routing in the future.
        if (mem_xbar_rsp_cb_)
            mem_xbar_rsp_cb_(0, rsp);
    }
    inflight_.erase(it);
}
