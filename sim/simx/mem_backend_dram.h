// mem_backend_dram.h
#pragma once
#include "mem_backend.h"
#include "dram_sim.h"
#include <unordered_map>
#include <functional>
#include "types.h"

namespace vortex {

class MemBackendDram : public IMemBackend {
public:
    static MemBackendDram* instance() { return inst_; }

    // Construct with the same parameters as MemSim::Config: number of banks,
    // block size in bytes, and clock ratio. These values are passed to
    // the underlying DramSim so that the external memory model matches.
    MemBackendDram(uint32_t num_banks, uint32_t block_size, float clock_ratio);

    void reset() override;
    void tick() override;
    void send_request(uint64_t addr, bool write,
                      uint32_t size, uint32_t tag,
                      uint32_t cid, uint64_t uuid) override;

    // Not used directly; completions are handled by dram_complete().
    void complete(uint64_t tag);

    // Set by MemSim to push completed responses back to the correct
    // bank in the crossbar.
    std::function<void(uint32_t bank, const MemRsp& rsp)> mem_xbar_rsp_cb_;

private:
    struct Info {
        uint32_t cid;
        uint64_t uuid;
        bool write;
        uint32_t bank;  // bank index computed from the address
    };
    std::unordered_map<uint64_t, Info> inflight_;
    uint32_t num_banks_;
    uint32_t block_size_;
    uint32_t lg2_block_size_;
    static MemBackendDram* inst_;
    DramSim dram_sim_;

    // Static callback invoked by DramSim when a request completes
    static void dram_complete(void* arg);
};

} // namespace vortex