// mem_backend_dram.cpp
#include "mem_backend_dram.h"

using namespace vortex;

namespace {
struct CallbackData {
    MemBackendDram* backend;
    uint64_t tag;
};
} // anonymous namespace

MemBackendDram* MemBackendDram::inst_ = nullptr;

MemBackendDram::MemBackendDram(uint32_t num_banks, uint32_t block_size, float clock_ratio)
    : num_banks_(num_banks)
    , block_size_(block_size)
    , lg2_block_size_(0)
    , dram_sim_(num_banks, block_size, clock_ratio)
{
    // Compute log2(block_size_) once; block_size_ is assumed to be a power of two.
    uint32_t tmp = block_size_;
    while (tmp > 1) {
        ++lg2_block_size_;
        tmp >>= 1;
    }
    inst_ = this;
}

void MemBackendDram::reset() {
    inflight_.clear();
    dram_sim_.reset();
}

void MemBackendDram::tick() {
    // Retire pending transactions in DramSim
    dram_sim_.tick();
}

void MemBackendDram::dram_complete(void* arg) {
    auto* data = static_cast<CallbackData*>(arg);
    MemBackendDram* backend = data->backend;
    uint64_t tag = data->tag;
    auto it = backend->inflight_.find(tag);
    if (it != backend->inflight_.end()) {
        const Info& info = it->second;
        if (!info.write) {
            // Form a MemRsp for reads only
            MemRsp rsp{tag, info.cid, info.uuid};
            // Route the response to the recorded bank
            uint32_t bank = info.bank;
            if (backend->mem_xbar_rsp_cb_)
                backend->mem_xbar_rsp_cb_(bank, rsp);
        }
        backend->inflight_.erase(it);
    }
    delete data;
}

void MemBackendDram::send_request(uint64_t addr, bool write,
                                  uint32_t size, uint32_t tag,
                                  uint32_t cid, uint64_t uuid) {
    // Compute bank index: (addr >> lg2(block_size)) mod num_banks
    uint32_t bank_idx = 0;
    if (num_banks_ > 0)
        bank_idx = static_cast<uint32_t>((addr >> lg2_block_size_) & (num_banks_ - 1));
    inflight_.emplace(tag, Info{cid, uuid, write, bank_idx});
    auto* cb_data = new CallbackData{this, tag};
    // The size is ignored by DramSim because it is configured with block_size_.
    dram_sim_.send_request(addr, write, &MemBackendDram::dram_complete, cb_data);
}

void MemBackendDram::complete(uint64_t tag) {
    // Not used; dram_complete() handles completions
}