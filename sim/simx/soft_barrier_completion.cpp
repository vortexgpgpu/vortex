// Copyright (c) 2019-2026
// Licensed under the Apache License, Version 2.0.

#include "soft_barrier_completion.h"

#include <cstring>
#include "mem/mem_block_pool.h"

using namespace vortex;

SoftBarrierCompletion::SoftBarrierCompletion(
    const SimContext& ctx, const char* name, uint32_t core_id)
  : SimObject<SoftBarrierCompletion>(ctx, name)
  , completion_in(this)
  , lmem_req_out(this)
  , lmem_rsp_in(this)
  , core_id_(core_id)
{}

SoftBarrierCompletion::~SoftBarrierCompletion() {}

void SoftBarrierCompletion::on_reset() {
  amo_inflight_ = false;
}

bool SoftBarrierCompletion::busy() const {
  return amo_inflight_ || !completion_in.empty()
      || !lmem_rsp_in.empty();
}

void SoftBarrierCompletion::on_tick() {
  if (!lmem_rsp_in.empty()) {
    if (!amo_inflight_)
      std::abort();
    amo_inflight_ = false;
    lmem_rsp_in.pop();
  }

  if (amo_inflight_ || completion_in.empty())
    return;

  uint32_t address = completion_in.peek();
  if (address > (1u << VX_CFG_LMEM_LOG_SIZE) - sizeof(uint32_t)
   || (address & 3u) != 0)
    std::abort();

  auto data = make_mem_block();
  std::memset(data->data(), 0, data->size());
  uint32_t offset = address & (VX_CFG_MEM_BLOCK_SIZE - 1);
  uint32_t decrement = uint32_t(-1);
  std::memcpy(data->data() + offset, &decrement, sizeof(decrement));

  MemReq request(
      MemOp::AMO_ADD, address, data, uint64_t(0xf) << offset,
      0, core_id_, 0);
  request.flags.local = 1;
  if (lmem_req_out.try_send(request)) {
    amo_inflight_ = true;
    completion_in.pop();
  }
}
