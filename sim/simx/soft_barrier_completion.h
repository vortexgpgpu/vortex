// Copyright (c) 2019-2026
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <simobject.h>
#include "types.h"

namespace vortex {

class SoftBarrierCompletion : public SimObject<SoftBarrierCompletion> {
public:
  SimChannel<uint32_t> completion_in;
  SimChannel<MemReq> lmem_req_out;
  SimChannel<MemRsp> lmem_rsp_in;

  SoftBarrierCompletion(const SimContext& ctx, const char* name,
                        uint32_t core_id);
  ~SoftBarrierCompletion();

  bool busy() const;

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t core_id_;
  bool amo_inflight_ = false;

  friend class SimObject<SoftBarrierCompletion>;
};

} // namespace vortex
