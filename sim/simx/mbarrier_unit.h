// Copyright (c) 2019-2026
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <array>
#include <deque>
#include <simobject.h>
#include "types.h"

namespace vortex {

class MBarrierUnit : public SimObject<MBarrierUnit> {
public:
  SimChannel<MbarrierReq> request_in;
  SimChannel<MbarrierRsp> response_out;
#ifdef VX_CFG_DXA_MBAR_ENABLE
  SimChannel<uint32_t> completion_in;
#endif
  SimChannel<uint32_t> unlock_out;
  SimChannel<MemReq> lmem_req_out;
  SimChannel<MemRsp> lmem_rsp_in;

  MBarrierUnit(const SimContext& ctx, const char* name, uint32_t core_id);
  ~MBarrierUnit();

  bool busy() const;

protected:
  void on_reset();
  void on_tick();

private:
  struct Entry {
    bool valid = false;
    uint32_t address = 0;
    uint32_t phase = 0;
    uint32_t pending_arrivals = 0;
    uint32_t expected_arrivals = 0;
    uint32_t pending_transactions = 0;
  };

  struct Result {
    uint32_t phase = 0;
    bool wait = false;
  };

  uint32_t decode_instruction_address(uint64_t address) const;
  void validate_address(uint32_t address) const;
  int find_entry(uint32_t address) const;
  uint32_t find_victim();
  bool start_miss(uint32_t address);
  Result execute(Entry& entry, MbarrierType op, uint32_t value,
                 uint32_t wid);
  void complete_phase(Entry& entry);
  void enqueue_write(const Entry& entry);

  uint32_t core_id_;
  std::array<Entry, VX_CFG_MBAR_CACHE_SIZE> entries_;
  uint32_t victim_ptr_ = 0;
  std::array<bool, VX_CFG_NUM_WARPS> waiter_valid_;
  std::array<uint32_t, VX_CFG_NUM_WARPS> waiter_address_;
  std::deque<uint32_t> unlocks_;

  bool write_pending_ = false;
  MemReq write_request_;
  bool miss_active_ = false;
  bool miss_sent_ = false;
  uint32_t miss_address_ = 0;
  uint32_t miss_entry_ = 0;

  friend class SimObject<MBarrierUnit>;
};

} // namespace vortex
