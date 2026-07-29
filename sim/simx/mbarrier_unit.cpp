// Copyright (c) 2019-2026
// Licensed under the Apache License, Version 2.0.

#include "mbarrier_unit.h"

#include <cstring>
#include <iostream>
#include "instr_trace.h"
#include "mem/mem_block_pool.h"

using namespace vortex;

namespace {

constexpr uint32_t kObjectSize = sizeof(uint32_t);

constexpr uint32_t bits_for_limit(uint32_t limit) {
  return limit <= 1 ? 1 : 1 + bits_for_limit(limit >> 1);
}

constexpr uint32_t kArrivalBits = bits_for_limit(VX_CFG_NUM_WARPS);
constexpr uint32_t kTransactionBits =
    bits_for_limit(VX_CFG_MAX_BAR_EVENTS);
constexpr uint32_t kPendingArrivalShift = 1;
constexpr uint32_t kExpectedArrivalShift =
    kPendingArrivalShift + kArrivalBits;
constexpr uint32_t kTransactionShift =
    kExpectedArrivalShift + kArrivalBits;
constexpr uint32_t kStateBits =
    kTransactionShift + kTransactionBits;

constexpr uint32_t bit_mask(uint32_t width) {
  return (uint32_t(1) << width) - 1;
}

[[noreturn]] void invalid_mbarrier(const char* message,
                                    uint64_t address) {
  std::cerr << "mbarrier error: " << message << ", address=0x"
            << std::hex << address << std::dec << std::endl;
  std::abort();
}

} // namespace

MBarrierUnit::MBarrierUnit(const SimContext& ctx, const char* name,
                           uint32_t core_id)
  : SimObject<MBarrierUnit>(ctx, name)
  , request_in(this)
  , response_out(this)
#ifdef VX_CFG_DXA_MBAR_ENABLE
  , completion_in(this)
#endif
  , unlock_out(this)
  , lmem_req_out(this)
  , lmem_rsp_in(this)
  , core_id_(core_id)
{
  static_assert(VX_CFG_MBAR_CACHE_SIZE > 0,
                "mbarrier cache must not be empty");
  static_assert((VX_CFG_MBAR_CACHE_SIZE
               & (VX_CFG_MBAR_CACHE_SIZE - 1)) == 0,
                "mbarrier cache size must be a power of two");
  static_assert(kStateBits <= 32,
                "mbarrier state must fit in one word");
}

MBarrierUnit::~MBarrierUnit() {}

void MBarrierUnit::on_reset() {
  entries_ = {};
  victim_ptr_ = 0;
  waiter_valid_.fill(false);
  waiter_address_.fill(0);
  unlocks_.clear();
  write_pending_ = false;
  miss_active_ = false;
  miss_sent_ = false;
}

void MBarrierUnit::validate_address(uint32_t address) const {
  constexpr uint32_t kLmemSize = 1u << VX_CFG_LMEM_LOG_SIZE;
  if (address > kLmemSize - kObjectSize
   || (address & (kObjectSize - 1)) != 0)
    invalid_mbarrier("object is outside LMEM or misaligned", address);
}

uint32_t MBarrierUnit::decode_instruction_address(
    uint64_t address) const {
  if (address < VX_MEM_LMEM_BASE_ADDR)
    invalid_mbarrier("object is outside LMEM", address);
  uint64_t offset = address - VX_MEM_LMEM_BASE_ADDR;
  if (offset > UINT32_MAX)
    invalid_mbarrier("object offset is too large", address);
  validate_address(uint32_t(offset));
  return uint32_t(offset);
}

int MBarrierUnit::find_entry(uint32_t address) const {
  for (uint32_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].valid && entries_[i].address == address)
      return int(i);
  }
  return -1;
}

uint32_t MBarrierUnit::find_victim() {
  for (uint32_t i = 0; i < entries_.size(); ++i) {
    if (!entries_[i].valid) {
      victim_ptr_ = (i + 1) % entries_.size();
      return i;
    }
  }
  uint32_t victim = victim_ptr_;
  victim_ptr_ = (victim_ptr_ + 1) % entries_.size();
  return victim;
}

bool MBarrierUnit::start_miss(uint32_t address) {
  if (miss_active_ || write_pending_)
    return false;
  miss_active_ = true;
  miss_sent_ = false;
  miss_address_ = address;
  miss_entry_ = find_victim();
  return true;
}

void MBarrierUnit::complete_phase(Entry& entry) {
  entry.phase ^= 1;
  entry.pending_arrivals = entry.expected_arrivals;
  for (uint32_t wid = 0; wid < VX_CFG_NUM_WARPS; ++wid) {
    if (waiter_valid_[wid]
     && waiter_address_[wid] == entry.address) {
      waiter_valid_[wid] = false;
      unlocks_.push_back(wid);
    }
  }
}

MBarrierUnit::Result MBarrierUnit::execute(
    Entry& entry, MbarrierType op, uint32_t value, uint32_t wid) {
  Result result{entry.phase, false};

  switch (op) {
  case MbarrierType::INIT:
    if (value == 0 || value > VX_CFG_NUM_WARPS)
      invalid_mbarrier("invalid initialization count", entry.address);
    for (uint32_t i = 0; i < VX_CFG_NUM_WARPS; ++i) {
      if (waiter_valid_[i] && waiter_address_[i] == entry.address)
        invalid_mbarrier("object still has waiters", entry.address);
    }
    entry.phase = 0;
    entry.pending_arrivals = value;
    entry.expected_arrivals = value;
    entry.pending_transactions = 0;
    break;
  case MbarrierType::ARRIVE:
    if (value == 0 || value > entry.pending_arrivals)
      invalid_mbarrier("invalid arrival count", entry.address);
    entry.pending_arrivals -= value;
    if (entry.pending_arrivals == 0
     && entry.pending_transactions == 0)
      complete_phase(entry);
    break;
  case MbarrierType::EXPECT_TX:
    if (value == 0
     || value > VX_CFG_MAX_BAR_EVENTS - entry.pending_transactions)
      invalid_mbarrier("invalid transaction expectation", entry.address);
    entry.pending_transactions += value;
    break;
  case MbarrierType::WAIT:
    result.wait = ((value & 1u) == entry.phase);
    if (result.wait) {
      if (waiter_valid_.at(wid))
        invalid_mbarrier("warp already has a waiter", entry.address);
      waiter_valid_[wid] = true;
      waiter_address_[wid] = entry.address;
    }
    break;
  }
  return result;
}

void MBarrierUnit::enqueue_write(const Entry& entry) {
  auto data = make_mem_block();
  std::memset(data->data(), 0, data->size());
  uint32_t state = (entry.phase & 1u)
                 | (entry.pending_arrivals << kPendingArrivalShift)
                 | (entry.expected_arrivals << kExpectedArrivalShift)
                 | (entry.pending_transactions << kTransactionShift);
  uint32_t offset = entry.address & (VX_CFG_MEM_BLOCK_SIZE - 1);
  std::memcpy(data->data() + offset, &state, sizeof(state));
  write_request_ = MemReq(
      MemOp::ST, entry.address, data, uint64_t(0xf) << offset,
      0, core_id_, 0);
  write_request_.flags.local = 1;
  write_pending_ = true;
}

bool MBarrierUnit::busy() const {
  return write_pending_ || miss_active_ || !request_in.empty()
#ifdef VX_CFG_DXA_MBAR_ENABLE
      || !completion_in.empty()
#endif
      || !unlocks_.empty() || !lmem_rsp_in.empty();
}

void MBarrierUnit::on_tick() {
  if (!lmem_rsp_in.empty()) {
    if (!miss_active_ || !miss_sent_)
      invalid_mbarrier("unexpected LMEM response", miss_address_);
    const auto& response = lmem_rsp_in.peek();
    if (!response.data)
      invalid_mbarrier("LMEM refill has no data", miss_address_);
    uint32_t state = 0;
    uint32_t offset = miss_address_ & (VX_CFG_MEM_BLOCK_SIZE - 1);
    std::memcpy(&state, response.data->data() + offset, sizeof(state));
    auto& entry = entries_[miss_entry_];
    entry = {};
    entry.valid = true;
    entry.address = miss_address_;
    entry.phase = state & 1u;
    entry.pending_arrivals =
        (state >> kPendingArrivalShift) & bit_mask(kArrivalBits);
    entry.expected_arrivals =
        (state >> kExpectedArrivalShift) & bit_mask(kArrivalBits);
    entry.pending_transactions =
        (state >> kTransactionShift) & bit_mask(kTransactionBits);
    if ((uint64_t(state) >> kStateBits) != 0
     || entry.pending_arrivals > entry.expected_arrivals
     || entry.expected_arrivals > VX_CFG_NUM_WARPS
     || entry.pending_transactions > VX_CFG_MAX_BAR_EVENTS)
      invalid_mbarrier("invalid backing state", miss_address_);
    miss_active_ = false;
    miss_sent_ = false;
    lmem_rsp_in.pop();
  }

  if (!unlocks_.empty() && unlock_out.try_send(unlocks_.front()))
    unlocks_.pop_front();

  if (write_pending_) {
    if (lmem_req_out.try_send(write_request_))
      write_pending_ = false;
    return;
  }

  if (miss_active_) {
    if (!miss_sent_) {
      MemReq request(MemOp::LD, miss_address_, nullptr, 0,
                     0, core_id_, 0);
      request.flags.local = 1;
      if (lmem_req_out.try_send(request))
        miss_sent_ = true;
    }
    return;
  }

#ifdef VX_CFG_DXA_MBAR_ENABLE
  if (!completion_in.empty()) {
    uint32_t address = completion_in.peek();
    validate_address(address);
    int index = find_entry(address);
    if (index < 0) {
      start_miss(address);
      return;
    }
    auto& entry = entries_[index];
    if (entry.pending_transactions == 0)
      invalid_mbarrier("transaction completion underflow", address);
    --entry.pending_transactions;
    if (entry.pending_arrivals == 0
     && entry.pending_transactions == 0)
      complete_phase(entry);
    enqueue_write(entry);
    completion_in.pop();
    return;
  }
#endif

  if (request_in.empty() || response_out.full())
    return;

  const auto& request = request_in.peek();
  uint32_t address = decode_instruction_address(request.address);
  int index = find_entry(address);
  if (index < 0 && request.op != MbarrierType::INIT) {
    start_miss(address);
    return;
  }
  if (index < 0) {
    index = int(find_victim());
    entries_[index] = {};
    entries_[index].valid = true;
    entries_[index].address = address;
  }

  auto& entry = entries_[index];
  Result result = execute(entry, request.op, request.value, request.wid);
  if (request.op != MbarrierType::WAIT)
    enqueue_write(entry);
  response_out.send(
      MbarrierRsp{result.phase, result.wait, request.block_id,
                  request.trace});
  request_in.pop();
}
