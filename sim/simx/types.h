// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <bitset>
#include <queue>
#include <vector>
#include <unordered_map>
#include <util.h>
#include <stringutil.h>
#include <VX_config.h>
#include <VX_types.h>
#include <simobject.h>
#include <bitvector.h>
#include <iostream>
#include "debug.h"
#include "constants.h"

namespace vortex {

typedef uint8_t Byte;
#if (XLEN == 32)
typedef uint32_t Word;
typedef int32_t  WordI;
typedef uint64_t DWord;
typedef int64_t  DWordI;
#elif (XLEN == 64)
typedef uint64_t Word;
typedef int64_t  WordI;
typedef __uint128_t DWord;
typedef __int128_t DWordI;
#else
#error unsupported XLEN
#endif

typedef std::bitset<MAX_NUM_CORES>   CoreMask;
typedef std::bitset<MAX_NUM_REGS>    RegMask;
typedef BitVector<Word>              ThreadMask;
typedef std::bitset<MAX_NUM_WARPS>   WarpMask;

///////////////////////////////////////////////////////////////////////////////

union reg_data_t {
  uint8_t  u8;
  uint16_t u16;
  Word     u;
  WordI    i;
  float    f32;
  double   f64;
  uint32_t u32;
  uint64_t u64;
  int32_t  i32;
  int64_t  i64;
};

///////////////////////////////////////////////////////////////////////////////

enum class RegType {
  None,
  Integer,
  Float,
#ifdef EXT_V_ENABLE
  Vector,
#endif
  Count
};

inline std::ostream &operator<<(std::ostream &os, const RegType& type) {
  switch (type) {
  case RegType::None: break;
  case RegType::Integer: os << "x"; break;
  case RegType::Float:   os << "f"; break;
#ifdef EXT_V_ENABLE
  case RegType::Vector:  os << "v"; break;
#endif
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct RegOpd {
  RegType type = RegType::None;
  uint32_t idx = 0;

  uint32_t id() const {
    if (type == RegType::None)
      return 0;
    // unique register id embedding the type
    return (((int)(type)-1) << LOG_NUM_REGS) | idx;
  }

  friend std::ostream &operator<<(std::ostream &os, const RegOpd& reg) {
    os << reg.type << reg.idx;
    return os;
  }

  constexpr static uint32_t ID_BITS = log2ceil((int)RegType::Count) + LOG_NUM_REGS;
};

///////////////////////////////////////////////////////////////////////////////

enum class FUType {
  ALU,
  LSU,
  FPU,
  SFU,
#ifdef EXT_TPU_ENABLE
  TPU,
#endif
#ifdef EXT_V_ENABLE
  VPU,
#endif
  Count
};

inline std::ostream &operator<<(std::ostream &os, const FUType& type) {
  switch (type) {
  case FUType::ALU: os << "ALU"; break;
  case FUType::LSU: os << "LSU"; break;
  case FUType::FPU: os << "FPU"; break;
  case FUType::SFU: os << "SFU"; break;
#ifdef EXT_TPU_ENABLE
  case FUType::TPU: os << "TPU"; break;
#endif
#ifdef EXT_V_ENABLE
  case FUType::VPU: os << "VPU"; break;
#endif
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AluType {
  ARITH,
  BRANCH,
  SYSCALL,
  IMUL,
  IDIV
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  case AluType::ARITH:   os << "ARITH"; break;
  case AluType::BRANCH:  os << "BRANCH"; break;
  case AluType::SYSCALL: os << "SYSCALL"; break;
  case AluType::IMUL:    os << "IMUL"; break;
  case AluType::IDIV:    os << "IDIV"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class LsuType {
  LOAD,
  STORE,
#ifdef EXT_V_ENABLE
  VLOAD,
  VSTORE,
#endif
  FENCE
};

inline std::ostream &operator<<(std::ostream &os, const LsuType& type) {
  switch (type) {
  case LsuType::LOAD:  os << "LOAD"; break;
  case LsuType::STORE: os << "STORE"; break;
#ifdef EXT_V_ENABLE
  case LsuType::VLOAD: os << "VLOAD"; break;
  case LsuType::VSTORE:os << "VSTORE"; break;
#endif
  case LsuType::FENCE: os << "FENCE"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AddrType {
  Global,
  Shared,
  IO
};

inline AddrType get_addr_type(uint64_t addr) {
  if (addr >= IO_BASE_ADDR && addr < IO_END_ADDR) {
     return AddrType::IO;
  }
  if (LMEM_ENABLED) {
    if (addr >= LMEM_BASE_ADDR && (addr-LMEM_BASE_ADDR) < (1 << LMEM_LOG_SIZE)) {
        return AddrType::Shared;
    }
  }
  return AddrType::Global;
}

inline std::ostream &operator<<(std::ostream &os, const AddrType& type) {
  switch (type) {
  case AddrType::Global: os << "Global"; break;
  case AddrType::Shared: os << "Shared"; break;
  case AddrType::IO:     os << "IO"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct mem_addr_size_t {
  uint64_t addr;
  uint32_t size;
};

///////////////////////////////////////////////////////////////////////////////

enum class FpuType {
  FNCP,
  FMA,
  FDIV,
  FSQRT,
  FCVT
};

inline std::ostream &operator<<(std::ostream &os, const FpuType& type) {
  switch (type) {
  case FpuType::FNCP:  os << "FNCP"; break;
  case FpuType::FMA:   os << "FMA"; break;
  case FpuType::FDIV:  os << "FDIV"; break;
  case FpuType::FSQRT: os << "FSQRT"; break;
  case FpuType::FCVT:  os << "FCVT"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class SfuType {
  TMC,
  WSPAWN,
  SPLIT,
  JOIN,
  BAR,
  PRED,
  CSRRW,
  CSRRS,
  CSRRC
};

inline std::ostream &operator<<(std::ostream &os, const SfuType& type) {
  switch (type) {
  case SfuType::TMC:    os << "TMC"; break;
  case SfuType::WSPAWN: os << "WSPAWN"; break;
  case SfuType::SPLIT:  os << "SPLIT"; break;
  case SfuType::JOIN:   os << "JOIN"; break;
  case SfuType::BAR:    os << "BAR"; break;
  case SfuType::PRED:   os << "PRED"; break;
  case SfuType::CSRRW:  os << "CSRRW"; break;
  case SfuType::CSRRS:  os << "CSRRS"; break;
  case SfuType::CSRRC:  os << "CSRRC"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class TpuType {
  WMMA = 0,
};

inline std::ostream &operator<<(std::ostream &os, const TpuType& type) {
  switch (type) {
  case TpuType::WMMA: os << "WMMA"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VpuType {
  VSET    = 0,

  ARITH   = 1,
  IMUL    = 2,
  IDIV    = 3,

  FMA     = 4,
  FDIV    = 5,
  FSQRT   = 6,
  FCVT    = 7,
  FNCP    = 8,

  // reduction
  ARITH_R = 9,
  FMA_R   = 10,
  FNCP_R  = 11
};

inline std::ostream &operator<<(std::ostream &os, const VpuType& type) {
  switch (type) {
  case VpuType::VSET:   os << "VSET"; break;
  case VpuType::ARITH:  os << "ARITH"; break;
  case VpuType::IMUL:   os << "IMUL"; break;
  case VpuType::IDIV:   os << "IDIV"; break;
  case VpuType::FMA:    os << "FMA"; break;
  case VpuType::FDIV:   os << "FDIV"; break;
  case VpuType::FSQRT:  os << "FSQRT"; break;
  case VpuType::FCVT:   os << "FCVT"; break;
  case VpuType::FNCP:   os << "FNCP"; break;
  case VpuType::ARITH_R:os << "ARITH_R"; break;
  case VpuType::FMA_R:  os << "FMA_R"; break;
  case VpuType::FNCP_R: os << "FNCP_R"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class ArbiterType {
  Priority,
  RoundRobin,
  Matrix
};

inline std::ostream &operator<<(std::ostream &os, const ArbiterType& type) {
  switch (type) {
  case ArbiterType::Priority:   os << "Priority"; break;
  case ArbiterType::RoundRobin: os << "RoundRobin"; break;
  case ArbiterType::Matrix:     os << "Matrix"; break;
  default: assert(false);
  }
  return os;
}

class IArbiterImpl {
public:
  IArbiterImpl() {}
  virtual ~IArbiterImpl() {}
  virtual uint32_t grant(const BitVector<>& requests) = 0;
  virtual void reset() = 0;
};

class PriorityArbiter : public IArbiterImpl {
public:
  PriorityArbiter(uint32_t size) : size_(size) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    for (uint32_t i = 0; i < size_; ++i) {
      if (requests.test(i)) {
        return i;
      }
    }
    return -1;
  }

  void reset() override {
    //--
  }
private:
  uint32_t size_;
};

class RoundRobinArbiter : public IArbiterImpl {
public:
  RoundRobinArbiter(uint32_t size) : size_(size) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    uint32_t start = (last_grant_ + 1) % size_;
    for (uint32_t i = 0; i < size_; ++i) {
      uint32_t idx = (start + i) % size_;
      if (requests.test(idx)) {
        last_grant_ = idx;
        return idx;
      }
    }
    return -1;
  }

  void reset() override {
    last_grant_ = 0;
  }

private:
  uint32_t size_;
  uint32_t last_grant_;
};

class MatrixArbiter : public IArbiterImpl {
public:
  MatrixArbiter(uint32_t size)
    : size_(size)
    , priority_matrix_(size, std::vector<bool>(size)) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    for (uint32_t i = 0; i < size_; ++i) {
      if (requests[i]) {
        // Check if this request has the highest priority by comparing it
        bool highest_priority = true;
        for (uint32_t j = 0; j < size_; ++j) {
          if (requests[j] && priority_matrix_[i][j]) {
            // If there is any active request with higher priority, this is not the highest
            highest_priority = false;
            break;
          }
        }

        if (highest_priority) {
          // Update the priority matrix: clear the row and set the column
          for (uint32_t j = 0; j < size_; ++j) {
            if (i != j) {
              priority_matrix_[i][j] = false;
              priority_matrix_[j][i] = true;
            }
          }
          return i; // Return the granted request index
        }
      }
    }
    return -1;
  }

  void reset() override {
    // Initialize the priority matrix
    for (uint32_t i = 0; i < size_; ++i) {
      priority_matrix_[i].resize(size_);
      // Initialize only the upper triangle to true
      for (uint32_t j = i + 1; j < size_; ++j) {
        priority_matrix_[i][j] = true;
      }
    }
  }

private:
  uint32_t size_;
  std::vector<std::vector<bool>> priority_matrix_;
};

class Arbiter {
public:
  Arbiter(ArbiterType type = ArbiterType::Priority, uint32_t size = 0) {
    switch (type) {
    case ArbiterType::Priority:
      impl_ = std::make_shared<PriorityArbiter>(size);
      break;
    case ArbiterType::RoundRobin:
      impl_ = std::make_shared<RoundRobinArbiter>(size);
      break;
    case ArbiterType::Matrix:
      impl_ = std::make_shared<MatrixArbiter>(size);
      break;
    default:
      assert(false); // Should never reach here
    }
  }

  virtual ~Arbiter() {}

  uint32_t grant(const BitVector<>& requests) {
    return impl_->grant(requests);
  }

  void reset() {
    impl_->reset();
  }

private:
  std::shared_ptr<IArbiterImpl> impl_;
};

///////////////////////////////////////////////////////////////////////////////

struct LsuReq {
  BitVector<> mask;
  std::vector<uint64_t> addrs;
  bool     write;
  uint32_t tag;
  uint32_t cid;
  uint64_t uuid;

  LsuReq(uint32_t size)
    : mask(size)
    , addrs(size, 0)
    , write(false)
    , tag(0)
    , cid(0)
    , uuid(0)
  {}

  friend std::ostream &operator<<(std::ostream &os, const LsuReq& req) {
    os << "rw=" << req.write << ", mask=" << req.mask << ", addr={";
    bool first_addr = true;
    for (size_t i = 0; i < req.mask.size(); ++i) {
      if (!first_addr) os << ", ";
      first_addr = false;
      if (req.mask.test(i)) {
        os << "0x" << std::hex << req.addrs.at(i) << std::dec;
      } else {
        os << "-";
      }
    }
    os << "}, tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct LsuRsp {
  BitVector<> mask;
  uint64_t tag;
  uint32_t cid;
  uint64_t uuid;

 LsuRsp(uint32_t size)
    : mask(size)
    , tag (0)
    , cid(0)
    , uuid(0)
  {}

  friend std::ostream &operator<<(std::ostream &os, const LsuRsp& rsp) {
    os << "mask=" << rsp.mask << ", tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct MemReq {
  uint64_t addr;
  bool     write;
  AddrType type;
  uint32_t tag;
  uint32_t cid;
  uint64_t uuid;

  MemReq(uint64_t _addr = 0,
          bool _write = false,
          AddrType _type = AddrType::Global,
          uint64_t _tag = 0,
          uint32_t _cid = 0,
          uint64_t _uuid = 0
  ) : addr(_addr)
    , write(_write)
    , type(_type)
    , tag(_tag)
    , cid(_cid)
    , uuid(_uuid)
  {}

  friend std::ostream &operator<<(std::ostream &os, const MemReq& req) {
    os << "rw=" << req.write << ", ";
    os << "addr=0x" << std::hex << req.addr << std::dec << ", type=" << req.type;
    os << ", tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct MemRsp {
  uint64_t tag;
  uint32_t cid;
  uint64_t uuid;

  MemRsp(uint64_t _tag = 0, uint32_t _cid = 0, uint64_t _uuid = 0)
    : tag (_tag)
    , cid(_cid)
    , uuid(_uuid)
  {}

  friend std::ostream &operator<<(std::ostream &os, const MemRsp& rsp) {
    os << "tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class HashTable {
public:
  typedef T DataType;

  HashTable(uint32_t capacity)
    : entries_(capacity)
    , size_(0)
  {}

  bool empty() const {
    return (0 == size_);
  }

  bool full() const {
    return (size_ == entries_.size());
  }

  uint32_t size() const {
    return size_;
  }

  bool contains(uint32_t index) const {
    return entries_.at(index).first;
  }

  const T& at(uint32_t index) const {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  T& at(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  uint32_t allocate(const T& value) {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      if (!entry.first) {
        entry.first = true;
        entry.second = value;
        ++size_;
        return i;
      }
    }
    assert(false);
    return -1;
  }

  void release(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    entry.first = false;
    --size_;
  }

  void clear() {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      entry.first = false;
    }
    size_ = 0;
  }

private:
  std::vector<std::pair<bool, T>> entries_;
  uint32_t size_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class TxArbiter : public SimObject<TxArbiter<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input = 0)
      : data(_data)
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  std::vector<SimPort<ReqType>> Inputs;
  std::vector<SimPort<RspType>> Outputs;

  TxArbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1
  ) : SimObject<TxArbiter<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , delay_(delay)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
    , arbiters_(num_outputs, {type, 1u << lg2_num_reqs_})
  {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(num_inputs >= num_outputs);

    // bypass mode
    if (num_inputs == num_outputs) {
      for (uint32_t i = 0; i < num_inputs; ++i) {
        Inputs.at(i).bind(&Outputs.at(i));
      }
    }
  }

  void reset() {
    for (auto& arb : arbiters_) {
      arb.reset();
    }
  }

  void tick() {
    uint32_t I = Inputs.size();
    uint32_t O = Outputs.size();
    uint32_t R = 1 << lg2_num_reqs_;

    // skip bypass mode
    if (I == O)
      return;

    // process inputs
    for (uint32_t o = 0; o < O; ++o) {
      BitVector<> requests(R);
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t i = o * R + r;
        if (i >= I)
          continue;
        requests.set(r, !Inputs.at(i).empty());
      }
      if (requests.any()) {
        uint32_t g = arbiters_.at(o).grant(requests);
        uint32_t i = o * R + g;
        auto& req_in = Inputs.at(i);
        auto& req = req_in.front();
        DT(4, this->name() << "-req" << i << "_" << o << ": " << req);
        Outputs.at(o).push(RspType(req, i), delay_);
        req_in.pop();
      }
    }
  }

protected:

  uint32_t delay_;
  uint32_t lg2_num_reqs_;
  std::vector<Arbiter> arbiters_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class TxCrossBar : public SimObject<TxCrossBar<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input= 0)
      : data(_data)
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  std::vector<SimPort<ReqType>> Inputs;
  std::vector<SimPort<RspType>> Outputs;

  TxCrossBar(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs,
    uint32_t num_outputs,
    std::function<uint32_t(const Type& req)> output_sel,
    uint32_t delay = 1
  )
    : SimObject<TxCrossBar<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , delay_(delay)
    , lg2_inputs_(log2ceil(num_inputs))
    , lg2_outputs_(log2ceil(num_outputs))
    , output_sel_(output_sel)
    , collisions_(0) {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(ispow2(num_outputs));
    assert(output_sel != nullptr);

    // bypass mode
    if (num_inputs == 1 && num_outputs == 1) {
      Inputs.at(0).bind(&Outputs.at(0));
    }
  }

  void reset() {
    //--
  }

  void tick() {
    uint32_t I = Inputs.size();
    uint32_t O = Outputs.size();
    if (I == 1 && O == 1)
      return;

    // process incoming requests
    for (uint32_t o = 0; o < O; ++o) {
      int32_t input_idx = -1;
      bool has_collision = false;
      for (uint32_t i = 0; i < I; ++i) {
        auto& req_in = Inputs.at(i);
        if (req_in.empty())
          continue;
        auto& req = req_in.front();
        uint32_t output_idx = 0;
        if (lg2_outputs_ != 0) {
          // select output index
          output_idx = output_sel_(req);
          // skip if input is not going to current output
          if (output_idx != o)
            continue;
        }
        if (input_idx != -1) {
          has_collision = true;
          break;
        }
        input_idx = i;
      }
      if (input_idx != -1) {
        auto& req_in = Inputs.at(input_idx);
        auto& req = req_in.front();
        DT(4, this->name() << "-req" << input_idx << "_" << o << ": " << req);
        Outputs.at(o).push(RspType(req, input_idx), delay_);
        req_in.pop();
        collisions_ += has_collision;
      }
    }
  }

  uint64_t collisions() const {
    return collisions_;
  }

protected:

  uint32_t delay_;
  uint32_t lg2_inputs_;
  uint32_t lg2_outputs_;
  std::function<uint32_t(const Type& req)> output_sel_;
  uint64_t collisions_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxRxArbiter : public SimObject<TxRxArbiter<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>>  ReqIn;
  std::vector<SimPort<Rsp>>  RspIn;

  std::vector<SimPort<Req>>  ReqOut;
  std::vector<SimPort<Rsp>>  RspOut;

  TxRxArbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1
  )
    : SimObject<TxRxArbiter<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , arbiter_(nullptr)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
  {
    if (num_inputs != num_outputs) {
      // allocate arbiter
      arbiter_ = ReqArb::Create(name, type, num_inputs, num_outputs, delay);
      // bind arbiter inputs and outputs
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&arbiter_->Inputs.at(i));
      }
      for (uint32_t o = 0; o < num_outputs; ++o) {
        arbiter_->Outputs.at(o).bind(&ReqOut.at(o),
          [lg2_num_reqs = lg2_num_reqs_](const typename ReqArb::RspType& arb_rsp) {
            Req req(arb_rsp.data);
            if (lg2_num_reqs != 0) {
              uint32_t r = arb_rsp.input & ((1 << lg2_num_reqs) - 1);
              req.tag = (req.tag << lg2_num_reqs) | r;
            }
            return req;
          });
      }
    } else {
      // bypass mode
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&ReqOut.at(i));
        RspOut.at(i).bind(&RspIn.at(i));
      }
    }
  }

  void reset() {
    //--
  }

  void tick() {
    if (!arbiter_)
      return;

    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_num_reqs_;

    // process outgoing responses
    for (uint32_t o = 0; o < O; ++o) {
      auto& rsp_out = RspOut.at(o);
      if (!rsp_out.empty()) {
        auto& rsp = rsp_out.front();
        uint32_t r = 0;
        Rsp in_rsp(rsp);
        if (lg2_num_reqs_ != 0) {
          r = rsp.tag & (R-1);
          in_rsp.tag = rsp.tag >> lg2_num_reqs_;
        }
        uint32_t i = o * R + r;
        DT(4, this->name() << "-rsp" << o << "_" << i << ": " << in_rsp);
        RspIn.at(i).push(in_rsp, 1);
        rsp_out.pop();
      }
    }
  }

protected:
  typedef TxArbiter<Req> ReqArb;

  typename ReqArb::Ptr arbiter_;
  uint32_t lg2_num_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxRxCrossBar : public SimObject<TxRxCrossBar<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>> ReqIn;
  std::vector<SimPort<Rsp>> RspIn;

  std::vector<SimPort<Req>> ReqOut;
  std::vector<SimPort<Rsp>> RspOut;

  TxRxCrossBar(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs,
    std::function<uint32_t(const Req& req)> output_sel,
    uint32_t delay = 1
  )
    : SimObject<TxRxCrossBar<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , crossbar_(nullptr)
    , arbiter_(type, num_outputs)
    , lg2_inputs_(log2ceil(num_inputs)) {

    if (num_inputs != 1 || num_outputs != 1) {
      // allocate crossbar
      crossbar_ = ReqXbar::Create(name, num_inputs, num_outputs, output_sel, delay);
      // bind crossbar inputs and outputs
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&crossbar_->Inputs.at(i));
      }
      for (uint32_t o = 0; o < num_outputs; ++o) {
        crossbar_->Outputs.at(o).bind(&ReqOut.at(o),
          [lg2_inputs = lg2_inputs_](const typename ReqXbar::RspType& xbar_rsp) {
            Req req(xbar_rsp.data);
            if (lg2_inputs != 0) {
              req.tag = (req.tag << lg2_inputs) | xbar_rsp.input;
            }
            return req;
          });
      }
    } else {
      // bypass mode
      ReqIn.at(0).bind(&ReqOut.at(0));
      RspOut.at(0).bind(&RspIn.at(0));
    }
  }

  void reset() {
    arbiter_.reset();
  }

  void tick() {
    if (!crossbar_)
      return;

    uint32_t I = ReqIn.size();
    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_inputs_;

    // process outgoing responses
    for (uint32_t i = 0; i < I; ++i) {
      BitVector<> requests(O);
      for (uint32_t o = 0; o < O; ++o) {
        auto& rsp_out = RspOut.at(o);
        if (rsp_out.empty())
          continue;
        auto& rsp = rsp_out.front();
        // skip if response is not going to current input
        if (lg2_inputs_ != 0) {
          uint32_t input_idx = rsp.tag & (R-1);
          if (input_idx != i)
            continue;
        }
        requests.set(o);
      }
      if (requests.any()) {
        uint32_t g = arbiter_.grant(requests);
        auto& rsp_out = RspOut.at(g);
        auto& rsp = rsp_out.front();
        Rsp in_rsp(rsp);
        if (lg2_inputs_ != 0) {
          in_rsp.tag = rsp.tag >> lg2_inputs_;
        }
        DT(4, this->name() << "-rsp" << g << "_" << i << ": " << in_rsp);
        RspIn.at(i).push(in_rsp, 1);
        rsp_out.pop();
      }
    }
  }

  uint64_t collisions() const {
    return crossbar_->collisions();
  }

protected:
  typedef TxCrossBar<Req> ReqXbar;

  typename ReqXbar::Ptr crossbar_;
  Arbiter arbiter_;
  uint32_t lg2_inputs_;
};

///////////////////////////////////////////////////////////////////////////////

class LocalMemSwitch : public SimObject<LocalMemSwitch> {
public:
  SimPort<LsuReq> ReqIn;
  SimPort<LsuRsp> RspIn;

  SimPort<LsuReq> ReqLmem;
  SimPort<LsuRsp> RspLmem;

  SimPort<LsuReq> ReqDC;
  SimPort<LsuRsp> RspDC;

  LocalMemSwitch(
    const SimContext& ctx,
    const char* name,
    uint32_t delay
  );

  void reset();

  void tick();

private:
  uint32_t delay_;
};

///////////////////////////////////////////////////////////////////////////////

class LsuMemAdapter : public SimObject<LsuMemAdapter> {
public:
  SimPort<LsuReq> ReqIn;
  SimPort<LsuRsp> RspIn;

  std::vector<SimPort<MemReq>> ReqOut;
  std::vector<SimPort<MemRsp>> RspOut;

  LsuMemAdapter(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs,
    uint32_t delay
  );

  void reset();

  void tick();

private:
  uint32_t delay_;
};

using LsuArbiter  = TxRxArbiter<LsuReq, LsuRsp>;
using MemArbiter  = TxRxArbiter<MemReq, MemRsp>;
using MemCrossBar = TxRxCrossBar<MemReq, MemRsp>;

}
