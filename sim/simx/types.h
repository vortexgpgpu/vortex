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
#include "debug.h"
#include <iostream>

namespace vortex {

typedef uint8_t Byte;
#if (XLEN == 32)
typedef uint32_t Word;
typedef int32_t  WordI;
typedef uint64_t DWord;
typedef int64_t  DWordI;
typedef uint32_t WordF;
#elif (XLEN == 64)
typedef uint64_t Word;
typedef int64_t  WordI;
typedef __uint128_t DWord;
typedef __int128_t DWordI;
typedef uint64_t WordF;
#else
#error unsupported XLEN
#endif

#define MAX_NUM_CORES   1024
#define MAX_NUM_THREADS 32
#define MAX_NUM_WARPS   32
#define MAX_NUM_REGS    32
#define NUM_SRC_REGS    3

typedef std::bitset<MAX_NUM_CORES>   CoreMask;
typedef std::bitset<MAX_NUM_REGS>    RegMask;
typedef std::bitset<MAX_NUM_THREADS> ThreadMask;
typedef std::bitset<MAX_NUM_WARPS>   WarpMask;

///////////////////////////////////////////////////////////////////////////////

class ThreadMaskOS {
public:
  ThreadMaskOS(const ThreadMask& mask, int size)
    : mask_(mask)
    , size_(size)
  {}

  friend std::ostream& operator<<(std::ostream& os, const ThreadMaskOS& wrapper) {
    for (int i = 0; i < wrapper.size_; ++i) {
      os << wrapper.mask_[i];
    }
    return os;
  }

private:
  const ThreadMask& mask_;
  int size_;
};

///////////////////////////////////////////////////////////////////////////////

enum class RegType {
  None,
  Integer,
  Float,
  Count,
  Vector
};

inline std::ostream &operator<<(std::ostream &os, const RegType& type) {
  switch (type) {
  case RegType::None: break;
  case RegType::Integer: os << "x"; break;
  case RegType::Float:   os << "f"; break;
  case RegType::Vector:  os << "v"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class FUType {
  ALU,
  LSU,
  FPU,
  SFU,
  TCU,
  Count
};

inline std::ostream &operator<<(std::ostream &os, const FUType& type) {
  switch (type) {
  case FUType::ALU: os << "ALU"; break;
  case FUType::LSU: os << "LSU"; break;
  case FUType::FPU: os << "FPU"; break;
  case FUType::SFU: os << "SFU"; break;
  case FUType::TCU: os << "TCU"; break;
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
  TCU_LOAD,
  STORE,
  TCU_STORE,
  FENCE
};

enum class TCUType {
  TCU_MUL
};

inline std::ostream &operator<<(std::ostream &os, const TCUType& type) {
  switch (type) {
  case TCUType::TCU_MUL: os << "TCU MUL"; break;
  default: assert(false);
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const LsuType& type) {
  switch (type) {
  case LsuType::LOAD:  os << "LOAD"; break;
  case LsuType::TCU_LOAD: os << "TCU_LOAD"; break;
  case LsuType::STORE: os << "STORE"; break;
  case LsuType::TCU_STORE: os << "TCU_STORE"; break;
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

enum class ArbiterType {
  Priority,
  RoundRobin
};

inline std::ostream &operator<<(std::ostream &os, const ArbiterType& type) {
  switch (type) {
  case ArbiterType::Priority:   os << "Priority"; break;
  case ArbiterType::RoundRobin: os << "RoundRobin"; break;
  default: assert(false);
  }
  return os;
}///////////////////////////////////////////////////////////////////////////////

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
};

inline std::ostream &operator<<(std::ostream &os, const LsuReq& req) {
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
};

inline std::ostream &operator<<(std::ostream &os, const LsuRsp& rsp) {
  os << "mask=" << rsp.mask << ", tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
  os << " (#" << rsp.uuid << ")";
  return os;
}

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
};

inline std::ostream &operator<<(std::ostream &os, const MemReq& req) {
  os << "rw=" << req.write << ", ";
  os << "addr=0x" << std::hex << req.addr << std::dec << ", type=" << req.type;
  os << ", tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
  os << " (#" << req.uuid << ")";
  return os;
}

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
};

inline std::ostream &operator<<(std::ostream &os, const MemRsp& rsp) {
  os << "tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
  os << " (#" << rsp.uuid << ")";
  return os;
}

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
class Arbiter : public SimObject<Arbiter<Type>> {
public:
  typedef Type ReqType;

  std::vector<SimPort<Type>> Inputs;
  std::vector<SimPort<Type>> Outputs;

  Arbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1
  ) : SimObject<Arbiter<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , type_(type)
    , delay_(delay)
    , grants_(num_outputs, 0)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
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
    for (auto& grant : grants_) {
      grant = 0;
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
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t g = (grants_.at(o) + r) & (R-1);
        uint32_t j = o * R + g;
        if (j >= I)
          continue;

        auto& req_in = Inputs.at(j);
        if (!req_in.empty()) {
          auto& req = req_in.front();
          DT(4, this->name() << "-req" << o << ": " << req);
          Outputs.at(o).push(req, delay_);
          req_in.pop();
          this->update_grant(o, g);
          break;
        }
      }
    }
  }

protected:

  void update_grant(uint32_t index, uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      grants_.at(index) = grant + 1;
    }
  }

  ArbiterType type_;
  uint32_t delay_;
  std::vector<uint32_t> grants_;
  uint32_t lg2_num_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class CrossBar : public SimObject<CrossBar<Type>> {
public:
  typedef Type ReqType;

  std::vector<SimPort<Type>> Inputs;
  std::vector<SimPort<Type>> Outputs;

  CrossBar(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1,
    std::function<uint32_t(const Type& req)> output_sel = nullptr
  )
    : SimObject<CrossBar<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , type_(type)
    , delay_(delay)
    , grants_(num_outputs, 0)
    , lg2_inputs_(log2ceil(num_inputs))
    , lg2_outputs_(log2ceil(num_outputs))
    , collisions_(0) {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(ispow2(num_outputs));
    if (output_sel != nullptr) {
      output_sel_ = output_sel;
    } else {
      output_sel_ = [this](const Type& req) {
        return (uint32_t)bit_getw(req.addr, 0, (lg2_outputs_-1));
      };
    }
  }

  void reset() {
    for (auto& grant : grants_) {
      grant = 0;
    }
  }

  void tick() {
    uint32_t I = Inputs.size();
    uint32_t O = Outputs.size();
    uint32_t R = 1 << lg2_inputs_;

    // process incoming requests
    for (uint32_t o = 0; o < O; ++o) {
      int32_t input_idx = -1;
      bool has_collision = false;
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t i = (grants_.at(o) + r) & (R-1);
        if (i >= I)
          continue;
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
          continue;
        }
        input_idx = i;
      }
      if (input_idx != -1) {
        auto& req_in = Inputs.at(input_idx);
        auto& req = req_in.front();
        DT(4, this->name() << "-req" << o << ": " << req);
        Outputs.at(o).push(req, delay_);
        req_in.pop();
        this->update_grant(o, input_idx);
        collisions_ += has_collision;
      }
    }
  }

  uint64_t collisions() const {
    return collisions_;
  }

protected:

  void update_grant(uint32_t index, uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      grants_.at(index) = grant + 1;
    }
  }

  ArbiterType type_;
  uint32_t delay_;
  std::vector<uint32_t> grants_;
  uint32_t lg2_inputs_;
  uint32_t lg2_outputs_;
  std::function<uint32_t(const Type& req)> output_sel_;
  uint64_t collisions_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxArbiter : public SimObject<TxArbiter<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>>  ReqIn;
  std::vector<SimPort<Rsp>>  RspIn;

  std::vector<SimPort<Req>>  ReqOut;
  std::vector<SimPort<Rsp>>  RspOut;

  TxArbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1
  )
    : SimObject<TxArbiter<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , type_(type)
    , delay_(delay)
    , grants_(num_outputs, 0)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
  {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(num_inputs >= num_outputs);

    // bypass mode
    if (num_inputs == num_outputs) {
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&ReqOut.at(i));
        RspOut.at(i).bind(&RspIn.at(i));
      }
    }
  }

  void reset() {
    for (auto& grant : grants_) {
      grant = 0;
    }
  }

  void tick() {
    uint32_t I = ReqIn.size();
    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_num_reqs_;

    // skip bypass mode
    if (I == O)
      return;

    // process outgoing responses
    for (uint32_t o = 0; o < O; ++o) {
      auto& rsp_out = RspOut.at(o);
      if (!rsp_out.empty()) {
        auto& rsp = rsp_out.front();
        uint32_t g = 0;
        if (lg2_num_reqs_ != 0) {
          g = rsp.tag & (R-1);
          rsp.tag >>= lg2_num_reqs_;
        }
        uint32_t j = o * R + g;
        DT(4, this->name() << "-rsp" << j << ": " << rsp);
        RspIn.at(j).push(rsp, 1);
        rsp_out.pop();
      }
    }

    // process incoming requests
    for (uint32_t o = 0; o < O; ++o) {
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t g = (grants_.at(o) + r) & (R-1);
        uint32_t j = o * R + g;
        if (j >= I)
          continue;

        auto& req_in = ReqIn.at(j);
        if (!req_in.empty()) {
          auto& req = req_in.front();
          if (lg2_num_reqs_ != 0) {
            req.tag = (req.tag << lg2_num_reqs_) | g;
          }
          DT(4, this->name() << "-req" << o << ": " << req);
          ReqOut.at(o).push(req, delay_);
          req_in.pop();
          this->update_grant(o, g);
          break;
        }
      }
    }
  }

protected:

  void update_grant(uint32_t index, uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      grants_.at(index) = grant + 1;
    }
  }

  ArbiterType type_;
  uint32_t delay_;
  std::vector<uint32_t> grants_;
  uint32_t lg2_num_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxCrossBar : public SimObject<TxCrossBar<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>> ReqIn;
  std::vector<SimPort<Rsp>> RspIn;

  std::vector<SimPort<Req>> ReqOut;
  std::vector<SimPort<Rsp>> RspOut;

  TxCrossBar(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1,
    std::function<uint32_t(const Req& req)> output_sel = nullptr
  )
    : SimObject<TxCrossBar<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , type_(type)
    , delay_(delay)
    , req_grants_(num_outputs, 0)
    , rsp_grants_(num_inputs, 0)
    , lg2_inputs_(log2ceil(num_inputs))
    , lg2_outputs_(log2ceil(num_outputs))
    , req_collisions_(0)
    , rsp_collisions_(0) {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(ispow2(num_inputs));
    assert(ispow2(num_outputs));
    if (output_sel != nullptr) {
      output_sel_ = output_sel;
    } else {
      output_sel_ = [this](const Req& req) {
        return (uint32_t)bit_getw(req.addr, 0, (lg2_outputs_-1));
      };
    }
  }

  void reset() {
    for (auto& grant : req_grants_) {
      grant = 0;
    }
    for (auto& grant : rsp_grants_) {
      grant = 0;
    }
  }

  void tick() {
    uint32_t I = ReqIn.size();
    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_inputs_;
    uint32_t T = 1 << lg2_outputs_;

    // process outgoing responses
    for (uint32_t i = 0; i < I; ++i) {
      int32_t output_idx = -1;
      bool has_collision = false;
      for (uint32_t t = 0; t < T; ++t) {
        uint32_t o = (rsp_grants_.at(i) + t) & (T-1);
        if (o >= O)
          continue;
        auto& rsp_out = RspOut.at(o);
        if (rsp_out.empty())
          continue;
        auto& rsp = rsp_out.front();
        uint32_t input_idx = 0;
        if (lg2_inputs_ != 0) {
          input_idx = rsp.tag & (R-1);
          // skip if response is not going to current input
          if (input_idx != i)
            continue;
        }
        if (output_idx != -1) {
          has_collision = true;
          continue;
        }
        output_idx = o;
      }
      if (output_idx != -1) {
        auto& rsp_out = RspOut.at(output_idx);
        auto& rsp = rsp_out.front();
        if (lg2_inputs_ != 0) {
          rsp.tag >>= lg2_inputs_;
        }
        DT(4, this->name() << "-rsp" << i << ": " << rsp);
        RspIn.at(i).push(rsp, 1);
        rsp_out.pop();
        this->update_rsp_grant(i, output_idx);
        rsp_collisions_ += has_collision;
      }
    }

    // process incoming requests
    for (uint32_t o = 0; o < O; ++o) {
      int32_t input_idx = -1;
      bool has_collision = false;
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t i = (req_grants_.at(o) + r) & (R-1);
        if (i >= I)
          continue;
        auto& req_in = ReqIn.at(i);
        if (req_in.empty())
          continue;
        auto& req = req_in.front();
        uint32_t output_idx = 0;
        if (lg2_outputs_ != 0) {
          // select output index
          output_idx = output_sel_(req);
          // skip if request is not going to current output
          if (output_idx != o)
            continue;
        }
        if (input_idx != -1) {
          has_collision = true;
          continue;
        }
        input_idx = i;
      }
      if (input_idx != -1) {
        auto& req_in = ReqIn.at(input_idx);
        auto& req = req_in.front();
        if (lg2_inputs_ != 0) {
          req.tag = (req.tag << lg2_inputs_) | input_idx;
        }
        DT(4, this->name() << "-req" << o << ": " << req);
        ReqOut.at(o).push(req, delay_);
        req_in.pop();
        this->update_req_grant(o, input_idx);
        req_collisions_ += has_collision;
      }
    }
  }

  uint64_t req_collisions() const {
    return req_collisions_;
  }

  uint64_t rsp_collisions() const {
    return rsp_collisions_;
  }

protected:

  void update_req_grant(uint32_t index, uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      req_grants_.at(index) = grant + 1;
    }
  }

  void update_rsp_grant(uint32_t index, uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      rsp_grants_.at(index) = grant + 1;
    }
  }

  ArbiterType type_;
  uint32_t delay_;
  std::vector<uint32_t> req_grants_;
  std::vector<uint32_t> rsp_grants_;
  uint32_t lg2_inputs_;
  uint32_t lg2_outputs_;
  std::function<uint32_t(const Req& req)> output_sel_;
  uint64_t req_collisions_;
  uint64_t rsp_collisions_;
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

using LsuArbiter  = TxArbiter<LsuReq, LsuRsp>;
using MemArbiter  = TxArbiter<MemReq, MemRsp>;
using MemCrossBar = TxCrossBar<MemReq, MemRsp>;

}
