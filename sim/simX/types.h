#pragma once

#include <stdint.h>
#include <bitset>
#include <queue>
#include <unordered_map>
#include <util.h>
#include <VX_config.h>
#include <simobject.h>

namespace vortex {

typedef uint8_t  Byte;
typedef uint32_t Word;
typedef int32_t  WordI;

typedef uint32_t Addr;
typedef uint32_t Size;

typedef std::bitset<32> RegMask;
typedef std::bitset<32> ThreadMask;
typedef std::bitset<32> WarpMask;

///////////////////////////////////////////////////////////////////////////////

enum class RegType {
  None,
  Integer,
  Float,
  Vector
};

inline std::ostream &operator<<(std::ostream &os, const RegType& type) {
  switch (type) {
  case RegType::None: break;
  case RegType::Integer: os << "r"; break;
  case RegType::Float:   os << "fr"; break;
  case RegType::Vector:  os << "vr"; break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class ExeType {
  NOP,
  ALU,
  LSU,
  CSR,
  FPU,
  GPU,
  MAX,
};

inline std::ostream &operator<<(std::ostream &os, const ExeType& type) {
  switch (type) {
  case ExeType::NOP: os << "NOP"; break;
  case ExeType::ALU: os << "ALU"; break;
  case ExeType::LSU: os << "LSU"; break;
  case ExeType::CSR: os << "CSR"; break;
  case ExeType::FPU: os << "FPU"; break;
  case ExeType::GPU: os << "GPU"; break;
  case ExeType::MAX: break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AluType {
  ARITH,
  BRANCH,
  IMUL,
  IDIV,    
  CMOV,
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  case AluType::ARITH:  os << "ARITH"; break;
  case AluType::BRANCH: os << "BRANCH"; break;
  case AluType::IMUL:   os << "IMUL"; break;
  case AluType::IDIV:   os << "IDIV"; break;
  case AluType::CMOV:   os << "CMOV"; break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class LsuType {
  LOAD,
  STORE,
  FENCE,
  PREFETCH,    
};

inline std::ostream &operator<<(std::ostream &os, const LsuType& type) {
  switch (type) {
  case LsuType::LOAD:     os << "LOAD"; break;
  case LsuType::STORE:    os << "STORE"; break;
  case LsuType::FENCE:    os << "FENCE"; break;
  case LsuType::PREFETCH: os << "PREFETCH"; break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AddrType {
  Global,
  Shared,
  IO,
};

inline std::ostream &operator<<(std::ostream &os, const AddrType& type) {
  switch (type) {
  case AddrType::Global: os << "Global"; break;
  case AddrType::Shared: os << "Shared"; break;
  case AddrType::IO:     os << "IO"; break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct mem_addr_size_t {
  uint64_t addr;
  uint32_t size;
};

inline AddrType get_addr_type(Word addr, uint32_t size) {
  __unused (size);
  if (SM_ENABLE) {
    if (addr >= (SMEM_BASE_ADDR - SMEM_SIZE)
    &&  addr < SMEM_BASE_ADDR) {      
      assert((addr + size) <= SMEM_BASE_ADDR);
      return AddrType::Shared;
    }
  }
  if (addr >= IO_BASE_ADDR) {
     return AddrType::IO;
  }
  return AddrType::Global;
}

///////////////////////////////////////////////////////////////////////////////

enum class FpuType {
  FNCP,
  FMA,
  FDIV,
  FSQRT,
  FCVT,
};

inline std::ostream &operator<<(std::ostream &os, const FpuType& type) {
  switch (type) {
  case FpuType::FNCP:  os << "FNCP"; break;
  case FpuType::FMA:   os << "FMA"; break;
  case FpuType::FDIV:  os << "FDIV"; break;
  case FpuType::FSQRT: os << "FSQRT"; break;
  case FpuType::FCVT:  os << "FCVT"; break;
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class GpuType {
  TMC,
  WSPAWN,
  SPLIT,
  JOIN,
  BAR,
  TEX,
};

inline std::ostream &operator<<(std::ostream &os, const GpuType& type) {
  switch (type) {
  case GpuType::TMC:    os << "TMC"; break;
  case GpuType::WSPAWN: os << "WSPAWN"; break;
  case GpuType::SPLIT:  os << "SPLIT"; break;
  case GpuType::JOIN:   os << "JOIN"; break;
  case GpuType::BAR:    os << "BAR"; break;
  case GpuType::TEX:    os << "TEX"; break;
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
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct MemReq {
    uint64_t addr;
    uint32_t tag;
    bool write;
    bool is_io;

    MemReq(uint64_t _addr = 0, 
           uint64_t _tag = 0, 
           bool _write = false, 
           bool _is_io = false
    )   : addr(_addr)
        , tag(_tag)
        , write(_write)
        , is_io(_is_io) 
    {}
};

struct MemRsp {
    uint64_t tag;    
    MemRsp(uint64_t _tag = 0) : tag (_tag) {}
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class Queue {
protected:
  std::queue<T> queue_;

public:
  Queue() {}

  bool empty() const {
    return queue_.empty();
  }

  const T& front() const {
    return queue_.front();
  }

  T& front() {
    return queue_.front();
  }

  const T& back() const {
    return queue_.back();
  }

  T& back() {
    return queue_.back();
  }

  void push(const T& value) {    
    queue_.push(value);
  }

  void pop() {
    queue_.pop();
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class HashTable {
private:
  std::vector<std::pair<bool, T>> entries_;
  uint32_t size_;

public:    
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
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp, uint32_t MaxInputs = 32>
class Switch : public SimObject<Switch<Req, Rsp>> {
private:
  ArbiterType type_;
  uint32_t delay_;  
  uint32_t cursor_;
  uint32_t tag_shift_;

public:
  Switch(
    const SimContext& ctx, 
    const char* name, 
    ArbiterType type, 
    uint32_t num_inputs, 
    uint32_t delay = 1
  ) 
    : SimObject<Switch<Req, Rsp, MaxInputs>>(ctx, name)    
    , type_(type)
    , delay_(delay)
    , cursor_(0)
    , tag_shift_(log2ceil(num_inputs))
    , ReqIn(num_inputs, this)
    , ReqOut(this)
    , RspIn(this)    
    , RspOut(num_inputs, this)
  {
    assert(delay_ != 0);
    assert(num_inputs <= MaxInputs);
    if (num_inputs == 1) {
      // bypass
      ReqIn.at(0).bind(&ReqOut);
      RspIn.bind(&RspOut.at(0));
    }
  }

  void step(uint64_t /*cycle*/) {  
    if (ReqIn.size() == 1)
      return;
        
    // process incomming requests    
    for (uint32_t i = 0, n = ReqIn.size(); i < n; ++i) {      
      uint32_t j = (cursor_ + i) % n;
      auto& req_in = ReqIn.at(j);      
      if (!req_in.empty()) {
        auto& req = req_in.front();
        if (tag_shift_) {
          req.tag = (req.tag << tag_shift_) | j;
        }
        ReqOut.send(req, delay_);                
        req_in.pop();
        this->update_cursor(j);
        break;
      }
    } 

    // process incoming reponses
    if (!RspIn.empty()) {
      auto& rsp = RspIn.front();    
      uint32_t port_id = 0;
      if (tag_shift_) {
        port_id = rsp.tag & ((1 << tag_shift_)-1);
        rsp.tag >>= tag_shift_;
      }      
      RspOut.at(port_id).send(rsp, 1);
      RspIn.pop();
    }
  }

  void update_cursor(uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      cursor_ = grant + 1;
    }
  }

  std::vector<SimPort<Req>>  ReqIn;
  SimPort<Req>              ReqOut;
  SimPort<Rsp>               RspIn;    
  std::vector<SimPort<Rsp>> RspOut;
};

}