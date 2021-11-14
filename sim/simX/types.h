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

enum class AluType {
  ARITH,
  BRANCH,
  IMUL,
  IDIV,    
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  case AluType::ARITH:  os << "ARITH"; break;
  case AluType::BRANCH: os << "BRANCH"; break;
  case AluType::IMUL:   os << "IMUL"; break;
  case AluType::IDIV:   os << "IDIV"; break;
  }
  return os;
}

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

template <typename T>
class Queue {
protected:
  std::queue<T> queue_;

  uint32_t count;

public:
  Queue() {}

  bool empty() const {
    return queue_.empty();
  }

  const T& top() const {
    return queue_.front();
  }

  void push(const T& value) {
    ++count;
    queue_.push(value);
  }

  void pop() {
    queue_.pop();
  }

  bool try_pop(T* value) {
    if (queue_.empty())
      return false;
    *value = queue_.front();
    queue_.pop();
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class HashTable {
private:
  std::vector<std::pair<bool, T>> entries_;
  uint32_t capacity_;

public:    
  HashTable(uint32_t size)
    : entries_(size)
    , capacity_(0) 
  {}

  bool empty() const {
    return (0 == capacity_);
  }
  
  bool full() const {
    return (capacity_ == entries_.size());
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
        ++capacity_;              
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
    --capacity_;
  }

  void remove(uint32_t index, T* value) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    *value = entry.second;
    entry.first = false;
    --capacity_;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp, uint32_t MaxInputs = 32>
class Switch : public SimObject<Switch<Req, Rsp>> {
private:
  struct req_batch_t {  
    std::vector<Req>       data;
    std::bitset<MaxInputs> valid;
    req_batch_t() {} 
    req_batch_t(uint32_t size) 
      : data(size)
      , valid(0)
    {} 
  };

  ArbiterType type_;
  std::queue<req_batch_t> reqq_;
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
  }

  void step(uint64_t /*cycle*/) {    
    // process incomming requests
    {
      req_batch_t req_batch(ReqIn.size());
      for (uint32_t i = 0, n = ReqIn.size(); i < n; ++i) {
        Req req;
        if (ReqIn.at(i).read(&req)) {
          req_batch.data.at(i) = req;
          req_batch.valid.set(i);
        }
      }
      if (req_batch.valid.any()) {
        reqq_.push(req_batch);
      }
    }

    // apply arbitration
    if (!reqq_.empty()) {
      auto& req_batch = reqq_.front();
      for (uint32_t i = 0, n = req_batch.data.size(); i < n; ++i) {
        auto j = (cursor_ + i) % n;        
        if (req_batch.valid.test(j)) {
          auto& req = req_batch.data.at(j);
          if (tag_shift_) {
            req.tag = (req.tag << tag_shift_) | j;
          }
          ReqOut.send(req, delay_);
          req_batch.valid.reset(j);
          this->update_cursor(j);
          if (!req_batch.valid.any())
            reqq_.pop(); // pop when empty
          break;
        }
      }      
    } 

    // process incoming reponses
    {
      Rsp rsp;
      if (RspIn.read(&rsp)) {    
        uint32_t port_id = 0;
        if (tag_shift_) {
          port_id = rsp.tag & ((1 << tag_shift_)-1);
          rsp.tag >>= tag_shift_;
        }      
        RspOut.at(port_id).send(rsp, 1);
      }
    }
  }

  void update_cursor(uint32_t grant) {
    if (type_ == ArbiterType::RoundRobin) {
      cursor_ = grant + 1;
    }
  }

  std::vector<SlavePort<Req>>  ReqIn;
  MasterPort<Req>              ReqOut;
  SlavePort<Rsp>               RspIn;    
  std::vector<MasterPort<Rsp>> RspOut;
};

}