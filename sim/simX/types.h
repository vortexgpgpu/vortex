#pragma once

#include <stdint.h>
#include <bitset>
#include <queue>
#include <unordered_map>
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

enum class ExeType {
  ALU,
  LSU,
  CSR,
  FPU,
  GPU,
  MAX,
};

enum class AluType {
  ARITH,
  BRANCH,
  IMUL,
  IDIV,    
};

enum class FpuType {
  FNCP,
  FMA,
  FDIV,
  FSQRT,
  FCVT,
};

enum class GpuType {
  TMC,
  WSPAWN,
  SPLIT,
  JOIN,
  BAR,
  TEX,
};

enum class ArbiterType {
  Priority,
  RoundRobin
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

  const T& top() const {
    return queue_.front();
  }

  void push(const T& value) {
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
    return -1;
  }

  void release(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    entry.first = false;
  }

  void remove(uint32_t index, T* value) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    *value = entry.second;
    entry.first = false;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp, uint32_t MaxInputs = 32>
class Switch : public SimObject<Switch<Req, Rsp>> {
private:
  struct req_t {  
    std::vector<Req>       data;
    std::bitset<MaxInputs> valid;
    req_t() {} 
    req_t(uint32_t size) : data(size) {} 
  };

  void handleIncomingRequest(const Req& req, uint32_t port_id) {
    cur_req_.data.at(port_id) = req;
    cur_req_.valid.set(port_id);
  }

  void handleIncomingResponse(const Rsp& rsp, uint32_t) {
    rsps_.push(rsp);
  }

  ArbiterType type_;
  std::queue<req_t> reqs_;
  std::queue<Rsp> rsps_;
  req_t cur_req_; 
  uint32_t delay_;  
  uint32_t cursor_;
  std::unordered_map<uint32_t, uint32_t> addr_table_;

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
    , cur_req_(num_inputs)
    , delay_(delay)
    , cursor_(0)
    , ReqIn(num_inputs, {this, this, &Switch<Req, Rsp, MaxInputs>::handleIncomingRequest})
    , ReqOut(this)
    , RspIn(this, this, &Switch<Req, Rsp, MaxInputs>::handleIncomingResponse)    
    , RspOut(num_inputs, this)
  {
    assert(delay_ != 0);
    assert(num_inputs <= MaxInputs);
  }

  void step(uint64_t /*cycle*/) {    
    if (cur_req_.valid.any()) {
      reqs_.push(cur_req_);      
      cur_req_.valid.reset();
    }

    while (!reqs_.empty()) {
      auto& entry = reqs_.front();
      bool found = false;
      for (uint32_t i = 0, n = entry.data.size(); i < n; ++i) {
        auto j = (cursor_ + i) % n;        
        if (entry.valid.test(j)) {
          auto& req = entry.data.at(j);
          addr_table_[req.tag] = j;
          ReqOut.send(req, delay_);
          entry.valid.reset(j);
          this->update_cursor(j);
          found = true;
          break;
        }
      }
      if (found)
        break;
      reqs_.pop();
    } 

    if (!rsps_.empty()) {
      auto& rsp = rsps_.front();
      auto port_id = addr_table_.at(rsp.tag);
      RspOut.at(port_id).send(rsp, 1);
      rsps_.pop();
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