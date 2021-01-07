#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

extern "C" {
  int dpi_register();
  void dpi_assert(int inst, bool cond, int delay);
}

class ShiftRegister {
public:
  ShiftRegister() : init_(false), depth_(0) {}

  void ensure_init(int depth) {
    if (!init_) {
      buffer_.resize(depth);
      init_  = true;
      depth_ = depth;
    }
  }

  void push(int value, bool enable) {
    if (!enable)
      return;      
    for (unsigned i = 0; i < depth_-1; ++i) {
      buffer_[i] = buffer_[i+1];
    }
    buffer_[depth_-1] = value;
  }

  int top() const {
    return buffer_[0];
  }

private:

  std::vector<int> buffer_;
  bool init_;
  unsigned depth_;  
};

class Instances {
public:
  ShiftRegister& get(int inst) {
    return instances_.at(inst);
  }

  int allocate() {
    mutex_.lock();   
    int inst = instances_.size();
    instances_.resize(inst + 1); 
    mutex_.unlock();
    return inst;
  }

private:
  std::vector<ShiftRegister> instances_;
  std::mutex mutex_;
};

Instances instances;

int dpi_register() {
  return instances.allocate();
}

void dpi_assert(int inst, bool cond, int delay) {
  ShiftRegister& sr = instances.get(inst);

  sr.ensure_init(delay);
  sr.push(!cond, 1);

  auto status = sr.top();
  if (status) {
    printf("delayed assertion at %s!\n", svGetNameFromScope(svGetScope()));
    std::abort();
  }
}