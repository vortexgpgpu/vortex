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
  void dpi_imul(bool enable, int a, int b, bool is_signed_a, bool is_signed_b, int* resultl, int* resulth);
  void dpi_idiv(bool enable, int a, int b, bool is_signed, int* quotient, int* remainder);

  int dpi_register();
  void dpi_assert(int inst, bool cond, int delay);

  void dpi_trace(const char* format, ...);
  void dpi_trace_start();
  void dpi_trace_stop();
}

bool sim_trace_enabled();
void sim_trace_enable(bool enable);

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

void dpi_imul(bool enable, int a, int b, bool is_signed_a, bool is_signed_b, int* resultl, int* resulth) {
  if (!enable)
    return;
    
  uint64_t first  = *(uint32_t*)&a;
  uint64_t second = *(uint32_t*)&b;
    
  if (is_signed_a && (first & 0x80000000)) {
    first |= 0xFFFFFFFF00000000;
  }

  if (is_signed_b && (second & 0x80000000)) {
    second |= 0xFFFFFFFF00000000;
  }

  uint64_t result;
  if (is_signed_a || is_signed_b) {
    result = (int64_t)first * (int64_t)second;
  } else {
    result = first * second;
  }    
    
  *resultl = result & 0xFFFFFFFF;
  *resulth = (result >> 32) & 0xFFFFFFFF;
}

void dpi_idiv(bool enable, int a, int b, bool is_signed, int* quotient, int* remainder) {
  if (!enable)
    return;

  uint32_t dividen = *(uint32_t*)&a;
  uint32_t divisor = *(uint32_t*)&b;

  if (is_signed) {
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else if (dividen == 0x80000000 && divisor == 0xffffffff) {
      *remainder = 0;
      *quotient  = dividen;
    } else { 
      *quotient  = (int32_t)dividen / (int32_t)divisor;
      *remainder = (int32_t)dividen % (int32_t)divisor;      
    }
  } else {    
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else {
      *quotient  = dividen / divisor;
      *remainder = dividen % divisor;
    }
  }
}

void dpi_trace(const char* format, ...) { 
  if (!sim_trace_enabled())
    return;
  va_list va;
	va_start(va, format);  
	vprintf(format, va);
	va_end(va);		  
}

void dpi_trace_start() { 
  sim_trace_enable(true);
}

void dpi_trace_stop() { 
  sim_trace_enable(false);
}