#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>

#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

#ifdef XLEN_64
#define INT_TYPE int64_t
#define UINT_TYPE uint64_t
#else
#define INT_TYPE int32_t
#define UINT_TYPE uint32_t
#endif

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 3
#endif


extern "C" {
  void dpi_imul(bool enable, bool is_signed_a, bool is_signed_b, INT_TYPE a, INT_TYPE b, INT_TYPE* resultl, INT_TYPE* resulth);
  void dpi_idiv(bool enable, bool is_signed, INT_TYPE a, INT_TYPE b, INT_TYPE* quotient, INT_TYPE* remainder);

  int dpi_register();
  void dpi_assert(int inst, bool cond, int delay);

  void dpi_trace(int level, const char* format, ...);
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

///////////////////////////////////////////////////////////////////////////////

void dpi_imul(bool enable, bool is_signed_a, bool is_signed_b, INT_TYPE a, INT_TYPE b, INT_TYPE* resultl, INT_TYPE* resulth) {
  if (!enable)
    return;
#ifdef XLEN_64  
  uint64_t a_lo = (uint64_t)(uint32_t)a;
  uint64_t a_hi = a >> 32;
  uint64_t b_lo = (uint64_t)(uint32_t)b;
  uint64_t b_hi = b >> 32;

  uint64_t p0 = a_lo * b_lo;
  uint64_t p1 = a_lo * b_hi;
  uint64_t p2 = a_hi * b_lo;
  uint64_t p3 = a_hi * b_hi;

  uint32_t cy = (uint32_t)(((p0 >> 32) + (uint32_t)p1 + (uint32_t)p2) >> 32);

  *resultl = p0 + (p1 << 32) + (p2 << 32);
  *resulth = p3 + (p1 >> 32) + (p2 >> 32) + cy;
#else
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
#endif
}

void dpi_idiv(bool enable, bool is_signed, INT_TYPE a, INT_TYPE b, INT_TYPE* quotient, INT_TYPE* remainder) {
  if (!enable)
    return;

  UINT_TYPE dividen = a;
  UINT_TYPE divisor = b;

  auto inf_neg = UINT_TYPE(1) << (XLEN-1); 

  if (is_signed) {
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else if (dividen == inf_neg && divisor == -1) {
      *remainder = 0;
      *quotient  = dividen;
    } else { 
      *quotient  = (INT_TYPE)dividen / (INT_TYPE)divisor;
      *remainder = (INT_TYPE)dividen % (INT_TYPE)divisor;      
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

///////////////////////////////////////////////////////////////////////////////

void dpi_trace(int level, const char* format, ...) { 
  if (level > DEBUG_LEVEL)
    return;
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
