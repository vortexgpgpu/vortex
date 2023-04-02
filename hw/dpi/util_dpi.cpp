#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 3
#endif

extern "C" {
  void dpi_imul(bool enable, long int a, long int b, bool is_signed_a, bool is_signed_b, long int* resultl, long int* resulth);
  void dpi_idiv(bool enable, long int a, long int b, bool is_signed, long int* quotient, long int* remainder);

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

void umul64wide (uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
    uint64_t a_lo = (uint64_t)(uint32_t)a;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = (uint64_t)(uint32_t)b;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint32_t cy = (uint32_t)(((p0 >> 32) + (uint32_t)p1 + (uint32_t)p2) >> 32);

    *lo = p0 + (p1 << 32) + (p2 << 32);
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;
}


void dpi_imul(bool enable, long int a, long int b, bool is_signed_a, bool is_signed_b, long int* resultl, long int* resulth) {
  if (!enable)
    return;
    
  uint64_t first  = *(long int*)&a;
  uint64_t second = *(long int*)&b;

  umul64wide (a, b, (uint64_t *)resulth, (uint64_t *)resultl);
    
  //if (a < 0LL) *resulth -= b;
  //if (b < 0LL) *resultl -= a;  
  dpi_trace(1, "MUL - %lld %lld %lld %lld\n", a, b, *resulth, *resultl);
  /*if (is_signed_a && (first & 0x80000000)) {
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
  *resulth = (result >> 32) & 0xFFFFFFFF;*/
}

void dpi_idiv(bool enable, long int a, long int b, bool is_signed, long int* quotient, long int* remainder) {
  if (!enable)
    return;

  uint64_t dividen = *(long int*)&a;
  uint64_t divisor = *(long int*)&b;

  if (is_signed) {
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else if (dividen == 0x8000000000000000 && divisor == 0xffffffffffffffff) {
      *remainder = 0;
      *quotient  = dividen;
    } else { 
      *quotient  = (int64_t)dividen / (int64_t)divisor;
      *remainder = (int64_t)dividen % (int64_t)divisor;      
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
    dpi_trace(1, "DIV - %d %lld %lld %lld %lld %lld %lld\n",is_signed , a, b, dividen, divisor, *quotient, *remainder);
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
