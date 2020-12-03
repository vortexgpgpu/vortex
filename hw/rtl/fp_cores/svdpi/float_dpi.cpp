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
  void dpi_fadd(int inst, bool enable, int a, int b, int* result);
  void dpi_fsub(int inst, bool enable, int a, int b, int* result);
  void dpi_fmul(int inst, bool enable, int a, int b, int* result);
  void dpi_fmadd(int inst, bool enable, int a, int b, int c, int* result);
  void dpi_fmsub(int inst, bool enable, int a, int b, int c, int* result);
  void dpi_fdiv(int inst, bool enable, int a, int b, int* result);
  void dpi_fsqrt(int inst, bool enable, int a, int* result);
  void dpi_ftoi(int inst, bool enable, int a, int* result);
  void dpi_ftou(int inst, bool enable, int a, int* result);
  void dpi_itof(int inst, bool enable, int a, int* result);
  void dpi_utof(int inst, bool enable, int a, int* result);
  void dpi_delayed_assert(int inst, bool cond);
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

union Float_t {    
    float f;
    int   i;
    struct {
        uint32_t man  : 23;
        uint32_t exp  : 8;
        uint32_t sign : 1;
    } parts;
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

void dpi_fadd(int inst, bool enable, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f + fb.f;

  sr.ensure_init(LATENCY_FADDMUL);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fsub(int inst, bool enable, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f - fb.f;

  sr.ensure_init(LATENCY_FADDMUL);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fmul(int inst, bool enable, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f * fb.f;

  sr.ensure_init(LATENCY_FADDMUL);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fmadd(int inst, bool enable, int a, int b, int c, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = fa.f * fb.f + fc.f;

  sr.ensure_init(LATENCY_FMADD);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fmsub(int inst, bool enable, int a, int b, int c, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = fa.f * fb.f - fc.f;

  sr.ensure_init(LATENCY_FMADD);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fdiv(int inst, bool enable, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f / fb.f;

  sr.ensure_init(LATENCY_FDIV);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_fsqrt(int inst, bool enable, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fr;

  fa.i = a;
  fr.f = sqrtf(fa.f);

  sr.ensure_init(LATENCY_FSQRT);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_ftoi(int inst, bool enable, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fr;

  fa.i = a;
  fr.i = int(fa.f);   

  sr.ensure_init(LATENCY_FTOI);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_ftou(int inst, bool enable, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fr;

  fa.i = a;
  fr.i = unsigned(fa.f);   

  sr.ensure_init(LATENCY_FTOI);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_itof(int inst, bool enable, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fr;

  fr.f = (float)a;   

  sr.ensure_init(LATENCY_ITOF);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_utof(int inst, bool enable, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  Float_t fa, fr;

  unsigned ua = a;
  fr.f = (float)ua;   

  sr.ensure_init(LATENCY_ITOF);
  sr.push(fr.i, enable);
  *result = sr.top();
}

void dpi_delayed_assert(int inst, bool cond) {
  ShiftRegister& sr = instances.get(inst);

  sr.ensure_init(2);
  sr.push(!cond, 1);

  auto status = sr.top();
  if (status) {
    printf("delayed assertion at %s!\n", svGetNameFromScope(svGetScope()));
    std::abort();
  }
}