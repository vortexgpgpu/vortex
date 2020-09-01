#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

extern "C" {
  void dpi_fadd(int inst, bool enable, bool valid, int a, int b, int* result);
  void dpi_fsub(int inst, bool enable, bool valid, int a, int b, int* result);
  void dpi_fmul(int inst, bool enable, bool valid, int a, int b, int* result);
  void dpi_fmadd(int inst, bool enable, bool valid, int a, int b, int c, int* result);
  void dpi_fmsub(int inst, bool enable, bool valid, int a, int b, int c, int* result);
  void dpi_fdiv(int inst, bool enable, bool valid, int a, int b, int* result);
  void dpi_fsqrt(int inst, bool enable, bool valid, int a, int* result);
  void dpi_ftoi(int inst, bool enable, bool valid, int a, int* result);
  void dpi_ftou(int inst, bool enable, bool valid, int a, int* result);
  void dpi_itof(int inst, bool enable, bool valid, int a, int* result);
  void dpi_utof(int inst, bool enable, bool valid, int a, int* result);
}

extern double sc_time_stamp();

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

  void push(int value, bool enable, bool valid) {
    if (!enable)
      return;      
    for (unsigned i = 0; i < depth_-1; ++i) {
      buffer_[i] = buffer_[i+1];
    }
    buffer_[depth_-1].value = value;
    buffer_[depth_-1].valid = valid;
  }

  int top() const {
    return buffer_[0].value;
  }

  bool valid() const { 
    return buffer_[0].valid;
  }

private:

  struct entry_t {
    int value;
    bool valid;
  };

  std::vector<entry_t> buffer_;
  int top_;
  unsigned depth_;
  bool init_;
};

class Instances {
public:
  ShiftRegister& get(int inst) {
    mutex_.lock();
    ShiftRegister& sr = instances_[inst];
    mutex_.unlock();
    return sr;
  }

private:
  std::unordered_map<int, ShiftRegister> instances_;
  std::mutex mutex_;
};

Instances instances;

void dpi_fadd(int inst, bool enable, bool valid, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa + fb;   

  sr.ensure_init(LATENCY_FMADD);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fsub(int inst, bool enable, bool valid, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa - fb;   

  sr.ensure_init(LATENCY_FMADD);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fmul(int inst, bool enable, bool valid, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa * fb;   

  sr.ensure_init(LATENCY_FMADD);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fmadd(int inst, bool enable, bool valid, int a, int b, int c, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fc = *(float*)&c;
  float fr = fa * fb + fc;   

  sr.ensure_init(LATENCY_FMADD);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fmsub(int inst, bool enable, bool valid, int a, int b, int c, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fc = *(float*)&c;
  float fr = fa * fb - fc;   

  sr.ensure_init(LATENCY_FMADD);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fdiv(int inst, bool enable, bool valid, int a, int b, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa / fb;   

  sr.ensure_init(LATENCY_FDIV);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_fsqrt(int inst, bool enable, bool valid, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  float fr = sqrtf(fa);

  sr.ensure_init(LATENCY_FSQRT);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_ftoi(int inst, bool enable, bool valid, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  int ir = int(fa);   

  sr.ensure_init(LATENCY_FTOI);
  sr.push(ir, enable, valid);
  *result = sr.top();
}

void dpi_ftou(int inst, bool enable, bool valid, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fa = *(float*)&a;
  unsigned ir = unsigned(fa);   

  sr.ensure_init(LATENCY_FTOI);
  sr.push(ir, enable, valid);
  *result = sr.top();
}

void dpi_itof(int inst, bool enable, bool valid, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  float fr = (float)a;   

  sr.ensure_init(LATENCY_ITOF);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}

void dpi_utof(int inst, bool enable, bool valid, int a, int* result) {
  ShiftRegister& sr = instances.get(inst);

  unsigned ua = *(unsigned*)&a;
  float fr = (float)ua;   

  sr.ensure_init(LATENCY_ITOF);
  sr.push(*(int*)&fr, enable, valid);
  *result = sr.top();
}