#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "svdpi.h"
#include "VX_config.h"

extern "C" {
  void dpi_fadd(bool clk, bool enable, int a, int b, int* result);
  void dpi_fsub(bool clk, bool enable, int a, int b, int* result);
  void dpi_fmul(bool clk, bool enable, int a, int b, int* result);
  void dpi_fmadd(bool clk, bool enable, int a, int b, int c, int* result);
  void dpi_fmsub(bool clk, bool enable, int a, int b, int c, int* result);
  void dpi_fdiv(bool clk, bool enable, int a, int b, int* result);
  void dpi_fsqrt(bool clk, bool enable, int a, int* result);
  void dpi_ftoi(bool clk, bool enable, int a, int* result);
  void dpi_ftou(bool clk, bool enable, int a, int* result);
  void dpi_itof(bool clk, bool enable, int a, int* result);
  void dpi_utof(bool clk, bool enable, int a, int* result);
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

  void push(int value, bool clk, bool enable) {
    if (clk || !enable)
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
  unsigned depth_;
  bool init_;
};

class Instances {
public:
  ShiftRegister& get(svScope scope) {
    mutex_.lock();
    ShiftRegister& reg = instances_[scope];
    mutex_.unlock();
    return reg;
  }

private:
  std::unordered_map<svScope, ShiftRegister> instances_;
  std::mutex mutex_;
};

Instances instances;

void dpi_fadd(bool clk, bool enable, int a, int b, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa + fb;   

  inst.ensure_init(LATENCY_FMADD);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_fsub(bool clk, bool enable, int a, int b, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa - fb;   

  inst.ensure_init(LATENCY_FMADD);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_fmul(bool clk, bool enable, int a, int b, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa * fb;   

  inst.ensure_init(LATENCY_FMADD);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_fmadd(bool clk, bool enable, int a, int b, int c, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fc = *(float*)&c;
  float fr = fa * fb + fc;   

  inst.ensure_init(LATENCY_FMADD);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_fmsub(bool clk, bool enable, int a, int b, int c, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fc = *(float*)&c;
  float fr = fa * fb - fc;   

  inst.ensure_init(LATENCY_FMADD);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_fdiv(bool clk, bool enable, int a, int b, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fb = *(float*)&b;
  float fr = fa / fb;   

  inst.ensure_init(LATENCY_FDIV);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.
  
  top();
}

void dpi_fsqrt(bool clk, bool enable, int a, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  float fr = sqrt(fa);   

  inst.ensure_init(LATENCY_FSQRT);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_ftoi(bool clk, bool enable, int a, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  int ir = int(fa);   

  inst.ensure_init(LATENCY_FTOI);
  inst.push(ir, clk, enable);
  *result = inst.top();
}

void dpi_ftou(bool clk, bool enable, int a, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fa = *(float*)&a;
  unsigned ir = unsigned(fa);   

  inst.ensure_init(LATENCY_FTOI);
  inst.push(ir, clk, enable);
  *result = inst.top();
}

void dpi_itof(bool clk, bool enable, int a, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  float fr = float(a);   

  inst.ensure_init(LATENCY_ITOF);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}

void dpi_utof(bool clk, bool enable, int a, int* result) {
  auto scope = svGetScope();
  ShiftRegister& inst = instances.get(scope);

  unsigned ua = *(unsigned*)&a;
  float fr = float(ua);   

  inst.ensure_init(LATENCY_ITOF);
  inst.push(*(int*)&fr, clk, enable);
  *result = inst.top();
}