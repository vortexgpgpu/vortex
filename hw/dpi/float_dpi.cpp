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
  void dpi_fadd(int a, int b, int frm, int* result, int* fflags);
  void dpi_fsub(int a, int b, int frm, int* result, int* fflags);
  void dpi_fmul(int a, int b, int frm, int* result, int* fflags);
  void dpi_fmadd(int a, int b, int c, int frm, int* result, int* fflags);
  void dpi_fmsub(int a, int b, int c, int frm, int* result, int* fflags);
  void dpi_fnmadd(int a, int b, int c, int frm, int* result, int* fflags);
  void dpi_fnmsub(int a, int b, int c, int frm, int* result, int* fflags);

  void dpi_fdiv(int a, int b, int frm, int* result, int* fflags);
  void dpi_fsqrt(int a, int frm, int* result, int* fflags);
  
  void dpi_ftoi(int a, int frm, int* result, int* fflags);
  void dpi_ftou(int a, int frm, int* result, int* fflags);
  void dpi_itof(int a, int frm, int* result, int* fflags);
  void dpi_utof(int a, int frm, int* result, int* fflags);

  void dpi_fclss(int a, int* result);
  void dpi_fsgnj(int a, int b, int* result);
  void dpi_fsgnjn(int a, int b, int* result);
  void dpi_fsgnjx(int a, int b, int* result);

  void dpi_flt(int a, int b, int* result, int* fflags);
  void dpi_fle(int a, int b, int* result, int* fflags);
  void dpi_feq(int a, int b, int* result, int* fflags);
  void dpi_fmin(int a, int b, int* result, int* fflags);
  void dpi_fmax(int a, int b, int* result, int* fflags);
}

union Float_t {    
    float f;
    int   i;
    struct {
        uint32_t man  : 23;
        uint32_t exp  : 8;
        uint32_t sign : 1;
    } parts;
};

void dpi_fadd(int a, int b, int frm, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f + fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fsub(int a, int b, int frm, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f - fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fmul(int a, int b, int frm, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f * fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fmadd(int a, int b, int c, int frm, int* result, int* fflags) {
  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = fa.f * fb.f + fc.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fmsub(int a, int b, int c, int frm, int* result, int* fflags) {
  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = fa.f * fb.f - fc.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fnmadd(int a, int b, int c, int frm, int* result, int* fflags) {
  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = -(fa.f * fb.f + fc.f);

  *result = fr.i;
  *fflags = 0;
}

void dpi_fnmsub(int a, int b, int c, int frm, int* result, int* fflags) {
  Float_t fa, fb, fc, fr;

  fa.i = a;
  fb.i = b;
  fc.i = c;
  fr.f = -(fa.f * fb.f - fc.f);

  *result = fr.i;
  *fflags = 0;
}

void dpi_fdiv(int a, int b, int frm, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f / fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fsqrt(int a, int frm, int* result, int* fflags) {
  Float_t fa, fr;

  fa.i = a;
  fr.f = sqrtf(fa.f);

  *result = fr.i;
  *fflags = 0;
}

void dpi_ftoi(int a, int frm, int* result, int* fflags) {
  Float_t fa, fr;

  fa.i = a;
  fr.i = int(fa.f);   

  *result = fr.i;
  *fflags = 0;
}

void dpi_ftou(int a, int frm, int* result, int* fflags) {
  Float_t fa, fr;

  fa.i = a;
  fr.i = unsigned(fa.f);   

  *result = fr.i;
  *fflags = 0;
}

void dpi_itof(int a, int frm, int* result, int* fflags) {
  Float_t fa, fr;

  fr.f = (float)a;   

  *result = fr.i;
  *fflags = 0;
}

void dpi_utof(int a, int frm, int* result, int* fflags) {
  Float_t fa, fr;

  unsigned ua = a;
  fr.f = (float)ua;   

  *result = fr.i;
  *fflags = 0;
}

void dpi_flt(int a, int b, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f < fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fle(int a, int b, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f <= fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_feq(int a, int b, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = fa.f == fb.f;

  *result = fr.i;
  *fflags = 0;
}

void dpi_fmin(int a, int b, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = std::min<float>(fa.f, fb.f);

  *result = fr.i;
  *fflags = 0;
}

void dpi_fmax(int a, int b, int* result, int* fflags) {
  Float_t fa, fb, fr;

  fa.i = a;
  fb.i = b;
  fr.f = std::max<float>(fa.f, fb.f);

  *result = fr.i;
  *fflags = 0;
}

void dpi_fclss(int a, int* result) {

  int r = 0; // clear all bits

  bool fsign = (a >> 31);
  uint32_t expo = (a >> 23) & 0xFF;
  uint32_t fraction = a & 0x7FFFFF;

  if ((expo == 0) && (fraction == 0)) {
    r = fsign ? (1 << 3) : (1 << 4); // +/- 0
  } else if ((expo == 0) && (fraction != 0)) {
    r = fsign ? (1 << 2) : (1 << 5); // +/- subnormal
  } else if ((expo == 0xFF) && (fraction == 0)) {
    r = fsign ? (1<<0) : (1<<7); // +/- infinity
  } else if ((expo == 0xFF ) && (fraction != 0)) { 
    if (!fsign && (fraction == 0x00400000)) {
      r = (1 << 9);               // quiet NaN
    } else { 
      r = (1 << 8);               // signaling NaN
    }
  } else {
    r = fsign ? (1 << 1) : (1 << 6); // +/- normal
  }

  *result = r;
}

void dpi_fsgnj(int a, int b, int* result) {
  
  int sign = b & 0x80000000;
  int r = sign | (a & 0x7FFFFFFF);

  *result = r;
}

void dpi_fsgnjn(int a, int b, int* result) {
  
  int sign = ~b & 0x80000000;
  int r = sign | (a & 0x7FFFFFFF);

  *result = r;
}

void dpi_fsgnjx(int a, int b, int* result) {
  
  int sign1 = a & 0x80000000;
  int sign2 = b & 0x80000000;
  int r = (sign1 ^ sign2) | (a & 0x7FFFFFFF);

  *result = r;
}