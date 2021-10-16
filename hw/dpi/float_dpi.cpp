#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include <rvfloats.h>
#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

extern "C" {
  void dpi_fadd(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fsub(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fmul(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fnmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fnmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags);

  void dpi_fdiv(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_fsqrt(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  
  void dpi_ftoi(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_ftou(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_itof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags);
  void dpi_utof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags);

  void dpi_fclss(bool enable, int a, int* result);
  void dpi_fsgnj(bool enable, int a, int b, int* result);
  void dpi_fsgnjn(bool enable, int a, int b, int* result);
  void dpi_fsgnjx(bool enable, int a, int b, int* result);

  void dpi_flt(bool enable, int a, int b, int* result, svBitVecVal* fflags);
  void dpi_fle(bool enable, int a, int b, int* result, svBitVecVal* fflags);
  void dpi_feq(bool enable, int a, int b, int* result, svBitVecVal* fflags);
  void dpi_fmin(bool enable, int a, int b, int* result, svBitVecVal* fflags);
  void dpi_fmax(bool enable, int a, int b, int* result, svBitVecVal* fflags);
}

void dpi_fadd(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fadd(a, b, (*frm & 0x7), fflags);
}

void dpi_fsub(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fsub(a, b, (*frm & 0x7), fflags);
}

void dpi_fmul(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmul(a, b, (*frm & 0x7), fflags);
}

void dpi_fmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmadd(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmsub(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fnmadd(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fnmsub(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fdiv(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fdiv(a, b, (*frm & 0x7), fflags);
}

void dpi_fsqrt(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fsqrt(a, (*frm & 0x7), fflags);
}

void dpi_ftoi(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_ftoi(a, (*frm & 0x7), fflags);
}

void dpi_ftou(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_ftou(a, (*frm & 0x7), fflags);
}

void dpi_itof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_itof(a, (*frm & 0x7), fflags);
}

void dpi_utof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_utof(a, (*frm & 0x7), fflags);
}

void dpi_flt(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_flt(a, b, fflags);
}

void dpi_fle(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fle(a, b, fflags);
}

void dpi_feq(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_feq(a, b, fflags);
}

void dpi_fmin(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmin(a, b, fflags);
}

void dpi_fmax(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmax(a, b, fflags);
}

void dpi_fclss(bool enable, int a, int* result) {
  if (!enable) 
    return;
  *result = rv_fclss(a);
}

void dpi_fsgnj(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnj(a, b);
}

void dpi_fsgnjn(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnjn(a, b);
}

void dpi_fsgnjx(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnjx(a, b);
}