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
  *result = rv_fadd_s(a, b, (*frm & 0x7), fflags);
}

void dpi_fsub(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fsub_s(a, b, (*frm & 0x7), fflags);
}

void dpi_fmul(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmul_s(a, b, (*frm & 0x7), fflags);
}

void dpi_fmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmadd_s(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmsub_s(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmadd(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fnmadd_s(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmsub(bool enable, int a, int b, int c, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fnmsub_s(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fdiv(bool enable, int a, int b, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fdiv_s(a, b, (*frm & 0x7), fflags);
}

void dpi_fsqrt(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fsqrt_s(a, (*frm & 0x7), fflags);
}

void dpi_ftoi(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_ftoi_s(a, (*frm & 0x7), fflags);
}

void dpi_ftou(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_ftou_s(a, (*frm & 0x7), fflags);
}

void dpi_itof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_itof_s(a, (*frm & 0x7), fflags);
}

void dpi_utof(bool enable, int a, const svBitVecVal* frm, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_utof_s(a, (*frm & 0x7), fflags);
}

void dpi_flt(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_flt_s(a, b, fflags);
}

void dpi_fle(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fle_s(a, b, fflags);
}

void dpi_feq(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_feq_s(a, b, fflags);
}

void dpi_fmin(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmin_s(a, b, fflags);
}

void dpi_fmax(bool enable, int a, int b, int* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = rv_fmax_s(a, b, fflags);
}

void dpi_fclss(bool enable, int a, int* result) {
  if (!enable) 
    return;
  *result = rv_fclss_s(a);
}

void dpi_fsgnj(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnj_s(a, b);
}

void dpi_fsgnjn(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnjn_s(a, b);
}

void dpi_fsgnjx(bool enable, int a, int b, int* result) {
  if (!enable) 
    return;
  *result = rv_fsgnjx_s(a, b);
}