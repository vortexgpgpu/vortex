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

#ifdef XLEN_64
#define INT_TYPE int64_t
#define API_CALL(x) rv_ ## x ## _d
#else
#define INT_TYPE int32_t
#define API_CALL(x) rv_ ## x ## _s
#endif

extern "C" {
  void dpi_fadd(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fsub(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fmul(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fmadd(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fmsub(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fnmadd(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fnmsub(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);

  void dpi_fdiv(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fsqrt(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  
  void dpi_ftoi(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_ftou(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_itof(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_utof(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags);

  void dpi_fclss(bool enable, INT_TYPE a, INT_TYPE* result);
  void dpi_fsgnj(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result);
  void dpi_fsgnjn(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result);
  void dpi_fsgnjx(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result);

  void dpi_flt(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fle(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_feq(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fmin(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags);
  void dpi_fmax(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags);
}

void dpi_fadd(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fadd)(a, b, (*frm & 0x7), fflags);
}

void dpi_fsub(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fsub)(a, b, (*frm & 0x7), fflags);
}

void dpi_fmul(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fmul)(a, b, (*frm & 0x7), fflags);
}

void dpi_fmadd(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fmadd)(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fmsub(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fmsub)(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmadd(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fnmadd)(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fnmsub(bool enable, INT_TYPE a, INT_TYPE b, int c, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fnmsub)(a, b, c, (*frm & 0x7), fflags);
}

void dpi_fdiv(bool enable, INT_TYPE a, INT_TYPE b, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fdiv)(a, b, (*frm & 0x7), fflags);
}

void dpi_fsqrt(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fsqrt)(a, (*frm & 0x7), fflags);
}

void dpi_ftoi(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(ftoi)(a, (*frm & 0x7), fflags);
}

void dpi_ftou(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(ftou)(a, (*frm & 0x7), fflags);
}

void dpi_itof(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(itof)(a, (*frm & 0x7), fflags);
}

void dpi_utof(bool enable, int a, const svBitVecVal* frm, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(utof)(a, (*frm & 0x7), fflags);
}

void dpi_flt(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(flt)(a, b, fflags);
}

void dpi_fle(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fle)(a, b, fflags);
}

void dpi_feq(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(feq)(a, b, fflags);
}

void dpi_fmin(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fmin)(a, b, fflags);
}

void dpi_fmax(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  *result = API_CALL(fmax)(a, b, fflags);
}

void dpi_fclss(bool enable, INT_TYPE a, INT_TYPE* result) {
  if (!enable) 
    return;
  *result = API_CALL(fclss)(a);
}

void dpi_fsgnj(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result) {
  if (!enable) 
    return;
  *result = API_CALL(fsgnj)(a, b);
}

void dpi_fsgnjn(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result) {
  if (!enable) 
    return;
  *result = API_CALL(fsgnjn)(a, b);
}

void dpi_fsgnjx(bool enable, INT_TYPE a, INT_TYPE b, INT_TYPE* result) {
  if (!enable) 
    return;
  *result = API_CALL(fsgnjx)(a, b);
}