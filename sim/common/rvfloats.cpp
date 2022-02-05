#include "rvfloats.h"
#include <stdio.h>

extern "C" {
#include <softfloat.h>
#include <internals.h>
#include <../RISCV/specialize.h>
}

#define F32_SIGN 0x80000000
#define F64_SIGN 0x8000000000000000

inline float32_t to_float32_t(uint32_t x) { return float32_t{x}; }
inline float64_t to_float64_t(uint64_t x) { return float64_t{x}; }

inline uint32_t from_float32_t(float32_t x) { return uint32_t(x.v); }
inline uint64_t from_float64_t(float64_t x) { return uint64_t(x.v); }

inline uint32_t get_fflags() {
  uint32_t fflags = softfloat_exceptionFlags;
  if (fflags) {
    softfloat_exceptionFlags = 0; 
  }
  return fflags;
}

#ifdef __cplusplus
extern "C" {
#endif

uint32_t rv_fadd_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_add(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fadd_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_add(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fsub_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_sub(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fsub_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_sub(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fmul_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_mul(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fmul_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_mul(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fmadd_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_mulAdd(to_float32_t(a), to_float32_t(b), to_float32_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_mulAdd(to_float64_t(a), to_float64_t(b), to_float64_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fmsub_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto c_neg = c ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a), to_float32_t(b), to_float32_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto c_neg = c ^ F64_SIGN;
  auto r = f64_mulAdd(to_float64_t(a), to_float64_t(b), to_float64_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fnmadd_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto a_neg = a ^ F32_SIGN;
  auto c_neg = c ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a_neg), to_float32_t(b), to_float32_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fnmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto a_neg = a ^ F64_SIGN;
  auto c_neg = c ^ F64_SIGN;
  auto r = f64_mulAdd(to_float64_t(a_neg), to_float64_t(b), to_float64_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fnmsub_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto a_neg = a ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a_neg), to_float32_t(b), to_float32_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fnmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto a_neg = a ^ F64_SIGN;
  auto r = f64_mulAdd(to_float64_t(a_neg), to_float64_t(b), to_float64_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fdiv_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_div(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fdiv_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_div(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_fsqrt_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_sqrt(to_float32_t(a));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_fsqrt_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_sqrt(to_float64_t(a));
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_ftoi_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_i32(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_ftoi_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_to_i32(to_float64_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_ftou_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_ui32(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_ftou_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_to_ui32(to_float64_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_ftol_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_i64(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_ftol_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_to_i64(to_float64_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_ftolu_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_ui64(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_ftolu_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f64_to_ui64(to_float64_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_itof_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = i32_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_itof_d(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = i32_to_f64(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_utof_s(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = ui32_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_utof_d(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = ui32_to_f64(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_ltof_s(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = i64_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_ltof_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = i64_to_f64(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

uint32_t rv_lutof_s(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = ui64_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint64_t rv_lutof_d(uint64_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = ui64_to_f64(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float64_t(r);
}

bool rv_flt_s(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_lt(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

bool rv_flt_d(uint64_t a, uint64_t b, uint32_t* fflags) {
  auto r = f64_lt(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

bool rv_fle_s(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_le(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

bool rv_fle_d(uint64_t a, uint64_t b, uint32_t* fflags) {
  auto r = f64_le(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

bool rv_feq_s(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_eq(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }  
  return r;
}

bool rv_feq_d(uint64_t a, uint64_t b, uint32_t* fflags) {
  auto r = f64_eq(to_float64_t(a), to_float64_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_fmin_s(uint32_t a, uint32_t b, uint32_t* fflags) {  
  uint32_t r;
  if (isNaNF32UI(a) && isNaNF32UI(b)) {
    r = defaultNaNF32UI;   
  } else {
    auto fa = to_float32_t(a);
    auto fb = to_float32_t(b);
    if ((f32_lt_quiet(fa, fb) || (f32_eq(fa, fb) && (a & F32_SIGN)))
     || isNaNF32UI(b)) {
      r = a;
    } else {
      r = b;
    }
  }
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_fmin_d(uint64_t a, uint64_t b, uint32_t* fflags) {  
  uint64_t r;
  if (isNaNF64UI(a) && isNaNF64UI(b)) {
    r = defaultNaNF64UI;   
  } else {
    auto fa = to_float64_t(a);
    auto fb = to_float64_t(b);
    if ((f64_lt_quiet(fa, fb) || (f64_eq(fa, fb) && (a & F64_SIGN)))
     || isNaNF64UI(b)) {
      r = a;
    } else {
      r = b;
    }
  }
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_fmax_s(uint32_t a, uint32_t b, uint32_t* fflags) {
  uint32_t r;
  if (isNaNF32UI(a) && isNaNF32UI(b)) {
    r = defaultNaNF32UI;   
  } else {
    auto fa = to_float32_t(a);
    auto fb = to_float32_t(b);
    if ((f32_lt_quiet(fb, fa) || (f32_eq(fb, fa) && (b & F32_SIGN)))
     || isNaNF32UI(b)) {
      r = a;
    } else {
      r = b;
    }
  }
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint64_t rv_fmax_d(uint64_t a, uint64_t b, uint32_t* fflags) {
  uint64_t r;
  if (isNaNF64UI(a) && isNaNF64UI(b)) {
    r = defaultNaNF64UI;   
  } else {
    auto fa = to_float64_t(a);
    auto fb = to_float64_t(b);
    if ((f64_lt_quiet(fb, fa) || (f64_eq(fb, fa) && (b & F64_SIGN)))
     || isNaNF64UI(b)) {
      r = a;
    } else {
      r = b;
    }
  }
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_fclss_s(uint32_t a) {
  auto infOrNaN      = (0xff == expF32UI(a));
  auto subnormOrZero = (0 == expF32UI(a));
  bool sign          = signF32UI(a);
  bool fracZero      = (0 == fracF32UI(a));
  bool isNaN         = isNaNF32UI(a);
  bool isSNaN        = softfloat_isSigNaNF32UI(a);

  uint32_t r =
      (  sign && infOrNaN && fracZero )        << 0 |
      (  sign && !infOrNaN && !subnormOrZero ) << 1 |
      (  sign && subnormOrZero && !fracZero )  << 2 |
      (  sign && subnormOrZero && fracZero )   << 3 |
      ( !sign && infOrNaN && fracZero )        << 7 |
      ( !sign && !infOrNaN && !subnormOrZero ) << 6 |
      ( !sign && subnormOrZero && !fracZero )  << 5 |
      ( !sign && subnormOrZero && fracZero )   << 4 |
      ( isNaN &&  isSNaN )                     << 8 |
      ( isNaN && !isSNaN )                     << 9;  
  
  return r;
}

uint32_t rv_fclss_d(uint64_t a) {
  auto infOrNaN      = (0x7ff == expF64UI(a));
  auto subnormOrZero = (0 == expF64UI(a));
  bool sign          = signF64UI(a);
  bool fracZero      = (0 == fracF64UI(a));
  bool isNaN         = isNaNF64UI(a);
  bool isSNaN        = softfloat_isSigNaNF64UI(a);

  uint32_t r =
      (  sign && infOrNaN && fracZero )        << 0 |
      (  sign && !infOrNaN && !subnormOrZero ) << 1 |
      (  sign && subnormOrZero && !fracZero )  << 2 |
      (  sign && subnormOrZero && fracZero )   << 3 |
      ( !sign && infOrNaN && fracZero )        << 7 |
      ( !sign && !infOrNaN && !subnormOrZero ) << 6 |
      ( !sign && subnormOrZero && !fracZero )  << 5 |
      ( !sign && subnormOrZero && fracZero )   << 4 |
      ( isNaN &&  isSNaN )                     << 8 |
      ( isNaN && !isSNaN )                     << 9;  
  
  return r;
}

uint32_t rv_fsgnj_s(uint32_t a, uint32_t b) {
  auto sign = b & F32_SIGN;
  auto r = sign | (a & ~F32_SIGN);
  return r;
}

uint64_t rv_fsgnj_d(uint64_t a, uint64_t b) {  
  auto sign = b & F64_SIGN;
  auto r = sign | (a & ~F64_SIGN);
  return r;
}

uint32_t rv_fsgnjn_s(uint32_t a, uint32_t b) {
  auto sign = ~b & F32_SIGN;
  auto r = sign | (a & ~F32_SIGN);
  return r;
}

uint64_t rv_fsgnjn_d(uint64_t a, uint64_t b) {  
  auto sign = ~b & F64_SIGN;
  auto r = sign | (a & ~F64_SIGN);
  return r;
}

uint32_t rv_fsgnjx_s(uint32_t a, uint32_t b) {
  auto sign1 = a & F32_SIGN;
  auto sign2 = b & F32_SIGN;
  auto r = (sign1 ^ sign2) | (a & ~F32_SIGN);
  return r;
}

uint64_t rv_fsgnjx_d(uint64_t a, uint64_t b) {  
  auto sign1 = a & F64_SIGN;
  auto sign2 = b & F64_SIGN;
  auto r = (sign1 ^ sign2) | (a & ~F64_SIGN);
  return r;
}

uint32_t rv_dtof(uint64_t a) {
  auto r = f64_to_f32(to_float64_t(a));
  return from_float32_t(r);
}

uint64_t rv_ftod(uint32_t a) {
  auto r = f32_to_f64(to_float32_t(a));
  return from_float64_t(r);
}

#ifdef __cplusplus
}
#endif
