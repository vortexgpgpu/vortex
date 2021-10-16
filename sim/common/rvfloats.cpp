#include "rvfloats.h"
#include <stdio.h>

extern "C" {
#include <softfloat.h>
#include <softfloat/source/include/internals.h>
#include <softfloat/source/RISCV/specialize.h>
}

#define F32_SIGN 0x80000000

inline float32_t to_float32_t(uint32_t x) { return float32_t{x}; }

inline uint32_t from_float32_t(float32_t x) { return uint32_t(x.v); }

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

uint32_t rv_fadd(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_add(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fsub(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_sub(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fmul(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_mul(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_mulAdd(to_float32_t(a), to_float32_t(b), to_float32_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  int c_neg = c ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a), to_float32_t(b), to_float32_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fnmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  int a_neg = a ^ F32_SIGN;
  int c_neg = c ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a_neg), to_float32_t(b), to_float32_t(c_neg));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fnmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  int a_neg = a ^ F32_SIGN;
  auto r = f32_mulAdd(to_float32_t(a_neg), to_float32_t(b), to_float32_t(c));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fdiv(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_div(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_fsqrt(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_sqrt(to_float32_t(a));
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_ftoi(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_i32(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_ftou(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = f32_to_ui32(to_float32_t(a), frm, true);
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_itof(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = i32_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_utof(uint32_t a, uint32_t frm, uint32_t* fflags) {
  softfloat_roundingMode = frm;
  auto r = ui32_to_f32(a);
  if (fflags) { *fflags = get_fflags(); }
  return from_float32_t(r);
}

uint32_t rv_flt(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_lt(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_fle(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_le(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }
  return r;
}

uint32_t rv_feq(uint32_t a, uint32_t b, uint32_t* fflags) {
  auto r = f32_eq(to_float32_t(a), to_float32_t(b));
  if (fflags) { *fflags = get_fflags(); }  
  return r;
}

uint32_t rv_fmin(uint32_t a, uint32_t b, uint32_t* fflags) {  
  int r;
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

uint32_t rv_fmax(uint32_t a, uint32_t b, uint32_t* fflags) {
  int r;
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

uint32_t rv_fclss(uint32_t a) {
  auto infOrNaN      = (0xff == expF32UI(a));
  auto subnormOrZero = (0 == expF32UI(a));
  bool sign          = signF32UI(a);
  bool fracZero      = (0 == fracF32UI(a));
  bool isNaN         = isNaNF32UI(a);
  bool isSNaN        = softfloat_isSigNaNF32UI(a);

  int r =
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

uint32_t rv_fsgnj(uint32_t a, uint32_t b) {
  
  int sign = b & F32_SIGN;
  int r = sign | (a & ~F32_SIGN);

  return r;
}

uint32_t rv_fsgnjn(uint32_t a, uint32_t b) {
  
  int sign = ~b & F32_SIGN;
  int r = sign | (a & ~F32_SIGN);

  return r;
}

uint32_t rv_fsgnjx(uint32_t a, uint32_t b) {
  
  int sign1 = a & F32_SIGN;
  int sign2 = b & F32_SIGN;
  int r = (sign1 ^ sign2) | (a & ~F32_SIGN);

  return r;
}

#ifdef __cplusplus
}
#endif
