#ifndef RVFLOATS_H
#define RVFLOATS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t rv_fadd(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fsub(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmul(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fnmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fnmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);

uint32_t rv_fdiv(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fsqrt(uint32_t a, uint32_t frm, uint32_t* fflags);

uint32_t rv_ftoi(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_ftou(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_itof(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_utof(uint32_t a, uint32_t frm, uint32_t* fflags);

uint32_t rv_fclss(uint32_t a);
uint32_t rv_fsgnj(uint32_t a, uint32_t b);
uint32_t rv_fsgnjn(uint32_t a, uint32_t b);
uint32_t rv_fsgnjx(uint32_t a, uint32_t b);

uint32_t rv_flt(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_fle(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_feq(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_fmin(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_fmax(uint32_t a, uint32_t b, uint32_t* fflags);

#ifdef __cplusplus
}
#endif

#endif