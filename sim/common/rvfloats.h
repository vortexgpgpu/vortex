#ifndef RVFLOATS_H
#define RVFLOATS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t rv_fadd(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsub(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmul(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmsub(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fdiv(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsqrt(uint32_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_ftoi(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ftou(uint32_t a, uint32_t frm, uint32_t* fflags);
// simx64
uint64_t rv_ftol(uint32_t a, uint32_t frm, uint32_t* fflags);
// simx64
uint64_t rv_ftolu(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_itof(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_utof(uint32_t a, uint32_t frm, uint32_t* fflags);
// simx64
uint64_t rv_ltof(uint64_t a, uint32_t frm, uint32_t* fflags);
// simx64
uint64_t rv_lutof(uint64_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_fclss(uint32_t a);
uint64_t rv_fsgnj(uint32_t a, uint32_t b);
uint64_t rv_fsgnjn(uint32_t a, uint32_t b);
uint64_t rv_fsgnjx(uint32_t a, uint32_t b);

uint64_t rv_flt(uint32_t a, uint32_t b, uint32_t* fflags);
uint64_t rv_fle(uint32_t a, uint32_t b, uint32_t* fflags);
uint64_t rv_feq(uint32_t a, uint32_t b, uint32_t* fflags);
uint64_t rv_fmin(uint32_t a, uint32_t b, uint32_t* fflags);
uint64_t rv_fmax(uint32_t a, uint32_t b, uint32_t* fflags);



// simx64
uint64_t rv_fadd_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsub_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmul_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fdiv_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsqrt_d(uint64_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_fmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);

uint64_t rv_ftoi_d(uint64_t a, uint64_t frm, uint32_t* fflags);
uint64_t rv_ftou_d(uint64_t a, uint64_t frm, uint32_t* fflags);
uint64_t rv_ftol_d(uint64_t a, uint64_t frm, uint32_t* fflags);
uint64_t rv_ftolu_d(uint64_t a, uint64_t frm, uint32_t* fflags);
uint64_t rv_itof_d(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_utof_d(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ltof_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_lutof_d(uint64_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_fclss_d(uint64_t a);
uint64_t rv_fsgnj_d(uint64_t a, uint64_t b);
uint64_t rv_fsgnjn_d(uint64_t a, uint64_t b);
uint64_t rv_fsgnjx_d(uint64_t a, uint64_t b);

uint64_t rv_flt_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_fle_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_feq_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_fmin_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_fmax_d(uint64_t a, uint64_t b, uint32_t* fflags);

uint64_t rv_dtof(uint64_t a);
uint64_t rv_ftod(uint32_t a);

#ifdef __cplusplus
}
#endif

#endif