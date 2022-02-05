#ifndef RVFLOATS_H
#define RVFLOATS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t rv_fadd_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fsub_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmul_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmadd_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fmsub_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fnmadd_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fnmsub_s(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);
uint32_t rv_fdiv_s(uint32_t a, uint32_t b, uint32_t frm, uint32_t* fflags);
uint32_t rv_fsqrt_s(uint32_t a, uint32_t frm, uint32_t* fflags);

uint32_t rv_ftoi_s(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_ftou_s(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_itof_s(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_utof_s(uint32_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_ftol_s(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ftolu_s(uint32_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_ltof_s(uint64_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_lutof_s(uint64_t a, uint32_t frm, uint32_t* fflags);

uint32_t rv_fclss_s(uint32_t a);

uint32_t rv_fsgnj_s(uint32_t a, uint32_t b);
uint32_t rv_fsgnjn_s(uint32_t a, uint32_t b);
uint32_t rv_fsgnjx_s(uint32_t a, uint32_t b);

bool rv_flt_s(uint32_t a, uint32_t b, uint32_t* fflags);
bool rv_fle_s(uint32_t a, uint32_t b, uint32_t* fflags);
bool rv_feq_s(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_fmin_s(uint32_t a, uint32_t b, uint32_t* fflags);
uint32_t rv_fmax_s(uint32_t a, uint32_t b, uint32_t* fflags);

///////////////////////////////////////////////////////////////////////////////

uint64_t rv_fadd_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsub_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmul_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fdiv_d(uint64_t a, uint64_t b, uint32_t frm, uint32_t* fflags);
uint64_t rv_fsqrt_d(uint64_t a, uint32_t frm, uint32_t* fflags);

uint64_t rv_fmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmadd_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);
uint64_t rv_fnmsub_d(uint64_t a, uint64_t b, uint64_t c, uint32_t frm, uint32_t* fflags);

uint32_t rv_ftoi_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint32_t rv_ftou_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ftol_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ftolu_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_itof_d(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_utof_d(uint32_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_ltof_d(uint64_t a, uint32_t frm, uint32_t* fflags);
uint64_t rv_lutof_d(uint64_t a, uint32_t frm, uint32_t* fflags);

uint32_t rv_fclss_d(uint64_t a);
uint64_t rv_fsgnj_d(uint64_t a, uint64_t b);
uint64_t rv_fsgnjn_d(uint64_t a, uint64_t b);
uint64_t rv_fsgnjx_d(uint64_t a, uint64_t b);

bool rv_flt_d(uint64_t a, uint64_t b, uint32_t* fflags);
bool rv_fle_d(uint64_t a, uint64_t b, uint32_t* fflags);
bool rv_feq_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_fmin_d(uint64_t a, uint64_t b, uint32_t* fflags);
uint64_t rv_fmax_d(uint64_t a, uint64_t b, uint32_t* fflags);

uint32_t rv_dtof(uint64_t a);
uint64_t rv_ftod(uint32_t a);

#ifdef __cplusplus
}
#endif

#endif