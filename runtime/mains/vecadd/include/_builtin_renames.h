/* pocl/_kernel_renames.h - Rename OpenCL builtin functions to avoid name
   clashes with libm functions which are called in implementation.

   Copyright (c) 2011-2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                           Perimeter Institute for Theoretical Physics
   Copyright (c) 2011-2017 Pekka Jääskeläinen / TUT

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef _KERNEL_RENAMES_H
#define _KERNEL_RENAMES_H

/* Move built-in declarations and libm functions out of the way.
  (There should be a better way of doing so. These functions are
  built-in math functions for OpenCL (see Clang's "Builtins.def").
  Functions defined in libc or libm may also
  interfere with OpenCL's functions, since their prototypes will be
  wrong. */
#define abs            _cl_abs
#define abs_diff       _cl_abs_diff
#define acos           _cl_acos
#define acosh          _cl_acosh
#define acospi         _cl_acospi
#define add_sat        _cl_add_sat
#define all            _cl_all
#define any            _cl_any
#define asin           _cl_asin
#define asinh          _cl_asinh
#define asinpi         _cl_asinpi
#define atan           _cl_atan
#define atan2          _cl_atan2
#define atan2pi        _cl_atan2pi
#define atanh          _cl_atanh
#define atanpi         _cl_atanpi
#define bitselect      _cl_bitselect
#define cbrt           _cl_cbrt
#define ceil           _cl_ceil
#define clamp          _cl_clamp
#define clz            _cl_clz
#define copysign       _cl_copysign
#define cos            _cl_cos
#define cosh           _cl_cosh
#define cospi          _cl_cospi
#define cross          _cl_cross
#define degrees        _cl_degrees
#define distance       _cl_distance
#define dot            _cl_dot
#define erf            _cl_erf
#define erfc           _cl_erfc
#define exp            _cl_exp
#define exp10          _cl_exp10
#define exp2           _cl_exp2
#define expm1          _cl_expm1
#define fabs           _cl_fabs
#define fast_distance  _cl_fast_distance
#define fast_length    _cl_fast_length
#define fast_normalize _cl_fast_normalize
#define fdim           _cl_fdim
#define floor          _cl_floor
#define fma            _cl_fma
#define fmax           _cl_fmax
#define fmin           _cl_fmin
#define fmod           _cl_fmod
#define fract          _cl_fract
#define frexp          _cl_frexp
#define hadd           _cl_hadd
#define half_cos       _cl_half_cos
#define half_divide    _cl_half_divide
#define half_exp       _cl_half_exp
#define half_exp10     _cl_half_exp10
#define half_exp2      _cl_half_exp2
#define half_log       _cl_half_log
#define half_log10     _cl_half_log10
#define half_log2      _cl_half_log2
#define half_powr      _cl_half_powr
#define half_recip     _cl_half_recip
#define half_rsqrt     _cl_half_rsqrt
#define half_sin       _cl_half_sin
#define half_sqrt      _cl_half_sqrt
#define half_tan       _cl_half_tan
#define hypot          _cl_hypot
#define ilogb          _cl_ilogb
#define isequal        _cl_isequal
#define isfinite       _cl_isfinite
#define isgreater      _cl_isgreater
#define isgreaterequal _cl_isgreaterequal
#define isinf          _cl_isinf
#define isless         _cl_isless
#define islessequal    _cl_islessequal
#define islessgreater  _cl_islessgreater
#define isnan          _cl_isnan
#define isnormal       _cl_isnormal
#define isnotequal     _cl_isnotequal
#define isordered      _cl_isordered
#define isunordered    _cl_isunordered
#define ldexp          _cl_ldexp
#define length         _cl_length
#define lgamma         _cl_lgamma
#define lgamma_r       _cl_lgamma_r
#define log            _cl_log
#define log10          _cl_log10
#define log1p          _cl_log1p
#define log2           _cl_log2
#define logb           _cl_logb
#define mad            _cl_mad
#define mad24          _cl_mad24
#define mad_hi         _cl_mad_hi
#define mad_sat        _cl_mad_sat
#define max            _cl_max
#define maxmag         _cl_maxmag
#define min            _cl_min
#define minmag         _cl_minmag
#define mix            _cl_mix
#define modf           _cl_modf
#define mul24          _cl_mul24
#define mul_hi         _cl_mul_hi
#define nan            _cl_nan
#define native_cos     _cl_native_cos
#define native_divide  _cl_native_divide
#define native_exp     _cl_native_exp
#define native_exp10   _cl_native_exp10
#define native_exp2    _cl_native_exp2
#define native_log     _cl_native_log
#define native_log10   _cl_native_log10
#define native_log2    _cl_native_log2
#define native_powr    _cl_native_powr
#define native_recip   _cl_native_recip
#define native_rsqrt   _cl_native_rsqrt
#define native_sin     _cl_native_sin
#define native_sqrt    _cl_native_sqrt
#define native_tan     _cl_native_tan
#define nextafter      _cl_nextafter
#define normalize      _cl_normalize
#define popcount       _cl_popcount
#define pow            _cl_pow
#define pown           _cl_pown
#define powr           _cl_powr
#define radians        _cl_radians
#define remainder      _cl_remainder
#define remquo         _cl_remquo
#define rhadd          _cl_rhadd
#define rint           _cl_rint
#define rootn          _cl_rootn
#define rotate         _cl_rotate
#define round          _cl_round
#define rsqrt          _cl_rsqrt
#define select         _cl_select
#define sign           _cl_sign
#define signbit        _cl_signbit
#define sin            _cl_sin
#define sincos         _cl_sincos
#define sinh           _cl_sinh
#define sinpi          _cl_sinpi
#define smoothstep     _cl_smoothstep
#define sqrt           _cl_sqrt
#define step           _cl_step
#define sub_sat        _cl_sub_sat
#define tan            _cl_tan
#define tanh           _cl_tanh
#define tanpi          _cl_tanpi
#define tgamma         _cl_tgamma
#define trunc          _cl_trunc
#define upsample       _cl_upsample
#define atom_add     atomic_add
#define atom_sub     atomic_sub
#define atom_xchg    atomic_xchg
#define atom_inc     atomic_inc
#define atom_dec     atomic_dec
#define atom_cmpxchg atomic_cmpxchg
#define atom_min     atomic_min
#define atom_max     atomic_max
#define atom_and     atomic_and
#define atom_or      atomic_or
#define atom_xor     atomic_xor

#endif
