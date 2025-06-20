// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include <rvfloats.h>
#include <util.h>
#include "svdpi.h"
#include "verilated_vpi.h"
#include "VX_config.h"

using namespace vortex;

extern "C" {
  void dpi_fadd(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fsub(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmul(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fnmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fnmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);

  void dpi_fdiv(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fsqrt(bool enable, int dst_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);

  void dpi_ftoi(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_ftou(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_itof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_utof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_f2f(bool enable, int dst_fmt, int src_fmt, int64_t a, int64_t* result);

  void dpi_fclss(bool enable, int dst_fmt, int64_t a, int64_t* result);
  void dpi_fsgnj(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);
  void dpi_fsgnjn(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);
  void dpi_fsgnjx(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);

  void dpi_flt(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fle(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_feq(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fmin(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fmax(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
}

inline uint64_t nan_box(uint32_t value) {
#ifdef XLEN_64
  return value | 0xffffffff00000000;
#else
  return value;
#endif
}

inline uint64_t nan_box16(uint32_t value) {
#ifdef XLEN_64
  return value | 0xffffffffffff0000;
#else
  return value;
#endif
}

inline bool is_nan_boxed(uint64_t value) {
#ifdef XLEN_64
  return (uint32_t(value >> 32) == 0xffffffff);
#else
  return true;
#endif
}

inline bool is_nan_boxed16(uint64_t value) {
#ifdef XLEN_64
  return (uint64_t(value >> 16) == 0xffffffffffff);
#else
  return true;
#endif
}

inline int64_t check_boxing(int64_t a) {
  if (is_nan_boxed(a))
    return a;
  return nan_box(0x7fc00000); // NaN
}

inline int64_t check_boxing16(int64_t a) {
  if (is_nan_boxed16(a))
    return a;
  return nan_box16(0x7fc0); // NaN
}

void dpi_fadd(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fadd_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fadd_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fsub(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fsub_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fsub_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fmul(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fmul_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fmul_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fmadd_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fmadd_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fmsub_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fmsub_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fnmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fnmadd_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fnmadd_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fnmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fnmsub_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fnmsub_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fdiv(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fdiv_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fdiv_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fsqrt(bool enable, int dst_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fsqrt_d(a, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fsqrt_s(check_boxing(a), (*frm & 0x7), fflags));
  }
}

void dpi_ftoi(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    if (src_fmt) {
      *result = rv_ftol_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_ftol_s(check_boxing(a), (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) {
      *result = sext<uint64_t>(rv_ftoi_d(a, (*frm & 0x7), fflags), 32);
    } else {
      *result = sext<uint64_t>(rv_ftoi_s(check_boxing(a), (*frm & 0x7), fflags), 32);
    }
  }
}

void dpi_ftou(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    if (src_fmt) {
      *result = rv_ftolu_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_ftolu_s(check_boxing(a), (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) {
      *result = sext<uint64_t>(rv_ftou_d(a, (*frm & 0x7), fflags), 32);
    } else {
      *result = sext<uint64_t>(rv_ftou_s(check_boxing(a), (*frm & 0x7), fflags), 32);
    }
  }
}

void dpi_itof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    if (src_fmt) {
      *result = rv_ltof_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_itof_d(a, (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) {
      *result = nan_box(rv_ltof_s(a, (*frm & 0x7), fflags));
    } else {
      *result = nan_box(rv_itof_s(a, (*frm & 0x7), fflags));
    }
  }
}

void dpi_utof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    if (src_fmt) {
      *result = rv_lutof_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_utof_d(a, (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) {
      *result = nan_box(rv_lutof_s(a, (*frm & 0x7), fflags));
    } else {
      *result = nan_box(rv_utof_s(a, (*frm & 0x7), fflags));
    }
  }
}

void dpi_f2f(bool enable, int dst_fmt, int src_fmt, int64_t a, int64_t* result) {
  if (!enable)
    return;
  switch (dst_fmt) {
  case 0: // fp32
    switch (src_fmt) {
    case 1: // fp64
      *result = nan_box(rv_dtof(a));
      break;
    case 2: // fp16
      *result = nan_box(rv_htof_s(check_boxing16(a), 0, nullptr));
      break;
    case 3: // bf16
      *result = nan_box(rv_btof_s(check_boxing16(a), 0, nullptr));
      break;
    default:
      std::abort();
    }
    break;
  case 1: // fp64
    switch (src_fmt) {
    case 0: // fp32
      *result = rv_ftod(check_boxing(a));
      break;
    default:
      std::abort();
    }
    break;
  case 2: // fp16
    switch (src_fmt) {
    case 0: // fp32
      *result = nan_box16(rv_ftoh_s(check_boxing(a), 0, nullptr));
      break;
    default:
      std::abort();
    }
    break;
  case 3: // bf16
    switch (src_fmt) {
    case 0: // fp32
      *result = nan_box16(rv_ftob_s(check_boxing(a), 0, nullptr));
      break;
    default:
      std::abort();
    }
    break;
  default:
    std::abort();
  }
}

void dpi_fclss(bool enable, int dst_fmt, int64_t a, int64_t* result) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fclss_d(a);
  } else {
    *result = rv_fclss_s(check_boxing(a));
  }
}

void dpi_fsgnj(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fsgnj_d(a, b);
  } else {
    *result = nan_box(rv_fsgnj_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_fsgnjn(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fsgnjn_d(a, b);
  } else {
    *result = nan_box(rv_fsgnjn_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_fsgnjx(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fsgnjx_d(a, b);
  } else {
    *result = nan_box(rv_fsgnjx_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_flt(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_flt_d(a, b, fflags);
  } else {
    *result = rv_flt_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_fle(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fle_d(a, b, fflags);
  } else {
    *result = rv_fle_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_feq(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_feq_d(a, b, fflags);
  } else {
    *result = rv_feq_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_fmin(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fmin_d(a, b, fflags);
  } else {
    *result = nan_box(rv_fmin_s(check_boxing(a), check_boxing(b), fflags));
  }
}

void dpi_fmax(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable)
    return;
  if (dst_fmt) {
    *result = rv_fmax_d(a, b, fflags);
  } else {
    *result = nan_box(rv_fmax_s(check_boxing(a), check_boxing(b), fflags));
  }
}