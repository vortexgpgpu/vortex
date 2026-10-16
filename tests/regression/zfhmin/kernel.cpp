#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#if __riscv_xlen == 64
typedef uint64_t raw_t;
#define FMV_X_RAW "fmv.x.d"
#else
typedef uint32_t raw_t;
#define FMV_X_RAW "fmv.x.w"
#endif

#define CVT(insn, out, in, rm) __asm__ volatile(insn " %0, %1, " rm : "=f"(out) : "f"(in))

static inline __attribute__((always_inline)) void store_h(uint16_t* p, float h) {
  __asm__ volatile("fsh %0, 0(%1)" :: "f"(h), "r"(p) : "memory");
}

static inline __attribute__((always_inline)) float load_h(const uint16_t* p) {
  float h;
  __asm__ volatile("flh %0, 0(%1)" : "=f"(h) : "r"(p) : "memory");
  return h;
}

static inline __attribute__((always_inline)) long mv_x_h(float h) {
  long r;
  __asm__ volatile("fmv.x.h %0, %1" : "=r"(r) : "f"(h));
  return r;
}

static inline __attribute__((always_inline)) float mv_h_x(long x) {
  float h;
  __asm__ volatile("fmv.h.x %0, %1" : "=f"(h) : "r"(x));
  return h;
}

static inline __attribute__((always_inline)) float mv_w_x(long x) {
  float f;
  __asm__ volatile("fmv.w.x %0, %1" : "=f"(f) : "r"(x));
  return f;
}

static inline __attribute__((always_inline)) raw_t mv_x_raw(float f) {
  raw_t r;
  __asm__ volatile(FMV_X_RAW " %0, %1" : "=r"(r) : "f"(f));
  return r;
}

static inline __attribute__((always_inline)) uint32_t mv_x_w(float f) {
  long r;
  __asm__ volatile("fmv.x.w %0, %1" : "=r"(r) : "f"(f));
  return (uint32_t)r;
}

static void narrow_s(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const float*>(arg->src_addr);
  auto dst = reinterpret_cast<uint16_t*>(arg->dst_addr) + i * ZF_NUM_RM;
  auto mvx = reinterpret_cast<int64_t*>(arg->aux_addr);
  float x = src[i];
  float h;
  CVT("fcvt.h.s", h, x, "rne"); store_h(dst + 0, h); mvx[i] = mv_x_h(h);
  CVT("fcvt.h.s", h, x, "rtz"); store_h(dst + 1, h);
  CVT("fcvt.h.s", h, x, "rdn"); store_h(dst + 2, h);
  CVT("fcvt.h.s", h, x, "rup"); store_h(dst + 3, h);
  CVT("fcvt.h.s", h, x, "rmm"); store_h(dst + 4, h);
  CVT("fcvt.h.s", h, x, "dyn"); store_h(dst + 5, h);
}

static void widen_s(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const uint16_t*>(arg->src_addr);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  float h = load_h(src + i);
  float y;
  __asm__ volatile("fcvt.s.h %0, %1" : "=f"(y) : "f"(h));
  dst[i] = mv_x_w(y);
}

#if __riscv_xlen == 64
static void narrow_d(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const double*>(arg->src_addr);
  auto dst = reinterpret_cast<uint16_t*>(arg->dst_addr) + i * ZF_NUM_RM;
  double x = src[i];
  float h;
  CVT("fcvt.h.d", h, x, "rne"); store_h(dst + 0, h);
  CVT("fcvt.h.d", h, x, "rtz"); store_h(dst + 1, h);
  CVT("fcvt.h.d", h, x, "rdn"); store_h(dst + 2, h);
  CVT("fcvt.h.d", h, x, "rup"); store_h(dst + 3, h);
  CVT("fcvt.h.d", h, x, "rmm"); store_h(dst + 4, h);
  CVT("fcvt.h.d", h, x, "dyn"); store_h(dst + 5, h);
}

static void widen_d(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const uint16_t*>(arg->src_addr);
  auto dst = reinterpret_cast<uint64_t*>(arg->dst_addr);
  float h = load_h(src + i);
  double y;
  __asm__ volatile("fcvt.d.h %0, %1" : "=f"(y) : "f"(h));
  uint64_t r;
  __asm__ volatile("fmv.x.d %0, %1" : "=r"(r) : "f"(y));
  dst[i] = r;
}
#endif

static void flag_case(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const zf_flag_case_t*>(arg->src_addr);
  auto dst = reinterpret_cast<zf_flag_out_t*>(arg->dst_addr);
  uint32_t op = src[i].op;
  uint64_t bits = src[i].bits;
  uint64_t res = 0;
  long fflags;
  __asm__ volatile("fsrm %0" :: "r"(src[i].rm));
  __asm__ volatile("csrw fflags, zero");
  if (op == ZF_OP_H_S) {
    float x = mv_w_x((long)(uint32_t)bits);
    float h;
    CVT("fcvt.h.s", h, x, "dyn");
    res = (uint16_t)mv_x_h(h);
  } else if (op == ZF_OP_S_H) {
    float h = mv_h_x((long)bits);
    float y;
    __asm__ volatile("fcvt.s.h %0, %1" : "=f"(y) : "f"(h));
    res = mv_x_w(y);
  }
#if __riscv_xlen == 64
  else if (op == ZF_OP_H_D) {
    double x;
    __asm__ volatile("fmv.d.x %0, %1" : "=f"(x) : "r"(bits));
    float h;
    CVT("fcvt.h.d", h, x, "dyn");
    res = (uint16_t)mv_x_h(h);
  } else if (op == ZF_OP_D_H) {
    float h = mv_h_x((long)bits);
    double y;
    __asm__ volatile("fcvt.d.h %0, %1" : "=f"(y) : "f"(h));
    __asm__ volatile("fmv.x.d %0, %1" : "=r"(res) : "f"(y));
  }
#endif
  __asm__ volatile("csrr %0, fflags" : "=r"(fflags));
  dst[i].result = res;
  dst[i].fflags = (uint32_t)fflags;
}

static void box(const kernel_arg_t* arg, uint32_t i) {
  auto hsrc = reinterpret_cast<const uint16_t*>(arg->src_addr);
  auto gsrc = reinterpret_cast<const uint32_t*>(arg->aux_addr);
  auto dst = reinterpret_cast<uint64_t*>(arg->dst_addr) + i * ZF_BOX_OUTS;
  long g = (long)gsrc[i];
  dst[0] = (uint64_t)mv_x_raw(load_h(hsrc + i));
  dst[1] = (uint64_t)mv_x_raw(mv_h_x(g));
  dst[2] = (uint64_t)(int64_t)mv_x_h(load_h(hsrc + i));
  float y;
  __asm__ volatile("fcvt.s.h %0, %1" : "=f"(y) : "f"(mv_w_x(g)));
  dst[3] = mv_x_w(y);
  dst[4] = (uint64_t)(int64_t)mv_x_h(mv_w_x(g));
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (arg->mode == ZF_MODE_FLAGS) {
    int nt = vx_num_threads();
    for (int l = 0; l < nt; ++l) {
      if (vx_thread_id() == l && i < arg->count)
        flag_case(arg, i);
    }
    return;
  }
  if (i >= arg->count)
    return;
  __asm__ volatile("fsrm %0" :: "r"(arg->frm));
  switch (arg->mode) {
  case ZF_MODE_NARROW_S: narrow_s(arg, i); break;
  case ZF_MODE_WIDEN_S:  widen_s(arg, i); break;
#if __riscv_xlen == 64
  case ZF_MODE_NARROW_D: narrow_d(arg, i); break;
  case ZF_MODE_WIDEN_D:  widen_d(arg, i); break;
#endif
  case ZF_MODE_BOX:      box(arg, i); break;
  default: break;
  }
}
