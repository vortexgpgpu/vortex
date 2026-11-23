#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#define FOP_S(insn, r, a, b) __asm__ volatile(insn " %0, %1, %2" : "=f"(r) : "f"(a), "f"(b))
#define FCMP(insn, r, a, b)  __asm__ volatile(insn " %0, %1, %2" : "=r"(r) : "f"(a), "f"(b))

static inline __attribute__((always_inline)) float mv_w_x(uint64_t x) {
  float f;
  __asm__ volatile("fmv.w.x %0, %1" : "=f"(f) : "r"((long)x));
  return f;
}

static inline __attribute__((always_inline)) uint64_t mv_x_w(float f) {
  long r;
  __asm__ volatile("fmv.x.w %0, %1" : "=r"(r) : "f"(f));
  return (uint32_t)r;
}

#if __riscv_xlen == 64
// A single-precision op on raw 64-bit register values, result read in full.
#define FOP_S_RAW(insn, r, a, b) __asm__ volatile( \
    "fmv.d.x ft0, %1\n\tfmv.d.x ft1, %2\n\t" insn " ft2, ft0, ft1\n\tfmv.x.d %0, ft2" \
    : "=r"(r) : "r"(a), "r"(b) : "ft0", "ft1", "ft2")
#define FCMP_S_RAW(insn, r, a, b) __asm__ volatile( \
    "fmv.d.x ft0, %1\n\tfmv.d.x ft1, %2\n\t" insn " %0, ft0, ft1" \
    : "=r"(r) : "r"(a), "r"(b) : "ft0", "ft1")

static uint64_t run_raw(uint32_t op, uint64_t a, uint64_t b) {
  uint64_t r = 0;
  switch (op) {
  case FF_OP_FSGNJ:  FOP_S_RAW("fsgnj.s", r, a, b);  break;
  case FF_OP_FSGNJN: FOP_S_RAW("fsgnjn.s", r, a, b); break;
  case FF_OP_FSGNJX: FOP_S_RAW("fsgnjx.s", r, a, b); break;
  case FF_OP_FMIN:   FOP_S_RAW("fmin.s", r, a, b);   break;
  case FF_OP_FMAX:   FOP_S_RAW("fmax.s", r, a, b);   break;
  case FF_OP_FEQ:    FCMP_S_RAW("feq.s", r, a, b);   break;
  case FF_OP_FLT:    FCMP_S_RAW("flt.s", r, a, b);   break;
  case FF_OP_FLE:    FCMP_S_RAW("fle.s", r, a, b);   break;
  default: break;
  }
  return r;
}

static inline __attribute__((always_inline)) double mv_d_x(uint64_t x) {
  double d;
  __asm__ volatile("fmv.d.x %0, %1" : "=f"(d) : "r"(x));
  return d;
}

static inline __attribute__((always_inline)) uint64_t mv_x_d(double d) {
  uint64_t r;
  __asm__ volatile("fmv.x.d %0, %1" : "=r"(r) : "f"(d));
  return r;
}
#endif

static void run_case(const kernel_arg_t* arg, uint32_t i) {
  auto src = reinterpret_cast<const ff_case_t*>(arg->src_addr);
  auto dst = reinterpret_cast<ff_out_t*>(arg->dst_addr);
  uint32_t op = src[i].op;
  float fa = mv_w_x(src[i].a);
  float fb = mv_w_x(src[i].b);
  float fr = 0;
  long cmp = 0;
  uint64_t res = 0;
  long fflags;
#if __riscv_xlen == 64
  if (src[i].raw) {
    __asm__ volatile("csrw fflags, zero");
    res = run_raw(op, src[i].a, src[i].b);
    __asm__ volatile("csrr %0, fflags" : "=r"(fflags));
    dst[i].result = res;
    dst[i].fflags = (uint32_t)fflags;
    return;
  }
#endif
  __asm__ volatile("csrw fflags, zero");
  switch (op) {
  case FF_OP_FSGNJ:  FOP_S("fsgnj.s", fr, fa, fb);  res = mv_x_w(fr); break;
  case FF_OP_FSGNJN: FOP_S("fsgnjn.s", fr, fa, fb); res = mv_x_w(fr); break;
  case FF_OP_FSGNJX: FOP_S("fsgnjx.s", fr, fa, fb); res = mv_x_w(fr); break;
  case FF_OP_FMIN:   FOP_S("fmin.s", fr, fa, fb);   res = mv_x_w(fr); break;
  case FF_OP_FMAX:   FOP_S("fmax.s", fr, fa, fb);   res = mv_x_w(fr); break;
  case FF_OP_FEQ:    FCMP("feq.s", cmp, fa, fb);    res = cmp; break;
  case FF_OP_FLT:    FCMP("flt.s", cmp, fa, fb);    res = cmp; break;
  case FF_OP_FLE:    FCMP("fle.s", cmp, fa, fb);    res = cmp; break;
#if __riscv_xlen == 64
  default: {
    double da = mv_d_x(src[i].a);
    double db = mv_d_x(src[i].b);
    double dr = 0;
    switch (op - FF_NUM_OPS) {
    case FF_OP_FSGNJ:  FOP_S("fsgnj.d", dr, da, db);  res = mv_x_d(dr); break;
    case FF_OP_FSGNJN: FOP_S("fsgnjn.d", dr, da, db); res = mv_x_d(dr); break;
    case FF_OP_FSGNJX: FOP_S("fsgnjx.d", dr, da, db); res = mv_x_d(dr); break;
    case FF_OP_FMIN:   FOP_S("fmin.d", dr, da, db);   res = mv_x_d(dr); break;
    case FF_OP_FMAX:   FOP_S("fmax.d", dr, da, db);   res = mv_x_d(dr); break;
    case FF_OP_FEQ:    FCMP("feq.d", cmp, da, db);    res = cmp; break;
    case FF_OP_FLT:    FCMP("flt.d", cmp, da, db);    res = cmp; break;
    case FF_OP_FLE:    FCMP("fle.d", cmp, da, db);    res = cmp; break;
    default: break;
    }
  } break;
#else
  default: break;
#endif
  }
  __asm__ volatile("csrr %0, fflags" : "=r"(fflags));
  dst[i].result = res;
  dst[i].fflags = (uint32_t)fflags;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // fflags is shared by the warp: run one lane at a time so each case sees only its own flags.
  int nt = vx_num_threads();
  for (int l = 0; l < nt; ++l) {
    if (vx_thread_id() == l && i < arg->count) {
      run_case(arg, i);
    }
  }
}
