#ifndef _CGO27_EPILOGUE_H_
#define _CGO27_EPILOGUE_H_

// app id -> epilogue dispatch, shared by the kernel and the host CPU reference.
//
// Apps (see 260718_moti_RFC.md §3):
//   1 baseline D = C+A*B                     (no epilogue)
//   2 + ReLU                                  elementwise      [WIRED]
//   3 + GELU                                  elementwise      [WIRED]
//   4 + Residual (+R)                          needs R matrix   [Phase B]
//   5 + Scale (per-channel)                    needs s vector   [Phase B]
//   6 + Softmax (row-wise)                     cross-row pass   [Phase B]
//   7 dequant(int8->fp16) + bias + GELU        prologue + bias  [Phase B]
//   8 dequant(int8->fp16) + softmax            prologue + pass  [Phase B]
//
// Only apps 1-3 can be expressed as a pure float->float map, which is what makes
// them fusible on the accumulator fragment while it is still in registers. The
// others need either extra operand data (4,5,7) or a reduction across the row
// (6,8), so they are applied as separate passes and are wired in Phase B; their
// math already lives in epilogue/*.h.
//
// epi_apply() is deliberately the ONLY place the app id is decoded, so the kernel
// paths and the host reference cannot drift apart.

#include "epilogue/relu.h"
#include "epilogue/gelu.h"
#include "epilogue/residual.h"
#include "epilogue/scale.h"
#include "epilogue/softmax.h"
#include "epilogue/dequant.h"

// The predicates and the map are all decided by MOTI_APP at compile time -- see common.h
// for why. They keep their old signatures so nothing downstream had to change shape, but
// the argument is now only ever checked against the build, never used to select.

static inline bool epi_is_elementwise(uint32_t) { return MOTI_APP_IS_ELEMENTWISE; }
static inline bool epi_needs_row_pass(uint32_t) { return MOTI_APP_NEEDS_ROW_PASS; }
static inline bool epi_needs_col_pass(uint32_t) { return MOTI_APP_NEEDS_COL_PASS; }
static inline bool epi_needs_aux(uint32_t)      { return MOTI_APP_NEEDS_AUX; }

// Elements of the auxiliary operand this app reads. The host allocates C and this
// contiguously and passes the pair's base as C_addr; the kernel finds it at
// MOTI_AUX_ELEM_OFFSET(M, N). Zero when the app needs none.
static inline uint32_t epi_aux_elems(uint32_t, uint32_t M, uint32_t N) {
  (void)M; (void)N;
#if   MOTI_APP == 4
  return M * N;      // residual, one per output element
#elif MOTI_APP == 5
  return N;          // per-channel scale, one per column
#else
  return 0;
#endif
}

// ONE epilogue per binary. At MOTI_APP=1 and 6 this is the identity and the compiler
// deletes it outright: app 6's work is not elementwise and lives in moti_softmax instead.
static inline float epi_apply(uint32_t, float v);

// THE epilogue, applied where the element's (row, col) is known.
//
// Apps 2 and 3 do not need the position and apps 1/6/9 do nothing here, but apps 4 and 5
// do: a residual is indexed per element and a per-channel scale per column. Giving every
// call site the position costs nothing at MOTI_APP != 4,5 -- the arguments are unused and
// the compiler drops them -- and it means one entry point instead of two.
static inline float epi_apply_at(float v, const float* aux,
                                 uint32_t row, uint32_t col, uint32_t N) {
  // Which of these a given build reads depends on MOTI_APP: app 4 needs both indices,
  // app 5 only the column, the rest none. Silence them all rather than guarding each.
  (void)aux; (void)row; (void)col; (void)N;
#if   MOTI_APP == 4
  return v + aux[row * N + col];      // residual
#elif MOTI_APP == 5
  return v * aux[col];                // per-channel scale, broadcast down the column
#else
  return epi_apply(0, v);
#endif
}

static inline float epi_apply(uint32_t, float v) {
#if   MOTI_APP == 2
  return epi_relu(v);
#elif MOTI_APP == 3
  return epi_gelu(v);
#else
  return v;      // 1 baseline; 6 does its work in the row pass; 4/5/7/8 not built
#endif
}

#endif // _CGO27_EPILOGUE_H_
