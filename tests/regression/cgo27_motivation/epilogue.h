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

// True when app's epilogue is a pure elementwise map that epi_apply() handles.
// Apps outside this set fall through epi_apply() unchanged (baseline behavior)
// until their extra passes land in Phase B.
static inline bool epi_is_elementwise(uint32_t app) {
  return app == 2 || app == 3;
}

// True when app needs a ROW-WISE reduction, which no mode can fuse. A tile holds only
// tileN of a row's N columns, so the row max and the row sum are not available until every
// tile of that row is written -- the in-core modes lose their fusion advantage here just
// as completely as the engine does, and that is precisely why this app is in the sweep.
// Costs every mode one extra full pass over D; costs the DTCU modes a THIRD, since their
// elementwise pass is already a second launch.
static inline bool epi_needs_row_pass(uint32_t app) {
  return app == 6;
}

static inline float epi_apply(uint32_t app, float v) {
  if (app == 2) return epi_relu(v);
  if (app == 3) return epi_gelu(v);
  return v;   // app 1 baseline, and apps 4-8 until Phase B wires their passes
}

#endif // _CGO27_EPILOGUE_H_
