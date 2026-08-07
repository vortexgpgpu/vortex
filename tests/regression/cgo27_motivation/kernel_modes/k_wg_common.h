#ifndef _CGO27_K_WG_COMMON_H_
#define _CGO27_K_WG_COMMON_H_

// Geometry and epilogue shared by the workgroup pair, modes 12 and 13.
//
// Those two are a matched difference measurement: 12 stages its operands with a DXA
// descriptor, 13 with the CTA's own loads, and NOTHING else is allowed to differ, because
// the ratio between them is reported as what the copy engine is worth. An epilogue
// duplicated in both files is a place for that comparison to quietly stop being one, so
// the parts that must stay identical live here and each .cpp keeps only its own body.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>

// Workgroup MMA geometry. Separate from `ctx` (the per-warp wmma_context every other
// in-core mode uses) because the fragment and tile shapes differ.
using wgctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// The accumulator half of that context, spelled out because wgmma_context keeps its own
// alias private. Same template arguments, so wgctx::fragment_acc IS accctx::fragment_acc.
using accctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// K-steps held in ONE staged tile. S copies amortise into a single issue and a single
// barrier pair, which is the reuse axis: at S=1 a staged tile feeds one MMA, at S=4 it
// feeds four. lmem and the DXA descriptor scale with it, and the host sizes both from the
// same macro -- changing it here alone would silently mis-tile.
#ifndef MOTI_WG_KSTEPS
#define MOTI_WG_KSTEPS 1
#endif
static constexpr uint32_t kS   = MOTI_WG_KSTEPS;
static constexpr uint32_t kStK = kS * wgctx::tileK;   // columns per staged tile

// The accumulator tile must be the WGMMA output tile, or D is stored to the wrong
// rectangle -- which is silent, not a crash.
static_assert(accctx::tileM == wgctx::xtileM, "acc tileM != wgmma xtileM");
static_assert(accctx::tileN == wgctx::xtileN, "acc tileN != wgmma xtileN");

// ---------------------------------------------------------------------------------
// D = epi(app, C + A*B) for one CTA's cta_M x xtileN output tile.
//
// The awkward part is that A*B sits in a WGMMA accumulator whose thread->element mapping
// is not the WMMA one. vx_tensor.h:789 refuses to LOAD an accumulator for exactly that
// reason, so C cannot be preloaded the way wmma_seed_C does it for modes 1..6 -- tried,
// and it put C in the wrong lanes: 24,173 of 32,768 elements wrong, exactly one warp in
// four correct, identically for both modes. store_matrix_sync is the only thing that
// knows the layout, so the accumulator must be stored before C can be added.
//
// WHERE it goes is the entire cost question, and it used to be answered wrong. Storing
// to global D and then reading it back to add C makes FOUR M*N global accesses -- store
// D, read C, read D, write D -- against the TWO (read C, write D) the in-core modes make.
// At 512x256x128 that is C+D = 1,024 KB of read-write traffic against a 1,024 KB L2 with
// A (128 KB) and B (64 KB) also live, and the pair simply stopped completing: over four
// hours, against 335,171 / 494,464 cycles for the same kernels with the pass compiled
// out. Below that shape the same code merely cost 58-79 %.
//
// So C is folded in HERE, while the accumulator is still in registers, and D is written
// once. Two M*N accesses, no second pass, no scratch, no barrier -- each warp owns its
// own xtileM x xtileN rectangle and never looks at another's.
//
// The layout that defeated the preload is not actually a secret: store_matrix_sync
// (vx_tensor.h:944) computes it in the open, and the loop below is that computation with
// a read of C and an add spliced in. Mirroring the accumulator's OWN indexing is what
// makes this correct where seeding through the WMMA layout was not -- the failure there
// was reading C at WMMA addresses, not anything about registers.
//
// Two other placements were built and measured before this one, and both are worse:
//
//   global D + read back   the original. Four M*N accesses; does not complete at
//                          512x256x128.
//   store to LMEM, then    Two M*N global accesses, but it pays an LMEM round trip and a
//   a cooperative pass     barrier for every output element, and store_matrix_sync's
//                          lane->address map (lane/tcN rows, lane%tcN cols, 64 B apart)
//                          touches 8 of the 32 LMEM banks four times each. Correct, and
//                          slower than the original at both shapes that fit in L2:
//                          mode 13 63,339 -> 77,386 at 128x64x32 and 261,673 -> 281,189
//                          at 256x128x64.
//
// always_inline for the reason wmma_common.h documents: an out-of-line helper leaks the
// pointers and pushes them to the stack.
//
// -DMOTI_WG_NO_C drops ONLY the `pC[...] +` term, so the difference is the cost of
// streaming C in and nothing else. That is a narrower switch than it used to be: it used
// to remove a whole second pass over D. There is no second pass now, so the old NO_C
// numbers are not comparable to the new ones. That build's D is still wrong on purpose;
// only its cycle count means anything.
//
// out_row is the warp's own first row -- tile_row + warp_rank * xtileM -- not the CTA's.
static __attribute__((always_inline)) void wg_store_epilogue(
    const wgctx::fragment_acc& acc,
    const wgctx::output_t* pC, wgctx::output_t* pD,
    uint32_t out_row, uint32_t tile_col, uint32_t N, uint32_t app) {
  // store_matrix_sync's own base: lane covers row lane/tcN, column lane%tcN of the
  // micro-tile, and register r covers the (r % m_steps, r / m_steps) block of them.
  const uint32_t lane = vx_thread_id();
  const uint32_t o0   = (out_row + lane / wgctx::tcN) * N + tile_col + (lane % wgctx::tcN);

  for (uint32_t r = 0; r < wgctx::fragment_acc::NR; ++r) {
    const uint32_t o = o0 + (r % wgctx::m_steps) * wgctx::tcM * N
                          + (r / wgctx::m_steps) * wgctx::tcN;
#ifndef MOTI_WG_NO_C
    pD[o] = epi_apply(app, pC[o] + acc.data[r]);
#else
    pD[o] = epi_apply(app, acc.data[r]);
#endif
  }
#ifdef MOTI_WG_NO_C
  (void)pC;
#endif
}

#endif // _CGO27_K_WG_COMMON_H_
