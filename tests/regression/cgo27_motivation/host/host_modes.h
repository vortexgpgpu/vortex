#ifndef _CGO27_HOST_MODES_H_
#define _CGO27_HOST_MODES_H_

// The mode registry: what modes exist, what each is called, and which are runnable.
//
// Split out of main.cpp so the sweep scripts' companion tools and any future per-mode
// driver share ONE definition. Everything downstream keys off the names here rather than
// integer literals, so moving a mode is a one-line change in this file.
//
// How a mode actually runs lives in run_modes.h; this file only says which exist.

#include <cstdint>

// The HW paths under comparison.
//
// The numbering is grouped by WHAT EXECUTES, and is deliberately not dense:
//   0-2   in-core, increasing operand-staging sophistication
//   3-4   in-core, pipelined by DEPTH, operands staged by the LSU
//   5-6   in-core, pipelined by DEPTH, operands staged by DXA
//   7-8   descriptor engine alone, by PLACEMENT
//   9-11  hetero: cores and engine(s) on the same GEMM at once
//
// 3/5 and 4/6 are MATCHED PAIRS: identical tile geometry, stage count, barrier count and
// lmem footprint, differing only in who copies A and B into Local Memory. That is what
// isolates the copy engine's contribution at each pipeline depth -- neither pair could
// do it alone. (3 and 4 previously held the retired DTCU no-TMA / TMA pair, which moved
// to 7/8; DTENSOR_FLAG_NO_TMA itself stays in the ISA.)
//
// Everything downstream keys off these names rather than literals, so changing a value
// here moves the mode. Note 5/6 and 7/8 are each ordered differently than they were.
enum : uint32_t {
  MODE_SIMT           = 0,
  MODE_TCU            = 1,
  MODE_TCU_DXA        = 2,
  MODE_TCU_PIPE2      = 3,  // 2-stage LMEM pipeline, LSU-staged  (control for 5)
  MODE_TCU_PIPE3      = 4,  // 3-stage LMEM pipeline, LSU-staged  (control for 6)
  MODE_TCU_DXA_PIPE2  = 5,  // 2-stage LMEM pipeline, DXA-staged
  MODE_TCU_DXA_PIPE3  = 6,  // 3-stage LMEM pipeline, DXA-staged
  MODE_DTCU_SOCKET    = 7,  // engine at socket scope,  D -> that socket's L1
  MODE_DTCU_CLUSTER   = 8,  // engine at cluster scope, D -> L2
  MODE_HET_TCU_DSOCK  = 9,  // hetero: in-core TCU + DTCU_socket
  MODE_HET_TCU_DCLUS  = 10, // hetero: in-core TCU + DTCU_cluster
  MODE_HET_ALL        = 11, // hetero: in-core TCU + both engines
  // 12/13: the first modes whose geometry can actually make operand staging pay -- a
  // multi-warp CTA sharing one staged tile, a producer warp, and wgmma reading smem
  // directly. 12 vs 13 is the DXA engine with everything else held fixed.
  MODE_TCU_WG_DXA     = 12, // workgroup WGMMA + DXA, warp-specialised
  MODE_TCU_WG         = 13, // workgroup WGMMA, cooperative SW load (control for 12)
};
static const uint32_t NUM_MODES = 14;
static const uint32_t MODE_ALL  = 0xFFFFFFFFu;
static uint32_t g_mode = MODE_ALL;
// Warps per CTA for the workgroup modes (12/13); 0 = use every warp slot the device has.
// This is the knob the earlier staging modes were missing entirely.
static uint32_t g_wg_warps = 0;
static inline bool run_this(uint32_t m) { return g_mode == MODE_ALL || g_mode == m; }

// Three states, because "not runnable" has two different meanings and conflating them
// would let `-m 3` look like a temporary gap and `-m 9` look like a typo.
enum class ModeState {
  Implemented,
  Reserved, // a hole in the numbering; never was and never will be a mode
  Planned,  // a real mode in the map, not built yet (hetero, Phase C)
};
static ModeState mode_state(uint32_t m) {
  switch (m) {
  case MODE_SIMT: case MODE_TCU: case MODE_TCU_DXA:
  case MODE_TCU_PIPE2: case MODE_TCU_PIPE3:
  case MODE_TCU_DXA_PIPE2: case MODE_TCU_DXA_PIPE3:
  case MODE_DTCU_SOCKET: case MODE_DTCU_CLUSTER:
  case MODE_TCU_WG_DXA: case MODE_TCU_WG:
    return ModeState::Implemented;
  // Numbered but NOT built. A first attempt was backed out: the claim and the start
  // instruction both executed -- a readback showed claimed[]=1 and the descriptor
  // intact -- yet the engine never went busy, so D was never written. It also needed
  // four extra kernel_arg_t fields, and growing that struct moves every mode's cycle
  // count (see 260824_DTCU_update_RFC.md 2.6), so it could not stay in the tree
  // disabled. Planned keeps the suite green and the number out of the results.
  // 12/13 build and run but do not verify yet: at ISSUE_WIDTH=4 warps exactly one of
  // the four warps' rows come out right (24,173 of 32,768 wrong), and replacing the
  // NRC=8 wgmma/accumulator contexts with the default wmma ctx makes the same kernel
  // pass at one warp. So the remaining fault is the WGMMA_NRC=8 accumulator layout,
  // not the staging, not the B smem layout, and not the DXA descriptors -- all three
  // were bisected clean. Planned until that is settled; a wrong number is worse than
  // no number.
  case MODE_HET_TCU_DSOCK: case MODE_HET_TCU_DCLUS: case MODE_HET_ALL:
    return ModeState::Planned;   // numbered, not built
  default:
    return ModeState::Reserved;
  }
}

// Engine-only modes: the whole GEMM goes to the descriptor engine and no core computes.
static inline bool is_dtcu_mode(uint32_t m) {
  return m == MODE_DTCU_CLUSTER || m == MODE_DTCU_SOCKET;
}
// Every mode that builds descriptors. Modes 9-11 will join this once they work.
static inline bool uses_engine(uint32_t m) { return is_dtcu_mode(m); }
// Whether a mode feeds the socket engines, the cluster engine, or both.
static inline bool wants_socket(uint32_t m) {
  return m == MODE_DTCU_SOCKET || m == MODE_HET_TCU_DSOCK || m == MODE_HET_ALL;
}
static inline bool wants_cluster(uint32_t m) {
  return m == MODE_DTCU_CLUSTER || m == MODE_HET_TCU_DCLUS || m == MODE_HET_ALL;
}


// Whitespace-free mode names for the machine-readable [MOTI] line. Separate from the
// human-readable names[] below, which contains spaces. The sweep scripts cross-check
// their own table against these, so a renumbering hard-errors there instead of
// silently mislabelling a CSV column.
static const char* const kShortNames[NUM_MODES] = {
  "SIMT", "TCU", "TCU+DXA", "TCU-pipe2", "TCU-pipe3",
  "TCU+DXA-pipe2", "TCU+DXA-pipe3",
  "DTCU_socket", "DTCU_cluster",
  "TCU+DTCU_socket", "TCU+DTCU_cluster", "TCU+DTCU_both",
  "TCU_wg+DXA", "TCU_wg"
};

#endif // _CGO27_HOST_MODES_H_
