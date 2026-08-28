#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Threads per warp. MUST equal the HW/kernel configuration: the host uses this
// for the launch geometry (block_dim) AND for the WMMA tile shapes
// (wmma_config_t<NUM_THREADS> in main.cpp), while the kernel uses
// VX_CFG_NUM_THREADS directly (wmma_context<VX_CFG_NUM_THREADS> in kernel.cpp).
// A hardcoded fallback here silently disagrees with the kernel whenever the
// Makefile sets a different -DVX_CFG_NUM_THREADS: the launch then spawns the
// wrong number of lanes (e.g. 4 of 32 -> uninitialized registers in the inactive
// lanes) and host/kernel compute different tile sizes, so the grid indexes past
// the matrices. That was the NT=32 failure (mode 1 looping ~33M iterations on a
// clobbered loop bound, plus the earlier out-of-range writes and the spurious
// dtensor_start). Derive it from the build config instead.
#ifndef NUM_THREADS
#ifdef VX_CFG_NUM_THREADS
#define NUM_THREADS VX_CFG_NUM_THREADS
#else
#define NUM_THREADS 32
#endif
#endif

#define NUM_WARPS 16
#define NUM_CORES 4
#define NUM_SOCKETS 1

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// GEMM computed by every mode: D = C + A * B  (dtcu_compare convention).
//   A : row-major  [M x K]   (A[i*K + k])
//   B : col-major  [K x N]   (B[j*K + k])
//   C : row-major  [M x N]   (C[i*N + j])  -- accumulator preload
//   D : row-major  [M x N]   (D[i*N + j])
//
// mode selects the execution path; the GEMM and input are identical across all
// modes so the ONLY difference is which compute/memory unit runs it:
//   0 = in-core SIMT      (plain scalar MAC loop, no tensor unit)
//   1 = in-core TCU       (WMMA fragments, register path)
//   2 = in-core TCU + DXA (WMMA fed by DXA-staged smem tiles)
//   3 = workgroup TCU + DXA, 4 = its cooperative-copy control, 5 = A-resident DXA
//   7/8 = socket/cluster DTCU, 14/15 = their producer-consumer pipelines
typedef struct {
  uint32_t mode;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t D_addr;
  uint64_t desc_addr;   // DTCU descriptor array (modes 7,8,14,15)
  // NOTE: append new fields HERE, at the end. Inserting a field in the middle
  // shifts every following offset and measurably perturbs codegen/layout — see
  // the mode-2 investigation in 260718_moti_RFC.md.
  uint32_t app;         // build-selected app id; see MOTI_APP below
} kernel_arg_t;

// DXA descriptor slots programmed host-side (modes 2,3,5).
#define DESC_A 0
#define DESC_B 1

// ---------------------------------------------------------------------------------
// Tiling knobs the HOST and the KERNEL must agree on, defined ONCE here.
//
// Both of these used to carry an #ifndef default in two files at the same time -- the
// host's host_types.h and the kernel that reads them. They agreed only because nobody had
// edited one of the two, and a disagreement is silent: the kernel tiles one way, the host
// sizes the grid, the Local Memory and the DXA descriptors the other way, and D comes out
// wrong with no error. That is the same failure NUM_THREADS is defined here to prevent,
// and the reason this file exists.
//
// -D on the command line still overrides, and now overrides both sides at once.

// K-steps held in ONE staged tile (modes 3/4/5/6). Reuse along K: at S=1 a staged tile
// feeds one MMA, at S=4 it feeds four. Both the kernel's sub-tile indexing and the host's
// lmem/descriptor sizing scale with it. Measured to lose at S=2 and S=4; the reason is
// still unexplained, but it is NOT occupancy -- see README.
#ifndef MOTI_WG_KSTEPS
#define MOTI_WG_KSTEPS 1
#endif

// Column tiles one CTA sweeps against a resident A (mode 5). Reuse along N, and the one
// that pays: it divides global A traffic by this factor. The host derives the grid width
// and the lmem size from it, the kernel loops over it.
#ifndef MOTI_WG_NCOLS
#define MOTI_WG_NCOLS 4
#endif

// Modes 5/6 keep one A chunk and up to two streamed B tiles in LMEM. Return the
// largest kStK-aligned A chunk that fits; zero means even the fixed B footprint
// cannot be represented by this machine configuration.
static inline uint32_t moti_wg_acol_kchunk(uint64_t lmem_bytes,
                                           uint32_t elem_bytes,
                                           uint32_t cta_m,
                                           uint32_t xtile_n,
                                           uint32_t st_k,
                                           uint32_t b_stages) {
  const uint64_t lmem_elems = lmem_bytes / elem_bytes;
  const uint64_t b_elems = (uint64_t)b_stages * xtile_n * st_k;
  if (lmem_elems <= b_elems || cta_m == 0 || st_k == 0)
    return 0;
  const uint64_t raw_k = (lmem_elems - b_elems) / cta_m;
  return (uint32_t)(raw_k / st_k) * st_k;
}

// Descriptors one submitter splits its row range into, for the PIPELINED engine modes
// (14/15). At T=1 those modes degenerate to modes 7/8 with a bigger block.
//
// The non-pipelined modes issue ONE descriptor covering a whole row band and then spin on
// its single `done` transition, so there is nothing for a consumer to observe until the
// entire band is finished -- which is why the cores sit idle for the whole GEMM and why an
// epilogue can only ever run as a separate launch afterwards. T descriptors give the band
// T completion points, and the epilogue for slice t-1 can run while the engine is still
// producing slice t.
//
// The host sizes the descriptor array from this and the kernel indexes it, so it is ONE
// definition shared by both -- a disagreement would hand the engine a descriptor slot the
// host never zeroed, and a stale `done` reads as a completed GEMM.
//
// Bounded by the engine's descriptor queue on the socket side: DTCU_SOCKET_QUEUE_DEPTH is
// SOCKET_SIZE*2, which is 2 here, so a socket submitter cannot have more than two slices in
// flight and issues them one ahead. The cluster queue is NUM_CORES*2 = 8 and takes all of
// them at once.
#ifndef MOTI_PIPE_TILES
#define MOTI_PIPE_TILES 4
#endif

// Consumer row/column layout for the pipelined modes. 0 = rows striped by CORE, one row at
// a time, its columns split across that core's threads (the default, and the only one that
// does not livelock). 1 = rows striped by WARP, which puts 16 rows' worth of cache lines in
// flight per core and starves the engine on the strict-priority L2 arbiter. Kept as a knob
// because the failure is a measurement, not a bug to hide -- see k_epi_rows.h.
#ifndef MOTI_PIPE_STRIPE_WARP
#define MOTI_PIPE_STRIPE_WARP 0
#endif

// Consumer warps per core for the pipelined modes (14/15). Warp 0 is the producer, so
// this is clamped to `warps_per_core - 1`.
//
// This is a real tuning axis rather than a debug switch, because overlap here is not free
// parallelism -- the consumers and the engine are reading and writing through the SAME L2.
// The cluster engine has exactly one port for operands and D together, so every consumer
// warp added takes bandwidth from the GEMM it is waiting on. There is an optimum, and it
// is not 16; see the sweep in docs/260808_moti.md.
// WIDTH of the consumer, in warps: the `cw` warps after the producer split one row's
// columns. It is not a row count -- see k_epi_rows.h for why giving each warp its own rows
// livelocks against the strict-priority L2 arbiter.
//
// Fifteen is the maximum in the configured 16-warp core: one producer plus fifteen
// consumers. Keeping the roles disjoint is what makes this a producer/consumer pipeline;
// allowing warp 0 to consume serialized its next submission behind epilogue work.
#ifndef MOTI_PIPE_CONSUMER_WARPS
#define MOTI_PIPE_CONSUMER_WARPS 15
#endif

// Register-only spin between two polls of a descriptor's done flag (modes 14/15).
//
// A consumer waiting on a slice has nothing else to do, but the poll itself is not free:
// dtensor_check() is an AMO that resolves at the last-level cache by design, so an
// unthrottled spin puts a continuous stream of LLC transactions on the port the engine is
// fetching its operands through. See moti_wait_desc() in k_dtcu_desc.h for what that
// measured. Kernel-only; the host never reads it.
#ifndef MOTI_PIPE_BACKOFF
#define MOTI_PIPE_BACKOFF 32
#endif

// WHICH EPILOGUE THIS BINARY CONTAINS. Compile-time, deliberately.
//
// The apps used to be selected at RUNTIME from kernel_arg_t::app, which meant every app's
// code had to be present in every binary. That is not free here: adding two standalone
// epilogue kernels that a mode never calls moved mode 4 by +44.6 % (32,583 -> 47,118),
// because a mode's cycle count depends on where its code lands in a 16 KB icache and an
// unused kernel is enough to relocate it.
//
// So the selection is a preprocessor one. Exactly ONE epilogue is compiled -- epi_apply()
// collapses to a single expression and the standalone passes exist only for the app that
// needs them -- and comparing app N against app 1 means comparing two builds that each
// contain one epilogue rather than one build carrying all of them.
//
// The host reads this too, so `-a` is checked against it instead of selecting anything.
//   1 baseline (no epilogue)   2 ReLU   3 GELU   4 residual   5 per-channel scale
//   6 row-wise softmax
//   9 bias-gradient-style broadcast: D[i][j] += mean over i. A column reduction -- the
//     bias gradient's access pattern, and the one that crosses socket boundaries.
//   4 and 5 use an auxiliary operand behind C -- see MOTI_AUX_ELEM_OFFSET.
#ifndef MOTI_APP
#define MOTI_APP 1
#endif

// Apps 7/8 require mixed source formats (quantized A and fp16 B), but dtensor_desc_t has
// one fmt_s shared by A and B. Reject them instead of silently measuring the identity app
// under a dequant label. App 9 is the implemented column-reduction stressor.
#if MOTI_APP < 1 || (MOTI_APP > 6 && MOTI_APP != 9)
#error "cgo27_motivation supports MOTI_APP=1..6 or 9; apps 7/8 need mixed A/B formats"
#endif

// True when the app is an elementwise map -- one output element depends on one input
// element and nothing else. The in-core modes fuse it into the accumulator; the DTCU has
// no epilogue hardware and runs it as a second pass over D.
//
// Apps 4 and 5 belong here even though they read an auxiliary operand: a residual is
// indexed per element and a scale per column, so both are still elementwise. Leaving them
// out meant the DTCU modes never ran their epilogue at all -- D came back as bare
// C + A*B and mode 7 reported 7,022 mismatches at 128x64x32 while mode 1, which fuses,
// passed. A classification, not a capability, and it silently skipped work.
#define MOTI_APP_IS_ELEMENTWISE \
  (MOTI_APP == 2 || MOTI_APP == 3 || MOTI_APP == 4 || MOTI_APP == 5)
// True when the app needs a row-wise reduction, which nothing can fuse: a tile holds only
// part of a row, so the row max and row sum are not known until every tile is written.
#define MOTI_APP_NEEDS_ROW_PASS (MOTI_APP == 6)
// True when the app reduces down a COLUMN. Same "cannot be fused" property as the row
// pass, but the opposite access pattern, and that is the point: the socket engines write D
// into the producing socket's L1 and slice M four ways, so a column touches all four
// sockets while a row touches one. This is the shape that should punish DTCU_socket's
// placement and leave DTCU_cluster's (D in L2) alone.
#define MOTI_APP_NEEDS_COL_PASS (MOTI_APP == 9)
// True when the app reads an auxiliary operand alongside the GEMM: a residual matrix (4)
// or a per-channel scale (5). Both live behind C in the same buffer -- see
// MOTI_AUX_ELEM_OFFSET below -- so neither needs a kernel_arg_t field.
#define MOTI_APP_NEEDS_AUX (MOTI_APP == 4 || MOTI_APP == 5)

// Where the auxiliary epilogue operands live, for the apps that need one.
//
// Apps 4 and 5 need an operand the GEMM does not have -- a residual matrix or a
// per-channel scale. Neither gets a kernel_arg_t field: growing that struct
// reshuffles codegen in paths nobody touched (64 -> 80 B moved mode 2 by +15.8 % and mode
// 5 by -32.9 %), which is why the note above says to leave it alone.
//
// Instead the host allocates the auxiliary array immediately after C in the SAME buffer
// and still passes that buffer's base as C_addr. The kernel derives the address from
// C_addr, M and N, all of which it already has:
//
//     aux = (otype*)C_addr + MOTI_AUX_ELEM_OFFSET(M, N)
//
//   app 4  R    [M x N]   residual, added elementwise
//   app 5  s    [N]       per-channel scale, broadcast down the column
//
// Apps 6 and 9 need no auxiliary operand; they reduce D in a standalone pass.
#define MOTI_AUX_ELEM_OFFSET(M, N) ((M) * (N))

// The pointer itself, nullptr when this build's app reads no auxiliary operand. Written as
// a macro so the address arithmetic -- an arg->M load and a multiply -- is not emitted at
// all for the apps that never use it. Computing it unconditionally cost modes 1 and 2
// +0.37 % and +0.28 %, which is small but is exactly the kind of drift the per-mode
// binaries exist to keep out.
#if MOTI_APP_NEEDS_AUX
#define MOTI_AUX_PTR(C_addr, M, N) \
  (reinterpret_cast<const float*>(C_addr) + MOTI_AUX_ELEM_OFFSET(M, N))
#else
#define MOTI_AUX_PTR(C_addr, M, N) (nullptr)
#endif

#endif
