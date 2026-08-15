#ifndef _CGO27_K_DTCU_DESC_H_
#define _CGO27_K_DTCU_DESC_H_

// Descriptor construction for the two engine modes. The KERNEL builds its descriptor --
// the split follows vx_core_id(), which the host does not know at enqueue time, and
// every input is already in kernel_arg_t or is a build constant, so nothing had to be
// added to that struct.

#include "wmma_common.h"
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

// Engine tile-N. The cluster engine accepts 16..128 and the harness has always used 32;
// the socket engine has exactly one legal value because its output must fit the socket's
// L1 (see the RFC 1.2). Compile-time, so shape_n_size folds to a constant.
#define MOTI_CLUSTER_TILE_N  32u
#define MOTI_SOCKET_TILE_N   ((uint32_t)DTCU_SOCKET_TILE_N_MAX)

// Row band for slice `s` of `slices`, over `M` rows. Ragged tails are legal -- the engine
// clamps a partial tile in hardware -- so the only thing to avoid is an EMPTY slice,
// which would make the engine abort on M == 0.
struct moti_slice { uint32_t row0, nrow; };
static inline moti_slice moti_slice_of(uint32_t M, uint32_t slices, uint32_t s) {
  const uint32_t rows = (M + slices - 1) / slices;
  const uint32_t row0 = s * rows;
  const uint32_t end  = (row0 + rows <= M) ? (row0 + rows) : M;
  return moti_slice{ row0, (end > row0) ? (end - row0) : 0u };
}

// ---------------------------------------------------------------------------------
// Wait for a descriptor to complete, WITH BACKOFF.
//
// dtensor_check() is an AMO, and an AMO is deliberately not satisfiable from the core's own
// L1 -- it takes the AmoProbe path and resolves at the last-level cache, which is the only
// reason a core that never issued the descriptor can observe completion at all
// (vx_dtensor.h says so). The cost is that a bare `while (0 == dtensor_check(da));` is a
// continuous stream of LLC transactions, aimed at the same port the engine is fetching its
// operands through.
//
// Measured, at 128x64x32 with three cores polling: the engine's own compute is 4,144 cycles
// and it spent 39,892 stalled on next_tile_load with tma_mem_wait=37,960, while core 0's
// loads averaged 674 cycles of latency. The pollers were starving the producer they were
// waiting for. Unpipelined modes never hit this because each core polls exactly one
// descriptor that its OWN engine is filling, and there are no spectators.
//
// The backoff must be REGISTER-ONLY or it defeats itself. The first version used a
// `volatile uint32_t` loop counter, which puts the counter on the stack and turns every
// iteration into a load and a store -- a backoff made of exactly the traffic it exists to
// avoid. An empty `nop` with a plain counter is the right shape: the asm keeps the loop
// from being deleted, and nothing touches memory.
static inline void moti_wait_desc(uint64_t desc_addr) {
  while (0 == dtensor_check(desc_addr)) {
    for (uint32_t i = 0; i < MOTI_PIPE_BACKOFF; ++i)
      __asm__ volatile ("nop");
  }
}

// ---------------------------------------------------------------------------------
// Pipeline slicing, for modes 14/15. Same idea as moti_slice_of, with one extra rule: a
// slice must be a whole number of ENGINE TILES.
//
// The engine computes a padded tile whatever the descriptor's M says, so a slice shorter
// than tile_m costs a full tile and delivers part of one. Ignoring that is not a small
// inefficiency -- the first build cut M into MOTI_PIPE_TILES slices flat, and at
// 128x64x32 that gave mode 14 eight-row slices against a 32-row tile: instr_tcu went to
// 131,072 against mode 15's 65,536 for the identical GEMM, and the mode took 546,099
// cycles against mode 7's 20,359. Correct output, four times the arithmetic.
//
// So the unit of slicing is the tile, not the row.

// How many slices `rows` can actually support. Never zero, so a caller always has one
// descriptor to submit; when it comes back 1 the range is a single tile and the mode
// degenerates to its unpipelined twin, which is a real answer rather than a failure --
// mode 14's band at M=128 is exactly one 32-row tile.
static inline uint32_t moti_pipe_slices(uint32_t rows, uint32_t tile_m) {
  const uint32_t tiles = (rows + tile_m - 1) / tile_m;
  uint32_t s = MOTI_PIPE_TILES;
  if (s > tiles) s = tiles;
  return (s == 0) ? 1u : s;
}

// Slice `s` of `slices` over `rows` rows starting at `row0_base`, aligned to tile_m.
// A ragged final tile is fine -- the engine clamps a partial tile in hardware.
static inline moti_slice moti_pipe_slice_of(uint32_t row0_base, uint32_t rows,
                                            uint32_t slices, uint32_t s, uint32_t tile_m) {
  const uint32_t tiles = (rows + tile_m - 1) / tile_m;
  const uint32_t tps   = (tiles + slices - 1) / slices;   // whole tiles per slice
  const uint32_t off   = s * tps * tile_m;
  if (off >= rows)
    return moti_slice{ row0_base + off, 0 };
  const uint32_t end = (off + tps * tile_m <= rows) ? (off + tps * tile_m) : rows;
  return moti_slice{ row0_base + off, end - off };
}

// Fill a descriptor for one row band. Only the ROW origin moves between slices: A and
// C/D are row-major so a band is contiguous, and B is shared by every slice untouched.
static inline void moti_fill_desc(dtensor_desc_t* d, kernel_arg_t* arg,
                                  moti_slice sl, uint32_t tile_n) {
  using it = typename vortex::tensor::ITYPE;
  using ot = typename vortex::tensor::OTYPE;
  const uint32_t N = arg->N, K = arg->K;

  d->ptrA = arg->A_addr + (uint64_t)sl.row0 * K * sizeof(typename it::dtype);
  d->ptrB = arg->B_addr;
  d->ptrC = arg->C_addr + (uint64_t)sl.row0 * N * sizeof(typename ot::dtype);
  d->ptrD = arg->D_addr + (uint64_t)sl.row0 * N * sizeof(typename ot::dtype);
  d->ldmA = K; d->ldmB = K; d->ldmC = N; d->ldmD = N;   // elements, not bytes
  d->M = (uint16_t)sl.nrow; d->N = (uint16_t)N; d->K = (uint16_t)K;
  d->fmt_s = (uint8_t)it::id; d->fmt_d = (uint8_t)ot::id;
  d->flags = 0x0;                                        // D = C + A*B, TMA overlap on
  d->shape_n_size = dtcu_shape_n_size(tile_n);
  d->shape_policy = 0;
  d->done = 0;   // the engine sets this; a consumer tells "finished" from "not started"
                 // by that transition alone, so it must be zero BEFORE the submit
}

// Make the descriptor visible to the engine, which reads it back out of memory.
//
// A fence is NOT enough. Core stores are write-through and fire-and-forget -- nothing
// acknowledges them (that is what the `strsp` opt-in exists for, RFC 1.6), so `fence`
// has no completion to wait on and the engine's descriptor read can pass the fill. That
// is not theoretical: with only the fence, mode 8's four slices produced 6,144 errors --
// exactly the 3/4 of the output belonging to the three descriptors the engine read as
// still-zero, each of which it retired instantly and flagged done.
//
// The AMO is what closes it. dtensor_check() is `amoor.w rd, x0` on the descriptor's
// `done` field, which takes the cache's AmoProbe path: it invalidates the local line and
// resolves at the last-level cache. Issuing it after the fill forces this core's stores
// to that same 64-byte line to have reached the LLC before the start instruction goes
// out. The returned value is discarded -- we only just wrote zero there.
static inline void moti_publish_desc(uint64_t desc_addr) {
  vx_fence();                        // orders the fill's stores ahead of the AMO
  (void)dtensor_check(desc_addr);    // drains them to the LLC the engine reads from
}

// ---------------------------------------------------------------------------------
// Publish a descriptor and DO NOT RETURN UNTIL THE ENGINE COULD READ IT.
//
// moti_publish_desc() does a fence and one AMO, on the argument that the AMO drains the
// core's write-through stores to the LLC the engine reads from. That is enough for modes
// 7 and 8, where each submitter fills exactly one descriptor at the very start of the
// kernel. It is NOT enough for the pipelined modes, and the failure is spectacular rather
// than subtle: the engine reads a descriptor that is partly or entirely still zero.
//
//   [DTCU] Error: empty GEMM. M=32, N=0, K=0      (mode 14, 256x256x64, T=2)
//   [DTCU] Error: empty GEMM. M=64, N=0, K=0      (mode 14, 512x256x128, T=2 forced)
//   [DTCU] Error: shape_n_size must explicitly select N-size   (mode 14, T=1)
//
// Note what those say: neighbouring fields disagree. M is the correct slice height while N
// and K, four bytes away in the same 64 B descriptor, are zero -- so this is not a stale
// line, it is part of a line having arrived and part not. `Dtcu::start()` calls
// `begin_descriptor_()` synchronously when the engine is idle, so the read is issued
// essentially at the start instruction, with no slack for stores still in flight.
//
// The fix is to close the loop instead of arguing about it: read the descriptor back
// through the same AMO path the engine's reader resolves at, and spin until our own bytes
// are there. An AMO is the only read that can do this -- a plain load returns this core's
// own cached copy, which of course looks correct.
//
// Cost is a handful of AMOs per descriptor, once, against a GEMM of hundreds of thousands
// of cycles. Deliberately NOT folded into moti_publish_desc(): modes 7 and 8 are already
// measured and correct, and changing their submit path would move every number in the
// grid for no benefit. If they are relying on luck, that is worth knowing separately.
static inline void moti_publish_desc_verified(uint64_t desc_addr, const dtensor_desc_t* d) {
  moti_publish_desc(desc_addr);
#if MOTI_PIPE_VERIFY
  // The three words that carry the shape and the format selectors. Reading them with
  // fetch_or(0) is a read-only RMW: it returns the value and changes nothing.
  const uint32_t want_mn = (uint32_t)d->M | ((uint32_t)d->N << 16);
  const uint32_t want_kf = (uint32_t)d->K | ((uint32_t)d->fmt_s << 16)
                                          | ((uint32_t)d->fmt_d << 24);
  const uint32_t want_fs = (uint32_t)d->flags | ((uint32_t)d->shape_n_size << 8)
                                              | ((uint32_t)d->shape_policy << 16);
  auto amo = [](uint64_t a) {
    return __atomic_fetch_or((uint32_t*)(uintptr_t)a, 0u, __ATOMIC_ACQUIRE);
  };
  while (amo(desc_addr + 48) != want_mn
      || amo(desc_addr + 52) != want_kf
      || amo(desc_addr + 56) != want_fs)
    ;
#else
  (void)d;
#endif
}

#endif // _CGO27_K_DTCU_DESC_H_
