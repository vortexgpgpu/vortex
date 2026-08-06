// GPU-side program for mode 12 -- workgroup WGMMA + DXA, warp-specialised.
//
// This is the mode where operand staging is actually set up to pay. Modes 2/5/6 have a
// DXA copy but none of the three things that make one worth its cost, and they land
// within 7 % of mode 1, which stages nothing at all. This one has all three:
//
//   1. REUSE. The CTA is `warps` warps wide and stages ONE A tile of cta_M = warps *
//      xtileM rows plus ONE B tile that every warp reads. In modes 2/5/6 a block is a
//      single warp: it stages a tile, issues one mma_sync against it, and throws it away,
//      so there is nothing to amortise the copy over. Sixteen resident warps there are
//      sixteen unrelated CTAs each copying its own private tile.
//
//   2. WARP SPECIALISATION. Warp 0 issues the DXA and the others compute. Modes 2/5/6
//      already contain `is_dxa = (get_sub_group_id() == 0)`, but they launch 32 threads
//      per block -- one warp -- so producer and consumer are the same warp and the async
//      copy has nothing to overlap with.
//
//   3. THE CONSUMER READS SHARED MEMORY DIRECTLY. wgmma_sync takes B as a shared-memory
//      DESCRIPTOR, so the B fragment is never loaded into registers. With
//      load_matrix_sync (modes 2/5/6) every fragment is still an LSU load and the load
//      COUNT does not drop -- measured 49,632 -> 47,520, 4 % -- DXA only makes each load
//      cheaper (95.5 -> 65.8 cycles) while paying issue and barrier traffic on the SFU.
//      This is the same property Hopper's wgmma has and the DTCU's MAC array has.
//
// WGMMA_RS: A still comes from registers (the RS form), B from smem. That is the variant
// sgemm_tcu_wg_dxa uses at NRC <= 16 and is what this is modelled on.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>

// Workgroup MMA geometry. Separate from `ctx` (the per-warp wmma_context every other
// in-core mode uses) because the fragment and tile shapes differ.
using wgctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// The accumulator half of that context, spelled out because wgmma_context keeps its own
// alias private. Same template arguments, so wgctx::fragment_acc IS accctx::fragment_acc.
// Needed because a wgmma context refuses to load an accumulator from registers
// (vx_tensor.h:789) and D = C + A*B has to preload C.
using accctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// K-steps held in ONE staged tile. S copies amortise into a single DXA issue and a
// single barrier pair, which is the reuse axis: at S=1 a staged tile feeds one MMA, at
// S=4 it feeds four. lmem and the DXA descriptor scale with it, and the host sizes both
// from the same macro -- changing it here alone would silently mis-tile.
#ifndef MOTI_WG_KSTEPS
#define MOTI_WG_KSTEPS 1
#endif
static constexpr uint32_t kS   = MOTI_WG_KSTEPS;
static constexpr uint32_t kStK = kS * wgctx::tileK;   // columns per staged tile

// The accumulator tile must be the WGMMA output tile, or C is seeded from and D stored to
// the wrong rectangle -- which is silent, not a crash.
static_assert(accctx::tileM == wgctx::xtileM, "acc tileM != wgmma xtileM");
static_assert(accctx::tileN == wgctx::xtileN, "acc tileN != wgmma xtileN");

__kernel void moti_tcu_wg_dxa(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<wgctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<wgctx::output_t*>(arg->D_addr);

  const uint32_t tid       = threadIdx.x;
  const uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;

  // The CTA owns cta_M rows; each warp owns xtileM of them. One staged A tile covers all
  // of them and one staged B tile is read by every warp -- that is the reuse.
  const uint32_t cta_M    = num_warps * wgctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t tile_col = blockIdx.x * wgctx::xtileN;

  // smem: A [cta_M x tileK] row-major, then B [xtileN x tileK] K-major -- the layouts the
  // host's DXA descriptors produce, and the ones the smem descriptor below assumes.
  auto smem   = reinterpret_cast<wgctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * kStK;

  // The accumulator starts at ZERO, not at C. A wgmma context refuses to load an
  // accumulator from memory (vx_tensor.h:789) and that refusal is not arbitrary: the
  // warpgroup's accumulator is distributed across the group differently from a per-warp
  // WMMA fragment, so seeding it through the WMMA layout puts C in the wrong lanes and
  // the product lands on top of it. That is exactly what it did -- 24,173 of 32,768
  // elements wrong, one warp in four correct, identically for modes 12 and 13.
  //
  // So C is added afterwards, in a cooperative pass over the CTA's own output tile.
  // That is a real cost this pair pays and modes 1/2/5/6 do not (they fuse C into the
  // accumulator for free), and it must be read that way when comparing them.
  wgctx::fragment_acc fragC;
  wgctx::fill_fragment(fragC, 0);

  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  for (uint32_t k = 0; k < K; k += kStK) {
    if (is_dxa_warp) {
      bar.expect_tx(2);                                   // A and B in flight
      vx_dxa_issue_2d_wg(DESC_A, bar.id(), A_smem, k, tile_row);
      vx_dxa_issue_2d_wg(DESC_B, bar.id(), B_smem, k, tile_col);
    }
    bar.arrive_and_wait();                                // stage filled

    // S MMAs against one staged tile. Sub-tile s sits at column offset s * tileK inside
    // the stage; the row stride stays kStK, which is what both the fragment load and the
    // smem descriptor are told.
    for (uint32_t s = 0; s < kS; ++s) {
      auto A_warp = A_smem + warp_rank * wgctx::xtileM * kStK + s * wgctx::tileK;
      auto desc_b = vt::vx_make_smem_desc(B_smem + s * wgctx::tileK,
                                          kStK * sizeof(wgctx::input_t));
      wgctx::fragment_a fragA;
      wgctx::load_matrix_sync(fragA, A_warp, kStK);
      wgctx::wgmma_sync(fragC, fragA, desc_b, fragC);     // B straight from smem
    }

    bar.arrive_and_wait();                                // stage free to refill
  }

  // A*B out through the WGMMA store, which knows the warpgroup layout.
  const uint32_t out_row = tile_row + warp_rank * wgctx::xtileM;
  wgctx::store_matrix_sync(pD + out_row * N + tile_col, fragC, N);

  // Then D = epi(C + A*B), cooperatively over the CTA tile.
  // Compiled out by -DMOTI_WG_NO_C to price it: that build produces a WRONG D on
  // purpose and only its cycle count means anything. cycles(with) - cycles(without) is
  // what this pair pays that modes 1/2/5/6 do not, and it has to come off before the two
  // groups are compared.
#ifndef MOTI_WG_NO_C
  bar.arrive_and_wait();
  for (uint32_t i = tid; i < cta_M * wgctx::xtileN; i += blockDim.x) {
    const uint32_t r = i / wgctx::xtileN, c = i % wgctx::xtileN;
    const uint32_t o = (tile_row + r) * N + tile_col + c;
    pD[o] = epi_apply(app, pC[o] + pD[o]);
  }
#else
  (void)pC; (void)app;
#endif
}
