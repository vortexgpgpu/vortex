// GPU-side program for mode 4 -- workgroup WGMMA, cooperative SW load (the DXA control for mode 12).
//
// This is the mode where operand staging is actually set up to pay. Those modes have a
// DXA copy but none of the three things that make one worth its cost, and they land
// within 7 % of mode 1, which stages nothing at all. This one has all three:
//
//   1. REUSE. The CTA is `warps` warps wide and stages ONE A tile of cta_M = warps *
//      xtileM rows plus ONE B tile that every warp reads. In the retired single-warp staging modes a block is a
//      single warp: it stages a tile, issues one mma_sync against it, and throws it away,
//      so there is nothing to amortise the copy over. Sixteen resident warps there are
//      sixteen unrelated CTAs each copying its own private tile.
//
//   2. THE WHOLE CTA COPIES, cooperatively. There is no producer warp because there is
//      no engine to hand the work to -- that absence is the variable this mode isolates.
//
//   3. THE CONSUMER READS SHARED MEMORY DIRECTLY. wgmma_sync takes B as a shared-memory
//      DESCRIPTOR, so the B fragment is never loaded into registers. With
//      load_matrix_sync (the retired single-warp staging modes) every fragment is still an LSU load and the load
//      COUNT does not drop -- measured 49,632 -> 47,520, 4 % -- DXA only makes each load
//      cheaper (95.5 -> 65.8 cycles) while paying issue and barrier traffic on the SFU.
//      This is the same property Hopper's wgmma has and the DTCU's MAC array has.
//
// WGMMA_RS: A still comes from registers (the RS form), B from smem. That is the variant
// sgemm_tcu_wg_dxa uses at NRC <= 16 and is what this is modelled on.

#include "k_wg_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>


__kernel void moti_tcu_wg(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pA = reinterpret_cast<const wgctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<const wgctx::input_t*>(arg->B_addr);
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
  // So C is added afterwards, in a cooperative pass over the CTA's own output tile. That
  // pass used to be an extra cost this pair paid and modes 1/2/5/6 did not, because the
  // accumulator went to global D and the pass then read it back -- four M*N accesses
  // against their two. It stores to LMEM now, so the pass makes the same two, and the
  // two groups are directly comparable. See k_wg_common.h.
  wgctx::fragment_acc fragC;
  wgctx::fill_fragment(fragC, 0);

  vortex::barrier bar(0);

  for (uint32_t k = 0; k < K; k += kStK) {
    // The whole CTA copies, cooperatively, into the SAME layouts DXA would have
    // produced. This is mode 3 with exactly one thing removed, so the pair measures
    // what the copy engine is worth once the geometry can actually use it.
    for (uint32_t i = tid; i < cta_M * kStK; i += blockDim.x) {
      const uint32_t r = i / kStK, c = i % kStK;
      A_smem[i] = pA[(tile_row + r) * K + (k + c)];       // A row-major [M x K]
    }
    for (uint32_t i = tid; i < wgctx::xtileN * kStK; i += blockDim.x) {
      const uint32_t r = i / kStK, c = i % kStK;
      B_smem[i] = pB[(tile_col + r) * K + (k + c)];       // B col-major, stored [N x K]
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

  // D = epi(C + A*B), fused while the accumulator is still in registers: two M*N global
  // accesses, the same as modes 1/2/5/6, against the four this used to make. Warp-private
  // -- no second pass, no scratch, no barrier. See k_wg_common.h.
  const uint32_t out_row = tile_row + warp_rank * wgctx::xtileM;
  wg_store_epilogue(fragC, pC, pD, out_row, tile_col, N, app);
}
