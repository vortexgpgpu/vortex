// Broadcast-SAXPY device kernel: one DXA fetch per cluster (= K CTAs),
// not per CTA.
//
// Each CTA computes elems_per_cta = VX_CFG_NUM_THREADS output elements:
//   out[g] = scale[cluster_id] * x[g] + y[g]
// where g = blockIdx.x * elems_per_cta + threadIdx.x.
//
// The interesting line is the `dxa_multicast_1d` ctor + `sync_and_issue`:
// rank-0 of each cluster issues a single 1-element DXA fetch and the
// engine writes it into the same SMEM offset on all K members' LMEM
// pages (the cluster contract guarantees those K pages live on the
// same core, so the multicast can land in one bus transaction).

#include <vx_spawn2.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescScale = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
    const TYPE* x         = reinterpret_cast<const TYPE*>(arg->x_addr);
    const TYPE* y         = reinterpret_cast<const TYPE*>(arg->y_addr);
    TYPE*       out       = reinterpret_cast<TYPE*>      (arg->out_addr);
    const uint32_t K      = arg->cluster_size;
    const uint32_t elems  = arg->elems_per_cta;

    // SMEM layout: a single TYPE-sized slot for the cluster-shared
    // scale. (The same offset on every member's LMEM page is what
    // the multicast lands in.)
    auto smem  = reinterpret_cast<TYPE*>(__local_mem());
    TYPE* shared_scale = &smem[0];

    // Per-CTA barrier — DXA "tx-complete" signal arrives here.
    vortex::barrier       local_scale(0);
    // Cluster-wide barrier — K members rendezvous before rank-0 fires.
    vortex::group_barrier group_scale(1, K);

    const bool is_loader_warp = (get_sub_group_id() == 0);

    if (is_loader_warp) {
        // ctor declares one expected tx event on local_scale.
        vortex::dxa_multicast_1d mc(kDescScale, K, local_scale, group_scale);
        // K-way rendezvous, then rank-0 issues the multicast at
        // coord = cluster_id along scale[].
        const uint32_t cluster_id = blockIdx.x / K;
        mc.sync_and_issue(shared_scale, cluster_id);
    }

    // Every warp of every CTA must arrive_and_wait, in line with
    // dxa_multicast_1d's API contract (the helper deliberately omits a
    // wait() of its own — see vx_dxa.h:295).
    local_scale.arrive_and_wait();

    // Now `*shared_scale` is populated on every cluster member.
    TYPE scale = *shared_scale;

    uint32_t g = blockIdx.x * elems + threadIdx.x;
    out[g] = scale * x[g] + y[g];
}
