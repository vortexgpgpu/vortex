// Broadcast-SAXPY showcase: out[i] = scale[i / K] * x[i] + y[i].
//
// Every group of K consecutive output elements shares one scale value;
// `vx_launch_info_t::cluster_dim = (K, 1, 1)` is what makes the K CTAs
// that compute those K outputs co-resident on one core, so the rank-0
// CTA can fetch `scale[cluster_id]` ONCE over DXA and multicast it to
// its K-1 peers' LMEM. Without cluster_dim each CTA would have to
// fetch its own copy of the same scalar — K loads per cluster.

#ifndef CTA_CLUSTER_COMMON_H
#define CTA_CLUSTER_COMMON_H

#include <stdint.h>

typedef float TYPE;

typedef struct {
    uint64_t x_addr;        // const TYPE[N]       — per-element input
    uint64_t y_addr;        // const TYPE[N]       — per-element input
    uint64_t out_addr;      //       TYPE[N]       — per-element output
    uint64_t scale_addr;    // const TYPE[N/K]     — one value per cluster
    uint32_t cluster_size;  // K (must match launch cluster_dim[0])
    uint32_t elems_per_cta; // = VX_CFG_NUM_THREADS (one warp/CTA)
} kernel_arg_t;

#endif // CTA_CLUSTER_COMMON_H
