// cta_cluster — showcase for `vx_launch_info_t::cluster_dim`.
//
// Computes `out[i] = scale[i / K] * x[i] + y[i]`. Every group of K
// output elements shares one `scale` value: that's the structural
// reuse `cluster_dim` is built to exploit.
//
// What the launch does:
//   * grid     = (G * K, 1, 1)        — one CTA per K-element output group
//   * cluster  = (K, 1, 1)            — every K consecutive CTAs co-resident
//   * each CTA produces NUM_THREADS output elements via one warp
//
// What the cluster_dim buys you:
//   Without it,  each of (G * K) CTAs would independently fetch
//                its own copy of `scale[cluster_id]` — G * K loads.
//   With it,     the rank-0 CTA of each cluster issues ONE DXA fetch
//                and the engine multicasts the result into all K
//                members' SMEM at the same offset — G loads total.
//   Saving       (K - 1) / K of the scale-stream bandwidth.
//
// This is the same intra-core multicast pattern sgemm2_dxa_mcast uses
// for its B-tile, reduced to the minimum that still demonstrates the
// API — no matmul tile arithmetic to wade through.
//
// Back-to-back launches: each `-l K,G,L` runs one launch with cluster
// size K, G clusters and L bytes of shared memory per CTA, in order on one
// device. The CTA dispatcher keeps its slot pointer across launches, so
// K=2 then K=4 checks that every cluster window stays aligned to its own
// size (CTA_ID % K is the member's rank), and a larger L checks that each
// CTA is admitted against its own kernel's slot capacity.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex2.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(_expr)                                          \
    do {                                                         \
        int _ret = _expr;                                        \
        if (_ret == 0) break;                                    \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
        cleanup();                                               \
        exit(-1);                                                \
    } while (false)

static bool fp_close(TYPE a, TYPE b) {
    union { TYPE f; int32_t i; } fa, fb;
    fa.f = a; fb.f = b;
    return std::abs(fa.i - fb.i) <= 6;
}

struct launch_spec_t { uint32_t K, G, L; };

static const char* kernel_file = "kernel.vxbin";
static uint32_t num_clusters = 4;   // G
static uint32_t verify       = 1;
static std::vector<launch_spec_t> launch_specs;

static vx_device_h device  = nullptr;
static vx_queue_h  queue   = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel  = nullptr;
static vx_buffer_h X_buffer = nullptr;
static vx_buffer_h Y_buffer = nullptr;
static vx_buffer_h out_buffer = nullptr;
static vx_buffer_h scale_buffer = nullptr;

constexpr uint32_t kDescScale = 0;

static void show_usage() {
    std::cout << "Usage: [-k kernel] [-g num_clusters] [-l K,G,L]... [-q skip_verify] [-h]\n"
              << "  -l K,G,L  one launch of G clusters of K CTAs with L bytes of shared\n"
              << "            memory each; repeat for back-to-back launches. Default: one\n"
              << "            launch with K = warps per core and L = one memory block.\n";
}

static launch_spec_t parse_launch_spec(const char* s) {
    launch_spec_t r{};
    if (sscanf(s, "%u,%u,%u", &r.K, &r.G, &r.L) != 3 || r.K == 0 || r.G == 0) {
        std::cerr << "Error: bad launch spec '" << s << "', expected K,G,L\n";
        exit(-1);
    }
    return r;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "k:g:l:qh")) != -1) {
        switch (c) {
            case 'k': kernel_file  = optarg; break;
            case 'g': num_clusters = atoi(optarg); break;
            case 'l': launch_specs.push_back(parse_launch_spec(optarg)); break;
            case 'q': verify       = 0; break;
            case 'h': show_usage(); exit(0);
            default:  show_usage(); exit(-1);
        }
    }
}

static void release_buffers() {
    if (X_buffer)     vx_buffer_release(X_buffer);
    if (Y_buffer)     vx_buffer_release(Y_buffer);
    if (out_buffer)   vx_buffer_release(out_buffer);
    if (scale_buffer) vx_buffer_release(scale_buffer);
    X_buffer = Y_buffer = out_buffer = scale_buffer = nullptr;
}

static void cleanup() {
    if (device) {
        release_buffers();
        if (kernel)  vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue)   vx_queue_release(queue);
        vx_device_release(device);
    }
}

// One launch of G clusters of K single-warp CTAs, each with lmem_size bytes of
// shared memory; returns the number of mismatched outputs.
static int run_launch(uint32_t K, uint32_t G, uint32_t lmem_size,
                      uint32_t elems_per_cta, int tag) {
    const uint32_t num_ctas    = G * K;
    const uint32_t num_elems   = num_ctas * elems_per_cta;
    const uint32_t buf_bytes   = num_elems * sizeof(TYPE);
    const uint32_t scale_bytes = G * sizeof(TYPE);

    std::cout << "launch " << tag << ": N elements = " << num_elems
              << ", K (cluster) = " << K << ", clusters = " << G
              << ", lmem/CTA = " << lmem_size << " bytes\n";

    RT_CHECK(vx_buffer_create(device, buf_bytes,   VX_MEM_READ,  &X_buffer));
    RT_CHECK(vx_buffer_create(device, buf_bytes,   VX_MEM_READ,  &Y_buffer));
    RT_CHECK(vx_buffer_create(device, buf_bytes,   VX_MEM_WRITE, &out_buffer));
    RT_CHECK(vx_buffer_create(device, scale_bytes, VX_MEM_READ,  &scale_buffer));

    kernel_arg_t kargs = {};
    RT_CHECK(vx_buffer_address(X_buffer,     &kargs.x_addr));
    RT_CHECK(vx_buffer_address(Y_buffer,     &kargs.y_addr));
    RT_CHECK(vx_buffer_address(out_buffer,   &kargs.out_addr));
    RT_CHECK(vx_buffer_address(scale_buffer, &kargs.scale_addr));
    kargs.cluster_size  = K;
    kargs.elems_per_cta = elems_per_cta;

    std::vector<TYPE> hX(num_elems), hY(num_elems);
    std::vector<TYPE> hScale(G);
    for (uint32_t i = 0; i < num_elems; ++i) {
        hX[i] = (TYPE)std::rand() / RAND_MAX;
        hY[i] = (TYPE)std::rand() / RAND_MAX;
    }
    for (uint32_t c = 0; c < G; ++c) {
        hScale[c] = (TYPE)std::rand() / RAND_MAX;
    }

    RT_CHECK(vx_enqueue_write(queue, X_buffer,     0, hX.data(),     buf_bytes,   0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, Y_buffer,     0, hY.data(),     buf_bytes,   0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, scale_buffer, 0, hScale.data(), scale_bytes, 0, nullptr, nullptr));

    // DXA descriptor for scale: 1D, one element per fetch. Every
    // multicast invocation picks coord = cluster_id, so K cluster
    // members receive scale[cluster_id] in lockstep at the same SMEM
    // slot 0.
    RT_CHECK(vortex::dxa::program_1d(device, kDescScale, kargs.scale_addr,
        /*size0=*/G, /*tile0=*/1, /*elem_bytes=*/sizeof(TYPE)));
    // Multicast attribute: the engine writes into the same SMEM offset on
    // each receiver's LMEM page, one lmem_size stride apart.
    RT_CHECK(vortex::dxa::set_multicast(device, kDescScale, lmem_size));

    vx_event_h launch_ev = nullptr;
    {
        vx_launch_info_t li = {};
        li.struct_size    = sizeof(li);
        li.kernel         = kernel;
        li.args_host      = &kargs;
        li.args_size      = sizeof(kargs);
        li.ndim           = 1;
        li.grid_dim[0]    = num_ctas;
        li.block_dim[0]   = elems_per_cta;       // one warp per CTA
        li.lmem_size      = lmem_size;
        // The headline of this demo: K consecutive CTAs are required
        // to land on the same core so the DXA multicast in the kernel
        // can deliver scale[cluster_id] to all of them in one shot.
        li.cluster_dim[0] = K;
        RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    std::vector<TYPE> hOut(num_elems, 0);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, hOut.data(), out_buffer, 0, buf_bytes,
                             1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    int errors = 0;
    if (verify) {
        // Reference: out[i] = scale[i / (K * elems_per_cta)] * x[i] + y[i].
        // CTA blockIdx.x ∈ [c*K, c*K+K) all share scale[c].
        const uint32_t elems_per_cluster = K * elems_per_cta;
        for (uint32_t i = 0; i < num_elems; ++i) {
            uint32_t c = i / elems_per_cluster;
            TYPE ref = hScale[c] * hX[i] + hY[i];
            if (!fp_close(hOut[i], ref)) {
                if (errors < 16)
                    printf("*** launch %d error: [%u] expected=%f, actual=%f\n",
                           tag, i, ref, hOut[i]);
                ++errors;
            }
        }
    }

    release_buffers();
    return errors;
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv);
    std::srand(50);

    RT_CHECK(vx_device_open(0, &device));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t isa_flags = 0;
    RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
        std::cerr << "Error: DXA ISA extension is disabled in runtime CONFIGS.\n";
        cleanup();
        return -1;
    }

    uint64_t num_threads = 0, num_warps = 0;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS,   &num_warps));

    // Default: one launch where one cluster fills one core (= num_warps
    // single-warp CTAs). The cta_dispatcher rounds each CTA's LMEM up to
    // VX_CFG_MEM_BLOCK_SIZE so consecutive CTAs land at block-aligned
    // offsets; the DXA multicast descriptor's smem_stride must match that,
    // else peer CTAs receive the scale at the wrong SMEM offset.
    if (launch_specs.empty()) {
        const uint32_t K = (uint32_t)num_warps;
        launch_specs.push_back({K, num_clusters, VX_CFG_MEM_BLOCK_SIZE});
        std::cout << "cta_cluster — broadcast SAXPY: out[i] = scale[i/K] * x[i] + y[i]\n"
                  << "  scale stream bandwidth:\n"
                  << "    without cluster_dim : " << num_clusters * K
                  << " loads (each CTA fetches its own copy)\n"
                  << "    with    cluster_dim : " << num_clusters
                  << " loads (rank-0 fetches, multicasts to K-1 peers)\n"
                  << "    saving              : " << (K - 1) << "x on scale[]\n";
    }
    for (const auto& sp : launch_specs) {
        if (sp.K > num_warps) {
            std::cerr << "Error: cluster size " << sp.K << " exceeds the "
                      << num_warps << " warps of a core\n";
            cleanup();
            return -1;
        }
    }

    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    int errors = 0;
    for (size_t i = 0; i < launch_specs.size(); ++i) {
        const auto& sp = launch_specs[i];
        errors += run_launch(sp.K, sp.G, sp.L, (uint32_t)num_threads, (int)i);
    }

    cleanup();
    if (errors) { std::cout << "FAILED (" << errors << " errors)\n"; return errors != 0; }
    std::cout << "PASSED\n";
    return 0;
}
