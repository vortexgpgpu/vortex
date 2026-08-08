#ifndef _CGO27_HOST_RUN_H_
#define _CGO27_HOST_RUN_H_

// run_case(): the ONE piece of scaffolding every mode shares -- open the device, allocate
// and upload A/B/C/D, load that mode's device program, launch, read D back, collect the
// MPM counters, tear down.
//
// Nothing in here branches on a mode id. Everything mode-specific comes from the
// ModeSpec that run_modes.h's run_mode_N() supplies, so the modes cannot drift apart in
// how they are set up and timed -- which is the harness's whole premise: the same GEMM,
// the same scaffolding, one variable changed.

#include "host_types.h"
#include "host_modes.h"
#include "run_modes.h"
#include "epilogue.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

#include <VX_types.h>
#include <dtcu_cfg.h>
#include <vortex.h>
#include <dxa.h>

inline int run_case(uint32_t mode,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t tcu_tileM, uint32_t tcu_tileN, uint32_t tcu_tileK,
                    const std::vector<itype_t>& hA,
                    const std::vector<itype_t>& hB,
                    const std::vector<otype_t>& hC,
                    std::vector<otype_t>& out,
                    Stats& stats) {
  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Each path needs its engine present. Modes 2/5/6 stage operands through DXA;
  // modes 7/8 hand the whole GEMM to the DTCU. Marking the case skipped (rather
  // than just returning) keeps its all-zero output out of the verify pass, which
  // would otherwise report M*N mismatches and fail a run that never executed.
  // Everything mode-specific about this run comes from here; see run_modes.h.
  const ModeSpec spec = moti_mode_spec(mode);

  // Workgroup staging (modes 12/13): how many warps share one staged tile. This is the
  // knob modes 2/5/6 never had -- they launch one warp per block, so sixteen resident
  // warps are sixteen unrelated CTAs each copying its own private tile. Capped at what
  // the device reports and at the rows available.
  uint32_t wg_warps = 1;
  const bool wg_any = (spec.geom == ModeSpec::GEOM_WMMA_WG)
                  || (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL);
  if (wg_any) {
    // NOT tunable. A WGMMA group is exactly ISSUE_WIDTH warps -- the block issues that
    // many warps in parallel per uop, so a CTA with any other count computes garbage
    // silently (measured: every other warp count fails the verify, and only this one
    // passes). sgemm_tcu_wg_dxa derives it the same way and says so at main.cpp:284.
    uint64_t issue_width = 1, dev_warps = 1;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,   &dev_warps));
    wg_warps = (uint32_t)issue_width;
    if (wg_warps > dev_warps) {
      std::cerr << "cgo27_motivation: WGMMA group (" << wg_warps
                << ") exceeds warps per core (" << dev_warps << ")" << std::endl;
      return -1;
    }
  }
  const uint32_t cta_M = wg_warps * wgcfg::xtileM;
  // K-steps per staged tile -- the reuse knob. MUST match MOTI_WG_KSTEPS in the workgroup
  // kernels: the host sizes lmem and the DXA tile from it, the kernel indexes sub-tiles
  // with it, and a disagreement mis-tiles silently.
  const uint32_t wg_stK = MOTI_WG_KSTEPS * wgcfg::tileK;
  // A staged tile deeper than K reads past the end of A and B. DXA hides it -- the engine
  // clamps out-of-bounds in hardware -- so mode 12 passed while mode 13's cooperative
  // copy produced 8,192 wrong elements at 128x64x32 with S=4 (stK=64 > K=32). Reject the
  // configuration instead of letting one of the pair mask it.
  if (wg_any && (K % wg_stK) != 0) {
    std::cerr << "cgo27_motivation: MOTI_WG_KSTEPS=" << MOTI_WG_KSTEPS
              << " gives a staged tile of " << wg_stK << " columns, which does not divide K="
              << K << std::endl;
    return -1;
  }
  // Mode 5 gives one CTA NCOLS column tiles, so N must divide by all of them at once.
  const uint32_t acol_N = MOTI_WG_NCOLS * wgcfg::xtileN;
  if (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL && (N % acol_N) != 0) {
    std::cerr << "cgo27_motivation: MOTI_WG_NCOLS=" << MOTI_WG_NCOLS
              << " gives a CTA " << acol_N << " columns wide, which does not divide N="
              << N << std::endl;
    return -1;
  }
  if (spec.kentry == nullptr) {
    std::cerr << "cgo27_motivation: no kernel entry for mode " << mode << std::endl;
    vx_queue_release(queue); vx_device_release(device);
    return -1;
  }
  {
    const uint64_t need = spec.isa_need;   // ALL of them, not any one
    if (need != 0) {
      uint64_t isa_flags = 0;
      RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
      if ((isa_flags & need) != need) {   // ALL of them, not any one
        std::cerr << "  (skipped: " << ((need == VX_ISA_EXT_DXA) ? "DXA" : kShortNames[mode])
                  << " ISA extension disabled)" << std::endl;
        stats.skipped = true;
        vx_queue_release(queue); vx_device_release(device);
        return 0;
      }
    }
  }

  kernel_arg_t karg{};
  karg.mode = mode; karg.app = g_app; karg.M = M; karg.N = N; karg.K = K;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, D_buf = nullptr,
              desc_buf = nullptr;
  // Upload staging, function-scoped so it outlives every asynchronous enqueue below.
  const std::vector<otype_t> d_zeros(out.size(), otype_t{});
  std::vector<uint8_t> desc_zeros;
  RT_CHECK(vx_buffer_create(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &karg.B_addr));
  RT_CHECK(vx_buffer_create(device, hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf));
  RT_CHECK(vx_buffer_address(C_buf, &karg.C_addr));
  RT_CHECK(vx_buffer_create(device, out.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &karg.D_addr));

  // DTCU descriptor (modes 7, 8). BYTE-FOR-BYTE IDENTICAL between the two modes except
  // for shape_n_size, which each engine bounds differently: that is the point of
  // selecting the engine with the start INSTRUCTION rather than a descriptor field.
  //
  // ROW SLICING, and who does it. Both engine modes split the GEMM by rows and submit one
  // descriptor per submitter -- mode 7 one per socket (each socket has its own engine, so
  // they run concurrently), mode 8 one per core into the single cluster engine's queue.
  //
  // The DESCRIPTORS ARE BUILT BY THE KERNEL (k_dtcu_desc.h), not here. The host cannot know
  // which slice a block will run: a socket engine is only reachable from inside its own
  // socket, so the mapping comes from vx_core_id(). The host's job is to size and
  // allocate the array, and to reject a shape the engine cannot express before the run
  // starts rather than after it produces wrong output.
  uint32_t n_desc = 0;
  if (uses_engine(mode)) {
    uint64_t num_cores = 0, socket_size = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_SOCKET_SIZE, &socket_size));
    if (socket_size == 0) socket_size = 1;
    const uint32_t num_sockets = (uint32_t)((num_cores + socket_size - 1) / socket_size);

    // One slot per potential submitter. Slots whose slice comes out empty (M < the
    // submitter count) are simply never written or submitted.
    const int engine = wants_socket(mode) ? DTCU_ENGINE_SOCKET : DTCU_ENGINE_CLUSTER;
    n_desc = wants_socket(mode) ? num_sockets : (uint32_t)num_cores;

    // Must match MOTI_{CLUSTER,SOCKET}_TILE_N in k_dtcu_desc.h -- the kernel picks the value,
    // this is the guard, and a mismatch here is a silently wrong descriptor.
    const uint32_t tile_n = (engine == DTCU_ENGINE_CLUSTER)
                          ? 32u : dtcu_tile_n_max_of(engine);
    if (!dtcu_tile_n_valid_of(engine, tile_n)) {
      std::cerr << "cgo27_motivation: tile_n=" << tile_n << " illegal for "
                << kShortNames[mode] << std::endl;
      return -1;
    }

    RT_CHECK(vx_buffer_create(device, (size_t)n_desc * sizeof(dtensor_desc_t),
                              VX_MEM_READ_WRITE, &desc_buf));
    RT_CHECK(vx_buffer_address(desc_buf, &karg.desc_addr));
  }

  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, C_buf, 0, hC.data(), hC.size() * sizeof(otype_t), 0, nullptr, nullptr));
  // Zero D before every run. It is pure output (the accumulator preload is C), so nothing
  // reads it, and without this a mode that writes NOTHING is only caught if the allocator
  // happens to hand back memory that does not already contain a correct D from the
  // previous mode. That is luck, not a test. Zeroing makes "did not write" always fail.
  // Host-side, so it costs no measured cycles.
  //
  // These two staging vectors are at FUNCTION scope on purpose. vx_enqueue_write is
  // asynchronous: it records the source pointer and returns, so a vector declared in a
  // nested block is freed while the transfer still refers to it. That read-after-free
  // segfaulted every shape from 256x128x64 up while 128x64x32 happened to survive,
  // because the small allocation was still mapped.
  RT_CHECK(vx_enqueue_write(queue, D_buf, 0, d_zeros.data(),
                            d_zeros.size() * sizeof(otype_t), 0, nullptr, nullptr));

  // Zero the descriptor array rather than uploading a filled one. The kernel writes every
  // field it uses, but a slot whose slice is empty is never written at all, and leaving
  // that as whatever the allocator returned would make a stale `done` look like a
  // completed GEMM to anything that scanned the array.
  if (desc_buf) {
    desc_zeros.assign((size_t)n_desc * sizeof(dtensor_desc_t), 0);
    RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, desc_zeros.data(), desc_zeros.size(),
                              0, nullptr, nullptr));
  }

  // mode 2: program DXA descriptors (source layout -> smem tile).
  //   A: row-major [M x K], tile [tcu_tileM x tcu_tileK], row stride K.
  //   B: col-major [K x N] stored as [N x K] row-major, tile [tcu_tileN x tcu_tileK], row stride K.
  if (spec.dxa_desc) {
    // The A tile is as tall as the CTA: one copy feeds every warp in it. For the
    // single-warp modes cta_M == tcu_tileM, so this is the old descriptor unchanged.
    const bool wg   = wg_any;
    const bool acol = (spec.geom == ModeSpec::GEOM_WMMA_WG_ACOL);
    const uint32_t dK = wg ? wg_stK        : tcu_tileK;
    const uint32_t dM = wg ? cta_M         : tcu_tileM;
    const uint32_t dN = wg ? wgcfg::xtileN : tcu_tileN;
    // A and B no longer share a K tile: mode 5 stages A for the WHOLE K range in one
    // issue and keeps it, while B is still restaged per K step. That asymmetry IS the
    // mode -- it is what turns A into a reused operand instead of a streamed one.
    RT_CHECK(vortex::dxa::program_2d(device, DESC_A, karg.A_addr,
      /*size0=*/K, /*size1=*/M, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/(acol ? K : dK), /*tile1=*/dM, /*elem_bytes=*/sizeof(itype_t)));
    RT_CHECK(vortex::dxa::program_2d(device, DESC_B, karg.B_addr,
      /*size0=*/K, /*size1=*/N, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/dK, /*tile1=*/dN, /*elem_bytes=*/sizeof(itype_t)));
  }

  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  // One device program per mode, not one containing all of them. Only one kernel ever
  // runs, but in a combined binary they all occupy address space, and address decides
  // which icache set a line lands in -- adding modes 3/4 moved mode 2 from 15,468 to
  // 24,106 cycles with a byte-identical kernel body (k_select.h has the counters). With
  // a program per mode the code always starts at the same low address whatever else is
  // in the tree, so a cycle count is a property of the mode. It also shrinks each
  // program from 14,700 B (90 % of the 16 KB icache) to 536-2,940 B.
  char vxbin[32];
  std::snprintf(vxbin, sizeof(vxbin), "kernel_m%u.vxbin", mode);
  RT_CHECK(vx_module_load_file(device, vxbin, &module_));
  // Each HW path is a separate kernel entry (see kernel.cpp), selected by name. The two
  // DTCU modes get two entries rather than one entry branching on arg->mode, so which
  // start instruction executes is fixed at link time -- the engine choice is an opcode,
  // and an opcode cannot be selected by a runtime value.
  RT_CHECK(vx_module_get_kernel(module_, spec.kentry, &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel; li.args_host = &karg; li.args_size = sizeof(karg);
  switch (spec.geom) {
  case ModeSpec::GEOM_SIMT:
    // One thread per output element; a warp covers NUM_THREADS columns of one row.
    li.ndim = 2;
    li.grid_dim[0]  = N / NUM_THREADS; li.grid_dim[1]  = M;
    li.block_dim[0] = NUM_THREADS;     li.block_dim[1] = 1;
    break;
  case ModeSpec::GEOM_WMMA: {
    // One block (one warp) per output tile, plus however many Local Memory stages the
    // mode pipelines across. lmem_stages comes from the spec so the LSU-staged controls
    // get exactly the footprint of the DXA modes they are compared against.
    li.ndim = 2;
    li.grid_dim[0]  = N / tcu_tileN; li.grid_dim[1]  = M / tcu_tileM;
    li.block_dim[0] = NUM_THREADS;   li.block_dim[1] = 1;
    const uint32_t stage_bytes =
        (tcu_tileM * tcu_tileK + tcu_tileN * tcu_tileK) * sizeof(itype_t);
    li.lmem_size = spec.lmem_stages * stage_bytes;
    break;
  }
  case ModeSpec::GEOM_WMMA_WG: {
    // One MULTI-WARP block per CTA tile. block_dim is warps x NUM_THREADS, which is what
    // finally makes get_sub_group_id() mean something: warp 0 becomes the producer and
    // the rest consume. lmem holds ONE A tile of cta_M rows plus ONE B tile, shared by
    // the whole CTA -- against modes 2/5/6 where every warp is its own CTA with its own
    // private stage.
    li.ndim = 2;
    li.grid_dim[0]  = N / wgcfg::xtileN;      li.grid_dim[1]  = M / cta_M;
    li.block_dim[0] = wg_warps * NUM_THREADS; li.block_dim[1] = 1;
    // The staged A+B tile is the only thing in Local Memory: the epilogue folds C into
    // the accumulator in registers and writes D once, so it needs no scratch. An earlier
    // version staged the fp32 output tile here too and had to be sized for the larger of
    // the two (4,096 B against 2,560 B); it was correct and slower. See
    // kernel_modes/k_wg_common.h.
    li.lmem_size = (cta_M + wgcfg::xtileN) * wg_stK * sizeof(itype_t);
    break;
  }
  case ModeSpec::GEOM_WMMA_WG_ACOL: {
    // One CTA per (cta_M rows) x (NCOLS column tiles). Fewer, fatter CTAs than mode 3:
    // the grid is NCOLS times narrower along N, which is exactly how the A fetch gets
    // amortised. Local Memory holds the whole A block plus one B tile, so it is sized
    // from the runtime K -- 16 KB at cta_M=64, K=128, against mode 3's 2,560 B, which is
    // what drops the resident CTAs from 4 to 3.
    li.ndim = 2;
    li.grid_dim[0]  = N / (MOTI_WG_NCOLS * wgcfg::xtileN);
    li.grid_dim[1]  = M / cta_M;
    li.block_dim[0] = wg_warps * NUM_THREADS; li.block_dim[1] = 1;
    li.lmem_size = (cta_M * K + wgcfg::xtileN * wg_stK) * sizeof(itype_t);
    break;
  }
  case ModeSpec::GEOM_PER_CORE: {
    // One block per CORE. The kernel derives its row slice from vx_core_id(), so a core
    // with no block is a slice nobody submits -- silent wrong output, not a hang. See
    // run_modes.h.
    uint64_t num_cores = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
    li.ndim = 1;
    li.grid_dim[0] = (uint32_t)num_cores; li.block_dim[0] = 1;
    break;
  }
  }

  // DTCU epilogue pass (modes 7/8). The engine is GEMM-only, so an elementwise
  // epilogue cannot be fused into it the way the in-core modes fuse theirs; it runs
  // as a SECOND launch over the whole matrix. That extra M*N round-trip is the cost
  // asymmetry the app sweep measures, and it is deliberately inside the timed
  // region so the reported cycles include it.
  const bool dtcu_needs_epi = uses_engine(mode) && epi_is_elementwise(g_app);
  vx_kernel_h epi_kernel = nullptr;
  vx_launch_info_t epi_li = {};
  if (dtcu_needs_epi) {
    RT_CHECK(vx_module_get_kernel(module_, "moti_epilogue", &epi_kernel));
    epi_li.struct_size = sizeof(epi_li);
    epi_li.kernel = epi_kernel; epi_li.args_host = &karg; epi_li.args_size = sizeof(karg);
    epi_li.ndim = 2;                                   // same geometry as moti_simt
    epi_li.grid_dim[0]  = N / NUM_THREADS; epi_li.grid_dim[1]  = M;
    epi_li.block_dim[0] = NUM_THREADS;     epi_li.block_dim[1] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr, read_ev = nullptr, epi_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  if (dtcu_needs_epi) {
    RT_CHECK(vx_enqueue_launch(queue, &epi_li, 1, &launch_ev, &epi_ev));
    RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &epi_ev, &read_ev));
  } else {
    RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &launch_ev, &read_ev));
  }
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MCYCLE, 0, &stats.cycles));
  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MINSTRET, 0, &stats.instrs));
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_CORE;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_ALU,  0, &stats.instr_alu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_FPU,  0, &stats.instr_fpu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_LSU,  0, &stats.instr_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_SFU,  0, &stats.instr_sfu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_TCU,  0, &stats.instr_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_ALU,  0, &stats.stall_alu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_FPU,  0, &stats.stall_fpu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_LSU,  0, &stats.stall_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_SFU,  0, &stats.stall_sfu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_TCU,  0, &stats.stall_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_BRANCHES,   0, &stats.branches));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DIVERGENCE, 0, &stats.divergence));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_IFETCHES,   0, &stats.ifetches));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_IFETCH_LT,  0, &stats.ifetch_lt));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOADS,      0, &stats.loads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOAD_LT,    0, &stats.load_lt));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STORES,     0, &stats.stores));
  }
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_MEM;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_READS,  0, &stats.l2_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_WRITES, 0, &stats.l2_writes));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_READS,      0, &stats.mem_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_WRITES,     0, &stats.mem_writes));
  }
  // DTCU engine counters. The CLASS selects the scope; core_id selects the instance
  // within it. There is one cluster engine, so class 9 with any core in the cluster
  // reads it -- but there are NUM_SOCKETS socket engines, and class 10 with core_id 0
  // would report one of them and silently under-count by the socket count. Sum over one
  // representative core per socket instead. (core_id 0xffffffff would over-count by
  // SOCKET_SIZE, since every core in a socket reports the same engine.)
  if (uses_engine(mode)) {
    const bool socket_scope = wants_socket(mode);
    const uint32_t cls = socket_scope ? VX_DCR_MPM_CLASS_DTCU_SOCKET
                                      : VX_DCR_MPM_CLASS_DTCU_CLUSTER;

    std::vector<uint32_t> reps;
    if (socket_scope) {
      uint64_t num_cores = 0, socket_size = 0;
      RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
      RT_CHECK(vx_dev_caps(device, VX_CAPS_SOCKET_SIZE, &socket_size));
      if (socket_size == 0) socket_size = 1;
      for (uint64_t c = 0; c < num_cores; c += socket_size)
        reps.push_back((uint32_t)c);
    } else {
      reps.push_back(0); // one cluster here; core 0 is inside it
    }
    stats.d_engines = (uint32_t)reps.size();

    // Sum a counter across the representative cores. Counts (op_reqs, instr_tcu) sum
    // correctly. Cycle counters sum to ENGINE-cycles, which is not comparable to
    // MCYCLE when several engines ran concurrently -- d_busy_max is what is.
    auto sum = [&](uint32_t csr, uint64_t* dst) -> int {
      uint64_t total = 0;
      for (uint32_t rep : reps) {
        uint64_t v = 0;
        int rc = vx_mpm_query(device, cls, csr, rep, &v);
        if (rc != 0) return rc;
        total += v;
        if (csr == VX_CSR_MPM_DTCU_BUSY) {
          if (v > stats.d_busy_max) stats.d_busy_max = v;
          if (v > 0) ++stats.d_engines_active; // an engine that never went busy did nothing
        }
      }
      *dst = total;
      return 0;
    };
    RT_CHECK(sum(VX_CSR_MPM_DTCU_OP_REQS,                &stats.d_op_reqs));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_OUT_REQS,               &stats.d_out_reqs));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_COMPUTE,                &stats.d_compute));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_NEXT_K_LOAD_STALL,      &stats.d_next_k_load_stall));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_MEM_WAIT,           &stats.d_tma_mem_wait));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_BUF_STARVE,         &stats.d_tma_buf_starve));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_OP_FILL,            &stats.d_tma_op_fill));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_ADDRGEN,            &stats.d_tma_addrgen));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_STORE_ISSUE_STALL,  &stats.d_tma_store_issue_stall));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_STORE_DRAIN,            &stats.d_store_drain));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_SMEM_READ_MODEL,        &stats.d_smem_read_model));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_NEXT_TILE_LOAD_STALL,   &stats.d_next_tile_load_stall));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_PREV_TILE_STORE_STALL,  &stats.d_prev_tile_store_stall));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_DESC_WAIT,              &stats.d_desc_wait));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_BUSY,                   &stats.d_busy));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_TMA_ACC_INIT,           &stats.d_tma_acc_init));
    RT_CHECK(sum(VX_CSR_MPM_DTCU_INSTR_TCU,              &stats.d_instr_tcu));
  }

  vx_event_release(read_ev); vx_event_release(launch_ev);
  if (epi_ev) vx_event_release(epi_ev);
  vx_buffer_release(A_buf); vx_buffer_release(B_buf); vx_buffer_release(C_buf); vx_buffer_release(D_buf);
  if (desc_buf) vx_buffer_release(desc_buf);
  vx_kernel_release(kernel); vx_module_release(module_);
  vx_queue_release(queue); vx_device_release(device);
  return 0;
}




#endif // _CGO27_HOST_RUN_H_
