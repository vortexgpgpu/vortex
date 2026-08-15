#ifndef _CGO27_RUN_MODES_H_
#define _CGO27_RUN_MODES_H_

// One function per mode, saying how that mode runs.
//
// WHY A SPEC RATHER THAN A FULL run_case PER MODE. Nine copies of "open the device,
// allocate A/B/C/D, upload, launch, read back, read the MPM counters, tear down" would
// be nine places for those to drift apart, and the harness's whole premise is that every
// mode runs the SAME GEMM through the same scaffolding. So the scaffolding stays in one
// place (run_case) and each run_mode_N() below supplies only what genuinely differs:
// which kernel entry, which ISA extension it needs, whether the host programs DXA
// descriptors, how deep its Local Memory pipeline is, and what launch geometry it wants.
//
// Adding a mode is then: a kernel entry, a kernel_m<N>.cpp, a run_mode_N() here, and an
// entry in moti_mode_spec(). Nothing else in the harness needs to know it exists.

#include "host_modes.h"
#include <VX_types.h>
#include <vortex.h>

struct ModeSpec {
  // Launch geometry family. Each is a different shape of grid, not a tweak of one.
  enum Geom {
    GEOM_SIMT,     // one thread per output element
    GEOM_WMMA,     // one block (one warp) per output tile
    GEOM_PER_CORE, // one block per core; the kernel picks its slice from vx_core_id()
    GEOM_WMMA_WG,  // one MULTI-WARP block per CTA tile; warps share the staged tile
    GEOM_WMMA_WG_ACOL, // as above, but the CTA sweeps NCOLS column tiles against a
                       // resident A staged once for the whole K range
    GEOM_PER_CORE_PIPE, // one MULTI-WARP block per core: the block's warps are the
                        // epilogue consumers, so unlike GEOM_PER_CORE it cannot be 1 thread
  };

  const char* kentry;      // entry name in kernel_m<N>.vxbin; nullptr = not runnable
  uint64_t    isa_need;    // ALL of these bits must be present, or the mode is skipped
  Geom        geom;
  uint8_t     lmem_stages; // Local Memory tile stages, 0 = none
  bool        dxa_desc;    // host programs the DXA 2D descriptors before launch
};

// ---- in-core, no operand staging ----

static inline ModeSpec run_mode_0() {   // SIMT: scalar MAC loop, the no-tensor baseline
  return ModeSpec{ "moti_simt", 0, ModeSpec::GEOM_SIMT, 0, false };
}

static inline ModeSpec run_mode_1() {   // TCU: WMMA, operands loaded by the LSU
  return ModeSpec{ "moti_tcu", 0, ModeSpec::GEOM_WMMA, 0, false };
}

// ---- in-core, operands staged into Local Memory ----

static inline ModeSpec run_mode_2() {   // WMMA on DXA-staged smem, single buffer
  return ModeSpec{ "moti_tcu_dxa", VX_ISA_EXT_DXA, ModeSpec::GEOM_WMMA, 1, true };
}

// ---- descriptor engine ----
//
// GEOM_PER_CORE for BOTH, and that is load-bearing rather than incidental. The kernel
// derives its row slice from vx_core_id(), so a core that receives no block is a slice
// nobody submits -- which is silent wrong output, not a hang. Mode 8 kept a 1x1x1 launch
// after being switched to a per-core split and lost 6,144 of 8,192 elements with a
// perfectly plausible cycle count.
//
// No lmem: the engine has its own operand SRAM and never touches Local Memory.

static inline ModeSpec run_mode_7() {   // DTCU_socket: one engine per socket, D -> its L1
  return ModeSpec{ "moti_dtcu_socket", VX_ISA_EXT_DTCU_SOCKET,
                   ModeSpec::GEOM_PER_CORE, 0, false };
}

static inline ModeSpec run_mode_8() {   // DTCU_cluster: one engine per cluster, D -> L2
  return ModeSpec{ "moti_dtcu_cluster", VX_ISA_EXT_DTCU_CLUSTER,
                   ModeSpec::GEOM_PER_CORE, 0, false };
}

// ---- descriptor engine, PIPELINED ----
//
// 7 and 8 issue one descriptor per submitter covering that submitter's whole row band and
// then spin on its single `done`. So the cores do nothing for the entire GEMM, and an
// epilogue can only run afterwards, as a second launch. Measured, that second launch costs
// modes 7 and 8 the SAME number of cycles to the cycle (+232,561 each for softmax at
// 128x64x32) -- which is the proof that it cannot see where D landed, and therefore that no
// epilogue will ever separate the two placements.
//
// 14 and 15 cut the band into MOTI_PIPE_TILES descriptors so it has T completion points,
// and run the epilogue for slice t-1 while the engine produces slice t. That is the only
// concurrency available: cp_submit_launch() polls a launch to retirement, so overlap has to
// happen INSIDE one launch.
//
// The pair is not symmetric, and that asymmetry is the experiment:
//
//   14 socket   four engines, and a core can only reach the engine in its own socket. So
//               every core is a producer and every core consumes its own slices. Nobody is
//               free. Its engine writes D into that socket's L1 through a dedicated port,
//               which at SOCKET_SIZE=1 is the very L1 the consuming core is using.
//   15 cluster  one engine, so ONE producer suffices: core 0 submits every slice and cores
//               1..N-1 do nothing but consume. Three quarters of the machine is free for
//               the epilogue, and D goes to L2 rather than into any core's L1.
//
// GEOM_PER_CORE_PIPE rather than GEOM_PER_CORE because a consumer needs threads: the
// non-pipelined modes launch one thread per block, which is enough to fill a descriptor and
// spin and nothing else.

static inline ModeSpec run_mode_14() {  // DTCU_socket, pipelined; every core produces
  return ModeSpec{ "moti_dtcu_socket_pipe", VX_ISA_EXT_DTCU_SOCKET,
                   ModeSpec::GEOM_PER_CORE_PIPE, 0, false };
}

static inline ModeSpec run_mode_15() {  // DTCU_cluster, pipelined; core 0 alone produces
  return ModeSpec{ "moti_dtcu_cluster_pipe", VX_ISA_EXT_DTCU_CLUSTER,
                   ModeSpec::GEOM_PER_CORE_PIPE, 0, false };
}

// ---- workgroup staging: the geometry that lets a copy engine pay ----
//
// A CTA of `warps` warps stages one A tile spanning all of them plus one B tile they all
// read, warp 0 issues the copy, and wgmma_sync reads B out of shared memory instead of
// loading it into registers. 3 and 4 differ ONLY in whether that copy is a DXA
// descriptor or the CTA's own loads, so the pair is what the engine is worth.
//
// These held mode ids 12 and 13 while the single-warp staging modes occupied 3-6. Those
// were retired (see host_modes.h) and the pair took their place, so a number quoted as
// "mode 12" or "mode 13" anywhere older means mode 3 or mode 4 here.

static inline ModeSpec run_mode_3() {   // workgroup WGMMA + DXA, warp-specialised
  return ModeSpec{ "moti_tcu_wg_dxa", VX_ISA_EXT_DXA, ModeSpec::GEOM_WMMA_WG, 1, true };
}

static inline ModeSpec run_mode_4() {   // workgroup WGMMA, cooperative SW load
  return ModeSpec{ "moti_tcu_wg", 0, ModeSpec::GEOM_WMMA_WG, 1, false };
}

// 3 and 5 are the N-axis-reuse pair: same kernel body, same epilogue, same DXA, and 5
// sweeps MOTI_WG_NCOLS column tiles against an A block staged once for the whole K.
static inline ModeSpec run_mode_5() {   // workgroup WGMMA + DXA, A resident in LMEM
  return ModeSpec{ "moti_tcu_wg_acol", VX_ISA_EXT_DXA, ModeSpec::GEOM_WMMA_WG_ACOL, 1, true };
}

// Dispatch. A mode with no entry here is not runnable, and that is reported rather than
// defaulted: with reserved holes and not-yet-built modes in the map, a silent fallthrough
// would run the wrong kernel under the right label.
static inline ModeSpec moti_mode_spec(uint32_t mode) {
  switch (mode) {
  case MODE_SIMT:          return run_mode_0();
  case MODE_TCU:           return run_mode_1();
  case MODE_TCU_DXA:       return run_mode_2();
  case MODE_TCU_WG_DXA:    return run_mode_3();
  case MODE_TCU_WG:        return run_mode_4();
  case MODE_TCU_WG_ACOL:   return run_mode_5();
  case MODE_DTCU_SOCKET:   return run_mode_7();
  case MODE_DTCU_CLUSTER:  return run_mode_8();
  case MODE_DTCU_SOCKET_PIPE:  return run_mode_14();
  case MODE_DTCU_CLUSTER_PIPE: return run_mode_15();
  default:                 return ModeSpec{ nullptr, 0, ModeSpec::GEOM_SIMT, 0, false };
  }
}

#endif // _CGO27_RUN_MODES_H_
