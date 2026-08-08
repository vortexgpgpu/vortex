#ifndef _CGO27_HOST_ARGS_H_
#define _CGO27_HOST_ARGS_H_

// Command-line parsing and the shape/legality checks that must run BEFORE any device
// work: a shape the selected modes cannot express should be rejected up front, not after
// a run produces wrong output.

#include "host_types.h"
#include "host_modes.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

inline void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "a:m:w:M:N:K:h")) != -1) {
    switch (c) {
    case 'a': g_app = parse_u32(optarg, "-a"); break;
    case 'w': g_wg_warps = parse_u32(optarg, "-w"); break;
    case 'M': g_M = parse_u32(optarg, "-M"); break;
    case 'N': g_N = parse_u32(optarg, "-N"); break;
    case 'K': g_K = parse_u32(optarg, "-K"); break;
    case 'm':
      if (0 == strcmp(optarg, "all")) {
        g_mode = MODE_ALL;
      } else {
        uint32_t m = parse_u32(optarg, "-m", /*allow_zero=*/true);
        if (m >= NUM_MODES) {
          std::cerr << "cgo27_motivation: invalid -m '" << optarg
                    << "' (expected 0.." << (NUM_MODES - 1) << " or 'all')\n";
          exit(-1);
        }
        if (mode_state(m) == ModeState::Reserved) {
          std::cerr << "cgo27_motivation: mode " << m << " is a reserved hole in the "
                       "numbering, not a path. Run -h for the map.\n";
          exit(-1);
        }
        g_mode = m;
      }
      break;
    case 'h':
      std::cout << "Usage: [-M m] [-N n] [-K k] [-a app_id] [-m mode]\n"
                   "  -M m   GEMM M (rows of A/C/D)     (default " << kDefaultM << ")\n"
                   "  -N n   GEMM N (cols of B/C/D)     (default " << kDefaultN << ")\n"
                   "  -K k   GEMM K (reduction depth)   (default " << kDefaultK << ")\n"
                   "         e.g. -M 1024 -N 512 -K 64.\n"
                   "         The DTCU modes take any shape -- both engines clamp ragged\n"
                   "         edges in hardware. The in-core modes do not: for those, each\n"
                   "         dimension must be a multiple of their tile (the harness prints\n"
                   "         the requirement).\n"
                   "  -a N   app id 1..8 (epilogue; default 1)\n"
                   "  -m X   which HW path to run: 'all' (default) or one mode:\n"
                   "           in-core        0=SIMT  1=TCU  2=TCU+DXA\n"
                   "           in-core, piped 5=TCU+DXA 2-stage  6=TCU+DXA 3-stage\n"
                   "           engine only    7=DTCU_socket (D->socket L1, tile 32x16)\n"
                   "                          8=DTCU_cluster (D->L2, tile 64x32)\n"
                   "           hetero         9=TCU+DTCU_socket  10=TCU+DTCU_cluster\n"
                   "                          11=TCU+both        (9-11 not built yet)\n"
                   "           6 is a reserved hole, not a path.\n";
      exit(0);
    default: exit(-1);
    }
  }
}

// ---------------------------------------------------------------------------
// Shape validation.
//
// Only the DTCU handles a ragged edge. The in-core modes do not, and each breaks
// differently when M/N/K are not exact multiples of its tile:
//   * modes 7/8 -- FINE for any shape. The engine rounds its tile counts up and the
//     TMA clamps the trailing tile: operands past the matrix are never fetched (the
//     scratchpad is zero-filled instead) and the D store leaves those bytes disabled.
//     Only the descriptor's uint16_t M/N/K field width binds.
//   * modes 1/2/5/6 -- `for (i = 0; i < K; i += tileK)` overruns K on the last step,
//     and the grid `(N / tileN, M / tileM)` truncates, leaving output tiles never
//     written.
//   * mode 0 -- the grid `(N / NUM_THREADS, M)` truncates the same way.
// The truncating cases produce a VERIFICATION MISMATCH rather than a diagnostic, so
// they need an up-front check -- but only against the modes that will actually run:
// `-m 4 -M 100 -N 48 -K 20` is legal, while the same shape on `-m 1` is not.
// ---------------------------------------------------------------------------
static uint32_t gcd_u32(uint32_t a, uint32_t b) { while (b) { uint32_t t = a % b; a = b; b = t; } return a; }
static uint32_t lcm_u32(uint32_t a, uint32_t b) { return (a / gcd_u32(a, b)) * b; }

static bool check_shape(uint32_t M, uint32_t N, uint32_t K,
                        uint32_t tcu_tileM, uint32_t tcu_tileN, uint32_t tcu_tileK) {
  bool ok = true;
  uint32_t need_M = 1, need_N = 1, need_K = 1;   // running LCM of every active constraint

  auto need = [&](uint32_t v, uint32_t mult, const char* dim, uint32_t* acc, const char* who) {
    *acc = lcm_u32(*acc, mult);
    if (v % mult) {
      std::cerr << "cgo27_motivation: " << dim << "=" << v << " is not a multiple of "
                << mult << " (" << who << "); nearest legal " << dim << ": ";
      if (v > mult) std::cerr << (v / mult) * mult << " or ";   // 0 is not a size
      std::cerr << (v / mult + 1) * mult << std::endl;
      ok = false;
    }
  };

  // mode 0, and the DTCU epilogue pass, which reuses the mode-0 launch geometry.
  const bool simt_geom = run_this(MODE_SIMT)
      || ((run_this(MODE_DTCU_CLUSTER) || run_this(MODE_DTCU_SOCKET)
           || run_this(MODE_HET_TCU_DSOCK) || run_this(MODE_HET_TCU_DCLUS)
           || run_this(MODE_HET_ALL)) && epi_is_elementwise(g_app));
  if (simt_geom)
    need(N, NUM_THREADS, "N", &need_N, "SIMT grid width NUM_THREADS -- mode 0 / DTCU epilogue pass");

  // modes 1, 2, 5, 6: one warp per output tile, K stepped by the WMMA tile.
  if (run_this(MODE_TCU) || run_this(MODE_TCU_DXA) ||
      run_this(MODE_HET_TCU_DSOCK) || run_this(MODE_HET_TCU_DCLUS) ||
      run_this(MODE_HET_ALL)) {
    need(M, tcu_tileM, "M", &need_M, "in-core TCU tileM");
    need(N, tcu_tileN, "N", &need_N, "in-core TCU tileN");
    need(K, tcu_tileK, "K", &need_K, "in-core TCU tileK");
  }

  // modes 7, 8: both DTCU engines round their tile counts UP and handle the ragged
  // trailing tile in hardware -- the operand fetch clamps past the matrix and
  // zero-fills, and the D store masks the bytes outside D (sim/simx/dtcu/dtcu_tma.cpp).
  // So M/N/K need no relation to either native tile; only the descriptor's field width
  // binds. This is also why check_shape() no longer takes a DTCU tile at all.
  if (run_this(MODE_DTCU_CLUSTER) || run_this(MODE_DTCU_SOCKET)
      || run_this(MODE_HET_TCU_DSOCK) || run_this(MODE_HET_TCU_DCLUS)
      || run_this(MODE_HET_ALL)) {
    // dtensor_desc_t holds M/N/K as uint16_t (dtcu_cfg.h), so a larger GEMM would
    // wrap silently and the engine would compute a different shape than we verify.
    if (M > 0xFFFFu || N > 0xFFFFu || K > 0xFFFFu) {
      std::cerr << "cgo27_motivation: M/N/K must each be <= 65535 for modes 7/8"
                   " (dtensor_desc_t stores them as uint16_t)" << std::endl;
      ok = false;
    }
  }

  if (!ok)
    std::cerr << "cgo27_motivation: for the selected mode(s), M must be a multiple of "
              << need_M << ", N of " << need_N << ", K of " << need_K << std::endl;
  return ok;
}


#endif // _CGO27_HOST_ARGS_H_
