// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Phase-2 in-process smoke driver for libvortex-gem5.so.
//
// Exercises the C ABI from a native x86 binary — no gem5 involvement.
// If a kernel completes here, the library is sound; any subsequent
// failure under gem5 is on the SimObject side, not the library.
//
// Usage:
//   LD_LIBRARY_PATH=$(dirname $(realpath gem5_smoke)) ./gem5_smoke kernel.vxbin

#include "vortex_gpgpu.h"
#include "constants.h"
#include <VX_config.h>
#include <VX_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <kernel.vxbin>\n"
                 "  Runs the kernel through libvortex-gem5's C ABI to confirm\n"
                 "  the library is wired up correctly before exposing it to\n"
                 "  the gem5 SimObject.\n",
                 argv[0]);
    return 1;
  }
  const char* kernel_path = argv[1];

  std::printf("[gem5_smoke] %s\n", vortex_gem5_build_info());
  std::printf("[gem5_smoke] kernel: %s\n", kernel_path);

  vortex_gem5_handle_t h = vortex_gem5_create();
  if (h == nullptr) {
    std::fprintf(stderr, "[gem5_smoke] vortex_gem5_create failed\n");
    return 1;
  }

  if (vortex_gem5_load_kernel(h, kernel_path) != 0) {
    std::fprintf(stderr, "[gem5_smoke] vortex_gem5_load_kernel failed\n");
    vortex_gem5_destroy(h);
    return 1;
  }

  // Tick until the kernel completes. cycle() returns false when no
  // cluster is running AND no channel still holds an in-flight packet.
  // Belt-and-braces cap at 100M cycles so a runaway kernel doesn't
  // hang the smoke test (a real run hits the IO_EXIT_CODE check well
  // before).
  uint64_t cycles = 0;
  constexpr uint64_t MAX_CYCLES = 100ull * 1000 * 1000;
  while (vortex_gem5_tick(h)) {
    if (++cycles > MAX_CYCLES) {
      std::fprintf(stderr,
                   "[gem5_smoke] aborted after %llu cycles — kernel did not complete\n",
                   static_cast<unsigned long long>(cycles));
      vortex_gem5_destroy(h);
      return 1;
    }
  }

  // Drain dirty cache lines to VRAM so we can read IO_EXIT_CODE. Same
  // pattern as sim/simx/main.cpp's post-run cache flush — one DCR_READ
  // per core triggers Processor::flush_caches() inside the simulator.
  uint32_t dummy = 0;
  for (uint32_t cid = 0; cid < NUM_CORES * NUM_CLUSTERS; ++cid) {
    vortex_gem5_dcr_read(h, VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
  }

  // Read the kernel's exit code from IO_EXIT_CODE via the VRAM-read
  // path — same byte the simx main reads in sim/simx/main.cpp:213.
  uint32_t exit_code = 0;
  vortex_gem5_vram_read(h, IO_EXIT_CODE,
                        reinterpret_cast<uint8_t*>(&exit_code),
                        sizeof(exit_code));

  std::printf("[gem5_smoke] cycles=%llu exit_code=%u\n",
              static_cast<unsigned long long>(cycles), exit_code);

  vortex_gem5_destroy(h);
  return static_cast<int>(exit_code);
}
