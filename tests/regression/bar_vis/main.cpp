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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    exit(-1);                                                 \
  } while (false)

int main(int argc, char** argv) {
  // A CTA of 128 threads is 32 warps of 4: two LSU blocks. It must hold an even
  // number of warps.
  uint32_t grid = 1, block = 128, rounds = 64, shared = 0, junk = 0, mode = 0;
  int c;
  while ((c = getopt(argc, argv, "g:b:r:sj:m:")) != -1) {
    switch (c) {
    case 'g': grid = atoi(optarg); break;
    case 'b': block = atoi(optarg); break;
    case 'r': rounds = atoi(optarg); break;
    case 's': shared = 1; break;
    case 'j': junk = atoi(optarg); break;
    case 'm': mode = atoi(optarg); break;
    default: return -1;
    }
  }
  vx_device_h dev;
  vx_queue_h q;
  vx_buffer_h buf, err, jb;
  vx_module_h mod;
  vx_kernel_h k;
  kernel_arg_t arg = {};
  const uint32_t n = grid * block;
  RT_CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(dev, &qi, &q));
  RT_CHECK(vx_buffer_create(dev, (uint64_t)n * 8 + (mode == 1 ? 1024ull * (junk + 2) + n * 8 : 0), VX_MEM_READ_WRITE, &buf));
  RT_CHECK(vx_buffer_create(dev, n * 4, VX_MEM_READ_WRITE, &err));
  RT_CHECK(vx_buffer_address(buf, &arg.buf_addr));
  RT_CHECK(vx_buffer_address(err, &arg.err_addr));
  RT_CHECK(vx_buffer_create(dev, (uint64_t)n * 64 * (junk ? junk : 1), VX_MEM_READ_WRITE, &jb));
  RT_CHECK(vx_buffer_address(jb, &arg.junk_addr));
  arg.junk = junk;
  arg.rounds = rounds;
  arg.shared = shared;
  arg.mode = mode;
  std::vector<uint32_t> h(2 * n, 0);
  RT_CHECK(vx_enqueue_write(q, buf, 0, h.data(), n * 8, 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  RT_CHECK(vx_module_get_kernel(mod, "main", &k));
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = k;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = block;
  li.lmem_size = shared ? block * 8 : 0;
  RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, nullptr));
  RT_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_enqueue_read(q, h.data(), err, 0, n * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  uint64_t stale = 0;
  for (uint32_t i = 0; i < n; ++i) {
    stale += h[i];
  }
  printf("grid=%u block=%u rounds=%u shared=%u junk=%u mode=%u stale_reads=%llu\n", grid, block, rounds, shared, junk, mode,
         (unsigned long long)stale);
  vx_device_dump_perf(dev, stdout);
  vx_buffer_release(buf);
  vx_buffer_release(err);
  vx_buffer_release(jb);
  vx_kernel_release(k);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);
  if (stale != 0) {
    printf("Found %llu stale reads!\nFAILED!\n", (unsigned long long)stale);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
