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

#include <common.h>
#include <vortex.h>

void prepare_kernel_launch_params(uint32_t threads_per_warp, uint32_t num_warps,
  uint32_t ndim, const uint32_t *block_dim,
  uint32_t eff_block_dim[3],
  uint32_t* block_size, uint32_t* warp_step_x, uint32_t* warp_step_y, uint32_t* warp_step_z) {
  // auto-select block dimensions when not specified (maximize warp occupancy)
  uint32_t auto_block_dim[3] = {threads_per_warp, num_warps, 1};
  const uint32_t* src = block_dim ? block_dim : auto_block_dim;
  for (uint32_t i = 0; i < 3; ++i) {
    eff_block_dim[i] = (i < ndim) ? src[i] : 1;
  }

  uint32_t _block_size = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
    _block_size *= eff_block_dim[i];
  }
  *block_size = _block_size;

  *warp_step_x = threads_per_warp % eff_block_dim[0];
  *warp_step_y = (threads_per_warp / eff_block_dim[0]) % eff_block_dim[1];
  *warp_step_z = (threads_per_warp / (eff_block_dim[0] * eff_block_dim[1])) % eff_block_dim[2];
}
