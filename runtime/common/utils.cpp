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

void prepare_kernel_launch_params(uint32_t threads_per_warp, uint32_t dimension, const uint32_t *block_dim,
  uint32_t* block_size, uint32_t* warp_step_x, uint32_t* warp_step_y, uint32_t* warp_step_z) {
  // block size in number of threads
  uint32_t _block_size = 1;
  for (uint32_t i = 0; i < dimension; ++i) {
    _block_size *= block_dim ? block_dim[i] : 1;
  }
  *block_size = _block_size;
  uint32_t dim_x = (dimension > 0 && block_dim) ? block_dim[0] : 1;
  uint32_t dim_y = (dimension > 1 && block_dim) ? block_dim[1] : 1;
  uint32_t dim_z = (dimension > 2 && block_dim) ? block_dim[2] : 1;

  *warp_step_x = threads_per_warp % dim_x;
  *warp_step_y = (threads_per_warp / dim_x) % dim_y;
  *warp_step_z = (threads_per_warp / (dim_x * dim_y)) % dim_z;
}
