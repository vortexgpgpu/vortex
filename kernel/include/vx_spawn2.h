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

#ifndef __VX_SPAWN2_H__
#define __VX_SPAWN2_H__

#include <vx_intrinsics.h>
#include <stdint.h>

// flat local thread index = cta_rank * num_threads_per_warp + thread_id
struct ThreadIdx {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_X);
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_Y);
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_Z);
    }
  } z;
};

struct BlockIdx {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_X);
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Y);
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Z);
    }
  } z;
};

struct BlockDim {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_X);
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Y);
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Z);
    }
  } z;
};

struct GridDim {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_X);
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Y);
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Z);
    }
  } z;
};

static const ThreadIdx threadIdx;
static const BlockIdx  blockIdx;
static const BlockDim  blockDim;
static const GridDim   gridDim;

#define __local_mem() \
  (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))

#define __syncthreads() \
  vx_barrier(csr_read(VX_CSR_CTA_ID), csr_read(VX_CSR_CTA_SIZE))

#endif // __VX_SPAWN2_H__