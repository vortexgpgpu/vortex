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

#define __kernel extern "C" __attribute__((annotate("vortex.kernel")))

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
      uint32_t __UNIFORM__ value = (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_X);
      return value;
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Y);
      return value;
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Z);
      return value;
    }
  } z;
};

struct BlockDim {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_X);
      return value;
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Y);
      return value;
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Z);
      return value;
    }
  } z;
};

struct GridDim {
  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_X);
      return value;
    }
  } x;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Y);
      return value;
    }
  } y;

  struct {
    __attribute__((always_inline)) operator uint32_t() const {
      uint32_t __UNIFORM__ value =  (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Z);
      return value;
    }
  } z;
};

static const ThreadIdx threadIdx;
static const BlockIdx  blockIdx;
static const BlockDim  blockDim;
static const GridDim   gridDim;

static __attribute__((always_inline)) uint32_t get_local_cta_id() {
  uint32_t __UNIFORM__ v;
  __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(VX_CSR_CTA_ID));
  return v;
}

static __attribute__((always_inline)) uint32_t get_warps_per_cta() {
  uint32_t __UNIFORM__ v;
  __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(VX_CSR_CTA_SIZE));
  return v;
}

#define __local_group_id get_local_cta_id()
#define __warps_per_group get_warps_per_cta()

#define __local_mem() \
  (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))

#define __syncthreads() \
  vx_barrier(csr_read(VX_CSR_CTA_ID), csr_read(VX_CSR_CTA_SIZE))

#endif // __VX_SPAWN2_H__