#include <vx_spawn.h>
#include "common.h"

// always a power-of-2
#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

// 32-bit bitcasts for shuffle payloads
static inline size_t bits_from_f32(float x) { size_t u; __builtin_memcpy(&u, &x, 4); return u; }
static inline float f32_from_bits(size_t u) { float x; __builtin_memcpy(&x, &u, 4); return x; }

static inline float warp_reduce_sum(float x, uint32_t warp_size) {
  int clamp = warp_size - 1;
  int segmask = ~clamp & 0x3f;
  #pragma unroll
  for (uint32_t off = (warp_size >> 1); off > 0; off >>= 1) {
    size_t xb = bits_from_f32(x);
    size_t yb = vx_shfl_down(xb, off, clamp, segmask);
    x += f32_from_bits(yb);
  }
  return x;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* __restrict src0 = reinterpret_cast<float*>(arg->src0_addr);
  auto* __restrict src1 = reinterpret_cast<float*>(arg->src1_addr);
  auto* __restrict dst  = reinterpret_cast<float*>(arg->dst_addr);
  uint32_t n = arg->num_points;

  // Allocate local menory
  auto tbuf = reinterpret_cast<float*>(__local_mem(__warps_per_group * sizeof(float)));

  uint32_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t lane = threadIdx.x & (NUM_THREADS - 1);   // lane in subgroup
  uint32_t lwid = threadIdx.x / NUM_THREADS;         // subgroup index

  // grid-stride per-thread accumulate
  float acc(0.0f);
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = gid; i < n; i += stride) {
    acc += src0[i] * src1[i];
  }

  // In-warp reduction
  acc = warp_reduce_sum(acc, NUM_THREADS);

  // one partial per subgroup -> shared memory
  if (lane == 0) {
    tbuf[lwid] = acc;
  }
  __syncthreads();

  // finaly reduction by local warp 0
  if (lwid == 0) {
    uint32_t num_warps = (blockDim.x + NUM_THREADS - 1) / NUM_THREADS;
    float v = (lane < num_warps) ? tbuf[lane] : 0.0f;
    v = warp_reduce_sum(v, NUM_THREADS);
    if (lane == 0) {
      dst[blockIdx.x] = v;
    }
  }
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
