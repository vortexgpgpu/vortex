#include <vx_spawn2.h>
#include "common.h"

__kernel void produce_k(produce_arg_t* __UNIFORM__ arg) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  for (uint32_t i = 0; i < arg->num_words; ++i) {
    dst[i] = arg->words[i];
  }
}

static inline uint32_t global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__kernel void vecadd_k(vec_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->n) {
    return;
  }
  auto a = reinterpret_cast<const uint32_t*>(arg->a_addr);
  auto b = reinterpret_cast<const uint32_t*>(arg->b_addr);
  auto c = reinterpret_cast<uint32_t*>(arg->c_addr);
  c[i] = a[i] + b[i];
}

__kernel void scale_k(vec_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->n) {
    return;
  }
  auto a = reinterpret_cast<const uint32_t*>(arg->a_addr);
  auto c = reinterpret_cast<uint32_t*>(arg->c_addr);
  c[i] = a[i] * arg->k;
}

__kernel void poison_k(vec_arg_t* __UNIFORM__ arg) {
  uint32_t i = global_id();
  if (i >= arg->n) {
    return;
  }
  auto c = reinterpret_cast<uint32_t*>(arg->c_addr);
  c[i] = POISON;
}

__kernel void mark_k(mark_arg_t* __UNIFORM__ arg) {
  if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
    return;
  }
  auto hits = reinterpret_cast<uint32_t*>(arg->hits_addr);
  uint32_t b = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  hits[b] += 1;
  if (b == 0) {
    auto dims = reinterpret_cast<uint32_t*>(arg->dims_addr);
    dims[0] = gridDim.x;
    dims[1] = gridDim.y;
    dims[2] = gridDim.z;
    dims[3] += 1;
  }
}
