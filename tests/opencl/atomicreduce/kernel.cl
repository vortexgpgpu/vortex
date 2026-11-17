#include "common.h"

// Atomic reduction: every work-item accumulates its element into a single
// global counter. Maximum-contention atomic_add, lowering to a hardware RVA
// amoadd.w on Vortex (requires the A extension).
__kernel void atomicreduce(__global const int* data,
                           __global int* result) {
  int gid = get_global_id(0);
  atomic_add(&result[0], data[gid]);
}

// Every OpenCL 1.2 32-bit atomic, with all work-items on one counter per
// operation. The values returned by inc and the two exchanges are kept: inc
// must hand out each ticket once, and an exchange chain must return every value
// it replaced.
__kernel void atomic_global(__global const int* data,
                            __global int* result,
                            __global int* rets) {
  int gid = get_global_id(0);
  int v = data[gid];
  __global uint* uresult = (__global uint*)result;

  atomic_add(&result[OP_ADD], v);
  atomic_sub(&result[OP_SUB], v);
  rets[gid * NUM_RETS + RET_INC] = atomic_inc(&result[OP_INC]);
  atomic_dec(&result[OP_DEC]);
  atomic_min(&result[OP_MIN], v);
  atomic_max(&result[OP_MAX], v);
  atomic_min(&uresult[OP_UMIN], (uint)v);
  atomic_max(&uresult[OP_UMAX], (uint)v);
  atomic_and(&result[OP_AND], ~(1 << (gid & 31)));
  atomic_or(&result[OP_OR], v);
  atomic_xor(&result[OP_XOR], v);
  rets[gid * NUM_RETS + RET_XCHG] = atomic_xchg(&result[OP_XCHG], v);

  int old = result[OP_CMPXCHG];
  for (;;) {
    int seen = atomic_cmpxchg(&result[OP_CMPXCHG], old, old + v);
    if (seen == old) {
      break;
    }
    old = seen;
  }

  float fold = atomic_xchg((__global float*)&result[OP_FXCHG], (float)v);
  rets[gid * NUM_RETS + RET_FXCHG] = (int)fold;
}

// Counter values before the kernel runs. Unsigned blocks start min/max at the
// unsigned extremes.
static void init_local_counters(__local int* c, int is_unsigned) {
  for (int i = 0; i < NUM_OPS; ++i) {
    c[i] = 0;
  }
  c[OP_MIN]   = is_unsigned ? -1 : INT_MAX;
  c[OP_MAX]   = is_unsigned ? 0 : INT_MIN;
  c[OP_UMIN]  = -1;
  c[OP_AND]   = -1;
  c[OP_XCHG]  = XCHG_INIT;
  c[OP_FXCHG] = as_int((float)XCHG_INIT);
}

// The same operations on local memory, one set of counters per work-group.
// Work-item 0 publishes the group's counters after the barrier.
__kernel void atomic_local(__global const int* data,
                           __global int* result,
                           __local int* counters) {
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int v = data[gid];
  __local uint* ucounters = (__local uint*)counters;

  if (lid == 0) {
    init_local_counters(counters, 0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_add(&counters[OP_ADD], v);
  atomic_sub(&counters[OP_SUB], v);
  atomic_inc(&counters[OP_INC]);
  atomic_dec(&counters[OP_DEC]);
  atomic_min(&counters[OP_MIN], v);
  atomic_max(&counters[OP_MAX], v);
  atomic_min(&ucounters[OP_UMIN], (uint)v);
  atomic_max(&ucounters[OP_UMAX], (uint)v);
  atomic_and(&counters[OP_AND], ~(1 << (gid & 31)));
  atomic_or(&counters[OP_OR], v);
  atomic_xor(&counters[OP_XOR], v);
  atomic_xchg(&counters[OP_XCHG], v);

  int old = counters[OP_CMPXCHG];
  for (;;) {
    int seen = atomic_cmpxchg(&counters[OP_CMPXCHG], old, old + v);
    if (seen == old) {
      break;
    }
    old = seen;
  }

  atomic_xchg((__local float*)&counters[OP_FXCHG], (float)v);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (int i = 0; i < NUM_OPS; ++i) {
      result[get_group_id(0) * NUM_OPS + i] = counters[i];
    }
  }
}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

// Every atom_* function of the cl_khr_*_int32_*_atomics extensions on one
// block of counters, through a pointer to T (int or uint); the expected values
// are those of atomic_global, with min/max unsigned for uint.
#define ATOM_ALL(c, uc, T, v, gid)                                       \
  do {                                                                   \
    atom_add(&(c)[OP_ADD], (T)(v));                                      \
    atom_sub(&(c)[OP_SUB], (T)(v));                                      \
    atom_inc(&(c)[OP_INC]);                                              \
    atom_dec(&(c)[OP_DEC]);                                              \
    atom_min(&(c)[OP_MIN], (T)(v));                                      \
    atom_max(&(c)[OP_MAX], (T)(v));                                      \
    atom_min(&(uc)[OP_UMIN], (uint)(v));                                 \
    atom_max(&(uc)[OP_UMAX], (uint)(v));                                 \
    atom_and(&(c)[OP_AND], (T)~(1 << ((gid) & 31)));                     \
    atom_or(&(c)[OP_OR], (T)(v));                                        \
    atom_xor(&(c)[OP_XOR], (T)(v));                                      \
    atom_xchg(&(c)[OP_XCHG], (T)(v));                                    \
    T old_ = (c)[OP_CMPXCHG];                                            \
    for (;;) {                                                           \
      T seen_ = atom_cmpxchg(&(c)[OP_CMPXCHG], old_, old_ + (T)(v));     \
      if (seen_ == old_) {                                               \
        break;                                                           \
      }                                                                  \
      old_ = seen_;                                                      \
    }                                                                    \
  } while (0)

// Global memory: an int block at result[0] and a uint block at result[NUM_OPS].
// The float exchange is not an atom_* function and stays untouched.
__kernel void atom_global(__global const int* data, __global int* result) {
  int gid = get_global_id(0);
  int v = data[gid];
  __global uint* uresult = (__global uint*)result;
  ATOM_ALL(result, uresult, int, v, gid);
  ATOM_ALL(uresult + NUM_OPS, uresult + NUM_OPS, uint, v, gid);
}

// Local memory: the same two blocks per work-group, published by work-item 0.
__kernel void atom_local(__global const int* data,
                         __global int* result,
                         __local int* counters) {
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int v = data[gid];
  __local uint* ucounters = (__local uint*)counters;

  if (lid == 0) {
    init_local_counters(counters, 0);
    init_local_counters(counters + NUM_OPS, 1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  ATOM_ALL(counters, ucounters, int, v, gid);
  ATOM_ALL(ucounters + NUM_OPS, ucounters + NUM_OPS, uint, v, gid);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (int i = 0; i < 2 * NUM_OPS; ++i) {
      result[get_group_id(0) * 2 * NUM_OPS + i] = counters[i];
    }
  }
}
