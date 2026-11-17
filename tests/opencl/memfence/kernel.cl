#include "common.h"

// Work-group exchange through global memory: each round every work-item writes
// its slot, then reads a slot another warp wrote. barrier(CLK_GLOBAL_MEM_FENCE)
// orders both the read after the peer's write and, at the end of the round, the
// next round's write after the peer's read.
__kernel void gbarrier(__global uint* buf, __global uint* out, int rounds) {
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);
  uint gid = get_global_id(0);
  uint base = gid - lid;
  uint acc = 0;
  for (int r = 0; r < rounds; ++r) {
    buf[gid] = fence_value(gid, lid, r);
    barrier(CLK_GLOBAL_MEM_FENCE);
    acc = acc * 31u + buf[base + FENCE_PEER(lid, lsz, r)];
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  out[gid] = acc;
}

// The same exchange through local memory, with every local-memory fence
// between the write and the read.
__kernel void lbarrier(__global uint* out, __local uint* lbuf, int rounds) {
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);
  uint gid = get_global_id(0);
  uint acc = 0;
  for (int r = 0; r < rounds; ++r) {
    lbuf[lid] = fence_value(gid, lid, r);
    write_mem_fence(CLK_LOCAL_MEM_FENCE);
    mem_fence(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    read_mem_fence(CLK_LOCAL_MEM_FENCE);
    acc = acc * 31u + lbuf[FENCE_PEER(lid, lsz, r)];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[gid] = acc;
}

// wait_group_events() is the work-group synchronization point of an async
// copy, in both directions. The copy into global memory is read back by other
// work-items of the group right after the wait.
__kernel void async_copy(__global const uint* in, __global uint* tmp,
                         __global uint* out, __local uint* lbuf) {
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);
  uint gid = get_global_id(0);
  uint base = gid - lid;

  event_t ev = async_work_group_copy(lbuf, in + base, lsz, 0);
  wait_group_events(1, &ev);
  uint x = lbuf[(lid + 1) % lsz] * 3u + lid;
  barrier(CLK_LOCAL_MEM_FENCE);
  lbuf[lid] = x;
  barrier(CLK_LOCAL_MEM_FENCE);

  ev = async_work_group_copy(tmp + base, lbuf, lsz, 0);
  wait_group_events(1, &ev);
  out[gid] = tmp[base + FENCE_PEER(lid, lsz, 0)];
}

#ifdef HAS_ATOMICS

// Device-scope publication without spinning: every work-group publishes its
// partial sum, and the one that takes the last ticket adds them all up. The
// writer releases its partial sum before taking a ticket and the last
// work-group acquires before reading. Before anything is published, every
// work-group reads all partial sums once, so a core whose L1 keeps those lines
// returns stale values unless the acquire invalidates them.
// mode 0 uses write_mem_fence / read_mem_fence, mode 1 mem_fence.
__kernel void lastgroup(__global const int* data, __global int* partial,
                        __global uint* count, __global int* result,
                        __local int* scratch, int mode) {
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);
  uint grp = get_group_id(0);
  uint ngrp = get_num_groups(0);

  if (lid == 0) {
    volatile __global int* vpartial = partial;
    for (uint g = 0; g < ngrp; ++g) {
      (void)vpartial[g];
    }
  }

  scratch[lid] = data[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) {
      scratch[lid] += scratch[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    partial[grp] = scratch[0];
    if (mode) {
      mem_fence(CLK_GLOBAL_MEM_FENCE);
    } else {
      write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    }
    uint ticket = atomic_inc(count);
    if (ticket == ngrp - 1) {
      if (mode) {
        mem_fence(CLK_GLOBAL_MEM_FENCE);
      } else {
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
      }
      int sum = 0;
      for (uint g = 0; g < ngrp; ++g) {
        sum += partial[g];
      }
      result[0] = sum;
    }
  }
}

#endif
