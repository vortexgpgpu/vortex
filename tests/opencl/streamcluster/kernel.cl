/* ============================================================
   StreamCluster (Rodinia) pgain kernel — ported for Vortex.
   Original: Jianbin Fang, 02/03/2011.

   Three Vortex correctness fixes vs. the stock Rodinia kernel:
   (a) Point_Struct.assign is `int`, not `long`. `long` is 8 bytes on the
       x86 host but 4 bytes on the 32-bit RISC-V device, so a `long` field
       mis-aligns the struct and p[tid].assign reads garbage -> wild OOB.
       Using `int` on BOTH host and device makes the ABI match.
   (b) memset_kernel bounds-guards its write (thread_id < number_bytes);
       the global size is rounded up past number_bytes and the original
       wrote OOB, stalling the LSU.
   (c) pgain_kernel hoists the shared-mem fill + barrier OUTSIDE the
       `if (thread_id < num)` guard so ALL work-items in the group reach
       the barrier. A divergent barrier deadlocks on Vortex SIMT when
       num is not a multiple of the work-group size.
   ============================================================ */

typedef struct {
  float weight;
  int assign;  /* (a) int, not long: match 32-bit device ABI */
  float cost;  /* cost of that assignment, weight*distance */
} Point_Struct;

/* Byte-wise memset used to clear the work_mem / switch_membership buffers. */
__kernel void memset_kernel(__global char *mem_d, short val, int number_bytes) {
  const int thread_id = get_global_id(0);
  if (thread_id < number_bytes)          /* (b) bounds guard */
    mem_d[thread_id] = val;
}

/* pgain: per-point cost gain of opening a new center at point x. */
__kernel void pgain_kernel(
    __global Point_Struct *p,
    __global float *coord_d,
    __global float *work_mem_d,
    __global int *center_table_d,
    __global char *switch_membership_d,
    __local float *coord_s,
    int num,
    int dim,
    long x,
    int K) {
  const int thread_id = get_global_id(0);
  const int local_id = get_local_id(0);

  /* (c) coordinate of point[x] into shared mem, then a UNIFORM barrier:
     both live outside the thread_id<num guard so every work-item in the
     group reaches the barrier. */
  if (local_id == 0)
    for (int i = 0; i < dim; i++)
      coord_s[i] = coord_d[i * num + x];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (thread_id < num) {
    /* squared-euclidean distance to point[x], scaled by weight */
    float x_cost = 0.0f;
    for (int i = 0; i < dim; i++)
      x_cost += (coord_d[(i * num) + thread_id] - coord_s[i]) *
                (coord_d[(i * num) + thread_id] - coord_s[i]);
    x_cost = x_cost * p[thread_id].weight;

    float current_cost = p[thread_id].cost;

    int base = thread_id * (K + 1);
    if (x_cost < current_cost) {
      /* reassigning to x saves cost -> mark for switch */
      switch_membership_d[thread_id] = '1';
      work_mem_d[base + K] = x_cost - current_cost;
    } else {
      /* keeping current center costs more -> record the shortfall */
      int assign = p[thread_id].assign;
      work_mem_d[base + center_table_d[assign]] += current_cost - x_cost;
    }
  }
}
