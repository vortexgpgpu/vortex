// lavaMD (Rodinia) force kernel — ported for Vortex.
//
// Computes inter-particle forces (Lennard-Jones-like) over a 3-D grid of boxes.
// One work-group processes one home box; the local work-group size equals
// NUMBER_PAR_PER_BOX. Each thread owns one home-box particle and accumulates the
// force/potential contribution from every particle in the home box and its (up
// to 26) neighbour boxes. Physics is identical to the original benchmark.
//
// Notes for the Vortex port:
//   * NUMBER_PAR_PER_BOX / NUMBER_THREADS are overridable via clBuildProgram
//     -D flags so the host and device agree on the particle/thread counts.
//   * Offsets use `int` (not `long`) so the box_str/dim_str struct ABI matches
//     the 32-bit device and the host byte-for-byte. Offsets stay small here.

#ifdef __cplusplus
extern "C" {
#endif

#define fp float

#ifndef NUMBER_PAR_PER_BOX
#define NUMBER_PAR_PER_BOX 16
#endif

// Stride of the home-particle loop. Equal to the local work-group size, so each
// thread handles exactly one particle.
#ifndef NUMBER_THREADS
#define NUMBER_THREADS NUMBER_PAR_PER_BOX
#endif

#define DOT(A, B) ((A.x) * (B.x) + (A.y) * (B.y) + (A.z) * (B.z))

typedef struct {
  fp x, y, z;
} THREE_VECTOR;

typedef struct {
  fp v, x, y, z;
} FOUR_VECTOR;

typedef struct nei_str {
  int x, y, z;
  int number;
  int offset;
} nei_str;

typedef struct box_str {
  // home box
  int x, y, z;
  int number;
  int offset;
  // neighbor boxes
  int nn;
  nei_str nei[26];
} box_str;

typedef struct par_str {
  fp alpha;
} par_str;

typedef struct dim_str {
  int cur_arg;
  int arch_arg;
  int cores_arg;
  int boxes1d_arg;
  int number_boxes;
  int box_mem;
  int space_elem;
  int space_mem;
  int space_mem2;
} dim_str;

// alpha and number_boxes are passed as scalar args rather than the original
// by-value par_str/dim_str structs: pocl-vortex by-value struct kernel args are
// ABI-fragile, and these two fields are all the kernel actually needs.
__kernel void kernel_gpu_opencl(fp alpha,
                                int number_boxes,
                                __global box_str* d_box_gpu,
                                __global FOUR_VECTOR* d_rv_gpu,
                                __global fp* d_qv_gpu,
                                __global FOUR_VECTOR* d_fv_gpu) {
  int bx = get_group_id(0);   // home box index
  int tx = get_local_id(0);   // thread (particle) index within the box
  int wtx = tx;

  // Home/neighbour particle staging in local memory.
  __local FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];
  __local FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
  __local fp qB_shared[NUMBER_PAR_PER_BOX];

  if (bx < number_boxes) {
    fp a2 = 2 * alpha * alpha;

    int first_i;
    int pointer;
    int k = 0;
    int first_j;
    int j = 0;

    fp r2, u2, vij, fs, fxij, fyij, fzij;
    THREE_VECTOR d;

    // Home box particles into local memory.
    first_i = d_box_gpu[bx].offset;
    while (wtx < NUMBER_PAR_PER_BOX) {
      rA_shared[wtx] = d_rv_gpu[first_i + wtx];
      wtx = wtx + NUMBER_THREADS;
    }
    wtx = tx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the home box (k==0) and its neighbour boxes.
    for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {
      if (k == 0) {
        pointer = bx;
      } else {
        pointer = d_box_gpu[bx].nei[k - 1].number;
      }

      first_j = d_box_gpu[pointer].offset;

      // Neighbour box particles into local memory.
      while (wtx < NUMBER_PAR_PER_BOX) {
        rB_shared[wtx] = d_rv_gpu[first_j + wtx];
        qB_shared[wtx] = d_qv_gpu[first_j + wtx];
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;
      barrier(CLK_LOCAL_MEM_FENCE);

      // Accumulate force/potential contributions.
      while (wtx < NUMBER_PAR_PER_BOX) {
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
          r2 = rA_shared[wtx].v + rB_shared[j].v - DOT(rA_shared[wtx], rB_shared[j]);
          u2 = a2 * r2;
          vij = exp(-u2);
          fs = 2 * vij;
          d.x = rA_shared[wtx].x - rB_shared[j].x;
          fxij = fs * d.x;
          d.y = rA_shared[wtx].y - rB_shared[j].y;
          fyij = fs * d.y;
          d.z = rA_shared[wtx].z - rB_shared[j].z;
          fzij = fs * d.z;
          d_fv_gpu[first_i + wtx].v += qB_shared[j] * vij;
          d_fv_gpu[first_i + wtx].x += qB_shared[j] * fxij;
          d_fv_gpu[first_i + wtx].y += qB_shared[j] * fyij;
          d_fv_gpu[first_i + wtx].z += qB_shared[j] * fzij;
        }
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

#ifdef __cplusplus
}
#endif
