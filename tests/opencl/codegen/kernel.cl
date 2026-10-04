#include "common.h"

// By-value structs of several sizes and alignments, interleaved with scalars
// and __local pointers, so the argument block mixes inline aggregates, scalar
// slots and the local-memory size slot. The kernel also writes its own copy of
// one struct: every work-item must still see the value the host passed.
__kernel void k_byval(__global int* out, char c0, s12_t s12, __local int* lm,
                      s24_t s24, int k, s16_t s16, __local int* lb, s1_t s1) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  lm[lid] = s24.v[lid % 5] + s24.tag;
  lb[lid] = (int)s16.q + s16.i;
  barrier(CLK_LOCAL_MEM_FENCE);
  __global int* o = out + gid * BYVAL_OUTS;
  o[0] = c0;
  o[1] = s12.a + s12.b * 3 + s12.c * 5 + s12.d * 7;
  o[2] = lm[(lid + 1) % get_local_size(0)];
  o[3] = k;
  o[4] = (int)(s16.q >> 32) ^ s16.i;
  o[5] = lb[lid];
  o[6] = s1.c;
  s12.a += gid;              // writes this work-item's private copy
  o[7] = s12.a;
}

// Merge of two sorted runs of float4, in the shape of Rodinia hybridsort's
// mergeSortPass: loop-carried float4 values reassigned from loaded ones on
// each branch, so register allocation has to copy whole float4 register
// groups between iterations.
float4 sort_elem(float4 r) {
  float4 nr;
  nr.x = (r.x > r.y) ? r.y : r.x;
  nr.y = (r.y > r.x) ? r.y : r.x;
  nr.z = (r.z > r.w) ? r.w : r.z;
  nr.w = (r.w > r.z) ? r.w : r.z;
  r.x = (nr.x > nr.z) ? nr.z : nr.x;
  r.y = (nr.y > nr.w) ? nr.w : nr.y;
  r.z = (nr.z > nr.x) ? nr.z : nr.x;
  r.w = (nr.w > nr.y) ? nr.w : nr.y;
  nr.x = r.x;
  nr.y = (r.y > r.z) ? r.z : r.y;
  nr.z = (r.z > r.y) ? r.z : r.y;
  nr.w = r.w;
  return nr;
}

float4 get_lowest(float4 a, float4 b) {
  a.x = (a.x < b.w) ? a.x : b.w;
  a.y = (a.y < b.z) ? a.y : b.z;
  a.z = (a.z < b.y) ? a.z : b.y;
  a.w = (a.w < b.x) ? a.w : b.x;
  return a;
}

float4 get_highest(float4 a, float4 b) {
  b.x = (a.w >= b.x) ? a.w : b.x;
  b.y = (a.z >= b.y) ? a.z : b.y;
  b.z = (a.y >= b.z) ? a.y : b.z;
  b.w = (a.x >= b.w) ? a.x : b.w;
  return b;
}

__kernel void k_vec4(__global const float4* input, __global float4* result, int run_len) {
  int gid = get_global_id(0);
  int a_start = gid * 2 * run_len;
  int b_start = a_start + run_len;
  __global float4* res = result + a_start;
  int aidx = 0, bidx = 0, outidx = 0;
  float4 a = input[a_start];
  float4 b = input[b_start];
  while (true) {
    float4 next_a = input[a_start + aidx + 1];
    float4 next_b = input[b_start + bidx + 1];
    float4 na = get_lowest(a, b);
    float4 nb = get_highest(a, b);
    a = sort_elem(na);
    b = sort_elem(nb);
    res[outidx++] = a;
    bool left_a = aidx + 1 < run_len;
    bool left_b = bidx + 1 < run_len;
    if (left_a) {
      if (left_b) {
        if (next_a.x < next_b.x) { aidx += 1; a = next_a; }
        else { bidx += 1; a = next_b; }
      } else {
        aidx += 1; a = next_a;
      }
    } else {
      if (left_b) { bidx += 1; a = next_b; }
      else break;
    }
  }
  res[outidx++] = b;
}
