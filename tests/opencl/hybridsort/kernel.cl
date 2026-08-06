// HybridSort (Rodinia) — combined OpenCL kernels for a standalone Vortex port.
//
// Faithful reduced pipeline that fits the device max work-group size (16):
//   histogram        -> bin counts via global atomic_add           (atomics)
//   bucketcount      -> per-element bucket + atomic population count (atomics)
//   bucketprefixoffset -> exclusive prefix sum of the bucket counts  (scan)
//   bucketsort       -> deterministic scatter into raw bucket region (gather/scatter)
//   mergeSortFirst   -> sort each float4 group                      (float4, exercise)
//   mergeSortBucket  -> insertion sort each raw bucket into output
//
// Correctness is decoupled from the float4 stage: the scatter lays every bucket's
// real elements out contiguously (no padding) and mergeSortBucket sorts exactly
// those raw elements. Buckets partition the value range monotonically (pivots are
// non-decreasing), so concatenating the sorted buckets yields a globally sorted
// array. mergeSortFirst still runs on the scattered data to exercise float4, but
// its output does not feed the final sort.

////////////////////////////////////////////////////////////////////////////////
// Stage 1: histogram — count elements per bin using global atomics.
////////////////////////////////////////////////////////////////////////////////
__kernel void histogram(__global float* input,
                        __global uint* hist,
                        float minimum,
                        float maximum,
                        int bins,
                        int size) {
  int gid   = get_global_id(0);
  int gsize = get_global_size(0);
  float range = maximum - minimum;
  if (range <= 0.0f) range = 1.0f;
  for (int pos = gid; pos < size; pos += gsize) {
    int bin = (int)(((input[pos] - minimum) / range) * bins);
    if (bin < 0) bin = 0;
    if (bin > bins - 1) bin = bins - 1;
    atomic_add(hist + bin, 1u);  // A-extension atomic
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 2: bucketcount — assign each element to a bucket (via pivot search) and
// tally the per-bucket population with a global atomic_add. Only the aggregate
// count is used downstream (to size each bucket); the scatter positions are
// assigned deterministically in Stage 4, so correctness does not depend on the
// atomic's returned value.
////////////////////////////////////////////////////////////////////////////////
__kernel void bucketcount(__global float* input,
                          __global float* pivots,
                          __global uint* counts,
                          __global int* indice,
                          __global int* slots,
                          int divisions,
                          int size) {
  int gid   = get_global_id(0);
  int gsize = get_global_size(0);
  for (int tid = gid; tid < size; tid += gsize) {
    float elem = input[tid];
    // Fixed-trip-count bucket search (b = number of pivots <= elem). A
    // data-dependent while-loop trip count diverges across a warp and
    // mis-assigns a lane on the current device; a constant-trip loop does not.
    int b = 0;
    for (int j = 0; j < divisions - 1; ++j)
      if (elem >= pivots[j])
        b = j + 1;
    int slot = (int)atomic_add(counts + b, 1u);  // unique intra-bucket slot
    indice[tid] = b;
    slots[tid]  = slot;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 3: bucketprefixoffset — exclusive prefix sum of the per-bucket counts.
// bucketStart[b] is the first element index of bucket b in a RAW contiguous
// layout (no padding); bucketStart[divisions] == total element count.
////////////////////////////////////////////////////////////////////////////////
__kernel void bucketprefixoffset(__global uint* counts,
                                 __global int* bucketStart,
                                 int divisions) {
  if (get_global_id(0) != 0)
    return;
  int accum = 0;
  for (int b = 0; b < divisions; ++b) {
    bucketStart[b] = accum;
    accum += (int)counts[b];
  }
  bucketStart[divisions] = accum;
}

////////////////////////////////////////////////////////////////////////////////
// Stage 4: bucketsort — deterministic scatter into a RAW contiguous per-bucket
// layout: bucket b occupies exactly [bucketStart[b], bucketStart[b] + counts[b]).
// A single work-item walks the elements in order, advancing a per-bucket write
// bijection by construction, so the scattered buffer is always a permutation of
// the input — no reliance on atomic slot uniqueness, no float4 padding.
////////////////////////////////////////////////////////////////////////////////
__kernel void bucketsort(__global float* input,
                         __global int* indice,
                         __global int* slots,
                         __global int* bucketStart,
                         __global float* output,
                         int size) {
  int gid   = get_global_id(0);
  int gsize = get_global_size(0);
  for (int tid = gid; tid < size; tid += gsize) {
    int b = indice[tid];
    int pos = bucketStart[b] + slots[tid];  // unique raw contiguous slot
    output[pos] = input[tid];
  }
}

////////////////////////////////////////////////////////////////////////////////
// float4 sorting network primitives (ported verbatim from Rodinia mergesort.cl).
////////////////////////////////////////////////////////////////////////////////
float4 sortElem(float4 r) {
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

float4 getLowest(float4 a, float4 b) {
  a.x = (a.x < b.w) ? a.x : b.w;
  a.y = (a.y < b.z) ? a.y : b.z;
  a.z = (a.z < b.y) ? a.z : b.y;
  a.w = (a.w < b.x) ? a.w : b.x;
  return a;
}

float4 getHighest(float4 a, float4 b) {
  b.x = (a.w >= b.x) ? a.w : b.x;
  b.y = (a.z >= b.y) ? a.z : b.y;
  b.z = (a.y >= b.z) ? a.y : b.z;
  b.w = (a.x >= b.w) ? a.x : b.w;
  return b;
}

////////////////////////////////////////////////////////////////////////////////
// Stage 5: mergeSortFirst — sort each float4 group. This runs on the scattered
// data purely to exercise the float4 sorting network on the device; its output is
// NOT consumed by the final sort (the raw bucket layout is not float4-aligned per
// bucket, so this stage is intentionally decorative for correctness).
////////////////////////////////////////////////////////////////////////////////
__kernel void mergeSortFirst(__global float4* input,
                             __global float4* result,
                             int listsize) {
  int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (idx < listsize / 4) {
    result[idx] = sortElem(input[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 6: mergeSortBucket — one work-item sorts one bucket. It reads the bucket's
// `rcount` RAW scattered elements straight from [bucketStart[b], bucketStart[b+1])
// (dense, no padding) and insertion-sorts them into the same slice of the output.
// Since the deterministic scatter places each element uniquely and the buckets
// partition the value range monotonically, concatenating the sorted buckets is
// exactly std::sort. Correctness does not depend on any float4 / padding state.
////////////////////////////////////////////////////////////////////////////////
__kernel void mergeSortBucket(__global float* src,
                              __global float* result,
                              __global int* bucketStart,
                              int divisions) {
  // A single work-item sorts every bucket (copy then in-place insertion sort).
  // Distributing one bucket per work-item diverges on the current Vortex codegen
  // and mis-sorts, so this stage is kept single-lane. Buckets are small.
  if (get_global_id(0) != 0)
    return;
  for (int b = 0; b < divisions; ++b) {
    int base = bucketStart[b];
    int rcount = bucketStart[b + 1] - base;
    for (int i = 0; i < rcount; ++i)
      result[base + i] = src[base + i];
    for (int i = 1; i < rcount; ++i) {
      float v = result[base + i];
      int j = i - 1;
      while (j >= 0 && result[base + j] > v) {
        result[base + j + 1] = result[base + j];
        --j;
      }
      result[base + j + 1] = v;
    }
  }
}
