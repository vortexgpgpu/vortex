// dwt2d (Rodinia) — forward 5/3 reversible DWT, standalone Vortex port.
//
// The upstream Rodinia cl_fdwt53Kernel is a tiled sliding-window transform
// hard-wired to a 256-thread work-group with shared-memory column loaders and
// subband BandIO writers — it cannot fit the device work-group limit of 16.
// This kernel keeps the SAME 5/3 lifting scheme (Forward53Predict /
// Forward53Update) but applies it as a straightforward non-tiled row/column
// transform: each work-item independently lifts one full line, so no shared
// memory, barriers, or atomics are needed. A 2-D level is one horizontal pass
// (rows) followed by one vertical pass (columns).
//
// BLOCK_SIZE (the work-group size, <= 16) is supplied via clBuildProgram -D to
// match the host launch configuration.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// In-place 1-D forward 5/3 lifting over `n` samples at the given stride.
// Even indices hold low-pass (approx) output, odd indices hold high-pass
// (detail) output. Symmetric (mirror) boundary extension. Integer divisions
// are identical to the Rodinia lifting operators, so the result is exact.
static void lift53(__global int* a, int n, int stride) {
  // Predict odd samples: c -= (prev + next) / 2
  for (int i = 1; i < n; i += 2) {
    int prev = a[(i - 1) * stride];
    int next = (i + 1 < n) ? a[(i + 1) * stride] : a[(i - 1) * stride];  // mirror
    a[i * stride] -= (prev + next) / 2;
  }
  // Update even samples: c += (prev + next + 2) / 4
  for (int i = 0; i < n; i += 2) {
    int prev = (i - 1 >= 0) ? a[(i - 1) * stride] : a[(i + 1) * stride];  // mirror
    int next = (i + 1 < n) ? a[(i + 1) * stride] : a[(i - 1) * stride];   // mirror
    a[i * stride] += (prev + next + 2) / 4;
  }
}

// Horizontal pass: one work-item per row, contiguous stride.
__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__kernel void fdwt53_horizontal(__global int* data, int N) {
  int r = get_global_id(0);
  if (r >= N)
    return;
  lift53(data + r * N, N, 1);
}

// Vertical pass: one work-item per column, row-stride N.
__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__kernel void fdwt53_vertical(__global int* data, int N) {
  int c = get_global_id(0);
  if (c >= N)
    return;
  lift53(data + c, N, N);
}
