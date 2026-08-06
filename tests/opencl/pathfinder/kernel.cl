// PathFinder dynamic-programming kernel (ghost-zone / pyramid blocking).
// Ported from Rodinia. Each work-group advances `iteration` rows of the wall,
// caching a halo-padded strip in local memory so neighbouring rows are reused.

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__kernel void dynproc_kernel(int iteration,
                             __global int* gpuWall,
                             __global int* gpuSrc,
                             __global int* gpuResults,
                             int cols,
                             int rows,
                             int startStep,
                             int border,
                             int HALO,
                             __local int* prev,
                             __local int* result) {
  int BLOCK_SIZE = get_local_size(0);
  int bx = get_group_id(0);
  int tx = get_local_id(0);

  // Non-overlapping small block covered by this work-group after N iterations.
  int small_block_cols = BLOCK_SIZE - (iteration * HALO * 2);

  // Block boundary in global coordinates (halo-extended).
  int blkX = (small_block_cols * bx) - border;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  int xidx = blkX + tx;

  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == iteration - 1)
      break;
    if (computed)
      prev[tx] = result[tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}
