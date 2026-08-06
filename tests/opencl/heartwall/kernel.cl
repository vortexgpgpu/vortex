// Heartwall (Rodinia) core template-matching kernel — standalone Vortex port.
//
// Heartwall tracks features across ultrasound frames. Its computational core is
// template matching: for each feature it rotates the feature template 180 deg
// and full-2D-convolves it against a larger search window extracted from the
// frame. Convolution with the 180-rotated template equals the cross-correlation
// of the template with the search window — the correlation map whose peak gives
// the new feature position.
//
// Both kernels reproduce the exact upstream arithmetic (the ROTATION and ACTUAL
// CONVOLUTION blocks of kernel_gpu_opencl.cl). Buffers are laid out
// column-major per feature: element (row,col) of an R-row matrix is at col*R+row
// (matching the upstream d_in_mod_temp[in_rows*(ja-1)+ia-1] indexing).
//
// Work-item mapping (work-group size <= 16, device max = NUM_WARPS*NUM_THREADS):
//   rotate:   one work-item per template element  (feature*in_elem   + ei_new)
//   convolve: one work-item per conv output pixel (feature*conv_elem + ei_new)
// Each work-item writes a distinct output element, so no atomics and no barriers
// are needed. The convolve pass is enqueued after rotate on the same in-order
// queue, so the rotated templates are fully written before convolution reads.

// Rotate each feature template 180 degrees: d_in_mod[ei] = d_in[rotated ei].
__kernel void rotate_template(__global const float* d_in_all,   // [allPoints*in_elem]
                              __global float* d_in_mod_all,       // [allPoints*in_elem]
                              int in_rows,
                              int in_elem,
                              int allPoints) {
  int gid = get_global_id(0);
  int feature = gid / in_elem;
  int ei_new  = gid % in_elem;
  if (feature >= allPoints)
    return;

  __global const float* d_in     = &d_in_all[feature * in_elem];
  __global float*       d_in_mod = &d_in_mod_all[feature * in_elem];

  // figure out row/col location (0-n), exactly as upstream
  int row = (ei_new + 1) % in_rows - 1;
  int col = (ei_new + 1) / in_rows + 1 - 1;
  if ((ei_new + 1) % in_rows == 0) {
    row = in_rows - 1;
    col = col - 1;
  }

  int rot_row = (in_rows - 1) - row;
  int rot_col = (in_rows - 1) - col;
  d_in_mod[ei_new] = d_in[rot_col * in_rows + rot_row];
}

// Full 2D convolution of the rotated template against the search window.
// Identical arithmetic to the upstream "ACTUAL CONVOLUTION" block.
__kernel void convolve(__global const float* d_in_mod_all,  // [allPoints*in_elem]  rotated template
                       __global const float* d_in2_all,     // [allPoints*in2_elem] search window
                       __global float* d_conv_all,          // [allPoints*conv_elem] output
                       int in_rows,
                       int in_cols,
                       int in2_rows,
                       int in2_cols,
                       int conv_rows,
                       int conv_elem,
                       int in_elem,
                       int in2_elem,
                       int ioffset,
                       int joffset,
                       int allPoints) {
  int gid = get_global_id(0);
  int feature = gid / conv_elem;
  int ei_new  = gid % conv_elem;
  if (feature >= allPoints)
    return;

  __global const float* d_in_mod = &d_in_mod_all[feature * in_elem];
  __global const float* d_in2    = &d_in2_all[feature * in2_elem];
  __global float*       d_conv   = &d_conv_all[feature * conv_elem];

  // figure out row/col location in output (1-n)
  int ic = (ei_new + 1) % conv_rows;
  int jc = (ei_new + 1) / conv_rows + 1;
  if ((ei_new + 1) % conv_rows == 0) {
    ic = conv_rows;
    jc = jc - 1;
  }

  int j = jc + joffset;
  int jp1 = j + 1;
  int ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
  int ja2 = (in_cols < j) ? in_cols : j;

  int i = ic + ioffset;
  int ip1 = i + 1;
  int ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
  int ia2 = (in_rows < i) ? in_rows : i;

  float s = 0;
  for (int ja = ja1; ja <= ja2; ja++) {
    int jb = jp1 - ja;
    for (int ia = ia1; ia <= ia2; ia++) {
      int ib = ip1 - ia;
      s += d_in_mod[in_rows * (ja - 1) + ia - 1] * d_in2[in2_rows * (jb - 1) + ib - 1];
    }
  }

  d_conv[ei_new] = s;
}
