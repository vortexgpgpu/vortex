// SRAD (Speckle Reducing Anisotropic Diffusion) kernels, ported from Rodinia.
//
// The device runs the non-tiled srad variant: extract (log-uncompress), then
// per iteration srad (directional derivatives + diffusion coefficient) and
// srad2 (divergence + image update), then compress (log-recompress). The
// statistics reduction (q0sqr) is done on the host, so no shared-memory tree /
// atomics are needed here. Constants are float literals to keep the whole
// computation single-precision (parity with the host reference).
//
// Image is Nr x Nc stored column-major: element (row, col) lives at row + Nr*col.

#define fp float

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 16
#endif

// Extract: scale input image and log-uncompress (0-255 range -> positive intensity).
__kernel void extract_kernel(long d_Ne,
                             __global fp* d_I) {
  int bx = get_group_id(0);
  int tx = get_local_id(0);
  int ei = (bx * NUMBER_THREADS) + tx;

  if (ei < d_Ne) {
    d_I[ei] = exp(d_I[ei] / 255);
  }
}

// SRAD: directional derivatives, ICOV, and diffusion coefficient per pixel.
__kernel void srad_kernel(fp d_lambda,
                          int d_Nr,
                          int d_Nc,
                          long d_Ne,
                          __global int* d_iN,
                          __global int* d_iS,
                          __global int* d_jE,
                          __global int* d_jW,
                          __global fp* d_dN,
                          __global fp* d_dS,
                          __global fp* d_dE,
                          __global fp* d_dW,
                          fp d_q0sqr,
                          __global fp* d_c,
                          __global fp* d_I) {
  int bx = get_group_id(0);
  int tx = get_local_id(0);
  int ei = bx * NUMBER_THREADS + tx;
  int row;
  int col;

  fp d_Jc;
  fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
  fp d_c_loc;
  fp d_G2, d_L, d_num, d_den, d_qsqr;

  // figure out row/col location in new matrix (column-major)
  row = (ei + 1) % d_Nr - 1;
  col = (ei + 1) / d_Nr + 1 - 1;
  if ((ei + 1) % d_Nr == 0) {
    row = d_Nr - 1;
    col = col - 1;
  }

  if (ei < d_Ne) {
    // current element and directional derivatives
    d_Jc = d_I[ei];
    d_dN_loc = d_I[d_iN[row] + d_Nr * col] - d_Jc;  // north
    d_dS_loc = d_I[d_iS[row] + d_Nr * col] - d_Jc;  // south
    d_dW_loc = d_I[row + d_Nr * d_jW[col]] - d_Jc;  // west
    d_dE_loc = d_I[row + d_Nr * d_jE[col]] - d_Jc;  // east

    // normalized discrete gradient mag squared / laplacian
    d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
            d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) / (d_Jc * d_Jc);
    d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

    // ICOV
    d_num = (0.5f * d_G2) - ((1.0f / 16.0f) * (d_L * d_L));
    d_den = 1 + (0.25f * d_L);
    d_qsqr = d_num / (d_den * d_den);

    // diffusion coefficient
    d_den = (d_qsqr - d_q0sqr) / (d_q0sqr * (1 + d_q0sqr));
    d_c_loc = 1.0f / (1.0f + d_den);

    // saturate to 0-1 range
    if (d_c_loc < 0) {
      d_c_loc = 0;
    } else if (d_c_loc > 1) {
      d_c_loc = 1;
    }

    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
  }
}

// SRAD2: divergence of the flux and the diffusion image update.
__kernel void srad2_kernel(fp d_lambda,
                           int d_Nr,
                           int d_Nc,
                           long d_Ne,
                           __global int* d_iN,
                           __global int* d_iS,
                           __global int* d_jE,
                           __global int* d_jW,
                           __global fp* d_dN,
                           __global fp* d_dS,
                           __global fp* d_dE,
                           __global fp* d_dW,
                           __global fp* d_c,
                           __global fp* d_I) {
  int bx = get_group_id(0);
  int tx = get_local_id(0);
  int ei = bx * NUMBER_THREADS + tx;
  int row;
  int col;

  fp d_cN, d_cS, d_cW, d_cE;
  fp d_D;

  row = (ei + 1) % d_Nr - 1;
  col = (ei + 1) / d_Nr + 1 - 1;
  if ((ei + 1) % d_Nr == 0) {
    row = d_Nr - 1;
    col = col - 1;
  }

  if (ei < d_Ne) {
    // diffusion coefficients (north/west reuse current cell)
    d_cN = d_c[ei];
    d_cS = d_c[d_iS[row] + d_Nr * col];
    d_cW = d_c[ei];
    d_cE = d_c[row + d_Nr * d_jE[col]];

    // divergence and image update
    d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] + d_cE * d_dE[ei];
    d_I[ei] = d_I[ei] + 0.25f * d_lambda * d_D;
  }
}

// Compress: log-recompress the diffused image back to the 0-255 range.
__kernel void compress_kernel(long d_Ne,
                              __global fp* d_I) {
  int bx = get_group_id(0);
  int tx = get_local_id(0);
  int ei = (bx * NUMBER_THREADS) + tx;

  if (ei < d_Ne) {
    d_I[ei] = log(d_I[ei]) * 255;
  }
}
