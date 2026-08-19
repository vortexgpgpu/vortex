#ifndef _CGO27_EPI_RESIDUAL_H_
#define _CGO27_EPI_RESIDUAL_H_

// App 4 epilogue: residual add, D = f(C + A*B) + R.
//
// Axis stressed: extra memory traffic — R is a full M x N matrix, so this app adds
// one more full-matrix read on top of the GEMM's C read and D write.
//
// COORDINATE-DEPENDENT: the value added depends on the element's (row, col), so it
// cannot be applied by a plain float->float function on the accumulator. Two ways
// to apply it, and the difference is exactly what the experiment is meant to show:
//   - in-core (0..5): fuse at store time — the tile's (row,col) is known, so
//     R's tile is read while the accumulator is still in registers.
//   - DTCU 7/8: separate SIMT pass over D after the descriptor GEMM, i.e. an
//     extra full round-trip through memory (read D, read R, write D).
// Modes 14/15 apply it in their consumer warps as each slice completes. The host allocates
// R directly behind C, and the CPU reference uses the same row/column indexing.

static inline float epi_residual(float v, float r) {
  return v + r;
}

#endif // _CGO27_EPI_RESIDUAL_H_
