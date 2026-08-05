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
//   - in-core (0,1,2,5,6): fuse at store time — the tile's (row,col) is known, so
//     R's tile is read while the accumulator is still in registers.
//   - DTCU (3,4): separate SIMT pass over D after the descriptor GEMM, i.e. an
//     extra full round-trip through memory (read D, read R, write D).
//
// WIRING STATUS: math is here; the R buffer (host allocation + upload + CPU
// reference) is Phase B — see 260718_moti_RFC.md §8. Fusing it in-core also needs
// the per-register (row,col) mapping, which wmma_context keeps private (cfg is a
// private member), so the in-core path will either replicate that layout math or
// apply this in a per-tile post-pass over the just-stored tile.

static inline float epi_residual(float v, float r) {
  return v + r;
}

#endif // _CGO27_EPI_RESIDUAL_H_
