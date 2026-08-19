#ifndef _CGO27_EPI_SCALE_H_
#define _CGO27_EPI_SCALE_H_

// App 5 epilogue: per-channel scale, D[i][j] = f(C + A*B)[i][j] * s[j].
//
// Axis stressed: cheap broadcast — s is a length-N vector, so unlike residual
// (app 4) the extra traffic is negligible and it should be nearly free to fuse.
// It is in the sweep precisely as the CHEAP counterpart to app 4: if the DTCU
// still loses ground here, the cost is the round-trip itself, not the data volume.
//
// COORDINATE-DEPENDENT (on the column only): needs the element's column index.
//   - in-core: fuse at store; each lane already knows its column.
//   - DTCU: separate SIMT pass over D.
//
// The host allocates s directly behind C; all in-core paths fuse it and DTCU 7/8 use
// their standalone pass (14/15 use their consumer warps).

static inline float epi_scale(float v, float s) {
  return v * s;
}

static inline float epi_bias(float v, float b) {
  return v + b;
}

#endif // _CGO27_EPI_SCALE_H_
