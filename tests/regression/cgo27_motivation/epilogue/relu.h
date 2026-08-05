#ifndef _CGO27_EPI_RELU_H_
#define _CGO27_EPI_RELU_H_

// App 2 epilogue: ReLU.
//
// Shared by host and device on purpose — main.cpp's CPU reference calls the SAME
// function the kernel does, so verification compares identical arithmetic instead
// of two independent implementations that differ in the last bits.
//
// Coordinate-independent (needs only the accumulator value), so the in-core modes
// fuse it on the accumulator fragment before the store; the DTCU modes run it as a
// separate SIMT pass over D (the engine has no epilogue HW).
//
// CAVEAT on the "divergence" axis: the RFC lists this app as the branch/divergence
// stressor, but at -O3 the compiler flattens this ternary into a branchless
// select/fmax, so it does NOT actually produce warp divergence. If divergence is
// what we want to measure, the app needs a formulation the compiler cannot
// flatten (e.g. a data-dependent side effect); flagged in the RFC.

static inline __attribute__((always_inline)) float epi_relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

#endif // _CGO27_EPI_RELU_H_
