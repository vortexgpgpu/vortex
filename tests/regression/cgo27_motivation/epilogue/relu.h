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
// This is the light elementwise endpoint, not a divergence benchmark: at -O3 the
// compiler can flatten the ternary to a select/fmax.

static inline __attribute__((always_inline)) float epi_relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

#endif // _CGO27_EPI_RELU_H_
