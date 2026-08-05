#ifndef _CGO27_EPI_DEQUANT_H_
#define _CGO27_EPI_DEQUANT_H_

// Apps 7 and 8 PROLOGUE (not an epilogue — kept in this directory so the whole
// pro/epilogue family lives in one place): int8 -> fp16 dequantization of the A
// operand, D = f(C + dequant(Aq)*B) [+ bias].
//
// Axis stressed: work BEFORE the GEMM rather than after it. The in-core modes can
// fold it into the operand load (dequantize on the way into the fragment); the DTCU
// takes a descriptor pointing at already-typed operands, so it needs a separate
// SIMT pass to materialize a dequantized A in memory first — a full extra
// read+write of A before the GEMM even starts. Apps 7/8 therefore tax the DTCU on
// BOTH ends (prologue pass + epilogue pass), which is the strongest case for the
// in-core path in the sweep.
//
// WIRING STATUS: math is here; the int8 A buffer + scale (host allocation, upload,
// CPU reference) and the in-core fused-load variant are Phase B.

// Symmetric per-tensor dequantization: x = q * scale.
static inline float epi_dequant_i8(int8_t q, float scale) {
  return (float)q * scale;
}

// Per-channel (per-K-column) variant, for when the quantizer keeps one scale per
// input channel instead of one per tensor.
static inline float epi_dequant_i8_pc(int8_t q, float scale_k) {
  return (float)q * scale_k;
}

#endif // _CGO27_EPI_DEQUANT_H_
