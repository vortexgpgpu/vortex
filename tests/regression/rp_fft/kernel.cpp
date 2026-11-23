#include <vx_spawn2.h>
#include "common.h"

// Batched FFT, one transform of FFT_LEN points per thread, as a Stockham
// autosort of radix-RADIX passes: each pass loads RADIX points, twiddles them,
// runs a RADIX-point DFT in registers and stores them. A larger radix means
// fewer passes over memory but more live registers. The batch is interleaved
// (point i of transform f at i * num_ffts + f), so a warp's lanes read
// neighbouring words.

#define PI_F 3.14159265358979f

template <int N>
struct RegDFT {
  // In-register radix-2 decimation in time; twiddles fold to constants.
  static inline __attribute__((always_inline)) void run(float* re, float* im) {
    float er[N / 2], ei[N / 2], odr[N / 2], odi[N / 2];
    #pragma clang loop unroll(full)
    for (int i = 0; i < N / 2; ++i) {
      er[i] = re[2 * i];      ei[i] = im[2 * i];
      odr[i] = re[2 * i + 1]; odi[i] = im[2 * i + 1];
    }
    RegDFT<N / 2>::run(er, ei);
    RegDFT<N / 2>::run(odr, odi);
    #pragma clang loop unroll(full)
    for (int k = 0; k < N / 2; ++k) {
      float wr = __builtin_cosf(-2.0f * PI_F * k / N);
      float wi = __builtin_sinf(-2.0f * PI_F * k / N);
      float tr = wr * odr[k] - wi * odi[k];
      float ti = wr * odi[k] + wi * odr[k];
      re[k] = er[k] + tr;          im[k] = ei[k] + ti;
      re[k + N / 2] = er[k] - tr;  im[k + N / 2] = ei[k] - ti;
    }
  }
};

template <>
struct RegDFT<1> {
  static inline __attribute__((always_inline)) void run(float*, float*) {}
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= arg->num_ffts)
    return;
  uint32_t nb = arg->num_ffts;
  auto in  = reinterpret_cast<float*>(arg->data_addr) + 2 * id;
  auto out = reinterpret_cast<float*>(arg->tmp_addr) + 2 * id;
  auto tw  = reinterpret_cast<const float*>(arg->tw_addr);
  constexpr uint32_t STRIDE = FFT_LEN / RADIX;

  constexpr uint32_t LG_RADIX = __builtin_ctz(RADIX);
  for (uint32_t lg_ns = 0; (1u << lg_ns) < FFT_LEN; lg_ns += LG_RADIX) {
    uint32_t ns = 1u << lg_ns;
    uint32_t tw_step = FFT_LEN >> (lg_ns + LG_RADIX);
    for (uint32_t j = 0; j < STRIDE; ++j) {
      float re[RADIX], im[RADIX];
      #pragma clang loop unroll(full)
      for (uint32_t r = 0; r < RADIX; ++r) {
        re[r] = in[2 * (j + r * STRIDE) * nb];
        im[r] = in[2 * (j + r * STRIDE) * nb + 1];
      }
      uint32_t k = j & (ns - 1);
      #pragma clang loop unroll(full)
      for (uint32_t r = 1; r < RADIX; ++r) {
        uint32_t t = (k * r * tw_step) & (FFT_LEN - 1);
        float wr = tw[2 * t], wi = tw[2 * t + 1];
        float xr = re[r] * wr - im[r] * wi;
        float xi = re[r] * wi + im[r] * wr;
        re[r] = xr;
        im[r] = xi;
      }
      RegDFT<RADIX>::run(re, im);
      uint32_t base = ((j >> lg_ns) << (lg_ns + LG_RADIX)) + k;
      #pragma clang loop unroll(full)
      for (uint32_t r = 0; r < RADIX; ++r) {
        out[2 * (base + r * ns) * nb] = re[r];
        out[2 * (base + r * ns) * nb + 1] = im[r];
      }
    }
    auto t = in; in = out; out = t;
  }
  // with an odd number of passes the result ends in the scratch buffer
}
