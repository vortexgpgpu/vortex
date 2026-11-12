// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// FlashAttention-2 forward pass with both GEMMs on WGMMA: tf32, fp16 or bf16
// inputs, fp32 accumulation and softmax.

#include <flash_attn.h>
#include "common.h"

#include <cstring>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <type_traits>

namespace {

namespace vt = vortex::tensor;

using wg_cfg  = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;
using input_t = typename vt::ITYPE::dtype;

// block_size_r steps by xtileM. block_size_c and head_dim(_tile) each play N in
// one GEMM and K in the other, so they step by the larger of xtileN and tileK.
constexpr uint32_t kQuantR = wg_cfg::xtileM;
constexpr uint32_t kQuantC = (wg_cfg::xtileN > wg_cfg::tileK) ? wg_cfg::xtileN : wg_cfg::tileK;

static_assert(std::is_same_v<vt::ITYPE, vt::tf32> || std::is_same_v<vt::ITYPE, vt::fp16>
              || std::is_same_v<vt::ITYPE, vt::bf16>, "ITYPE must be tf32, fp16 or bf16");

// fp32 -> ITYPE, rounding as the device's fpcvt.h does
input_t pack(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if constexpr (std::is_same_v<vt::ITYPE, vt::tf32>) {
    return (input_t)rv_ftotf32_s(u, 0, nullptr);
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) {
    return (input_t)rv_ftoh_s(u, 0, nullptr);
  } else {
    return (input_t)rv_ftob_s(u, 0, nullptr);
  }
}

float unpack(input_t v) {
  uint32_t u;
  if constexpr (std::is_same_v<vt::ITYPE, vt::tf32>) {
    u = rv_tf32tof_s((uint32_t)v, 0, nullptr);
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) {
    u = rv_htof_s((uint16_t)v, 0, nullptr);
  } else {
    u = rv_btof_s((uint16_t)v, 0, nullptr);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// LMEM per CTA: Q, a K/V staging tile and packed P in ITYPE; S, O and the
// per-row m, l and rescale factor in fp32.
uint64_t lmem_bytes(uint32_t r, uint32_t c, uint32_t t, uint32_t D) {
  return sizeof(input_t) * (uint64_t(r) * D + uint64_t(c) * t + uint64_t(r) * c)
       + sizeof(float) * (uint64_t(r) * c + uint64_t(r) * D + 3ull * r);
}

} // namespace

int main(int argc, char* argv[]) {
  flash_attn::Args args;
  args.head_dim = 32;
  flash_attn::parse_args(argc, argv, args);
  const uint32_t N = args.seq_len, D = args.head_dim;
  std::printf("flash_attn_tcu_wg: seq_len=%u head_dim=%u ITYPE=%s OTYPE=%s NRC=%u\n",
              N, D, vt::ITYPE::name, vt::OTYPE::name, WGMMA_NRC);
  std::printf("WGMMA geometry: tcM=%u tcN=%u xtileM=%u xtileN=%u tileK=%u\n",
              wg_cfg::tcM, wg_cfg::tcN, wg_cfg::xtileM, wg_cfg::xtileN, wg_cfg::tileK);

  flash_attn::Session session;
  if ((session.caps(VX_CAPS_ISA_FLAGS) & VX_ISA_EXT_TCU) == 0) {
    std::printf("Error: TCU extension not supported\n");
    return 1;
  }
  uint32_t num_threads = session.caps(VX_CAPS_NUM_THREADS);
  uint32_t num_warps   = session.caps(VX_CAPS_NUM_WARPS);
  uint64_t lmem_size   = session.caps(VX_CAPS_LOCAL_MEM_SIZE);
  // a CTA is one warpgroup: ISSUE_WIDTH warps
  uint32_t cta_warps   = session.caps(VX_CAPS_ISSUE_WIDTH);
  if (num_threads != VX_CFG_NUM_THREADS) {
    std::printf("Error: device threads (%u) != VX_CFG_NUM_THREADS=%u\n",
                num_threads, VX_CFG_NUM_THREADS);
    return 1;
  }
  if (cta_warps > num_warps) {
    std::printf("Error: ISSUE_WIDTH (%u) exceeds the core's %u warps, so a warpgroup "
                "cannot be resident\n", cta_warps, num_warps);
    return 1;
  }

  const uint32_t r = cta_warps * kQuantR;
  const uint32_t c = args.c ? args.c : std::max(kQuantC, (num_threads + kQuantC - 1) / kQuantC * kQuantC);
  if (args.r && args.r != r) {
    std::printf("Error: block_size_r must be %u (%u warps x xtileM=%u)\n", r, cta_warps, kQuantR);
    return 1;
  }
  if (N % r || N % c || c % kQuantC || D % kQuantC || (args.t && (args.t % kQuantC || D % args.t))) {
    std::printf("Error: seq_len must be a multiple of block_size_r=%u and block_size_c=%u; "
                "block_size_c, head_dim and head_dim_tile must be multiples of %u, and "
                "head_dim_tile must divide head_dim\n", r, c, kQuantC);
    return 1;
  }
  // the largest head_dim_tile that divides head_dim and fits LMEM
  uint32_t t = 0;
  for (uint32_t cand = args.t ? args.t : D; cand >= kQuantC; cand -= kQuantC) {
    if (D % cand == 0 && lmem_bytes(r, c, cand, D) <= lmem_size) {
      t = cand;
      break;
    }
    if (args.t) {
      break;
    }
  }
  if (t == 0) {
    std::printf("Error: no head_dim_tile fits %llu bytes of LMEM\n", (unsigned long long)lmem_size);
    return 1;
  }
  const uint64_t lmem = lmem_bytes(r, c, t, D);
  std::printf("block_size_r=%u (%u warps) block_size_c=%u head_dim_tile=%u lmem=%llu\n",
              r, cta_warps, c, t, (unsigned long long)lmem);

  // Inputs are ITYPE values; the reference sees the same values in fp32.
  std::srand(50);
  auto Q = flash_attn::random_matrix(N, D);
  auto K = flash_attn::random_matrix(N, D);
  auto V = flash_attn::random_matrix(N, D);
  std::vector<input_t> hQ(Q.size()), hK(K.size()), hV(V.size());
  for (size_t i = 0; i < Q.size(); ++i) {
    hQ[i] = pack(Q[i]); Q[i] = unpack(hQ[i]);
    hK[i] = pack(K[i]); K[i] = unpack(hK[i]);
    hV[i] = pack(V[i]); V[i] = unpack(hV[i]);
  }
  std::vector<float> O(Q.size());

  kernel_arg_t kernel_arg{};
  kernel_arg.seq_len       = N;
  kernel_arg.head_dim      = D;
  kernel_arg.head_dim_tile = t;
  kernel_arg.block_size_r  = r;
  kernel_arg.block_size_c  = c;
  kernel_arg.Q_addr = session.upload(hQ.data(), hQ.size() * sizeof(input_t));
  kernel_arg.K_addr = session.upload(hK.data(), hK.size() * sizeof(input_t));
  kernel_arg.V_addr = session.upload(hV.data(), hV.size() * sizeof(input_t));
  kernel_arg.O_addr = session.output(O.size() * sizeof(float));

  session.run(args.kernel_file, &kernel_arg, sizeof(kernel_arg),
              N / r, num_threads, cta_warps, lmem, O.data());

  // The device packs P to ITYPE before the second GEMM; the reference does too.
  auto ref = flash_attn::reference(Q, K, V, N, D, [](float p) { return unpack(pack(p)); });
  return flash_attn::report(flash_attn::check_output(ref, O, 1e-3f, 1e-3f));
}
