// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// FlashAttention-2 forward pass on SIMT cores: one warp per Q row, K and V
// staged through LMEM in head_dim tiles.

#include "flash_attn.h"
#include "flash_attn_args.h"

namespace {

struct Tiles {
  uint32_t r = 0, c = 0, t = 0; // block_size_r, block_size_c, head_dim_tile
  uint64_t lmem = 0;
  uint32_t occupancy = 0;
};

// LMEM per CTA: Q and O rows, the S/P rows, and one K/V staging tile.
uint64_t lmem_bytes(uint32_t r, uint32_t c, uint32_t t, uint32_t D) {
  return sizeof(float) * (2ull * D * r + uint64_t(r) * c + 2ull * c * t);
}

// block_size_r starts at half the warps and halves until the tiles fit LMEM.
// For each, head_dim_tile is the largest that still leaves room for as many
// CTAs per core as there are warps for them.
Tiles choose_tiles(const flash_attn::Args& args, uint32_t num_threads, uint32_t num_warps,
                   uint64_t lmem_size) {
  const uint32_t N = args.seq_len, D = args.head_dim;
  const uint32_t r_start = args.r ? args.r : std::max(1u, std::min(num_warps / 2, N));
  for (uint32_t r = r_start; r >= 1; r /= 2) {
    // each lane takes c / num_threads of the block's columns
    uint32_t c = args.c ? args.c : std::max(r, num_threads);
    if (c % num_threads == 0) {
      if (args.t) {
        uint32_t t = std::min(args.t, D);
        uint64_t lmem = lmem_bytes(r, c, t, D);
        if (lmem <= lmem_size) {
          uint32_t occ = std::min<uint64_t>(lmem_size / lmem, num_warps / r);
          return {r, c, t, lmem, std::max(1u, occ)};
        }
      } else {
        for (uint32_t ctas = std::max(1u, num_warps / r); ctas >= 1; --ctas) {
          uint64_t budget = lmem_size / ctas;
          uint64_t fixed = lmem_bytes(r, c, 0, D);
          if (fixed < budget) {
            uint64_t t = std::min<uint64_t>(D, (budget - fixed) / (2ull * c * sizeof(float)));
            if (t >= 1) {
              return {r, c, uint32_t(t), lmem_bytes(r, c, uint32_t(t), D), ctas};
            }
          }
        }
      }
    }
    if (args.r) {
      break;
    }
  }
  return {};
}

} // namespace

int main(int argc, char* argv[]) {
  flash_attn::Args args;
  flash_attn::parse_args(argc, argv, args);
  const uint32_t N = args.seq_len, D = args.head_dim;
  std::printf("flash_attn: seq_len=%u head_dim=%u\n", N, D);

  flash_attn::Session session;
  uint32_t num_threads = session.caps(VX_CAPS_NUM_THREADS);
  uint32_t num_warps   = session.caps(VX_CAPS_NUM_WARPS);
  uint64_t lmem_size   = session.caps(VX_CAPS_LOCAL_MEM_SIZE);

  if (args.r > num_warps) {
    std::printf("Error: block_size_r %u exceeds NUM_WARPS %u\n", args.r, num_warps);
    return 1;
  }
  auto tiles = choose_tiles(args, num_threads, num_warps, lmem_size);
  if (tiles.t == 0) {
    std::printf("Error: no tiling fits %llu bytes of LMEM\n", (unsigned long long)lmem_size);
    return 1;
  }
  std::printf("num_threads=%u num_warps=%u lmem_size=%llu\n", num_threads, num_warps,
              (unsigned long long)lmem_size);
  std::printf("block_size_r=%u block_size_c=%u head_dim_tile=%u lmem=%llu occupancy=%u\n",
              tiles.r, tiles.c, tiles.t, (unsigned long long)tiles.lmem, tiles.occupancy);

  std::srand(50);
  auto Q = flash_attn::random_matrix(N, D);
  auto K = flash_attn::random_matrix(N, D);
  auto V = flash_attn::random_matrix(N, D);
  std::vector<float> O(size_t(N) * D);
  const size_t bytes = O.size() * sizeof(float);

  kernel_arg_t kernel_arg{};
  kernel_arg.seq_len       = N;
  kernel_arg.head_dim      = D;
  kernel_arg.head_dim_tile = tiles.t;
  kernel_arg.block_size_r  = tiles.r;
  kernel_arg.block_size_c  = tiles.c;
  kernel_arg.Q_addr = session.upload(Q.data(), bytes);
  kernel_arg.K_addr = session.upload(K.data(), bytes);
  kernel_arg.V_addr = session.upload(V.data(), bytes);
  kernel_arg.O_addr = session.output(bytes);

  session.run(args.kernel_file, &kernel_arg, sizeof(kernel_arg),
              (N + tiles.r - 1) / tiles.r, num_threads, tiles.r, tiles.lmem, O.data());

  // fp32 on the device against a double-precision reference
  auto ref = flash_attn::reference(Q, K, V, N, D);
  return flash_attn::report(flash_attn::check_output(ref, O, 1e-3f, 1e-3f));
}
