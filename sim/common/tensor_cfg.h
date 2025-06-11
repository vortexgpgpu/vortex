// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <type_traits>
#include <algorithm>
#include <cassert>
#include <array>

namespace vortex {
namespace tensor {



struct fp32 {
  using dtype = uint32_t;
  static constexpr uint32_t bits = 32;
  static constexpr uint32_t id = 0;
};

struct fp16 {
  using dtype = uint16_t;
  static constexpr uint32_t bits = 16;
  static constexpr uint32_t id = 1;
};
struct bf16 {
  using dtype = uint16_t;
  static constexpr uint32_t bits = 16;
  static constexpr uint32_t id = 2;
};

struct int32 {
  using dtype = uint32_t;
  static constexpr uint32_t bits = 32;
  static constexpr uint32_t id = 3;
};

struct int16 {
  using dtype = uint16_t;
  static constexpr uint32_t bits = 16;
  static constexpr uint32_t id = 4;
};

struct int8 {
  using dtype = uint8_t;
  static constexpr uint32_t bits = 8;
  static constexpr uint32_t id = 5;
};

template <uint32_t NT,      // number of threads per warp
          typename It = fp32, // input type (A,B)
          typename Ot = fp32, // output type (C,D)
          uint32_t XB = 4,  // vector element type size in bytes
          uint32_t NR = 8,  // registers per fragment
          uint32_t DP = 0   // Dot-Product Length (0 for auto)
          >
struct wmma_config_t {
private:
  static constexpr uint32_t clog2(uint32_t x) {
    return (x < 2) ? 0 : (1 + clog2(x / 2));
  }
  static constexpr uint32_t tile_cap = NT * NR;
  static constexpr uint32_t lg_tile_cap = clog2(tile_cap);
  static constexpr uint32_t tile_en = lg_tile_cap / 2;
  static constexpr uint32_t tile_em = lg_tile_cap - tile_en;

  static constexpr uint32_t block_cap = NT;
  static constexpr uint32_t lg_block_cap = clog2(block_cap);
  static constexpr uint32_t block_en = lg_block_cap / 2;
  static constexpr uint32_t block_em = lg_block_cap - block_en;

public:

  static constexpr uint32_t i_ratio = XB / sizeof(typename It::dtype);
  static constexpr uint32_t o_ratio = XB / sizeof(typename Ot::dtype);
  static_assert(i_ratio * sizeof(typename It::dtype) == XB, "XB must be multiple of sizeof(It)");
  static_assert(o_ratio * sizeof(typename Ot::dtype) == XB, "XB must be multiple of sizeof(Ot)");

  static constexpr uint32_t xtileM = 1u << tile_em;
  static constexpr uint32_t xtileN = 1u << tile_en;
  static constexpr uint32_t xtileK = tile_cap / ((xtileM > xtileN) ? xtileM : xtileN);

  static constexpr uint32_t tcM = 1u << block_em;
  static constexpr uint32_t tcN = 1u << block_en;
  static constexpr uint32_t tcK = (DP != 0) ? DP : (block_cap / ((tcM > tcN) ? tcM : tcN));

  static constexpr uint32_t m_steps = xtileM / tcM;  // number of M steps per register
  static constexpr uint32_t n_steps = xtileN / tcN;  // number of N steps per register
  static constexpr uint32_t k_steps = xtileK / tcK;  // number of K steps per register

  static constexpr uint32_t a_block_size = tcM * tcK;                 // size of A micro-tile
  static constexpr uint32_t a_sub_blocks = block_cap / a_block_size;  // number of A micro-tiles per register
  static constexpr uint32_t a_sub_steps  = m_steps / a_sub_blocks;    // number of A sub-steps per register

  static constexpr uint32_t b_block_size = tcK * tcN;                 // size of B micro-tile
  static constexpr uint32_t b_sub_blocks = block_cap / b_block_size;  // number of B micro-tiles per register
  static constexpr uint32_t b_sub_steps  = n_steps / b_sub_blocks;    // number of B sub-steps per register

  static constexpr uint32_t NRA = (xtileM * xtileK) / NT; // Number of A registers
  static constexpr uint32_t NRB = (xtileN * xtileK) / NT; // Number of B registers
  static constexpr uint32_t NRC = (xtileM * xtileN) / NT; // Number of C registers

  static_assert(a_sub_steps != 0, "tcK is too small for tile A");
  static_assert(b_sub_steps != 0, "tcK is too small for tile B");

  static_assert((xtileM * xtileK <= tile_cap), "xtileM * xtileK <= tile_cap");
  static_assert((xtileN * xtileK <= tile_cap), "xtileN * xtileK <= tile_cap");
  static_assert((xtileM * xtileN <= tile_cap), "xtileM * xtileN <= tile_cap");

  static_assert((tcM * tcK <= block_cap), "tcM * tcK <= block_cap");
  static_assert((tcN * tcK <= block_cap), "tcN * tcK <= block_cap");
  static_assert((tcM * tcN <= block_cap), "tcM * tcN <= block_cap");

  static_assert((xtileM % tcM) == 0, "M,m divisibility");
  static_assert((xtileN % tcN) == 0, "N,n divisibility");
  static_assert((xtileK % tcK) == 0, "K,k divisibility");

  static constexpr uint32_t tileM = xtileM;
  static constexpr uint32_t tileN = xtileN;
  static constexpr uint32_t tileK = xtileK * i_ratio; // Adjusted for input type size
};

} // namespace tensor
} // namespace vortex