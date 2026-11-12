// Copyright © 2019-2026
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

#include <cstdint>

namespace vortex {

// IPOLY index hashing (Rau, "Pseudo-randomly interleaved memory", ISCA 1991),
// with the GF(2) equations GPGPU-Sim uses for its memory partitions and DRAM
// banks. Each returns the mask f(hi) that is XORed into the index; f depends
// only on the higher bits, so the same XOR restores the original index.

inline uint32_t ipoly_bit(uint64_t v, int i) { return uint32_t(v >> i) & 1u; }

// IPOLY(37), 32 sets.
inline uint32_t ipoly32(uint64_t a) {
  uint32_t i0 = ipoly_bit(a,13)^ipoly_bit(a,12)^ipoly_bit(a,11)^ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,6)^ipoly_bit(a,5)^ipoly_bit(a,3)^ipoly_bit(a,0);
  uint32_t i1 = ipoly_bit(a,14)^ipoly_bit(a,13)^ipoly_bit(a,12)^ipoly_bit(a,11)^ipoly_bit(a,10)^ipoly_bit(a,7)^ipoly_bit(a,6)^ipoly_bit(a,4)^ipoly_bit(a,1);
  uint32_t i2 = ipoly_bit(a,14)^ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,7)^ipoly_bit(a,6)^ipoly_bit(a,3)^ipoly_bit(a,2)^ipoly_bit(a,0);
  uint32_t i3 = ipoly_bit(a,11)^ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,7)^ipoly_bit(a,4)^ipoly_bit(a,3)^ipoly_bit(a,1);
  uint32_t i4 = ipoly_bit(a,12)^ipoly_bit(a,11)^ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,5)^ipoly_bit(a,4)^ipoly_bit(a,2);
  return i0 | (i1 << 1) | (i2 << 2) | (i3 << 3) | (i4 << 4);
}

// IPOLY(5), 16 sets.
inline uint32_t ipoly16(uint64_t a) {
  uint32_t i0 = ipoly_bit(a,11)^ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,6)^ipoly_bit(a,4)^ipoly_bit(a,3)^ipoly_bit(a,0);
  uint32_t i1 = ipoly_bit(a,12)^ipoly_bit(a,8)^ipoly_bit(a,7)^ipoly_bit(a,6)^ipoly_bit(a,5)^ipoly_bit(a,3)^ipoly_bit(a,1)^ipoly_bit(a,0);
  uint32_t i2 = ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,7)^ipoly_bit(a,6)^ipoly_bit(a,4)^ipoly_bit(a,2)^ipoly_bit(a,1);
  uint32_t i3 = ipoly_bit(a,10)^ipoly_bit(a,9)^ipoly_bit(a,8)^ipoly_bit(a,7)^ipoly_bit(a,5)^ipoly_bit(a,3)^ipoly_bit(a,2);
  return i0 | (i1 << 1) | (i2 << 2) | (i3 << 3);
}

}
