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
#include <cstdlib>
#include "tensor_cfg.h"
#include "VX_config.h"

// The register layout of a warp's matrix fragments: the matrix loads and
// stores (vx_ldm/vx_stm) move fragments in it and the tensor core reads its
// operands from it.
namespace vortex {
namespace frag {

using cfg = vortex::tensor::wmma_config_t<VX_CFG_NUM_THREADS>;

constexpr uint32_t clog2(uint32_t x) {
  return (x < 2) ? 0 : (1 + clog2(x / 2));
}

constexpr uint32_t kLgM   = clog2(cfg::m_steps);
constexpr uint32_t kLgN   = clog2(cfg::n_steps);
constexpr uint32_t kLgK   = clog2(cfg::k_steps);
constexpr uint32_t kLgBSb = clog2(cfg::b_sub_blocks);

constexpr uint32_t hi_bit(uint32_t x, uint32_t lg) {
  return (x >> (lg ? (lg - 1) : 0)) & 1;
}

// Register offsets of the operands of MMA step (m, n, k), permuted so the
// three operands of a step never share a register bank.
//   one B sub-block:   A={m[0],~m[hi],k}, B={n^k,~k}, C={m[0],~m[hi],XNOR(m[hi],n)}
//   two B sub-blocks:  A={k[0],~m,m^k[hi]}, B={k[0],k[hi]^np,~np}, C={n[0],~m,n[hi]}
//   asymmetric sparse: B={n[hi],~(n[0]^k),~k}, C={n[hi],m,~(m^n[0])}
inline uint32_t ra_off(uint32_t m, uint32_t k) {
  if (kLgBSb == 0) {
    return ((m & 1) << 2) | ((1 - hi_bit(m, kLgM)) << 1) | (k & 1);
  }
  return ((k & 1) << 2) | ((1 - (m & 1)) << 1) | ((m & 1) ^ hi_bit(k, kLgK));
}

inline uint32_t rb_off(uint32_t n, uint32_t k) {
  if (kLgBSb == 0) {
    return (((n ^ k) & 1) << 1) | (1 - (k & 1));
  }
  uint32_t np = hi_bit(n, kLgN);
  return ((k & 1) << 2) | ((hi_bit(k, kLgK) ^ np) << 1) | (1 - np);
}

inline uint32_t rc_off(uint32_t m, uint32_t n) {
  if (kLgBSb == 0) {
    uint32_t m_hi = hi_bit(m, kLgM);
    return ((m & 1) << 2) | ((1 - m_hi) << 1) | (1 - (m_hi ^ (n & 1)));
  }
  return ((n & 1) << 2) | ((1 - (m & 1)) << 1) | hi_bit(n, kLgN);
}

inline uint32_t rb_off_sp(uint32_t n, uint32_t k) {
  return (hi_bit(n, kLgN) << 2) | ((1 - ((n ^ k) & 1)) << 1) | (1 - (k & 1));
}

inline uint32_t rc_off_sp(uint32_t m, uint32_t n) {
  return (hi_bit(n, kLgN) << 2) | ((m & 1) << 1) | (1 - ((m ^ n) & 1));
}

// vx_ldm/vx_stm shapes, {ctx, role[1:0], trans, sparse}
constexpr uint32_t kLdmA   = 0x00; // A, row-major
constexpr uint32_t kLdmASp = 0x01; // sparse A, row-major
constexpr uint32_t kLdmB   = 0x06; // B, col-major
constexpr uint32_t kLdmBSp = 0x07; // symmetric sparse B, col-major
constexpr uint32_t kLdmC   = 0x08; // C, row-major
constexpr uint32_t kLdmWgC = 0x18; // warp-group C, row-major, one 8-register chunk

// Byte strides between the words a lane holds along a fragment row
constexpr uint32_t kLdmLgCsK = clog2(cfg::tcK * 4);
constexpr uint32_t kLdmLgCsN = clog2(cfg::tcN * 4);
// Row groups of a col-major B fragment
constexpr uint32_t kLdmLgBRows = kLgN - kLgBSb;
constexpr uint32_t kLdmLgKSp   = (kLgK > 0) ? (kLgK - 1) : 0;

// Uop e of a fragment access moves register fd/fs2 + reg_off to or from
// x[rs1 | grp] + (offset | col).
struct LdmUop {
  uint32_t reg_off;
  uint32_t grp;
  uint32_t col;
};

inline bool ldm_valid(uint32_t shape) {
  switch (shape) {
  case kLdmA:
  case kLdmASp:
  case kLdmB:
  case kLdmC:
  case kLdmWgC:
    return true;
  case kLdmBSp:
    return cfg::sym_sparse;
  default:
    return false;
  }
}

inline uint32_t ldm_count(uint32_t shape) {
  switch (shape) {
  case kLdmA:   return 1u << (kLgM + kLgK);
  case kLdmASp: return 1u << (kLgM + kLdmLgKSp);
  case kLdmB:
  case kLdmBSp: return cfg::NRB;
  default:      return 8;
  }
}

// Element e of a fragment: its register offset, row group and column
inline LdmUop ldm_elem(uint32_t shape, uint32_t e) {
  LdmUop u{0, 0, 0};
  switch (shape) {
  case kLdmA: // e = {m, k}: row group m, word column k
    u.grp     = e >> kLgK;
    u.col     = (e & ((1u << kLgK) - 1)) << kLdmLgCsK;
    u.reg_off = ra_off(e >> kLgK, e);
    break;
  case kLdmASp: // e = {m, k} over half the K steps; registers in order
    u.grp     = e >> kLdmLgKSp;
    u.col     = (e & ((1u << kLdmLgKSp) - 1)) << kLdmLgCsK;
    u.reg_off = e;
    break;
  case kLdmB: // e = {k, n}: a col-major B row is one of its columns (a pair with two sub-blocks)
    u.grp     = e & ((1u << kLdmLgBRows) - 1);
    u.col     = (e >> kLdmLgBRows) << kLdmLgCsK;
    u.reg_off = rb_off(u.grp << kLgBSb, e >> kLdmLgBRows);
    break;
  case kLdmBSp: // symmetric sparse: one row per register
    u.grp     = e;
    u.reg_off = e;
    break;
  case kLdmC: // e = {m, n}
    u.grp     = e >> kLgN;
    u.col     = (e & ((1u << kLgN) - 1)) << kLdmLgCsN;
    u.reg_off = rc_off(e >> kLgN, e);
    break;
  default: // warp-group C, e = {n, m}: n-major registers, two row groups
    u.grp     = e & 1;
    u.col     = (e >> 1) << kLdmLgCsN;
    u.reg_off = e;
    break;
  }
  return u;
}

// Uop e moves register fd/fs2 + e, holding the element whose register offset is e
inline LdmUop ldm_uop(uint32_t shape, uint32_t e) {
  uint32_t count = ldm_count(shape);
  for (uint32_t r = 0; r < count; ++r) {
    auto u = ldm_elem(shape, r);
    if (u.reg_off == e) {
      return u;
    }
  }
  std::abort();
}

} // namespace frag
} // namespace vortex
