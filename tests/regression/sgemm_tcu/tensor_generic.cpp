#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
//#include "float16.h"

#ifndef USE_SPARSE_A
/* 0 = dense A
 * 1 = sparse A             */
#define USE_SPARSE_A 1

struct int4_t {
  uint8_t data;
};

using float32_t = float;

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

#ifndef XLENB
#define XLENB 2
#endif

#ifndef ITYPE
#define ITYPE int16_t
#endif

#ifndef OTYPE
#define OTYPE int16_t
#endif

#ifndef DPLEN
#define DPLEN 0
#endif

#ifdef NDEBUG
#define DBG_PRINT(fmt, ...)
#else
#define DBG_PRINT(fmt, ...)            \
  do {                                 \
    fprintf(stderr, fmt, __VA_ARGS__); \
  } while (0)
#endif

#ifdef NDEBUG
class NullStream {
public:
  template <typename T> NullStream &operator<<(const T &) { return *this; }
  NullStream &operator<<(std::ostream &(*)(std::ostream &)) { return *this; }
  void flush() {}
  static NullStream &instance() {
    static NullStream null_stream;
    return null_stream;
  }
};
#define dbg_out NullStream::instance()
#else
#define dbg_out std::cout
#endif

template <uint32_t>
struct DebugPrint;

template <uint32_t NT,    // number of threads per warp
          uint32_t NR,    // registers per fragment
          uint32_t XB,    // vector element type size in bytes
          typename Ot,    // output type (C,D)
          typename It,    // input type (A,B)
          uint32_t DP = 0 // Dot-Product Length (0 for auto)
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
  static_assert(XB >= 0 && XB <= 8, "invalid XB value!");

  static constexpr uint32_t i_ratio = XB / sizeof(It);
  static constexpr uint32_t o_ratio = XB / sizeof(Ot);
  static_assert(i_ratio * sizeof(It) == XB, "XB must be multiple of sizeof(It)");
  static_assert(o_ratio * sizeof(Ot) == XB, "XB must be multiple of sizeof(Ot)");

  static constexpr uint32_t NumThreads = NT;
  static constexpr uint32_t NumRegs    = NR;

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

  static constexpr uint32_t tileM = xtileM;
  static constexpr uint32_t tileN = xtileN;
  static constexpr uint32_t tileK = xtileK * i_ratio; // Adjusted for input type size

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

  using vector_t = std::conditional_t<(XB == 1), uint8_t,
                    std::conditional_t<(XB == 2), uint16_t,
                      std::conditional_t<(XB == 4), uint32_t, uint64_t>>>;
  using input_t  = It;
  using output_t = Ot;
};

template <typename T>
struct raw_unsigned {
  static_assert(
    sizeof(T) == 1 || sizeof(T) == 2 ||
    sizeof(T) == 4 || sizeof(T) == 8,
    "raw_unsigned_t<T> only supports types of size 1, 2, 4 or 8 bytes"
  );

  using type = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<
      sizeof(T) == 2, uint16_t,
      std::conditional_t<
        sizeof(T) == 4, uint32_t,
        uint64_t  // sizeof(T) == 8
      >
    >
  >;
};

template <typename T>
using raw_unsigned_t = typename raw_unsigned<T>::type;

template <typename D, typename S>
D pack_row(const S *base, uint32_t ldm) {
  static_assert(sizeof(D) % sizeof(S) == 0, "D must be a multiple of S");
  constexpr uint32_t count = sizeof(D) / sizeof(S);
  using US = raw_unsigned_t<S>;
  D packed(0);
  auto src = base;
  for (uint32_t i = 0; i < count; ++i) {
    US bits;
    bits = *reinterpret_cast<const US *>(src);
    D elem = static_cast<D>(bits); // zero-extend S to D
    packed |= (elem << (i * (8u * sizeof(S))));
    src += ldm; // move to the next row
  }
  return packed;
}

template <typename T, uint32_t N>
struct vector_t {
private:
  std::array<T, N> data_;

public:
  vector_t() = default;

  vector_t(T value) {
    data_.fill(value);
  }

  T* data() {
    return data_.data();
  }

  const T* data() const {
    return data_.data();
  }

  T& operator[](size_t idx) {
    assert(idx < N);
    return data_[idx];
  }

  const T& operator[](size_t idx) const {
    assert(idx < N);
    return data_[idx];
  }

  friend std::ostream &operator<<(std::ostream &os, const vector_t &v) {
    os << std::hex << "{";
    for (size_t i = 0; i < N; ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << "0x" << +v.data_[i];
    }
    os << "}" << std::dec;
    return os;
  }
};

template <typename T, uint32_t R, uint32_t C>
struct array2d_t {
private:
  std::array<T, R * C> data_;

public:
  T* data() {
    return data_.data();
  }

  const T* data() const {
    return data_.data();
  }

  T &operator()(int row, int col) {
    assert(row >= 0 && row < R);
    assert(col >= 0 && col < C);
    return data_[row * C + col];
  }

  const T &operator()(int row, int col) const {
    assert(row >= 0 && row < R);
    assert(col >= 0 && col < C);
    return data_[row * C + col];
  }

  friend std::ostream &operator<<(std::ostream &os, const array2d_t &v) {
    os << "{";
    for (size_t j = 0; j < R; ++j) {
      if (j != 0) {
        os << ", ";
      }
      os << "{";
      for (size_t i = 0; i < C; ++i) {
        if (i != 0) {
          os << ", ";
        }
        os << +v(j,i);
      }
      os << "}";
    }
    os << "}";
    return os;
  }
};

template <typename Itype>
struct SparseMat {
  std::vector<Itype> values;   // non-zeros
  std::vector<uint8_t> meta;     // Array of row-masks: 1 byte marks the columns
                                  // of the 4 elements in the block that are non-zero.
                                  // e.g. 0b0101 means 2nd and 4th elements are non-zero.

  uint32_t rows, cols;           // original A dims (M × K)
};

template <typename Itype>
SparseMat<Itype>
dense_to_sparse(const std::vector<Itype>&, uint32_t, uint32_t);

template <typename Config>
class WMMA {
private:
  static constexpr uint32_t tileM = Config::tileM;
  static constexpr uint32_t tileN = Config::tileN;
  static constexpr uint32_t tileK = Config::tileK;

  static constexpr uint32_t tcM = Config::tcM;
  static constexpr uint32_t tcN = Config::tcN;
  static constexpr uint32_t tcK = Config::tcK;

  static constexpr uint32_t NT = Config::NumThreads;
  static constexpr uint32_t NRA = Config::NRA;
  static constexpr uint32_t NRB = Config::NRB;
  static constexpr uint32_t NRC = Config::NRC;

  static constexpr uint32_t m_steps = Config::m_steps;
  static constexpr uint32_t n_steps = Config::n_steps;
  static constexpr uint32_t k_steps = Config::k_steps;

  static constexpr uint32_t a_block_size = Config::a_block_size;
  static constexpr uint32_t a_sub_blocks = Config::a_sub_blocks;
  static constexpr uint32_t a_sub_steps  = Config::a_sub_steps;

  static constexpr uint32_t b_block_size = Config::b_block_size;
  static constexpr uint32_t b_sub_blocks = Config::b_sub_blocks;
  static constexpr uint32_t b_sub_steps  = Config::b_sub_steps;

  static constexpr uint32_t i_ratio = Config::i_ratio;
  static constexpr uint32_t o_ratio = Config::o_ratio;

  using Xt = typename Config::vector_t;
  using It = typename Config::input_t;
  using Ot = typename Config::output_t;

  using Vreg = vector_t<Xt, NT>; // Vector register type

  using FragA = array2d_t<It, tileM, tileK>; // A: M rows × K cols
  using FragB = array2d_t<It, tileK, tileN>; // B: K rows × N cols
  using FragC = array2d_t<Ot, tileM, tileN>; // C: M rows × N cols
  using FragD = array2d_t<Ot, tileM, tileN>; // D: M rows × N cols

  FragA fragA_;
  FragB fragB_;
  FragC fragC_;
  FragD fragD_;

  FragD fragRef_;

  void load_A(vector_t<Vreg, NRA> &vR, uint32_t lane, uint32_t ldm, const It *mdata) {
    uint32_t block_idx = lane / a_block_size;
    uint32_t lane_in_block = lane % a_block_size;
    uint32_t elem_row = lane_in_block / tcK;
    uint32_t elem_col = lane_in_block % tcK;
    DBG_PRINT("[load_A] lane=%u block_idx=%u lane_in_block=%u elem=[%u,%u], src=%p-%p\n", lane, block_idx, lane_in_block, elem_row, elem_col, mdata, mdata + tileM * tileK);

    for (uint32_t r = 0; r < NRA; ++r) {
      uint32_t block_m = (r / k_steps) * a_sub_blocks + block_idx;
      uint32_t block_k = r % k_steps;
      uint32_t row = block_m * tcM + elem_row;
      uint32_t col = block_k * tcK + elem_col;
      auto base = mdata + row * ldm + col * i_ratio;

      assert(reinterpret_cast<uintptr_t>(base) % alignof(Xt) == 0 && "Base pointer must be aligned to sizeof(Xt)");
      vR[r][lane] = *reinterpret_cast<const Xt *>(base);
      DBG_PRINT("  r=%u → block_m=%u block_k=%u → loads A[%u,%u] → %p → %u\n", r, block_m, block_k, row, col, base, vR[r][lane]);
    }
  }
template <typename Itype>
void load_A_sparse(vector_t<Vreg, NRA>& valR,
                   vector_t<Vreg, NRA>& maskR,
                   uint32_t lane,
                   uint32_t ldm,    // == tileK
                   const SparseMat<Itype>& SA)
{
  // 1) Compute this lane’s position within the tcM×tcK micro‑tile
  uint32_t block_idx     = lane / a_block_size;
  uint32_t lane_in_block = lane % a_block_size;
  uint32_t elem_row      = lane_in_block / tcK;
  uint32_t elem_col      = lane_in_block % tcK;

  // 2) For each A‑fragment register r
  for (uint32_t r = 0; r < NRA; ++r) {
    // 2.1) Global row & col indices
    uint32_t block_m  = (r / k_steps) * a_sub_blocks + block_idx;
    uint32_t block_k  = r % k_steps;
    uint32_t col_reg  = block_k * tcK + elem_col;
    uint32_t col_phys = col_reg * i_ratio;      // now == col_reg when i_ratio==1
    uint32_t row      = block_m * tcM + elem_row;

    // 2.2) Pointers into sparse metadata and values
    const uint8_t* meta = SA.meta.data()   + row * (SA.cols / 4);
    const Itype*   vals = SA.values.data() + row * (SA.cols / 2);

    // 2.3) Load the 4‑bit block mask
    uint32_t block_id    = col_phys >> 2;
    uint32_t bit_pos     = col_phys & 3;
    uint8_t  block_mask  = meta[block_id];
    maskR[r][lane]       = Xt(block_mask);

    // 2.4) If that bit is set, find and load the corresponding nonzero
    Itype v = 0;
    if ((block_mask >> bit_pos) & 1u) {
      uint32_t prefix = 0;
      for (uint32_t b = 0; b < block_id; ++b)
        prefix += __builtin_popcount(meta[b]);
      prefix += __builtin_popcount(block_mask & ((1u << bit_pos) - 1u));
      v = vals[prefix];
    }

    // 2.5) Store the element directly into the Xt register
    Xt packed;
    std::memcpy(&packed, &v, sizeof(packed));
    valR[r][lane] = packed;
  }
}


  void load_B(vector_t<Vreg, NRB> &vR, uint32_t lane, uint32_t ldm, const It *mdata) {
    uint32_t block_idx = lane / b_block_size;
    uint32_t lane_in_block = lane % b_block_size;
    uint32_t elem_col = lane_in_block / tcK;
    uint32_t elem_row = lane_in_block % tcK;
    DBG_PRINT("[load_B] lane=%u block_idx=%u lane_in_block=%u elem=[%u,%u], src=%p-%p\n", lane, block_idx, lane_in_block, elem_row, elem_col, mdata, mdata + tileK * tileN);

    for (uint32_t r = 0; r < NRB; ++r) {
      uint32_t block_k = r / b_sub_steps;
      uint32_t block_n = (r % b_sub_steps) * b_sub_blocks + block_idx;
      uint32_t row = block_k * tcK + elem_row;
      uint32_t col = block_n * tcN + elem_col;
      auto base = mdata + row * ldm  * i_ratio + col;

      if constexpr (sizeof(Xt) == sizeof(It)) {
        vR[r][lane] = *reinterpret_cast<const Xt *>(base);
      } else {
        vR[r][lane] = pack_row<Xt>(base, ldm);
      }
      DBG_PRINT("  r=%u → block_k=%u block_n=%u → loads B[%u,%u] → %p → %u\n", r, block_k, block_n, row, col, base, vR[r][lane]);
    }
  }

  void load_C(vector_t<Vreg, NRC> &vR, uint32_t lane, uint32_t ldm, const Ot *mdata) {
    uint32_t elem_row = lane / tcN;
    uint32_t elem_col = lane % tcN;
    DBG_PRINT("[load_C] lane=%u elem=[%u,%u], src=%p-%p\n", lane, elem_row, elem_col, mdata, mdata + tileM * tileN);

    for (uint32_t r = 0; r < NRC; ++r) {
      uint32_t block_m = r / n_steps;
      uint32_t block_n = r % n_steps;
      uint32_t row = block_m * tcM + elem_row;
      uint32_t col = block_n * tcN + elem_col;
      auto base = mdata + row * ldm + col;

      if constexpr (sizeof(Xt) == sizeof(Ot)) {
        vR[r][lane] = *reinterpret_cast<const Xt *>(base);
      } else {
        Xt tmp(0);
        *reinterpret_cast<Ot*>(&tmp) = *base;
        vR[r][lane] = tmp;
      }
      DBG_PRINT("  r=%u → block_m=%u block_n=%u → loads C[%u,%u] → %p → %u\n", r, block_m, block_n, row, col, base, vR[r][lane]);
    }
  }

  void store_D(Ot *mdata, uint32_t lane, uint32_t ldm, const vector_t<Vreg, NRC> &vR) {
    uint32_t elem_row = lane / tcN;
    uint32_t elem_col = lane % tcN;

    DBG_PRINT("[store_D] lane=%u elem=[%u,%u], dst=%p-%p\n", lane, elem_row, elem_col, mdata, mdata + tileM * tileN);

    for (uint32_t r = 0; r < NRC; ++r) {
      uint32_t block_m = r / n_steps;
      uint32_t block_n = r % n_steps;
      uint32_t row = block_m * tcM + elem_row;
      uint32_t col = block_n * tcN + elem_col;
      auto base = mdata + row * ldm + col;

      if constexpr (sizeof(Xt) == sizeof(Ot)) {
        *reinterpret_cast<Xt*>(base) = vR[r][lane];
      } else {
        Xt tmp(vR[r][lane]);
        *base = *reinterpret_cast<const Ot*>(&tmp);
      }
      DBG_PRINT("  r=%u → block_m=%u block_n=%u → store C[%u,%u] → %p → %u\n", r, block_m, block_n, row, col, base , vR[r][lane]);
    }
  }

  Xt FEDP(const Xt *a_row, const Xt *b_col, Xt c_val) {
    Ot acc(*reinterpret_cast<const Ot*>(&c_val));
    auto a = reinterpret_cast<const It *>(a_row);
    auto b = reinterpret_cast<const It *>(b_col);
    for (uint32_t z = 0; z < tcK * i_ratio; ++z) {
      auto a_val = static_cast<Ot>(a[z]);
      auto b_val = static_cast<Ot>(b[z]);
      acc = a_val * b_val + acc;
    }
    Xt ret(0);
    *reinterpret_cast<Ot*>(&ret) = acc;
    return ret;
  }

  Vreg MMA(uint32_t m, uint32_t n, const Vreg &va, const Vreg &vb, const Vreg &vc) {
    uint32_t a_off = (m % a_sub_blocks) * a_block_size;
    uint32_t b_off = (n % b_sub_blocks) * b_block_size;

    Vreg vd;
    for (uint32_t i = 0; i < tcM; ++i) {
      for (uint32_t j = 0; j < tcN; ++j) {
        auto a_row = &va[a_off + i * tcK];
        auto b_col = &vb[b_off + j * tcK];
        auto c = vc[i * tcN + j];
        auto d = FEDP(a_row, b_col, c);
        vd[i * tcN + j] = d;
      }
    }

    return vd;
  }

  FragD mmadd(const FragA &A, const FragB &B, const FragC &C) {
    FragD D;
    vector_t<Vreg, NRA> vA;
    vector_t<Vreg, NRB> vB;
    vector_t<Vreg, NRC> vC, vD;

    dbg_out << "A=" << A << "\n";
    dbg_out << "B=" << B << "\n";
    dbg_out << "C=" << C << "\n";

    // per-lane load
    #if USE_SPARSE_A
    /* ---------- ❸ replace on‑the‑fly converter with a tiny helper ---- */
    auto fragA_to_vec = [&](const FragA& f){
        std::vector<It> v(tileM*tileK);
        std::memcpy(v.data(), f.data(), v.size()*sizeof(It));
        return v;
    };

    vector_t<Vreg, NRA> vA_mask;
    auto Sp = dense_to_sparse(fragA_to_vec(A), tileM, tileK);
    for (uint32_t lane = 0; lane < NT; ++lane)
        load_A_sparse(vA, vA_mask, lane, tileK, Sp);
    #else
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_A(vA, lane, tileK, A.data());
    }
    #endif
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_B(vB, lane, tileN, B.data());
    }
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_C(vC, lane, tileN, C.data());
    }

    for (uint32_t i = 0; i < NRA; ++i) {
      dbg_out << "vA" << i << "=" << vA[i] << "\n";
    }
    for (uint32_t i = 0; i < NRB; ++i) {
      dbg_out << "vB" << i << "=" << vB[i] << "\n";
    }
    for (uint32_t i = 0; i < NRC; ++i) {
      dbg_out << "vC" << i << "=" << vC[i] << "\n";
    }

    // micro-ops
    for (uint32_t k = 0; k < k_steps; ++k) {
      for (uint32_t m = 0; m < m_steps; ++m) {
        for (uint32_t n = 0; n < n_steps; ++n) {
          uint32_t idxA = (m / a_sub_blocks) * k_steps + k;
          uint32_t idxB = (k * n_steps + n) / b_sub_blocks;
          uint32_t idxC = m * n_steps + n;

          auto &va = vA[idxA];
          auto &vb = vB[idxB];
          auto &vc = (k != 0) ? vD[idxC] : vC[idxC];

          auto vd = MMA(m, n, va, vb, vc);

          dbg_out << "[mmadd] m=" << m << " n=" << n << " k=" << k
                  << " → idxA=" << idxA << " idxB=" << idxB << " idxC=" << idxC
                  << " va=" << va << " vb=" << vb << " vc=" << vc << " vd=" << vd << "\n";

          vD[idxC] = vd;
        }
      }
    }

    dbg_out.flush();

    for (uint32_t i = 0; i < NRC; ++i) {
      dbg_out << "vD" << i << "=" << vD[i] << "\n";
    }

    // per-lane store
    for (uint32_t lane = 0; lane < NT; ++lane) {
      store_D(D.data(), lane, tileN, vD);
    }

    dbg_out << "D=" << D << "\n";
    return D;
  }

public:
  void init() {

    //fragA_.init();
    //fragB_.init();
    //fragC_.init();

    int x = 0;

    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileK; ++c) {
        fragA_(r, c) = x++;
      }
    }
    for (uint32_t r = 0; r < tileK; ++r) {
      for (uint32_t c = 0; c < tileN; ++c) {
        fragB_(r, c) = x++;
      }
    }
    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileN; ++c) {
        fragC_(r, c) = 0;
      }
    }

    for (uint32_t row = 0; row < tileM; ++row) {
      for (uint32_t col = 0; col < tileN; ++col) {
        Ot sum(0);
        for (uint32_t k = 0; k < tileK; ++k) {
          auto a = static_cast<Ot>(fragA_(row, k));
          auto b = static_cast<Ot>(fragB_(k, col));
          sum = a * b + sum;
        }
        fragRef_(row, col) = sum + fragC_(row, col);
      }
    }
  }

  float verify() {
    if constexpr (std::is_integral_v<Ot>) {
      int32_t err(0);
      for (uint32_t row = 0; row < tileM; ++row) {
        for (uint32_t col = 0; col < tileN; ++col) {
          auto curr = static_cast<int32_t>(fragD_(row, col));
          auto ref  = static_cast<int32_t>(fragRef_(row, col));
          auto diff = std::abs(curr - ref);
          err = std::max<int32_t>(err, diff);
        }
      }
      return err;
    } else {
      float err(0);
      for (uint32_t row = 0; row < tileM; ++row) {
        for (uint32_t col = 0; col < tileN; ++col) {
          auto curr = static_cast<float>(fragD_(row, col));
          auto ref  = static_cast<float>(fragRef_(row, col));
          auto diff = std::fabs(curr - ref);
          err = std::max<float>(err, diff);
        }
      }
      return err;
    }
  }

  void run() {
    fragD_ = mmadd(fragA_, fragB_, fragC_);
  }
static void unit_test_load_A_sparse() {
  // 1) Dimensions from this instantiation
  constexpr uint32_t M = tileM;
  constexpr uint32_t K = tileK;

  // 2) Build a random dense M×K block
  std::vector<It> dense(M * K);
  srand(0);
  for (auto &x : dense) {
    x = It((rand() & 0x7fff) - (rand() & 0x7fff));
  }

  // 3) Compress it
  auto sp = dense_to_sparse(dense, M, K);

  // 4) Run both loaders on every lane
  WMMA w;
  vector_t<Vreg, NRA> vd{}, vs{}, ms{};
  for (uint32_t lane = 0; lane < NT; ++lane) {
    w.load_A        (vd, lane, K, dense.data());
    w.load_A_sparse(vs, ms, lane, K, sp);
  }

  // 5) Verify per‑lane, per‑register
  for (uint32_t lane = 0; lane < NT; ++lane) {
    // figure out this lane’s micro‑tile position
    uint32_t block_idx     = lane / a_block_size;
    uint32_t lane_in_block = lane % a_block_size;
    uint32_t elem_row      = lane_in_block / tcK;
    uint32_t elem_col      = lane_in_block % tcK;

    for (uint32_t r = 0; r < NRA; ++r) {
      // global row/col in the original dense matrix
      uint32_t block_m = (r / k_steps) * a_sub_blocks + block_idx;
      uint32_t block_k = r % k_steps;
      uint32_t row     = block_m * tcM + elem_row;
      uint32_t col_reg = block_k * tcK   + elem_col;
      uint32_t col     = col_reg;  // because i_ratio==1

      // what mask bit did we keep?
      uint8_t mask = sp.meta[row * (K/4) + (col >> 2)];
      bool  kept = ((mask >> (col & 3)) & 1u) != 0;

      // dense value at (row,col)
      It dense_val = dense[row * K + col];

      // sparse loader’s output in that slice
      It sparse_val;
      std::memcpy(&sparse_val, &vs[r][lane], sizeof(sparse_val));

      if (kept) {
        assert(dense_val == sparse_val
               && "Kept element must round‑trip");
      } else {
        assert(sparse_val == It(0)
               && "Dropped element must be zero");
      }
    }
  }

  std::cout << "[unit] load_A_sparse PASSED\n";
}

};

using cfg = wmma_config_t<
    NUM_THREADS,
    8,
    XLENB,
    OTYPE,
    ITYPE,
    DPLEN>;

template <typename Itype>
SparseMat<Itype> dense_to_sparse(const std::vector<Itype>& dense,
                                 uint32_t rows, uint32_t cols)
{
  SparseMat<Itype> S;
  S.rows = rows;
  S.cols = cols;
  S.values.reserve(rows * cols / 2);
  S.meta   .reserve(rows * cols / 4);

  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; c += 4) {
      // build a length‐4 window of (value, index)
      std::array<std::pair<Itype,uint32_t>,4> tmp;
      for (uint32_t i = 0; i < 4; ++i)
        tmp[i] = { dense[r*cols + c + i], i };

      // pick the two largest‐magnitude entries
      std::sort(tmp.begin(), tmp.end(),
        [](auto &a, auto &b){
          return std::abs(int(a.first)) > std::abs(int(b.first));
        });

      // mask bit = union of their column‐indices
      uint32_t idx0 = tmp[0].second, idx1 = tmp[1].second;
      uint8_t m = (1u << idx0) | (1u << idx1);
      S.meta.push_back(m);

      // **NEW**: push values in ascending‐index order so that
      // your popcount‐prefix decompressor lines up correctly
      if (idx0 < idx1) {
        S.values.push_back(tmp[0].first);
        S.values.push_back(tmp[1].first);
      } else {
        S.values.push_back(tmp[1].first);
        S.values.push_back(tmp[0].first);
      }
    }
  }
  return S;
}


int main() {
  WMMA<cfg>::unit_test_load_A_sparse();

  WMMA<cfg> wmma;

  std::cout
      << "tileM = " << cfg::tileM << "\n"
      << "tileN = " << cfg::tileN << "\n"
      << "tileK = " << cfg::tileK << "\n"
      << "tcM   = " << cfg::tcM << "\n"
      << "tcN   = " << cfg::tcN << "\n"
      << "tcK   = " << cfg::tcK << "\n"
      << "m_steps = " << cfg::m_steps << "\n"
      << "n_steps = " << cfg::n_steps << "\n"
      << "k_steps = " << cfg::k_steps << "\n"
      << "a_block_size = " << cfg::a_block_size << "\n"
      << "a_sub_blocks = " << cfg::a_sub_blocks << "\n"
      << "a_sub_steps  = " << cfg::a_sub_steps << "\n"
      << "b_block_size = " << cfg::b_block_size << "\n"
      << "b_sub_blocks = " << cfg::b_sub_blocks << "\n"
      << "b_sub_steps  = " << cfg::b_sub_steps << "\n"
      << "NRA = " << cfg::NRA << "\n"
      << "NRB = " << cfg::NRB << "\n"
      << "NRC = " << cfg::NRC << "\n"
      ;

  //wmma.init();

  //wmma.run();

  //auto err = wmma.verify();

  //bool passed = (err < 1e-4f);

  //std::cout << "Max abs error: " << err << "\n"
  //          << (passed ? "PASSED!" : "FAILED!") << '\n';

  //return passed ? 0 : 1;
  return 0;
}

#endif
// README
// gcc -std=c++17 -o tensor_generic -O2 tensor_generic.cpp -lstdc++ 