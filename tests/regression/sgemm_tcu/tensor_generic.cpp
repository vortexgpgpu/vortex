#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

#define ENABLE_SPARSITY true
// Include random header only when sparsity is enabled
#ifdef ENABLE_SPARSITY
#include <random>
#endif

struct int4_t {
  uint8_t data;
};

using float32_t = float;

// ============================================================================
// Configuration Macros
// ============================================================================
#ifndef NUM_THREADS
#define NUM_THREADS 8      // Should be 32 for paper accuracy
#endif

#ifndef XLENB
#define XLENB 4
#endif

#ifndef ITYPE
#define ITYPE int8_t
#endif

#ifndef OTYPE
#define OTYPE int32_t
#endif

#ifndef DPLEN
#define DPLEN 0
#endif

// ============================================================================
// Debug Output Macros
// ============================================================================
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

// ============================================================================
// WMMA Configuration Template
// ============================================================================
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

  static constexpr uint32_t m_steps = xtileM / tcM;
  static constexpr uint32_t n_steps = xtileN / tcN;
  static constexpr uint32_t k_steps = xtileK / tcK;

  static constexpr uint32_t a_block_size = tcM * tcK;
  static constexpr uint32_t a_sub_blocks = block_cap / a_block_size;
  static constexpr uint32_t a_sub_steps  = m_steps / a_sub_blocks;

#ifdef ENABLE_SPARSITY
  // For 2:4 sparsity, B needs to provide both potential values
  static constexpr uint32_t SPARSITY_RATIO = 2;
  static constexpr uint32_t b_block_size = tcK * tcN * SPARSITY_RATIO;
  static constexpr uint32_t b_sub_blocks = block_cap / b_block_size;
  static constexpr uint32_t b_sub_steps  = n_steps / b_sub_blocks;
#else
  // Dense mode: standard B block configuration
  static constexpr uint32_t b_block_size = tcK * tcN;
  static constexpr uint32_t b_sub_blocks = block_cap / b_block_size;
  static constexpr uint32_t b_sub_steps  = n_steps / b_sub_blocks;
#endif

  static constexpr uint32_t NRA = (xtileM * xtileK) / NT;
  static constexpr uint32_t NRB = (xtileN * xtileK) / NT;
  static constexpr uint32_t NRC = (xtileM * xtileN) / NT;

  static constexpr uint32_t tileM = xtileM;
  static constexpr uint32_t tileN = xtileN;
  static constexpr uint32_t tileK = xtileK * i_ratio;

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

// ============================================================================
// Utility Types
// ============================================================================
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
        uint64_t
      >
    >
  >;
};

template <typename T>
using raw_unsigned_t = typename raw_unsigned<T>::type;

// ============================================================================
// Pack Row Function
// ============================================================================
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
    D elem = static_cast<D>(bits);
    packed |= (elem << (i * (8u * sizeof(S))));
    src += ldm;
  }
  return packed;
}

// ============================================================================
// Vector Register Type
// ============================================================================
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

// ============================================================================
// 2D Array Type
// ============================================================================
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

// ============================================================================
// WMMA Implementation (Dense or Sparse based on ENABLE_SPARSITY)
// ============================================================================
template <typename Config>
class WMMA {
private:
  // Configuration constants
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

#ifdef ENABLE_SPARSITY
  // Sparsity-specific constants
  static constexpr uint32_t SPARSITY_N = 2;  // 2 non-zero elements
  static constexpr uint32_t SPARSITY_M = 4;  // out of 4 elements (2:4 sparsity)
  static constexpr uint32_t METADATA_LANES = 2;  // Lanes 0,1 hold metadata
  static constexpr uint32_t COMPRESSION_RATE = SPARSITY_M / SPARSITY_N; // 2x compression
#endif

  using Xt = typename Config::vector_t;
  using It = typename Config::input_t;
  using Ot = typename Config::output_t;

  using Vreg = vector_t<Xt, NT>;

  using FragA = array2d_t<It, tileM, tileK>;
  using FragB = array2d_t<It, tileK, tileN>;
  using FragC = array2d_t<Ot, tileM, tileN>;
  using FragD = array2d_t<Ot, tileM, tileN>;

  // Matrix fragments
  FragA fragA_;
  FragB fragB_;
  FragC fragC_;
  FragD fragD_;

#ifdef ENABLE_SPARSITY
  // Sparsity-specific data structures
  using FragA_meta = array2d_t<uint8_t, tileM, tileK>;

  FragA fragA_compressed_;   // Compressed matrix A (50% storage)
  FragA_meta fragA_meta_;    // Metadata: 1 = non-zero, 0 = pruned
  vector_t<uint32_t, NRA> packed_bit_meta_;  // Packed bitmap metadata, 32 * 32bit Reg in RISC-V
#endif

  FragD fragRef_;

  uint32_t loop_iteration_count_;  // Counter for total loop iterations

  // ========================================================================
  // Sparsity Helper Functions (only compiled when ENABLE_SPARSITY is defined)
  // ========================================================================
#ifdef ENABLE_SPARSITY
  // Apply 2:4 structured pruning pattern
  void apply_2_4_pruning(std::mt19937 &gen) {
    std::vector<int> masks = {1, 1, 0, 0};  // 2 ones, 2 zeros

    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileK / SPARSITY_M; ++c) {
        // Shuffle the mask for this group of 4 elements
        std::shuffle(masks.begin(), masks.end(), gen);

        // Apply mask to each element in the group
        for (uint32_t c_4 = 0; c_4 < SPARSITY_M; ++c_4) {
          uint32_t col = c * SPARSITY_M + c_4;
          if (masks[c_4] == 0) {
            fragA_(r, col) = 0;
            fragA_meta_(r, col) = 0;
          } else {
            fragA_meta_(r, col) = 1;
          }
        }
      }
    }
  }

  // Compress matrix A by removing zeros
  void compress_matrix_A() {
    // Initialize compressed matrix to zero
    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileK; ++c) {
        fragA_compressed_(r, c) = 0;
      }
    }

    // Pack non-zero elements into compressed format
    uint32_t comp_cnt = 0;
    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileK; ++c) {
        if (fragA_meta_(r, c) == 1) {
          uint32_t comp_r = comp_cnt / (tileK / COMPRESSION_RATE);
          uint32_t comp_c = comp_cnt % (tileK / COMPRESSION_RATE);
          fragA_compressed_(comp_r, comp_c) = fragA_(r, c);
          comp_cnt++;
        }
      }
    }
  }

  // Pack metadata into compact bitmap format
  void pack_metadata_bitmap() {
    constexpr uint32_t ELEMENTS_PER_ROW = tcK * i_ratio * COMPRESSION_RATE;
    constexpr uint32_t ROWS_PER_CHUNK = tcM / COMPRESSION_RATE;

    for (uint32_t m = 0; m < m_steps; ++m) {
      for (uint32_t k = 0; k < k_steps / COMPRESSION_RATE; ++k) {
        for (uint32_t chunk = 0; chunk < COMPRESSION_RATE; ++chunk) {
          uint32_t tmp_bit = 0;

          // Pack metadata for this chunk
          for (uint32_t r_i = 0; r_i < ROWS_PER_CHUNK; ++r_i) {
            for (uint32_t c_i = 0; c_i < ELEMENTS_PER_ROW; ++c_i) {
              uint32_t row = r_i + chunk * ROWS_PER_CHUNK + m * tcM;
              uint32_t col = c_i + k * ELEMENTS_PER_ROW;

              if (fragA_meta_(row, col) == 1) {
                uint32_t bit_pos = 31 - (c_i + r_i * ELEMENTS_PER_ROW);
                tmp_bit |= (1ULL << bit_pos);
              }
            }
          }

          uint32_t idx = chunk + k * SPARSITY_N + m * (k_steps / SPARSITY_N) * SPARSITY_N;
          packed_bit_meta_[idx] = tmp_bit;
        }
      }
    }
  }

  // Extract bitmap for a specific row
  uint16_t extract_row_metadata(const Vreg &va_meta, uint32_t row_idx) const {
    uint32_t meta_reg_idx = row_idx / COMPRESSION_RATE;
    bool is_upper_half = (row_idx % COMPRESSION_RATE) == 0;
    return is_upper_half ?
           static_cast<uint16_t>(va_meta[meta_reg_idx] >> 16) :
           static_cast<uint16_t>(va_meta[meta_reg_idx]);
  }

  // Gather B column elements based on A's sparsity pattern
  void gather_sparse_B_column(
      It *b_collected,
      const Xt *b_col_0,
      const Xt *b_col_1,
      uint16_t a_row_meta) const {

    constexpr uint32_t TOTAL_ELEMENTS = tcK * i_ratio;
    uint32_t collect_idx = 0;

    // Gather from first half based on upper bits of metadata
    for (uint32_t bit_idx = 0; bit_idx < TOTAL_ELEMENTS; ++bit_idx) {
      uint32_t bit_pos = TOTAL_ELEMENTS * SPARSITY_N - bit_idx - 1;
      if ((a_row_meta & (1 << bit_pos)) != 0) {
        uint32_t element_idx = bit_idx / i_ratio;
        uint32_t byte_pos = (bit_idx % i_ratio) * 8;
        b_collected[collect_idx++] =
            static_cast<It>((b_col_0[element_idx] >> byte_pos) & 0xFF);
      }
    }

    // Gather from second half based on lower bits of metadata
    for (uint32_t bit_idx = 0; bit_idx < TOTAL_ELEMENTS; ++bit_idx) {
      if (collect_idx >= TOTAL_ELEMENTS) break;

      uint32_t bit_pos = TOTAL_ELEMENTS - bit_idx - 1;
      if ((a_row_meta & (1 << bit_pos)) != 0) {
        uint32_t element_idx = bit_idx / i_ratio;
        uint32_t byte_pos = (bit_idx % i_ratio) * 8;
        b_collected[collect_idx++] =
            static_cast<uint8_t>((b_col_1[element_idx] >> byte_pos) & 0xFF);
      }
    }
  }
#endif  // ENABLE_SPARSITY

  // ========================================================================
  // Load/Store Operations (different implementations for dense/sparse)
  // ========================================================================

#ifdef ENABLE_SPARSITY
  // Sparse version of load_A
  void load_A(vector_t<Vreg, NRA> &vR, uint32_t lane, uint32_t ldm,
              const It *mdata, const vector_t<uint32_t, 8> &A_meta) {
    uint32_t block_idx = lane / a_block_size;
    uint32_t lane_in_block = lane % a_block_size;
    uint32_t elem_row = lane_in_block / tcK;
    uint32_t elem_col = lane_in_block % tcK;

    // Load compressed data into first half of registers
    for (uint32_t r = 0; r < NRA / COMPRESSION_RATE; ++r) {
      uint32_t block_m = (r / (k_steps / COMPRESSION_RATE)) * a_sub_blocks + block_idx;
      uint32_t block_k = r % (k_steps / COMPRESSION_RATE);
      uint32_t row = block_m * tcM + elem_row;
      uint32_t col = block_k * tcK + elem_col;
      auto base = mdata + row * ldm + col * i_ratio;

      assert(reinterpret_cast<uintptr_t>(base) % alignof(Xt) == 0 &&
             "Base pointer must be aligned");
      vR[r][lane] = *reinterpret_cast<const Xt *>(base);
    }

    // Load metadata into second half (only for metadata lanes)
    if (lane < METADATA_LANES) {
      for (uint32_t r = NRA / COMPRESSION_RATE; r < NRA; ++r) {
        vR[r][lane] = A_meta.data()[COMPRESSION_RATE * (r - NRA / COMPRESSION_RATE) + lane];
      }
    } else {
      for (uint32_t r = NRA / COMPRESSION_RATE; r < NRA; ++r) {
        vR[r][lane] = 0;
      }
    }
  }
#else
  // Dense version of load_A
  void load_A(vector_t<Vreg, NRA> &vR, uint32_t lane, uint32_t ldm, const It *mdata) {
    uint32_t block_idx = lane / a_block_size;
    uint32_t lane_in_block = lane % a_block_size;
    uint32_t elem_row = lane_in_block / tcK;
    uint32_t elem_col = lane_in_block % tcK;
    //DBG_PRINT("[load_A] lane=%u block_idx=%u lane_in_block=%u elem=[%u,%u], src=%p-%p\n",
    //          lane, block_idx, lane_in_block, elem_row, elem_col, mdata, mdata + tileM * tileK);

    for (uint32_t r = 0; r < NRA; ++r) {
      uint32_t block_m = (r / k_steps) * a_sub_blocks + block_idx;
      uint32_t block_k = r % k_steps;
      uint32_t row = block_m * tcM + elem_row;
      uint32_t col = block_k * tcK + elem_col;
      auto base = mdata + row * ldm + col * i_ratio;

      assert(reinterpret_cast<uintptr_t>(base) % alignof(Xt) == 0 &&
             "Base pointer must be aligned to sizeof(Xt)");
      vR[r][lane] = *reinterpret_cast<const Xt *>(base);
      //DBG_PRINT("  r=%u → block_m=%u block_k=%u → loads A[%u,%u] → %p → %u\n",
      //          r, block_m, block_k, row, col, base, vR[r][lane]);
    }
  }
#endif

#ifdef ENABLE_SPARSITY
  // Sparse version of load_B (loads 2x data for sparse B access)
  void load_B(vector_t<Vreg, NRB> &vR, uint32_t lane, uint32_t ldm, const It *mdata) {
    uint32_t block_idx = lane / b_block_size;
    uint32_t lane_in_block = lane % b_block_size;
    uint32_t elem_col = lane_in_block / (tcK * COMPRESSION_RATE);
    uint32_t elem_row = lane_in_block % (tcK * COMPRESSION_RATE);

    for (uint32_t r = 0; r < NRB; ++r) {
      uint32_t block_k = r / b_sub_steps;
      uint32_t block_n = (r % b_sub_steps) * b_sub_blocks + block_idx;
      uint32_t row = block_k * tcK * COMPRESSION_RATE + elem_row;
      uint32_t col = block_n * tcN + elem_col;
      auto base = mdata + row * ldm * i_ratio + col;

      if constexpr (sizeof(Xt) == sizeof(It)) {
        vR[r][lane] = *reinterpret_cast<const Xt *>(base);
      } else {
        vR[r][lane] = pack_row<Xt>(base, ldm);
      }
    }
  }
#else
  // Dense version of load_B
  void load_B(vector_t<Vreg, NRB> &vR, uint32_t lane, uint32_t ldm, const It *mdata) {
    uint32_t block_idx = lane / b_block_size;
    uint32_t lane_in_block = lane % b_block_size;
    uint32_t elem_col = lane_in_block / tcK;
    uint32_t elem_row = lane_in_block % tcK;
    //DBG_PRINT("[load_B] lane=%u block_idx=%u lane_in_block=%u elem=[%u,%u], src=%p-%p\n",
    //          lane, block_idx, lane_in_block, elem_row, elem_col, mdata, mdata + tileK * tileN);

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
      //DBG_PRINT("  r=%u → block_k=%u block_n=%u → loads B[%u,%u] → %p → %u\n",
      //          r, block_k, block_n, row, col, base, vR[r][lane]);
    }
  }
#endif

  void load_C(vector_t<Vreg, NRC> &vR, uint32_t lane, uint32_t ldm, const Ot *mdata) {
    uint32_t elem_row = lane / tcN;
    uint32_t elem_col = lane % tcN;
    // DBG_PRINT("[load_C] lane=%u elem=[%u,%u], src=%p-%p\n",
    //           lane, elem_row, elem_col, mdata, mdata + tileM * tileN);

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
      // DBG_PRINT("  r=%u → block_m=%u block_n=%u → loads C[%u,%u] → %p → %u\n",
      //           r, block_m, block_n, row, col, base, vR[r][lane]);
    }
  }

  void store_D(Ot *mdata, uint32_t lane, uint32_t ldm, const vector_t<Vreg, NRC> &vR) {
    uint32_t elem_row = lane / tcN;
    uint32_t elem_col = lane % tcN;

    // DBG_PRINT("[store_D] lane=%u elem=[%u,%u], dst=%p-%p\n",
    //           lane, elem_row, elem_col, mdata, mdata + tileM * tileN);

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
      // DBG_PRINT("  r=%u → block_m=%u block_n=%u → store C[%u,%u] → %p → %u\n",
      //           r, block_m, block_n, row, col, base , vR[r][lane]);
    }
  }

  // ========================================================================
  // Core Computation Operations
  // ========================================================================

  // Fused Element-wise Dot Product
  Xt FEDP(const Xt *a_row, const Xt *b_col, Xt c_val) const {
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

#ifdef ENABLE_SPARSITY
  // Sparse Matrix Multiply-Accumulate micro-operation
  Vreg MMA(uint32_t m, uint32_t n, const Vreg &va, const Vreg &va_meta,
           const Vreg &vb, const Vreg &vc) {
    uint32_t a_off = (m % a_sub_blocks) * a_block_size;
    uint32_t b_off = (n % b_sub_blocks) * b_block_size;

    Vreg vd;
    It b_col_collected[tcK * i_ratio];

    for (uint32_t i = 0; i < tcM; ++i) {
      for (uint32_t j = 0; j < tcN; ++j) {
        auto a_row = &va[a_off + i * tcK];
        auto b_col_0 = &vb[b_off + j * tcK * COMPRESSION_RATE];
        auto b_col_1 = &vb[b_off + j * tcK * COMPRESSION_RATE + tcK];
        auto c = vc[i * tcN + j];

        // Extract metadata for this row
        uint16_t a_row_meta = extract_row_metadata(va_meta, i);

        // Gather sparse B elements based on A's metadata
        gather_sparse_B_column(b_col_collected, b_col_0, b_col_1, a_row_meta);

        // Compute dot product
        auto d = FEDP(a_row, reinterpret_cast<const Xt*>(b_col_collected), c);
        vd[i * tcN + j] = d;
      }
    }

    return vd;
  }
#else
  // Dense Matrix Multiply-Accumulate micro-operation
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
#endif

#ifdef ENABLE_SPARSITY
  // Sparse matrix multiply-add operation
  FragD mmadd(const FragA &A, const vector_t<uint32_t, 8> &A_meta,
              const FragB &B, const FragC &C) {
    FragD D;
    vector_t<Vreg, NRA> vA;
    vector_t<Vreg, NRB> vB;
    vector_t<Vreg, NRC> vC, vD;

    dbg_out << "A=" << A << "\n";
    dbg_out << "B=" << B << "\n";
    dbg_out << "C=" << C << "\n";

    // Load fragments into vector registers
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_A(vA, lane, tileK, A.data(), A_meta);
    }
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_B(vB, lane, tileN, B.data());
    }
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_C(vC, lane, tileN, C.data());
    }

    // Execute micro-operations
    for (uint32_t k = 0; k < k_steps / COMPRESSION_RATE; ++k) {
      for (uint32_t m = 0; m < m_steps; ++m) {
        for (uint32_t n = 0; n < n_steps; ++n) {
          loop_iteration_count_++;  // Count loop iterations
          uint32_t idxA = (m / a_sub_blocks) * (k_steps / COMPRESSION_RATE) + k;
          uint32_t idxA_meta = idxA + NRA / COMPRESSION_RATE;
          uint32_t idxB = (k * n_steps + n) / b_sub_blocks;
          uint32_t idxC = m * n_steps + n;

          auto &va = vA[idxA];
          auto &va_meta = vA[idxA_meta];
          auto &vb = vB[idxB];
          auto &vc = (k != 0) ? vD[idxC] : vC[idxC];

          auto vd = MMA(m, n, va, va_meta, vb, vc);
          vD[idxC] = vd;
        }
      }
    }

    // Store results back to fragment
    for (uint32_t lane = 0; lane < NT; ++lane) {
      store_D(D.data(), lane, tileN, vD);
    }

    dbg_out << "D=" << D << "\n";
    return D;
  }
#else
  // Dense matrix multiply-add operation
  FragD mmadd(const FragA &A, const FragB &B, const FragC &C) {
    FragD D;
    vector_t<Vreg, NRA> vA;
    vector_t<Vreg, NRB> vB;
    vector_t<Vreg, NRC> vC, vD;

    dbg_out << "A=" << A << "\n";
    dbg_out << "B=" << B << "\n";
    dbg_out << "C=" << C << "\n";

    // per-lane load
    for (uint32_t lane = 0; lane < NT; ++lane) {
      load_A(vA, lane, tileK, A.data());
    }
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
          loop_iteration_count_++;  // Count loop iterations
          uint32_t idxA = (m / a_sub_blocks) * k_steps + k;
          uint32_t idxB = (k * n_steps + n) / b_sub_blocks;
          uint32_t idxC = m * n_steps + n;

          auto &va = vA[idxA];
          auto &vb = vB[idxB];
          auto &vc = (k != 0) ? vD[idxC] : vC[idxC];

          auto vd = MMA(m, n, va, vb, vc);

          // dbg_out << "[mmadd] m=" << m << " n=" << n << " k=" << k
          //         << " → idxA=" << idxA << " idxB=" << idxB << " idxC=" << idxC
          //         << " va=" << va << " vb=" << vb << " vc=" << vc << " vd=" << vd << "\n";

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
#endif

public:
  // ========================================================================
  // Public Interface
  // ========================================================================

  void init() {
    int x = 0;

    // Initialize matrix A with sequential values
    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileK; ++c) {
        fragA_(r, c) = x++;
      }
    }

#ifdef ENABLE_SPARSITY
    // Apply 2:4 structured sparsity
    std::random_device rd;
    std::mt19937 gen(rd());
    apply_2_4_pruning(gen);

    // Compress sparse matrix A
    compress_matrix_A();

    // Pack metadata into bitmap format
    pack_metadata_bitmap();
#endif

    // Initialize matrix B with sequential values
    for (uint32_t r = 0; r < tileK; ++r) {
      for (uint32_t c = 0; c < tileN; ++c) {
        fragB_(r, c) = x++;
      }
    }

    // Initialize matrix C to zero
    for (uint32_t r = 0; r < tileM; ++r) {
      for (uint32_t c = 0; c < tileN; ++c) {
        fragC_(r, c) = 0;
      }
    }

    // Compute reference result
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

  float verify() const {
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
      return static_cast<float>(err);
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

  uint32_t get_loop_count() const {
    return loop_iteration_count_;
  }

  void run() {
    loop_iteration_count_ = 0;  // Initialize counter
#ifdef ENABLE_SPARSITY
    fragD_ = mmadd(fragA_compressed_, packed_bit_meta_, fragB_, fragC_);
#else
    fragD_ = mmadd(fragA_, fragB_, fragC_);
#endif
  }
};

// ============================================================================
// Main Test Driver
// ============================================================================
using cfg = wmma_config_t<
    NUM_THREADS,
    8,
    XLENB,
    OTYPE,
    ITYPE,
    DPLEN>;

int main() {
  WMMA<cfg> wmma;

#ifdef ENABLE_SPARSITY
  std::cout << "=== Sparse Tensor Core Configuration (2:4 Structured Sparsity) ===\n";
#else
  std::cout << "=== Dense Tensor Core Configuration ===\n";
#endif

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
      << "\n";

  wmma.init();
  wmma.run();

  auto err = wmma.verify();
  bool passed = (err < 1e-4f);

  std::cout << "Total loop iterations: " << wmma.get_loop_count() << "\n"
            << "Max abs error: " << err << "\n"
            << (passed ? "PASSED!" : "FAILED!") << '\n';

  return passed ? 0 : 1;
}

// ============================================================================
// Build Instructions
// ============================================================================
// Dense mode (default):
//   g++ -std=c++17 -O2 tensor_generic.cpp -o a.out
//
// Sparse mode (2:4 structured sparsity):
//   g++ -std=c++17 -O2 -DENABLE_SPARSITY tensor_generic.cpp -o a.out
//
// Debug builds:
//   g++ -std=c++17 -g tensor_generic.cpp -o a.out
//   g++ -std=c++17 -g -DENABLE_SPARSITY tensor_generic.cpp -o a.out
