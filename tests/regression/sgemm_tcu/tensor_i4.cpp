#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr int M = 8;
constexpr int N = 4;
constexpr int K = 32;

// Pack a vector of 4-bit unsigned values (0–15) into bytes (2 values per byte)
void pack_int4(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst) {
  int size = src.size();
  dst.resize((size + 1) / 2);
  for (int i = 0; i < size; i += 2) {
    uint8_t low = src[i] & 0xF;
    uint8_t high = (i + 1 < size) ? (src[i + 1] & 0xF) : 0;
    dst[i / 2] = low | (high << 4);
  }
}

// Unpack a single 4-bit element from the packed array
uint8_t unpack_int4(const std::vector<uint8_t> &packed, int idx) {
  uint8_t byte = packed[idx / 2];
  if (idx % 2 == 0) // even index → lower nibble
    return byte & 0xF;
  else // odd index → upper nibble
    return (byte >> 4) & 0xF;
}

// Print an M×K or K×N matrix stored in packed int4
void print_int4_mat(const std::vector<uint8_t> &packed, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      auto val = unpack_int4(packed, i * cols + j);
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << "0x" << std::hex << +val << std::dec;
    }
    std::cout << "\n";
  }
}

int main() {
  // 1) Create and initialize raw 4-bit data for A (M×K) and B (K×N)
  std::vector<uint8_t> A_raw(M * K), B_raw(K * N);
  int ctr = 0;
  for (int idx = 0; idx < M * K; ++idx) {
    A_raw[idx] = ctr++ % 16;
  }
  for (int idx = 0; idx < K * N; ++idx) {
    B_raw[idx] = ctr++ % 16;
  }

  // 2) Pack into int4 format
  std::vector<uint8_t> A_packed, B_packed;
  pack_int4(A_raw, A_packed);
  pack_int4(B_raw, B_packed);

  // 3) Allocate C as int32 and zero‐initialize
  std::vector<int32_t> C(M * N, 0);

  // 4) Compute C = A×B
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        uint8_t a = unpack_int4(A_packed, i * K + k);
        uint8_t b = unpack_int4(B_packed, k * N + j);
        sum += int32_t(a) * int32_t(b);
      }
      C[i * N + j] = sum;
    }
  }

  // 5) Print everything
  std::cout << "Matrix A (" << M << "×" << K << ") in int4:\n";
  print_int4_mat(A_packed, M, K);
  std::cout << "\nMatrix B (" << K << "×" << N << ") in int4:\n";
  print_int4_mat(B_packed, K, N);

  std::cout << "\nResult Matrix C (" << M << "×" << N << ") in int32:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << "0x" << std::hex << C[i * N + j] << std::dec;
    }
    std::cout << "\n";
  }

  return 0;
}

// README
// gcc -std=c++17 -O2 tensor_i4.cpp -lstdc++