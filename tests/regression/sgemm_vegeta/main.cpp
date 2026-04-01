#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();		                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h M_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static gemm_mode_t gemm_mode = GEMM_MODE_TGEMM;

static void show_usage() {
   std::cout << "Vortex SGEMM TILE Test (16x16 matrix operations)." << std::endl;
   std::cout << "Usage: [-m mode] [-h: help]" << std::endl;
   std::cout << "  -m mode: GEMM mode (0=TGEMM, 1=UGEMM, 2=VGEMM, 3=RGEMM) [default: 0]" << std::endl;
   std::cout << "    TGEMM (0): T × T -> T (dense × dense)" << std::endl;
   std::cout << "    UGEMM (1): T × U -> T (dense × 2:4 sparse)" << std::endl;
   std::cout << "    VGEMM (2): T × V -> T (dense × 1:4 sparse)" << std::endl;
   std::cout << "    RGEMM (3): T × U -> U (row-wise N:4 sparse × dense)" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:h")) != -1) {
    switch (c) {
    case 'm':
      gemm_mode = static_cast<gemm_mode_t>(atoi(optarg));
      if (gemm_mode < GEMM_MODE_TGEMM || gemm_mode > GEMM_MODE_RGEMM) {
        std::cerr << "Error: Invalid mode " << gemm_mode << std::endl;
        show_usage();
        exit(-1);
      }
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(M_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// Generate compressed 2:4 sparse tile and metadata from full logical matrix
// Input: logical_tile is M×K (e.g., 16×32), output: compressed_tile is M×(K/2) (e.g., 16×16)
// Metadata format: 16×16 nibbles stored as 128 bytes (8 bytes per row, 2 nibbles per byte)
static void compress_2_4_sparse(const std::vector<float>& logical_tile, int M, int K,
                                 std::vector<float>& compressed_tile, std::vector<uint8_t>& metadata) {
  compressed_tile.resize(M * (K / 2));
  metadata.resize(128);  // Fixed size: 16 rows × 8 bytes per row
  std::fill(metadata.begin(), metadata.end(), 0);
  
  for (int row = 0; row < M; ++row) {
    int compressed_col = 0;
    
    // Process K/4 groups of 4 elements
    for (int k_grp = 0; k_grp < K / 4; ++k_grp) {
      int k_base = k_grp * 4;
      
      // Find the 2 largest magnitude values in this group of 4
      std::pair<int, float> vals[4];
      for (int offset = 0; offset < 4; ++offset) {
        vals[offset] = {offset, logical_tile[row * K + k_base + offset]};
      }
      
      // Sort by magnitude to find top 2
      std::sort(vals, vals + 4, [](const auto& a, const auto& b) {
        return std::abs(a.second) > std::abs(b.second);
      });
      
      // Create bitmask for top 2 values
      uint8_t mask = 0;
      for (int i = 0; i < 2; ++i) {
        int offset = vals[i].first;
        mask |= (1u << offset);
      }
      
      // Store compressed values in POSITION ORDER (not magnitude order)
      // Hardware iterates through bit positions 0-3 and expects values in that order
      for (int offset = 0; offset < 4; ++offset) {
        if (mask & (1u << offset)) {
          compressed_tile[row * (K / 2) + compressed_col++] = logical_tile[row * K + k_base + offset];
        }
      }
      
      // Store metadata in 16×16 nibble format (128 bytes)
      // Each row has 8 bytes, each byte has 2 nibbles
      // Byte layout per row: byte 0 = cols 0,1; byte 1 = cols 2,3; ...; byte 7 = cols 14,15
      int byte_idx = row * 8 + k_grp / 2;
      if (k_grp % 2 == 0) {
        metadata[byte_idx] = (mask << 4);  // Upper nibble
      } else {
        metadata[byte_idx] |= mask;  // Lower nibble
      }
      
    }
  }
}

// Generate compressed 1:4 sparse tile and metadata from full logical matrix
// Input: logical_tile is M×K (e.g., 16×64), output: compressed_tile is M×(K/4) (e.g., 16×16)
// Metadata format: 16×16 nibbles stored as 128 bytes (8 bytes per row, 2 nibbles per byte)
static void compress_1_4_sparse(const std::vector<float>& logical_tile, int M, int K,
                                 std::vector<float>& compressed_tile, std::vector<uint8_t>& metadata) {
  compressed_tile.resize(M * (K / 4));
  metadata.resize(128);  // Fixed size: 16 rows × 8 bytes per row
  std::fill(metadata.begin(), metadata.end(), 0);
  
  for (int row = 0; row < M; ++row) {
    int compressed_col = 0;
    
    // Process K/4 groups of 4 elements
    for (int k_grp = 0; k_grp < K / 4; ++k_grp) {
      int k_base = k_grp * 4;
      
      // Find the largest magnitude value in this group of 4
      int max_offset = 0;
      float max_val = std::abs(logical_tile[row * K + k_base]);
      for (int offset = 1; offset < 4; ++offset) {
        float val = std::abs(logical_tile[row * K + k_base + offset]);
        if (val > max_val) {
          max_val = val;
          max_offset = offset;
        }
      }
      
      // Create bitmask and store compressed value
      uint8_t mask = (1u << max_offset);
      compressed_tile[row * (K / 4) + compressed_col++] = logical_tile[row * K + k_base + max_offset];
      
      // Store metadata in 16×16 nibble format (128 bytes)
      // Each row has 8 bytes, each byte has 2 nibbles
      int byte_idx = row * 8 + k_grp / 2;
      if (k_grp % 2 == 0) {
        metadata[byte_idx] = (mask << 4);  // Upper nibble
      } else {
        metadata[byte_idx] |= mask;  // Lower nibble
      }
    }
  }
}

// Generate compressed row-wise N:4 sparse tile and metadata from full logical matrix
// Input: logical_tile is M×K (16×32)
// Output: padded_tile is M×(K/2) (16×16), metadata is exactly 128 bytes
// Compression: For each 4-element block, keep top-2 values by magnitude (deterministic)
// Metadata layout: Must match TILE_LOAD_M format (8 bytes per row)
//   - 16 rows × 8 bytes/row = 128 bytes total
//   - Each byte stores 2 nibbles: upper nibble for col N, lower for col N+1
//   - For RGEMM: only first 8 nibbles (cols 0-7) are used, rest are zero
static void compress_rowwise_n4_sparse(const std::vector<float>& logical_tile, int M, int K,
                                        std::vector<float>& padded_tile, std::vector<uint8_t>& metadata) {
  // Output sizes: padded tile is M×(K/2), metadata is exactly 128 bytes
  padded_tile.resize(M * (K / 2));
  metadata.resize(128);  // 8 bytes per row × 16 rows = 128 bytes
  std::fill(metadata.begin(), metadata.end(), 0);
  
  for (int row = 0; row < M; ++row) {
    int padded_col = 0;
    
    // Process K/4 groups of 4 elements (8 groups for K=32)
    for (int k_grp = 0; k_grp < K / 4; ++k_grp) {
      int k_base = k_grp * 4;
      
      // Find the 2 largest magnitude values in this group of 4
      // Use index-value pairs for deterministic selection
      std::pair<int, float> vals[4];
      for (int offset = 0; offset < 4; ++offset) {
        vals[offset] = {offset, logical_tile[row * K + k_base + offset]};
      }
      
      // Sort by magnitude (descending) to find top 2
      // For equal magnitudes, lower index wins (stable, deterministic)
      std::sort(vals, vals + 4, [](const auto& a, const auto& b) {
        float abs_a = std::abs(a.second);
        float abs_b = std::abs(b.second);
        if (abs_a != abs_b) return abs_a > abs_b;
        return a.first < b.first;  // Tie-breaker: lower index first
      });
      
      // Create 4-bit bitmask for top 2 values
      uint8_t mask = 0;
      for (int i = 0; i < 2; ++i) {
        int offset = vals[i].first;
        mask |= (1u << offset);
      }
      
      // Store values in POSITION ORDER (not magnitude order)
      // Hardware iterates through bit positions 0-3 sequentially
      for (int offset = 0; offset < 4; ++offset) {
        if (mask & (1u << offset)) {
          padded_tile[row * (K / 2) + padded_col++] = logical_tile[row * K + k_base + offset];
        }
      }
      
      // Store metadata: 4 bits per block
      // Layout: 8 bytes per row (matching TILE_LOAD_M format)
      // Each byte stores 2 nibbles: upper for even col, lower for odd col
      // k_grp 0,1 -> byte 0 (cols 0,1), k_grp 2,3 -> byte 1 (cols 2,3), etc.
      int byte_idx = row * 8 + k_grp / 2;  // 8 bytes per row
      if (k_grp % 2 == 0) {
        metadata[byte_idx] = (mask << 4);  // Upper nibble (col N)
      } else {
        metadata[byte_idx] |= mask;  // Lower nibble (col N+1)
      }
    }
  }
  
  // Remaining bytes in each row (cols 8-15) are zero, already initialized
}

// CPU reference: C = A × B 
// A is MxK, B is KxN, C is MxN
// For TGEMM: A is 16x16, B is 16x16
// For UGEMM: A is 16x16 (but with 2:4 sparsity, effectively 16x32 positions), B is 16x32
// For VGEMM: A is 16x16 (but with 1:4 sparsity, effectively 16x64 positions), B is 16x64
static void matmul_cpu(float* C, const float* A, const float* B, int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

// Compare floats with ULP tolerance
static bool compare_float(float a, float b, int index, int& errors) {
  union { float f; int32_t i; } fa, fb;
  fa.f = a;
  fb.f = b;
  auto d = std::abs(fa.i - fb.i);
  if (d > FLOAT_ULP) {
    if (errors < 100) {
      printf("*** error: [%d] expected=%.6f, actual=%.6f\n", index, b, a);
    }
    ++errors;
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint32_t num_elements = TILE_SIZE * TILE_SIZE;  // 256 elements for T-reg
  uint32_t A_buf_size = T_TILE_BYTES;             // Always 1KB for A
  uint32_t C_buf_size = T_TILE_BYTES;             // Always 1KB for C (first T-reg of result)
  uint32_t B_buf_size, M_buf_size = 0;
  
  const char* mode_name;
  switch (gemm_mode) {
    case GEMM_MODE_TGEMM:
      mode_name = "TGEMM (T × T)";
      B_buf_size = T_TILE_BYTES;  // 1KB
      break;
    case GEMM_MODE_UGEMM:
      mode_name = "UGEMM (T × U, 2:4 sparse)";
      B_buf_size = U_TILE_BYTES;  // 2KB
      M_buf_size = M_TILE_BYTES;  // 1KB metadata
      break;
    case GEMM_MODE_VGEMM:
      mode_name = "VGEMM (T × V, 1:4 sparse)";
      B_buf_size = V_TILE_BYTES;  // 4KB
      M_buf_size = M_TILE_BYTES;  // 128 bytes metadata
      break;
    case GEMM_MODE_RGEMM:
      mode_name = "RGEMM (T × U -> U, row-wise N:4 sparse)";
      B_buf_size = U_TILE_BYTES;  // 2KB (B is dense U-reg)
      M_buf_size = M_TILE_BYTES;  // 128 bytes metadata
      break;
    default:
      std::cerr << "Invalid GEMM mode!" << std::endl;
      return -1;
  }

  std::cout << "SGEMM TILE Test: " << mode_name << std::endl;
  std::cout << "Matrix size: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
  std::cout << "A buffer: " << A_buf_size << " bytes" << std::endl;
  std::cout << "B buffer: " << B_buf_size << " bytes" << std::endl;
  if (M_buf_size > 0) {
    std::cout << "M buffer: " << M_buf_size << " bytes (metadata)" << std::endl;
  }
  std::cout << "C buffer: " << C_buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, A_buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, B_buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  
  kernel_arg.M_addr = 0;
  if (M_buf_size > 0) {
    RT_CHECK(vx_mem_alloc(device, M_buf_size, VX_MEM_READ, &M_buffer));
    RT_CHECK(vx_mem_address(M_buffer, &kernel_arg.M_addr));
  }
  
  RT_CHECK(vx_mem_alloc(device, C_buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  
  kernel_arg.mode = gemm_mode;

  std::cout << "dev_A=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "dev_B=0x" << std::hex << kernel_arg.B_addr << std::endl;
  if (kernel_arg.M_addr) {
    std::cout << "dev_M=0x" << std::hex << kernel_arg.M_addr << std::endl;
  }
  std::cout << "dev_C=0x" << std::hex << kernel_arg.C_addr << std::dec << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  
  // A's logical size depends on mode:
  // - TGEMM: 16x16 (dense)
  // - UGEMM/RGEMM: 16x32 (sparse compressed to 16x16)
  // - VGEMM: 16x64 (sparse 1:4 compressed to 16x16)
  uint32_t A_cols_logical = TILE_SIZE;
  if (gemm_mode == GEMM_MODE_UGEMM || gemm_mode == GEMM_MODE_RGEMM) A_cols_logical = 2 * TILE_SIZE;  // 32 logical cols
  else if (gemm_mode == GEMM_MODE_VGEMM) A_cols_logical = 4 * TILE_SIZE;  // 64 logical cols
  
  // B size matches A's logical K dimension
  uint32_t B_cols = TILE_SIZE;  // B is always 16 cols wide (output is 16x16)
  
  std::vector<float> h_A_logical(TILE_SIZE * A_cols_logical);  // Logical A before compression
  std::vector<float> h_A(num_elements);  // Compressed A (always 16x16 = 1KB for storage)
  std::vector<float> h_B(A_cols_logical * B_cols);  // B is K×N where K matches A's logical K
  std::vector<float> h_C(num_elements);  // Output is always 16×16 for tested modes
  std::vector<float> h_ref(num_elements);

  // Initialize logical matrix A
  for (uint32_t i = 0; i < TILE_SIZE * A_cols_logical; ++i) {
    h_A_logical[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  
  // Initialize matrix B (K×N where K = A's logical cols)
  for (uint32_t i = 0; i < A_cols_logical * B_cols; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // upload source buffers
  std::cout << "upload source buffers" << std::endl;
  
  std::vector<uint8_t> h_M;  // Metadata
  
  if (gemm_mode == GEMM_MODE_TGEMM) {
    // TGEMM: A (16x16) × B (16x16) = C (16x16)
    // Both dense in T-registers, no metadata
    h_A = h_A_logical;  // No compression needed
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_buf_size));
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, B_buf_size));
  }
  else if (gemm_mode == GEMM_MODE_UGEMM) {
    // UGEMM: A (16x32 logical, compressed to 16x16 with 2:4 sparsity) × B (32x16) = C (16x16)
    // A: logical 16x32 -> compressed 16x16 (1KB T-tile) with metadata
    // B: full 32x16 stored in U-register (2KB = 2 T-regs)
    
    // Compress A from 16x32 logical to 16x16 compressed
    compress_2_4_sparse(h_A_logical, TILE_SIZE, 2 * TILE_SIZE, h_A, h_M);
    
    std::cout << "2:4 sparse A: logical 16x32 -> compressed 16x16, metadata " << h_M.size() << " bytes" << std::endl;
    
    // Upload compressed A (1KB), metadata, and full B (2KB for U-reg)
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_buf_size));
    RT_CHECK(vx_copy_to_dev(M_buffer, h_M.data(), 0, M_buf_size));
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, B_buf_size));
  }
  else if (gemm_mode == GEMM_MODE_VGEMM) {
    // VGEMM: A (16x64 logical, compressed to 16x16 with 1:4 sparsity) × B (64x16) = C (16x16)
    // A: logical 16x64 -> compressed 16x16 (1KB T-tile) with metadata
    // B: full 64x16 stored in V-register (4KB = 4 T-regs)
    
    // Compress A from 16x64 logical to 16x16 compressed
    compress_1_4_sparse(h_A_logical, TILE_SIZE, 4 * TILE_SIZE, h_A, h_M);
    
    std::cout << "1:4 sparse A: logical 16x64 -> compressed 16x16, metadata " << h_M.size() << " bytes" << std::endl;
    
    // Upload compressed A (1KB), metadata, and full B (4KB for V-reg)
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_buf_size));
    RT_CHECK(vx_copy_to_dev(M_buffer, h_M.data(), 0, M_buf_size));
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, B_buf_size));
  }
  else if (gemm_mode == GEMM_MODE_RGEMM) {
    // RGEMM: A (16x32 logical, compressed to 16x16 via row-wise N:4) × B (32x16) = C (16x16)
    // A: logical 16x32 -> padded 16x16 (1KB T-tile) with metadata (128 bytes)
    // B: full 32x16 stored in U-register (2KB = 2 T-regs)
    
    // Compress A from 16x32 logical to 16x16 padded using row-wise N:4 compression
    compress_rowwise_n4_sparse(h_A_logical, TILE_SIZE, 2 * TILE_SIZE, h_A, h_M);
    
    std::cout << "Row-wise N:4 sparse A: logical 16x32 -> padded 16x16, metadata " << h_M.size() << " bytes" << std::endl;
    
    // Upload padded A (1KB), metadata (128B), and full B (2KB for U-reg)
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_buf_size));
    RT_CHECK(vx_copy_to_dev(M_buffer, h_M.data(), 0, M_buf_size));
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, B_buf_size));
  }

  // upload kernel binary
  std::cout << "upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download result
  std::cout << "download result" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, C_buf_size));

  // Zero out pruned values in h_A_logical based on metadata for CPU reference
  // This ensures CPU computes the same result as GPU (which only uses non-zero values)
  //
  // METADATA LAYOUT DIFFERENCES:
  // - UGEMM/VGEMM metadata: 8 bytes per row (16 nibbles = 16 4-element blocks)
  //   Total: 128 bytes (16 rows × 8 bytes/row)
  // - RGEMM metadata: 4 bytes per row (8 nibbles = 8 4-element blocks)
  //   Total: 64 bytes mask data + 64 bytes reserved = 128 bytes
  //
  if (gemm_mode == GEMM_MODE_UGEMM || gemm_mode == GEMM_MODE_VGEMM) {
    // UGEMM/VGEMM: 8 bytes per row (16 blocks of 4 elements for K=32/64)
    for (uint32_t row = 0; row < TILE_SIZE; ++row) {
      uint32_t k_groups = A_cols_logical / 4;
      for (uint32_t k_grp = 0; k_grp < k_groups; ++k_grp) {
        int k_base = k_grp * 4;
        // Get metadata nibble for this group (8 bytes per row layout)
        int byte_idx = row * 8 + k_grp / 2;
        uint8_t nibble = (k_grp % 2 == 0) ? (h_M[byte_idx] >> 4) : (h_M[byte_idx] & 0x0F);
        
        // Zero out positions not in metadata mask
        for (int offset = 0; offset < 4; ++offset) {
          if (!(nibble & (1u << offset))) {
            h_A_logical[row * A_cols_logical + k_base + offset] = 0.0f;
          }
        }
      }
    }
  }
  else if (gemm_mode == GEMM_MODE_RGEMM) {
    // RGEMM: 8 bytes per row (matching TILE_LOAD_M format)
    // Only first 8 nibbles (cols 0-7) are used for 8 blocks of 4 elements (K=32)
    for (uint32_t row = 0; row < TILE_SIZE; ++row) {
      uint32_t k_groups = A_cols_logical / 4;  // 8 groups for K=32
      for (uint32_t k_grp = 0; k_grp < k_groups; ++k_grp) {
        int k_base = k_grp * 4;
        // Get metadata nibble for this group (8 bytes per row layout)
        int byte_idx = row * 8 + k_grp / 2;
        uint8_t nibble = (k_grp % 2 == 0) ? (h_M[byte_idx] >> 4) : (h_M[byte_idx] & 0x0F);
        
        // Zero out positions not in metadata mask
        for (int offset = 0; offset < 4; ++offset) {
          if (!(nibble & (1u << offset))) {
            h_A_logical[row * A_cols_logical + k_base + offset] = 0.0f;
          }
        }
      }
    }
  }
  
  // compute CPU reference
  std::cout << "verify result" << std::endl;
  
  // For all modes: C = A_logical (with zeros in pruned positions) × B
  // For RGEMM: A_logical is 16×32 with zeros, B is 32×16, result is 16×16
  // M = TILE_SIZE (16), K = A_cols_logical, N = B_cols (16)
  matmul_cpu(h_ref.data(), h_A_logical.data(), h_B.data(), TILE_SIZE, A_cols_logical, B_cols);

  // verify result (always 256 elements = 16×16)
  int errors = 0;
  for (uint32_t i = 0; i < num_elements; ++i) {
    compare_float(h_C[i], h_ref[i], i, errors);
  }

  // write matrices to output file
  std::cout << "writing matrices to output file" << std::endl;
  std::ofstream output_file("matrices_output.txt");
  if (output_file.is_open()) {
    output_file << "GEMM Mode: " << mode_name << "\n\n";
    
    // 1. Print compressed/padded A matrix (what's actually sent to hardware)
    if (gemm_mode != GEMM_MODE_TGEMM) {
      output_file << "Matrix A Padded (Compressed " << TILE_SIZE << "x" << TILE_SIZE << "):\n";
      for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        for (uint32_t j = 0; j < TILE_SIZE; ++j) {
          output_file << h_A[i * TILE_SIZE + j];
          if (j < TILE_SIZE - 1) output_file << " ";
        }
        output_file << "\n";
      }
      output_file << "\n";
      
      // 2. Print metadata as 0/1 pattern
      output_file << "Metadata (" << TILE_SIZE << "x" << A_cols_logical << " sparsity pattern, 1=kept, 0=pruned):\n";
      for (uint32_t row = 0; row < TILE_SIZE; ++row) {
        uint32_t k_groups = A_cols_logical / 4;
        for (uint32_t k_grp = 0; k_grp < k_groups; ++k_grp) {
          // Get metadata nibble for this group (8 bytes per row layout)
          int byte_idx = row * 8 + k_grp / 2;
          uint8_t nibble = (k_grp % 2 == 0) ? (h_M[byte_idx] >> 4) : (h_M[byte_idx] & 0x0F);
          
          // Print 4 bits as 0/1
          for (int offset = 0; offset < 4; ++offset) {
            output_file << ((nibble & (1u << offset)) ? "1" : "0");
            if (k_grp < k_groups - 1 || offset < 3) output_file << " ";
          }
        }
        output_file << "\n";
      }
      output_file << "\n";
    }

    // 3. Print logical A matrix
    output_file << "Matrix A Logical (";
    if (gemm_mode == GEMM_MODE_TGEMM) {
      output_file << "Dense";
    } else if (gemm_mode == GEMM_MODE_UGEMM) {
      output_file << "2:4 Sparse";
    } else if (gemm_mode == GEMM_MODE_VGEMM) {
      output_file << "1:4 Sparse";
    } else if (gemm_mode == GEMM_MODE_RGEMM) {
      output_file << "Row-wise N:4 Sparse";
    }
    output_file << ", " << TILE_SIZE << "x" << A_cols_logical << "):\n";
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
      for (uint32_t j = 0; j < A_cols_logical; ++j) {
        output_file << h_A_logical[i * A_cols_logical + j];
        if (j < A_cols_logical - 1) output_file << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";

    // 4. Print B matrix
    output_file << "Matrix B (Dense, " << A_cols_logical << "x" << B_cols << "):\n";
    for (uint32_t i = 0; i < A_cols_logical; ++i) {
      for (uint32_t j = 0; j < B_cols; ++j) {
        output_file << h_B[i * B_cols + j];
        if (j < B_cols - 1) output_file << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";

    // 5. Print C matrices (GPU and CPU reference)
    output_file << "Matrix C (GPU Result, " << TILE_SIZE << "x" << TILE_SIZE << "):\n";
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
      for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        output_file << h_C[i * TILE_SIZE + j];
        if (j < TILE_SIZE - 1) output_file << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";

    output_file << "Matrix C (CPU Reference, " << TILE_SIZE << "x" << TILE_SIZE << "):\n";
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
      for (uint32_t j = 0; j < TILE_SIZE; ++j) {
        output_file << h_ref[i * TILE_SIZE + j];
        if (j < TILE_SIZE - 1) output_file << " ";
      }
      output_file << "\n";
    }

    output_file.close();
    std::cout << "Matrices written to 'matrices_output.txt'" << std::endl;
  } else {
    std::cerr << "Error: Unable to open output file" << std::endl;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
