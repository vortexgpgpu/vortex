#include <vx_spawn2.h>
#include <cmath>
#include "flash_attn_args.h"
#include "flash_attn_device.h"

using namespace flash_attn;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto Q = reinterpret_cast<float*>(arg->Q_addr);
    auto K = reinterpret_cast<float*>(arg->K_addr);
    auto V = reinterpret_cast<float*>(arg->V_addr);
    auto O = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto head_dim = arg->head_dim;
    auto head_dim_tile = arg->head_dim_tile;
    auto block_size_r = arg->block_size_r;
    auto block_size_c = arg->block_size_c;

    // Constant across a warp
    auto num_threads = blockDim.x;
    auto g_row = blockIdx.x * blockDim.y + threadIdx.y;
    auto l_row = threadIdx.y;
    auto g_row_offset = g_row * head_dim;
    auto l_row_offset = l_row * head_dim;

    auto local_ptr = __local_mem();
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + block_size_r * head_dim;
    auto local_V = (float*)local_K + block_size_c * head_dim_tile;
    auto local_O = (float*)local_V + block_size_c * head_dim_tile;
    auto local_P = (float*)local_O + block_size_r * head_dim;

    // Load Q tile from global memory
    for (uint32_t i = threadIdx.x; i < head_dim; i += num_threads) {
        if (g_row < seq_len) {
          local_Q[l_row_offset + i] = Q[g_row_offset + i];
        }
    }

    // Warp's row of Q tile
    float* Q_row = local_Q + l_row_offset;

    // Initialize O tile
    for (uint32_t i = threadIdx.x; i < head_dim; i += num_threads) {
        if (g_row < seq_len) {
          local_O[l_row_offset + i] = 0.0f;
        }
    }

    // Per-lane register state
    float m = -INFINITY;
    float l = 0.0f;

    // This warp's row of the S/P buffer in LMEM. Accumulates S and softmax
    // rewrites it in place as P.
    uint32_t sp_count = block_size_c / num_threads;
    float* P_row = local_P + l_row * block_size_c;

    // KV-block loop
    for (uint32_t j = 0; j < seq_len; j += block_size_c) {
        for (uint32_t i = 0; i < sp_count; ++i) {
          P_row[threadIdx.x + i * num_threads] = 0.0f;
        }

        // S = QK^T
        for (uint32_t h = 0; h < head_dim; h += head_dim_tile) {
            uint32_t tile_w = std::min(head_dim_tile, head_dim - h);
            // Load K tile
            for (uint32_t i = 0; i < sp_count; ++i) {
                uint32_t c = threadIdx.x + i * num_threads;
                for (uint32_t e = 0; e < tile_w; ++e) {
                  local_K[c * head_dim_tile + e] = K[(j + c) * head_dim + h + e];
                }
            }
            __syncthreads();
            for (uint32_t i = 0; i < sp_count; ++i) {
                uint32_t c = threadIdx.x + i * num_threads;
                float acc = P_row[c];
                for (uint32_t e = 0; e < tile_w; ++e) {
                  acc += Q_row[h + e] * local_K[c * head_dim_tile + e];
                }
                P_row[c] = acc;
            }
            __syncthreads();
        }

        // Softmax: reduce over this warp's row, then rewrite S as P in place.
        float local_max = P_row[threadIdx.x];
        for (uint32_t i = 1; i < sp_count; ++i) {
            float v = P_row[threadIdx.x + i * num_threads];
            local_max = (v > local_max) ? v : local_max;
        }
        float rowmax = warp_reduce_max(local_max, num_threads);

        float local_sum = 0.0f;
        for (uint32_t i = 0; i < sp_count; ++i) {
            uint32_t c = threadIdx.x + i * num_threads;
            float p = expf(P_row[c] - rowmax);
            P_row[c] = p;
            local_sum += p;
        }
        float rowsum = warp_reduce_sum(local_sum, num_threads);
        __syncthreads();

        // Compute new m and l
        float new_m = (m > rowmax) ? m : rowmax;
        float old_weight = expf(m - new_m), new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;

        // Update O
        for (uint32_t h = 0; h < head_dim; h += head_dim_tile) {
            uint32_t tile_w = std::min(head_dim_tile, head_dim - h);
            // Load V tile
            for (uint32_t i = 0; i < sp_count; ++i) {
                uint32_t c = threadIdx.x + i * num_threads;
                for (uint32_t e = 0; e < tile_w; ++e) {
                  local_V[e * block_size_c + c] = V[(j + c) * head_dim + h + e];
                }
            }
            __syncthreads();
            for (uint32_t k = threadIdx.x; k < tile_w; k += num_threads) {
                float dot = 0.0f;
                for (uint32_t c = 0; c < block_size_c; ++c) {
                    dot += P_row[c] * local_V[k * block_size_c + c];
                }
                auto& o = local_O[l_row * head_dim + h + k];
                o = old_weight * o + new_weight * dot;
            }
            __syncthreads();
        }

        m = new_m, l = new_l;
    }

    // Normalize O by softmax denominator and write back to global memory
    float inv_l = 1.0f / l;
    for (uint32_t i = threadIdx.x; i < head_dim; i += num_threads) {
        if (g_row < seq_len) {
          O[g_row_offset + i] = local_O[l_row_offset + i] * inv_l;
        }
    }
}
