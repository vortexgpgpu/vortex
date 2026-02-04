#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h layer_configs_buffer = nullptr;
vx_buffer_h buffer1 = nullptr;
vx_buffer_h buffer2 = nullptr;
std::vector<vx_buffer_h> weight_buffers;
std::vector<vx_buffer_h> bias_buffers;
std::vector<vx_buffer_h> metadata_buffers;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

static mlp_mode_t mlp_mode = MLP_MODE_TGEMM;

static void show_usage() {
   std::cout << "Vortex VEGETA MLP Test (Batched)." << std::endl;
//    std::cout << "Usage: [-m mode] [-h: help]" << std::endl;
//    std::cout << "  -m mode: GEMM mode (0=TGEMM, 1=UGEMM, 2=VGEMM) [default: 0]" << std::endl;
//    std::cout << "    TGEMM (0): Dense × Dense" << std::endl;
//    std::cout << "    UGEMM (1): Dense × 2:4 Sparse" << std::endl;
//    std::cout << "    VGEMM (2): Dense × 1:4 Sparse" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:h")) != -1) {
    switch (c) {
    case 'm':
      mlp_mode = static_cast<mlp_mode_t>(atoi(optarg));
      if (mlp_mode < MLP_MODE_TGEMM || mlp_mode > MLP_MODE_VGEMM) {
        std::cerr << "Error: Invalid mode " << mlp_mode << std::endl;
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
        vx_mem_free(input_buffer);
        vx_mem_free(output_buffer);
        vx_mem_free(layer_configs_buffer);
        vx_mem_free(buffer1);
        vx_mem_free(buffer2);
        for (auto& buf : weight_buffers) vx_mem_free(buf);
        for (auto& buf : bias_buffers) vx_mem_free(buf);
        for (auto& buf : metadata_buffers) vx_mem_free(buf);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Sparse Weight Compression
///////////////////////////////////////////////////////////////////////////////

// Compress weights to 2:4 sparsity (keep 2 largest of every 4)
// Input: dense weights [in_dim x out_dim]
// Output: compressed weights [in_dim/2 x out_dim], metadata
void compress_2_4_sparse(const std::vector<TYPE>& dense, int in_dim, int out_dim,
                         std::vector<TYPE>& compressed, std::vector<uint8_t>& metadata,
                         std::vector<TYPE>& logical) {
    // For the CPU reference, we need the "logical" dense weights with zeros
    logical.resize(in_dim * out_dim, 0.0f);
    compressed.resize((in_dim / 2) * out_dim);
    
    // Metadata: 128 bytes per 16x16 tile
    int in_tiles = in_dim / TILE_SIZE;
    int out_tiles = out_dim / TILE_SIZE;
    metadata.resize(in_tiles * out_tiles * M_TILE_BYTES, 0);
    
    // For each group of 4 consecutive values in K dimension, keep 2 largest
    for (int j = 0; j < out_dim; ++j) {
        int comp_idx = 0;
        for (int i = 0; i < in_dim; i += 4) {
            // Get 4 values
            std::pair<float, int> vals[4];
            for (int k = 0; k < 4; ++k) {
                vals[k] = {std::abs(dense[(i + k) * out_dim + j]), k};
            }
            // Sort by magnitude (descending)
            std::sort(vals, vals + 4, [](auto& a, auto& b) { return a.first > b.first; });
            
            // Keep top 2, generate mask
            uint8_t mask = 0;
            for (int k = 0; k < 2; ++k) {
                int pos = vals[k].second;
                mask |= (1 << pos);
                compressed[(comp_idx / 2) * out_dim + j] = dense[(i + pos) * out_dim + j];
                logical[(i + pos) * out_dim + j] = dense[(i + pos) * out_dim + j];
                comp_idx++;
            }
            
            // Store metadata (simplified - in real impl would be tile-based)
            int tile_in = i / TILE_SIZE;
            int tile_out = j / TILE_SIZE;
            int meta_offset = (tile_in * out_tiles + tile_out) * M_TILE_BYTES;
            int row_in_tile = (i % TILE_SIZE) / 4;
            int col_in_tile = j % TILE_SIZE;
            int byte_offset = meta_offset + row_in_tile * TILE_SIZE + col_in_tile;
            if (byte_offset < (int)metadata.size()) {
                metadata[byte_offset] = mask;
            }
        }
    }
}

// Compress weights to 1:4 sparsity (keep 1 largest of every 4)
void compress_1_4_sparse(const std::vector<TYPE>& dense, int in_dim, int out_dim,
                         std::vector<TYPE>& compressed, std::vector<uint8_t>& metadata,
                         std::vector<TYPE>& logical) {
    logical.resize(in_dim * out_dim, 0.0f);
    compressed.resize((in_dim / 4) * out_dim);
    
    int in_tiles = in_dim / TILE_SIZE;
    int out_tiles = out_dim / TILE_SIZE;
    metadata.resize(in_tiles * out_tiles * M_TILE_BYTES, 0);
    
    for (int j = 0; j < out_dim; ++j) {
        int comp_idx = 0;
        for (int i = 0; i < in_dim; i += 4) {
            // Find largest of 4
            int max_pos = 0;
            float max_val = std::abs(dense[i * out_dim + j]);
            for (int k = 1; k < 4; ++k) {
                float val = std::abs(dense[(i + k) * out_dim + j]);
                if (val > max_val) {
                    max_val = val;
                    max_pos = k;
                }
            }
            
            compressed[comp_idx * out_dim + j] = dense[(i + max_pos) * out_dim + j];
            logical[(i + max_pos) * out_dim + j] = dense[(i + max_pos) * out_dim + j];
            comp_idx++;
            
            // Store metadata
            uint8_t mask = (1 << max_pos);
            int tile_in = i / TILE_SIZE;
            int tile_out = j / TILE_SIZE;
            int meta_offset = (tile_in * out_tiles + tile_out) * M_TILE_BYTES;
            int row_in_tile = (i % TILE_SIZE) / 4;
            int col_in_tile = j % TILE_SIZE;
            int byte_offset = meta_offset + row_in_tile * TILE_SIZE + col_in_tile;
            if (byte_offset < (int)metadata.size()) {
                metadata[byte_offset] = mask;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// CPU Reference
///////////////////////////////////////////////////////////////////////////////

static inline TYPE relu_cpu(TYPE x) {
    return (x > 0) ? x : 0;
}

static void softmax_cpu(TYPE* sample, int dim) {
    TYPE max_val = sample[0];
    for (int i = 1; i < dim; ++i) {
        if (sample[i] > max_val) max_val = sample[i];
    }
    TYPE sum_exp = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sample[i] = expf(sample[i] - max_val);
        sum_exp += sample[i];
    }
    for (int i = 0; i < dim; ++i) {
        sample[i] /= sum_exp;
    }
}

static void layer_cpu_batch(const TYPE* input, const TYPE* weights, const TYPE* bias,
                            TYPE* output, int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < output_dim; ++j) {
            TYPE sum = bias[j];
            for (int i = 0; i < input_dim; ++i) {
                sum += input[b * input_dim + i] * weights[i * output_dim + j];
            }
            output[b * output_dim + j] = sum;
        }
    }
}

static void mlp_cpu_batch(const std::vector<TYPE>& input,
                          const std::vector<std::vector<TYPE>>& weights,
                          const std::vector<std::vector<TYPE>>& biases,
                          const std::vector<std::pair<int,int>>& dims,
                          std::vector<TYPE>& output) {
    std::vector<TYPE> layer_input = input;
    std::vector<TYPE> layer_output;
    
    for (size_t layer = 0; layer < dims.size(); ++layer) {
        int in_dim = dims[layer].first;
        int out_dim = dims[layer].second;
        layer_output.resize(BATCH_SIZE * out_dim);
        
        layer_cpu_batch(layer_input.data(), weights[layer].data(), 
                        biases[layer].data(), layer_output.data(),
                        BATCH_SIZE, in_dim, out_dim);
        
        if (layer < dims.size() - 1) {
            for (auto& v : layer_output) {
                v = relu_cpu(v);
            }
        }
        
        layer_input = layer_output;
    }
    
    output = layer_output;
    for (int b = 0; b < BATCH_SIZE; ++b) {
        softmax_cpu(&output[b * OUTPUT_DIM], OUTPUT_DIM);
    }
}

// Compare floats with relative tolerance
static bool compare_float(TYPE a, TYPE b, int batch, int idx, int& errors) {
    constexpr float REL_TOL = 0.05f;  // 5% relative tolerance
    constexpr float ABS_TOL = 1e-6f;
    
    float diff = std::abs(a - b);
    float max_val = std::max(std::abs(a), std::abs(b));
    
    if (diff > REL_TOL * max_val + ABS_TOL) {
        if (errors < 10) {
            printf("*** error: [batch=%d, idx=%d] expected=%.6f, actual=%.6f (diff=%.2f%%)\n", 
                   batch, idx, b, a, 100.0f * diff / max_val);
        }
        ++errors;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    
    std::srand(50);
    
    std::vector<std::pair<int,int>> layer_dims = {
        {INPUT_DIM, HIDDEN1_DIM},
        {HIDDEN1_DIM, HIDDEN2_DIM},
        {HIDDEN2_DIM, HIDDEN3_DIM},
        {HIDDEN3_DIM, OUTPUT_DIM}
    };
    
    const char* mode_name;
    switch (mlp_mode) {
        case MLP_MODE_TGEMM: mode_name = "TGEMM (Dense × Dense)"; break;
        case MLP_MODE_UGEMM: mode_name = "UGEMM (Dense × 2:4 Sparse)"; break;
        case MLP_MODE_VGEMM: mode_name = "VGEMM (Dense × 1:4 Sparse)"; break;
        default: mode_name = "Unknown";
    }

    std::cout << "=== Vortex VEGETA MLP Neural Network Test (Batched) ===" << std::endl;
    std::cout << "Network: " << INPUT_DIM << " -> " << HIDDEN1_DIM << " -> " 
              << HIDDEN2_DIM << " -> " << HIDDEN3_DIM << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "Mode: " << mode_name << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));
    
    // Generate batched input
    std::vector<TYPE> h_input(BATCH_SIZE * INPUT_DIM);
    for (auto& v : h_input) {
        v = static_cast<TYPE>(rand()) / RAND_MAX;
    }
    
    // Generate weights, biases, and optional sparse compression
    std::vector<std::vector<TYPE>> h_weights_dense(NUM_LAYERS);
    std::vector<std::vector<TYPE>> h_weights_compressed(NUM_LAYERS);
    std::vector<std::vector<TYPE>> h_weights_logical(NUM_LAYERS);
    std::vector<std::vector<TYPE>> h_biases(NUM_LAYERS);
    std::vector<std::vector<uint8_t>> h_metadata(NUM_LAYERS);
    
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        int in_dim = layer_dims[layer].first;
        int out_dim = layer_dims[layer].second;
        
        // Generate dense weights
        h_weights_dense[layer].resize(in_dim * out_dim);
        for (auto& v : h_weights_dense[layer]) {
            v = (static_cast<TYPE>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        
        h_biases[layer].resize(out_dim);
        for (auto& v : h_biases[layer]) {
            v = (static_cast<TYPE>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        
        // Compress weights for sparse modes
        if (mlp_mode == MLP_MODE_UGEMM) {
            compress_2_4_sparse(h_weights_dense[layer], in_dim, out_dim,
                               h_weights_compressed[layer], h_metadata[layer],
                               h_weights_logical[layer]);
        } else if (mlp_mode == MLP_MODE_VGEMM) {
            compress_1_4_sparse(h_weights_dense[layer], in_dim, out_dim,
                               h_weights_compressed[layer], h_metadata[layer],
                               h_weights_logical[layer]);
        } else {
            h_weights_logical[layer] = h_weights_dense[layer];
        }
    }
    
    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    
    RT_CHECK(vx_mem_alloc(device, BATCH_SIZE * INPUT_DIM * sizeof(TYPE), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_alloc(device, BATCH_SIZE * OUTPUT_DIM * sizeof(TYPE), VX_MEM_READ_WRITE, &output_buffer));
    
    int max_hidden = std::max({HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM});
    RT_CHECK(vx_mem_alloc(device, BATCH_SIZE * max_hidden * sizeof(TYPE), VX_MEM_READ_WRITE, &buffer1));
    RT_CHECK(vx_mem_alloc(device, BATCH_SIZE * max_hidden * sizeof(TYPE), VX_MEM_READ_WRITE, &buffer2));
    
    std::vector<layer_config_t> layer_configs(NUM_LAYERS);
    weight_buffers.resize(NUM_LAYERS);
    bias_buffers.resize(NUM_LAYERS);
    metadata_buffers.resize(NUM_LAYERS, nullptr);
    
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        int in_dim = layer_dims[layer].first;
        int out_dim = layer_dims[layer].second;
        
        layer_configs[layer].input_dim = in_dim;
        layer_configs[layer].output_dim = out_dim;
        
        // Upload weights (compressed for sparse modes, dense for TGEMM)
        std::vector<TYPE>& weights_to_upload = 
            (mlp_mode == MLP_MODE_TGEMM) ? h_weights_dense[layer] : h_weights_compressed[layer];
        
        size_t weight_size = weights_to_upload.size() * sizeof(TYPE);
        RT_CHECK(vx_mem_alloc(device, weight_size, VX_MEM_READ, &weight_buffers[layer]));
        RT_CHECK(vx_mem_address(weight_buffers[layer], &layer_configs[layer].weights_addr));
        RT_CHECK(vx_copy_to_dev(weight_buffers[layer], weights_to_upload.data(), 0, weight_size));
        
        // Bias
        size_t bias_size = h_biases[layer].size() * sizeof(TYPE);
        RT_CHECK(vx_mem_alloc(device, bias_size, VX_MEM_READ, &bias_buffers[layer]));
        RT_CHECK(vx_mem_address(bias_buffers[layer], &layer_configs[layer].bias_addr));
        RT_CHECK(vx_copy_to_dev(bias_buffers[layer], h_biases[layer].data(), 0, bias_size));
        
        // Metadata for sparse modes
        if (mlp_mode != MLP_MODE_TGEMM && !h_metadata[layer].empty()) {
            size_t meta_size = h_metadata[layer].size();
            RT_CHECK(vx_mem_alloc(device, meta_size, VX_MEM_READ, &metadata_buffers[layer]));
            RT_CHECK(vx_mem_address(metadata_buffers[layer], &layer_configs[layer].metadata_addr));
            RT_CHECK(vx_copy_to_dev(metadata_buffers[layer], h_metadata[layer].data(), 0, meta_size));
        } else {
            layer_configs[layer].metadata_addr = 0;
        }
    }
    
    RT_CHECK(vx_mem_alloc(device, layer_configs.size() * sizeof(layer_config_t), VX_MEM_READ, &layer_configs_buffer));
    RT_CHECK(vx_copy_to_dev(layer_configs_buffer, layer_configs.data(), 0, layer_configs.size() * sizeof(layer_config_t)));
    
    std::cout << "Uploading input data (" << BATCH_SIZE << " samples)..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, BATCH_SIZE * INPUT_DIM * sizeof(TYPE)));
    
    std::cout << "Uploading kernel binary..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    
    kernel_arg_t kernel_arg = {};
    kernel_arg.num_layers = NUM_LAYERS;
    kernel_arg.mode = mlp_mode;
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
    RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));
    RT_CHECK(vx_mem_address(layer_configs_buffer, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_mem_address(buffer1, &kernel_arg.buffer1_addr));
    RT_CHECK(vx_mem_address(buffer2, &kernel_arg.buffer2_addr));
    
    std::cout << "Uploading kernel arguments..." << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
    
    std::cout << "Starting VEGETA MLP inference on device..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    
    std::cout << "Downloading output (" << BATCH_SIZE << " samples)..." << std::endl;
    std::vector<TYPE> h_output(BATCH_SIZE * OUTPUT_DIM);
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, BATCH_SIZE * OUTPUT_DIM * sizeof(TYPE)));
    
    // CPU reference uses logical weights (with zeros for pruned elements)
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<TYPE> h_ref;
    mlp_cpu_batch(h_input, h_weights_logical, h_biases, layer_dims, h_ref);
    
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int i = 0; i < OUTPUT_DIM; ++i) {
            compare_float(h_output[b * OUTPUT_DIM + i], h_ref[b * OUTPUT_DIM + i], b, i, errors);
        }
    }
    
    std::cout << "Cleaning up..." << std::endl;
    cleanup();
    
    if (errors != 0) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }
    
    std::cout << "PASSED!" << std::endl;
    return 0;
}
