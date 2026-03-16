// MLP inference kernel using the Vortex sparse WMMA unit.

#include "common.h"
#include <vx_spawn.h>
#include <vx_sparse.h>
#include <math.h>

namespace vt = vortex::sparse;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// fp32 -> fp32: input_t == output_t == float
static_assert(sizeof(ctx::input_t) == sizeof(ctx::output_t),
              "ITYPE and OTYPE must have equal element size");

static inline size_t align_up_size(size_t value, size_t alignment) {
    if (!alignment) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

typedef struct {
    ctx::input_t  *W_buf;          // packed sparse weight buffer
    ctx::input_t  *X;              // dense activations [K × N]
    ctx::output_t *Y;              // output [M × N]
    uint32_t       M, N, K;
    uint32_t       sparsity_degree;
} sparse_fc_args_t;

void sparse_fc_thread(sparse_fc_args_t *__UNIFORM__ args) {
    uint32_t M = args->M;
    uint32_t N = args->N;
    uint32_t K = args->K;
    uint32_t sparsity_degree = args->sparsity_degree;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;
    if (tile_row >= M || tile_col >= N)
        return;

    // Locate metadata: mirrors host packing in main.cpp
    constexpr size_t meta_entry_bytes = sizeof(uint32_t);
    size_t values_per_row = (size_t)K * sparsity_degree / 4;
    size_t values_size    = (size_t)M * values_per_row * sizeof(ctx::input_t);
    size_t meta_offset    = align_up_size(values_size, meta_entry_bytes);
    const uint8_t  *base_ptr  = reinterpret_cast<const uint8_t *>(args->W_buf);
    const uint32_t *meta_base = reinterpret_cast<const uint32_t *>(base_ptr + meta_offset);

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, (ctx::output_t)0);

    auto pA_values = args->W_buf;
    auto pB        = args->X;
    auto pC        = args->Y;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += ctx::tileK) {

        ctx::load_matrix_sync(fragB, pB + k_tile * N + tile_col, N);

        size_t row_offset_vals = (size_t)tile_row * values_per_row;
        size_t col_offset_vals = (k_tile / 4) * sparsity_degree;
        auto pTileA = pA_values + row_offset_vals + col_offset_vals;
        ctx::load_matrix_sync(fragA, pTileA, K,
                              reinterpret_cast<const void *>(meta_base),
                              tile_row, k_tile, sparsity_degree);

        ctx::mma_sync(fragC, fragA, fragB, fragC, sparsity_degree);
    }

    ctx::store_matrix_sync(pC + tile_row * N + tile_col, fragC, N);
}

typedef struct {
    ctx::output_t *Y;
    ctx::output_t *bias;
    uint32_t       N;
    bool           apply_relu;
} bias_relu_args_t;

void bias_relu_thread(bias_relu_args_t *__UNIFORM__ args) {
    uint32_t j = blockIdx.x;
    uint32_t i = blockIdx.y;
    
    ctx::output_t *row = args->Y + i * args->N;
    ctx::output_t  b   = args->bias[i];
    
    ctx::output_t v = row[j] + b;
    if (args->apply_relu && v < (ctx::output_t)0) v = (ctx::output_t)0;
    row[j] = v;
}

typedef struct {
    ctx::output_t *data;
    uint32_t out_dim;
    uint32_t batch_n;
} softmax_args_t;

void softmax_thread(softmax_args_t *__UNIFORM__ args) {
    uint32_t b = blockIdx.x;
    if (b >= args->batch_n) return;
    
    ctx::output_t *data = args->data;
    uint32_t out_dim = args->out_dim;
    uint32_t batch_n = args->batch_n;
    
    ctx::output_t max_val = data[b];
    for (uint32_t i = 1; i < out_dim; ++i) {
        ctx::output_t v = data[i * batch_n + b];
        if (v > max_val) max_val = v;
    }

    ctx::output_t sum_exp = 0;
    for (uint32_t i = 0; i < out_dim; ++i) {
        ctx::output_t e = expf(data[i * batch_n + b] - max_val);
        data[i * batch_n + b] = e;
        sum_exp += e;
    }
    for (uint32_t i = 0; i < out_dim; ++i)
        data[i * batch_n + b] /= sum_exp;
}

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
    auto layer_configs = reinterpret_cast<layer_config_t *>(arg->layer_configs_addr);

    auto input   = reinterpret_cast<ctx::input_t  *>(arg->input_addr);
    auto buffer1 = reinterpret_cast<ctx::output_t *>(arg->buffer1_addr);
    auto buffer2 = reinterpret_cast<ctx::output_t *>(arg->buffer2_addr);
    auto output  = reinterpret_cast<ctx::output_t *>(arg->output_addr);

    // Dynamic ping-pong buffers
    ctx::input_t *layer_in;
    ctx::output_t *layer_out;

    uint32_t N          = arg->batch_size;
    uint32_t block_dim[2] = { NUM_THREADS, 1 };
    uint32_t sparsity_degree = arg->sparsity_degree;

    for (uint32_t layer = 0; layer < arg->num_layers; ++layer) {
        auto *lcfg = &layer_configs[layer];
        uint32_t M = lcfg->output_dim;
        uint32_t K = lcfg->input_dim;
        
        if (layer == 0) {
            layer_in = input;
            layer_out = buffer1;
        } else if (layer == arg->num_layers - 1) {
            layer_in = reinterpret_cast<ctx::input_t *>((layer % 2 == 1) ? buffer1 : buffer2);
            layer_out = output;
        } else {
            layer_in = reinterpret_cast<ctx::input_t *>((layer % 2 == 1) ? buffer1 : buffer2);
            layer_out = (layer % 2 == 1) ? buffer2 : buffer1;
        }

        // ── Sparse FC GEMM ──────────────────────────────────────────────────
        sparse_fc_args_t fc_args;
        fc_args.W_buf           = reinterpret_cast<ctx::input_t *>(lcfg->weights_addr);
        fc_args.X               = layer_in;
        fc_args.Y               = layer_out;
        fc_args.M               = M;
        fc_args.N               = N;
        fc_args.K               = K;
        fc_args.sparsity_degree = sparsity_degree;

        uint32_t grid_dim[2] = { N / ctx::tileN, M / ctx::tileM };
        vx_spawn_threads(2, grid_dim, block_dim,
                         (vx_kernel_func_cb)sparse_fc_thread, &fc_args);

        // ── Bias + conditional ReLU ─────────────────────────────────────────
        bias_relu_args_t br_args;
        br_args.Y          = layer_out;
        br_args.bias       = reinterpret_cast<ctx::output_t *>(lcfg->bias_addr);
        br_args.N          = N;
        br_args.apply_relu = (layer < arg->num_layers - 1);
        
        uint32_t br_grid[2] = { N, M };
        vx_spawn_threads(2, br_grid, nullptr,
                         (vx_kernel_func_cb)bias_relu_thread, &br_args);
    }

    // ── Softmax on final output ─────────────────────────────────────────────
    uint32_t out_dim = layer_configs[arg->num_layers - 1].output_dim;
    softmax_args_t sm_args;
    sm_args.data    = output;
    sm_args.out_dim = out_dim;
    sm_args.batch_n = N;
    vx_spawn_threads(1, &N, nullptr, (vx_kernel_func_cb)softmax_thread, &sm_args);
}

int main() {
    auto arg = reinterpret_cast<kernel_arg_t *>(csr_read(VX_CSR_MSCRATCH));
    uint32_t grid_dim = 1;
    return vx_spawn_threads(1, &grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
