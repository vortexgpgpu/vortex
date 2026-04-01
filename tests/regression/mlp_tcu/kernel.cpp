// MLP inference kernel using the Vortex TCU (Tensor Core Unit)

#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <math.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static_assert(sizeof(ctx::input_t) == sizeof(ctx::output_t), "expect fp16 in and out");


static inline float h2f(uint16_t h) {
    __fp16 tmp;
    __builtin_memcpy(&tmp, &h, sizeof(tmp));
    return (float)tmp;
}
static inline uint16_t f2h(float f) {
    __fp16 tmp = (__fp16)f;
    uint16_t h;
    __builtin_memcpy(&h, &tmp, sizeof(tmp));
    return h;
}

typedef struct {
    ctx::input_t *W;
    ctx::input_t *X;
    ctx::input_t *Y;   // output_t == input_t == uint16_t for fp16 to fp16
    uint32_t M, N, K;
} tcu_fc_args_t;

void tcu_fc_thread(tcu_fc_args_t *__UNIFORM__ args) {
    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, (ctx::output_t)0);

    for (uint32_t k = 0; k < args->K; k += ctx::tileK) {
        ctx::load_matrix_sync(fragA, args->W + tile_row * args->K + k, args->K);
        ctx::load_matrix_sync(fragB, args->X + k * args->N + tile_col, args->N);
        ctx::mma_sync(fragC, fragA, fragB, fragC);
    }

    ctx::store_matrix_sync(args->Y + tile_row * args->N + tile_col, fragC, args->N);
}

typedef struct {
    uint16_t *Y;
    uint16_t *bias;
    uint32_t  N;
    bool      apply_relu;
} bias_relu_args_t;

void bias_relu_thread(bias_relu_args_t *__UNIFORM__ args) {
    uint32_t j = blockIdx.x;
    uint32_t i = blockIdx.y;
    
    uint16_t *row = args->Y + i * args->N;
    float     b   = h2f(args->bias[i]);
    
    float v = h2f(row[j]) + b;
    if (args->apply_relu && v < 0.0f) v = 0.0f;
    row[j] = f2h(v);
}

typedef struct {
    uint16_t *data;
    uint32_t out_dim;
    uint32_t batch_n;
} softmax_args_t;

void softmax_thread(softmax_args_t *__UNIFORM__ args) {
    uint32_t b = blockIdx.x;
    if (b >= args->batch_n) return;
    
    uint16_t *data = args->data;
    uint32_t out_dim = args->out_dim;
    uint32_t batch_n = args->batch_n;
    
    float max_val = h2f(data[b]);
    for (uint32_t i = 1; i < out_dim; ++i) {
        float v = h2f(data[i * batch_n + b]);
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float e = expf(h2f(data[i * batch_n + b]) - max_val);
        data[i * batch_n + b] = f2h(e);
        sum_exp += e;
    }

    for (uint32_t i = 0; i < out_dim; ++i)
        data[i * batch_n + b] = f2h(h2f(data[i * batch_n + b]) / sum_exp);
}


void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
    auto layer_configs = reinterpret_cast<layer_config_t *>(arg->layer_configs_addr);

    //   Layer 0 : input   → buffer1
    //   Layer 1 : buffer1 → buffer2
    //   Layer 2 : buffer2 → buffer1   
    //   Layer 3 : buffer1 → output
    auto input   = reinterpret_cast<uint16_t *>(arg->input_addr);
    auto buffer1 = reinterpret_cast<uint16_t *>(arg->buffer1_addr);
    auto buffer2 = reinterpret_cast<uint16_t *>(arg->buffer2_addr);
    auto output  = reinterpret_cast<uint16_t *>(arg->output_addr);

    // Dynamic ping-pong buffers // @mitul: fixed now
    uint16_t *layer_in;
    uint16_t *layer_out;

    uint32_t N          = arg->batch_size;
    uint32_t block_dim[2] = { NUM_THREADS, 1 };

    for (uint32_t layer = 0; layer < arg->num_layers; ++layer) {
        auto     cfg = &layer_configs[layer];
        uint32_t M   = cfg->output_dim;
        uint32_t K   = cfg->input_dim;
        
        if (layer == 0) {
            layer_in = input;
            layer_out = buffer1;
        } else if (layer == arg->num_layers - 1) {
            layer_in = (layer % 2 == 1) ? buffer1 : buffer2;
            layer_out = output;
        } else {
            layer_in = (layer % 2 == 1) ? buffer1 : buffer2;
            layer_out = (layer % 2 == 1) ? buffer2 : buffer1;
        }

        // ── TCU GEMM ────────────────────────────────────────────────────
        tcu_fc_args_t gemm_args;
        gemm_args.W = reinterpret_cast<ctx::input_t *>(cfg->weights_addr);
        gemm_args.X = layer_in;
        gemm_args.Y = layer_out;
        gemm_args.M = M;
        gemm_args.N = N;
        gemm_args.K = K;

        uint32_t grid_dim[2] = { N / ctx::tileN, M / ctx::tileM };
        vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)tcu_fc_thread, &gemm_args);

        // ── Bias + optional ReLU ────────────────────────────────────────
        bias_relu_args_t br_args;
        br_args.Y          = layer_out;
        br_args.bias       = reinterpret_cast<uint16_t *>(cfg->bias_addr);
        br_args.N          = N;
        br_args.apply_relu = (layer < arg->num_layers - 1);

        uint32_t br_grid[2] = { N, M };
        vx_spawn_threads(2, br_grid, nullptr, (vx_kernel_func_cb)bias_relu_thread, &br_args);
    }

    // ── Softmax on final output ──────────────────────────────────────────
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
