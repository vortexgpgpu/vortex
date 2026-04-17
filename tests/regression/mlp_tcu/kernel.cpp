// MLP inference kernel using the Vortex TCU (Tensor Core Unit)

#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <math.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// ── Generic element ↔ float converters ──────────────────────────────────────
template<typename T> struct elem_cvt;

template<> struct elem_cvt<vt::fp32> {
    static float    to_f32(float    x) { return x; }
    static float  from_f32(float    x) { return x; }
};
template<> struct elem_cvt<vt::fp16> {
    static float to_f32(uint16_t x) {
        __fp16 tmp; __builtin_memcpy(&tmp, &x, sizeof(tmp)); return (float)tmp;
    }
    static uint16_t from_f32(float x) {
        __fp16 tmp = (__fp16)x; uint16_t h; __builtin_memcpy(&h, &tmp, sizeof(h)); return h;
    }
};
template<> struct elem_cvt<vt::bf16> {
    static float to_f32(uint16_t x) {
        uint32_t b = (uint32_t)x << 16; float f; __builtin_memcpy(&f, &b, sizeof(f)); return f;
    }
    static uint16_t from_f32(float x) {
        uint32_t b; __builtin_memcpy(&b, &x, sizeof(b));
        // Propagate NaN without turning it into Inf (set quiet bit)
        if ((b & 0x7F800000) == 0x7F800000 && (b & 0x007FFFFF))
            return (uint16_t)((b >> 16) | 0x0040);
        // Round to nearest-even (matches rv_ftob_s / softfloat f32_to_bf16)
        uint32_t lsb    = (b >> 16) & 1;
        uint32_t round  = (b >> 15) & 1;
        uint32_t sticky = b & 0x7FFF;
        if (round && (sticky || lsb)) b += 0x8000;
        return (uint16_t)(b >> 16);
    }
};

using  icvt = elem_cvt<vt::ITYPE>;  // input  element converter
using  ocvt = elem_cvt<vt::OTYPE>;  // output element converter

// ── Fully-connected GEMM via TCU ─────────────────────────────────────────────
// W and X are ITYPE; Y receives OTYPE from the accumulator.
typedef struct {
    ctx::input_t  *W;
    ctx::input_t  *X;
    ctx::output_t *Y;
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

// ── Bias + optional ReLU, with optional downcast to separate ITYPE buffer ────
// For mixed-precision intermediate layers (e.g. bf16→fp32), bias_relu writes
// the downcast ITYPE result into a SEPARATE buffer (Y_in) rather than in-place
// into Y.  This avoids the parallel race where thread 2k writing at byte 4k
// would corrupt the fp32 value that thread k is still reading from byte 4k.
typedef struct {
    ctx::output_t *Y;      // OTYPE buffer holding GEMM output
    ctx::input_t  *Y_in;   // ITYPE buffer for downcast result (nullptr on last layer)
    ctx::output_t *bias;   // bias in OTYPE
    uint32_t  N;
    bool      apply_relu;
    bool      downcast;    // true when sizeof(OTYPE) > sizeof(ITYPE) and not last layer
} bias_relu_args_t;

void bias_relu_thread(bias_relu_args_t *__UNIFORM__ args) {
    uint32_t j = blockIdx.x;
    uint32_t i = blockIdx.y;

    ctx::output_t *row = args->Y + i * args->N;
    float v = ocvt::to_f32(row[j]) + ocvt::to_f32(args->bias[i]);
    if (args->apply_relu && v < 0.0f) v = 0.0f;

    if (args->downcast) {
        // Write ITYPE result to separate buffer — no overlap with Y, no race
        args->Y_in[i * args->N + j] = icvt::from_f32(v);
    } else {
        row[j] = ocvt::from_f32(v);
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────
typedef struct {
    ctx::output_t *data;
    uint32_t out_dim;
    uint32_t batch_n;
} softmax_args_t;

void softmax_thread(softmax_args_t *__UNIFORM__ args) {
    uint32_t b = blockIdx.x;
    if (b >= args->batch_n) return;

    ctx::output_t *data  = args->data;
    uint32_t out_dim = args->out_dim;
    uint32_t batch_n = args->batch_n;

    float max_val = ocvt::to_f32(data[b]);
    for (uint32_t i = 1; i < out_dim; ++i) {
        float v = ocvt::to_f32(data[i * batch_n + b]);
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float e = expf(ocvt::to_f32(data[i * batch_n + b]) - max_val);
        data[i * batch_n + b] = ocvt::from_f32(e);
        sum_exp += e;
    }

    for (uint32_t i = 0; i < out_dim; ++i)
        data[i * batch_n + b] = ocvt::from_f32(ocvt::to_f32(data[i * batch_n + b]) / sum_exp);
}

// ── Top-level kernel ──────────────────────────────────────────────────────────
void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
    auto layer_configs = reinterpret_cast<layer_config_t *>(arg->layer_configs_addr);

    // buffer1/buffer2   : OTYPE-sized — hold the GEMM accumulator output.
    // buffer1_in/2_in   : ITYPE-sized — hold the bias+relu+downcast result that
    //                     the *next* layer reads as its ITYPE activations.
    // For same-precision (sizeof ITYPE == sizeof OTYPE), main.cpp aliases
    // buffer1_in_addr == buffer1_addr so the two pointer sets are identical.
    auto input      = reinterpret_cast<ctx::input_t  *>(arg->input_addr);
    auto buffer1    = reinterpret_cast<ctx::output_t *>(arg->buffer1_addr);
    auto buffer2    = reinterpret_cast<ctx::output_t *>(arg->buffer2_addr);
    auto buffer1_in = reinterpret_cast<ctx::input_t  *>(arg->buffer1_in_addr);
    auto buffer2_in = reinterpret_cast<ctx::input_t  *>(arg->buffer2_in_addr);
    auto output     = reinterpret_cast<ctx::output_t *>(arg->output_addr);

    ctx::input_t  *layer_in;
    ctx::output_t *layer_out;
    ctx::input_t  *layer_in_next;  // ITYPE buffer that receives the downcast result

    uint32_t N = arg->batch_size;
    uint32_t block_dim[2] = { NUM_THREADS, 1 };

    for (uint32_t layer = 0; layer < arg->num_layers; ++layer) {
        auto     cfg     = &layer_configs[layer];
        uint32_t M       = cfg->output_dim;
        uint32_t K       = cfg->input_dim;
        bool     is_last = (layer == arg->num_layers - 1);

        if (layer == 0) {
            layer_in      = input;
            layer_out     = is_last ? output : buffer1;
            layer_in_next = buffer1_in;
        } else {
            // Odd layers read from buffer1_in, write OTYPE to buffer2
            // Even non-zero layers read from buffer2_in, write OTYPE to buffer1
            layer_in      = (layer % 2 == 1) ? buffer1_in : buffer2_in;
            layer_out     = is_last ? output : ((layer % 2 == 1) ? buffer2 : buffer1);
            layer_in_next = (layer % 2 == 1) ? buffer2_in : buffer1_in;
        }

        // ── TCU GEMM ────────────────────────────────────────────────────
        tcu_fc_args_t gemm_args;
        gemm_args.W = reinterpret_cast<ctx::input_t *>(cfg->weights_addr);
        gemm_args.X = layer_in;
        gemm_args.Y = layer_out;
        gemm_args.M = M; gemm_args.N = N; gemm_args.K = K;

        uint32_t grid_dim[2] = { N / ctx::tileN, M / ctx::tileM };
        vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)tcu_fc_thread, &gemm_args);

        // ── Bias + ReLU (downcast result goes to separate ITYPE buffer) ──
        bias_relu_args_t br_args;
        br_args.Y          = layer_out;
        br_args.Y_in       = is_last ? nullptr : layer_in_next;
        br_args.bias       = reinterpret_cast<ctx::output_t *>(cfg->bias_addr);
        br_args.N          = N;
        br_args.apply_relu = !is_last;
        br_args.downcast   = !is_last && (sizeof(ctx::output_t) > sizeof(ctx::input_t));

        uint32_t br_grid[2] = { N, M };
        vx_spawn_threads(2, br_grid, nullptr, (vx_kernel_func_cb)bias_relu_thread, &br_args);
    }

    // ── Softmax on final OTYPE output ──────────────────────────────────────
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
