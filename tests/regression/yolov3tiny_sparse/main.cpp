// ---------------------------------------------------------------------------
// YOLOv3-Tiny Sparse Host Driver
// https://github.com/AlexeyAB/darknet/tree/master/src
// ---------------------------------------------------------------------------

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vortex.h>
#include <sparse_cfg.h>
#include <rvfloats.h>
#include <util.h>
#include <type_traits>
#include "common.h"

#define FLOAT_ULP 16000

#define RT_CHECK(_expr)                                          \
  do {                                                           \
    int _ret = (_expr);                                          \
    if (0 == _ret) break;                                        \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                   \
    exit(-1);                                                    \
  } while (false)

using namespace vortex;
namespace vt = vortex::sparse;

using sparse_cfg_t = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static uint32_t g_sparsity_degree = 1;

static inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

static inline uint32_t align_up32(uint32_t x, uint32_t a) {
    return (x + a - 1) / a * a;
}

// ── Type helpers ──────────────────────────────────────────────────────────────

template <typename S, typename D>
struct muladd_t {
    using stype = typename S::dtype;
    using dtype = typename D::dtype;
    static dtype eval(stype a, stype b, dtype c) {
        return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
    }
};

template <>
struct muladd_t<vt::fp16, vt::fp32> {
    static float eval(uint16_t a, uint16_t b, float c) {
        auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
        auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
        return fa * fb + c;
    }
};

template <>
struct muladd_t<vt::bf16, vt::fp32> {
    static float eval(uint16_t a, uint16_t b, float c) {
        auto fa = bit_cast<float>(static_cast<uint32_t>(a) << 16);
        auto fb = bit_cast<float>(static_cast<uint32_t>(b) << 16);
        return fa * fb + c;
    }
};

// Convert a host-side float to itype_t.
//   fp32  → itype  : identity
//   fp32  → fp16   : IEEE-correct via rv_ftoh_s
//   fp32  → bf16   : truncate mantissa (upper 16 bits of fp32 bit-pattern)
static inline itype_t to_itype(float v) {
    if constexpr (std::is_same_v<itype_t, float>) {
        return v;
    } else if constexpr (vt::ITYPE::id == vt::fp16::id) {
        return static_cast<itype_t>(rv_ftoh_s(bit_cast<uint32_t>(v), 0, nullptr));
    } else {
        // bf16
        return static_cast<itype_t>(bit_cast<uint32_t>(v) >> 16);
    }
}

struct SparseMat {
    std::vector<itype_t> values;  // packed non-zero weights as itype_t
    std::vector<uint8_t> meta;
    uint32_t rows, cols;
};

static SparseMat pruneAndPack(const std::vector<itype_t> &denseA,
                               uint32_t M, uint32_t K,
                               uint32_t sparsity_degree) {
    SparseMat out;
    out.rows = M;
    out.cols = K;
    out.values.reserve((size_t)M * K * sparsity_degree / 4);
    out.meta.reserve((size_t)M * K / 4);

    const itype_t *src = denseA.data();
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < K; c += 4) {
            itype_t blk[4] = {
                src[r * K + c + 0], src[r * K + c + 1],
                src[r * K + c + 2], src[r * K + c + 3]
            };
            uint32_t idx[4] = {0, 1, 2, 3};
           
            for (uint32_t i = 3; i > 0; --i) {
                uint32_t j = rand() % (i + 1);
                std::swap(idx[i], idx[j]);
            }
            uint8_t kept[2]; 
            uint8_t mask = 0;
            for (uint32_t i = 0; i < sparsity_degree; ++i) {
                kept[i] = static_cast<uint8_t>(idx[i]);
                mask |= (1u << idx[i]);
            }
            std::sort(kept, kept + sparsity_degree);
            for (uint32_t i = 0; i < sparsity_degree; ++i)
                out.values.push_back(blk[kept[i]]);
            out.meta.push_back(mask);
        }
    }
    return out;
}


// B is always float (fp32 im2col output from BN-normalised feature maps).
// A values are itype_t; convert B to itype_t before each multiply to mirror
// the GPU's workspace quantisation.
static void matmul_cpu_sparseA(float *C, const SparseMat &A, const float *B,
                                uint32_t N, uint32_t sparsity_degree) {
    const uint32_t M = A.rows;
    const uint32_t K = A.cols;
    const uint32_t values_per_row = K * sparsity_degree / 4;
    const uint32_t meta_per_row   = K / 4;

    for (uint32_t m = 0; m < M; ++m) {
        const itype_t *row_vals = A.values.data() + (size_t)m * values_per_row;
        const uint8_t *row_meta = A.meta.data()   + (size_t)m * meta_per_row;
        float         *crow     = C + (size_t)m * N;

        for (uint32_t n = 0; n < N; ++n) {
            size_t  v_idx = 0;
            otype_t sum   = otype_t(0);
            for (uint32_t blk = 0; blk < K; blk += 4) {
                uint8_t mask = row_meta[blk / 4];
                if (!mask) { continue; }
                for (uint32_t i = 0; i < 4; ++i) {
                    if (!(mask & (1u << i))) continue;
                    itype_t a_val = row_vals[v_idx++];
                    itype_t b_val = to_itype(B[(size_t)(blk + i) * N + n]);
                    sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a_val, b_val, sum);
                }
            }
            crow[n] += static_cast<float>(sum);
        }
    }
}

static inline float im2col_get_pixel_cpu(float *im, int height, int width,
                                          int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0.0f;
    return im[col + width * (row + height * channel)];
}

static void im2col_cpu(float *data_im,
                       int channels, int height, int width,
                       int ksize, int stride, int pad,
                       int height_col, int width_col,
                       std::vector<float> &data_col) {
    int channels_col = channels * ksize * ksize;
    int N = height_col * width_col;
    data_col.resize(channels_col * N);

    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im     = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                float val  = im2col_get_pixel_cpu(data_im, height, width,
                                                   im_row, im_col, c_im, pad);
                data_col[c * N + h * width_col + w] = val;
            }
        }
    }
}

static inline float leaky_activate_cpu(float x)    { return (x > 0) ? x : 0.1f * x; }
static inline float logistic_activate_cpu(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void bn_activate_cpu(float *output, int M, int N,
                             float *biases, float *scales,
                             float *mean, float *variance,
                             int batch_normalize, ACTIVATION activation) {
    for (int f = 0; f < M; ++f) {
        int base = f * N;
        if (batch_normalize) {
            float mn      = mean[f];
            float v       = variance[f];
            float s       = scales[f];
            float b       = biases[f];
            float inv_std = 1.0f / std::sqrt(v + 0.00001f);
            for (int i = 0; i < N; ++i) {
                float val = (output[base + i] - mn) * inv_std * s + b;
                if (activation == ACTIVATION_LEAKY)        val = leaky_activate_cpu(val);
                else if (activation == ACTIVATION_LOGISTIC) val = logistic_activate_cpu(val);
                output[base + i] = val;
            }
        } else {
            float b = biases[f];
            for (int i = 0; i < N; ++i) {
                float val = output[base + i] + b;
                if (activation == ACTIVATION_LEAKY)        val = leaky_activate_cpu(val);
                else if (activation == ACTIVATION_LOGISTIC) val = logistic_activate_cpu(val);
                output[base + i] = val;
            }
        }
    }
}

struct layer_def_t {
    LAYER_TYPE type;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;          
    int size;       
    int stride;
    int pad;
    int batch_normalize;
    ACTIVATION activation;
};

static layer_def_t network_def[NUM_LAYERS] = {
    // Layer 0: Conv 3×3, 16 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT, NET_WIDTH, NET_CHANNELS,
      NET_HEIGHT, NET_WIDTH, 16,
      16, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 1: MaxPool 2×2
    { LAYER_MAXPOOL, NET_HEIGHT, NET_WIDTH, 16,
      NET_HEIGHT/2, NET_WIDTH/2, 16,
      0, 2, 2, 0, 0, ACTIVATION_LINEAR },
    // Layer 2: Conv 3×3, 32 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT/2, NET_WIDTH/2, 16,
      NET_HEIGHT/2, NET_WIDTH/2, 32,
      32, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 3: MaxPool 2×2
    { LAYER_MAXPOOL, NET_HEIGHT/2, NET_WIDTH/2, 32,
      NET_HEIGHT/4, NET_WIDTH/4, 32,
      0, 2, 2, 0, 0, ACTIVATION_LINEAR },
    // Layer 4: Conv 3×3, 32 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT/4, NET_WIDTH/4, 32,
      NET_HEIGHT/4, NET_WIDTH/4, 32,
      32, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 5: Conv 1×1, YOLO_OUTPUT_DEPTH filters, no BN, Linear
    { LAYER_CONVOLUTIONAL, NET_HEIGHT/4, NET_WIDTH/4, 32,
      NET_HEIGHT/4, NET_WIDTH/4, YOLO_OUTPUT_DEPTH,
      YOLO_OUTPUT_DEPTH, 1, 1, 0, 0, ACTIVATION_LINEAR },
    // Layer 6: YOLO detection decode
    { LAYER_YOLO, NET_HEIGHT/4, NET_WIDTH/4, YOLO_OUTPUT_DEPTH,
      NET_HEIGHT/4, NET_WIDTH/4, YOLO_OUTPUT_DEPTH,
      0, 0, 0, 0, 0, ACTIVATION_LINEAR }
};


static void forward_conv_cpu_sparse(
    layer_def_t *def, float *input, float *output,
    const SparseMat &W_sparse,   
    float *biases, float *scales,
    float *mean, float *variance,
    uint32_t sparsity_degree)
{
    int M     = def->n;
    int K     = def->c * def->size * def->size;
    int N     = def->out_h * def->out_w;
    uint32_t M_pad = W_sparse.rows;  
    uint32_t K_pad = W_sparse.cols;

    std::vector<float> col;
    im2col_cpu(input, def->c, def->h, def->w,
               def->size, def->stride, def->pad,
               def->out_h, def->out_w, col);

    if ((uint32_t)K < K_pad)
        col.resize((size_t)K_pad * N, 0.0f);

    std::fill(output, output + (size_t)M_pad * N, 0.0f);
    matmul_cpu_sparseA(output, W_sparse, col.data(), N, sparsity_degree);

    bn_activate_cpu(output, M, N, biases, scales, mean, variance,
                    def->batch_normalize, def->activation);
}

static void forward_maxpool_cpu(layer_def_t *def, float *input, float *output) {
    int w = def->w, h = def->h, c = def->c;
    int ow = def->out_w, oh = def->out_h;
    int size = def->size, stride = def->stride, pad = def->pad;
    int w_off = -pad/2, h_off = -pad/2;

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                int out_idx = j + ow * (i + oh * k);
                float max_val = -1e38f;
                for (int n = 0; n < size; ++n) {
                    for (int m = 0; m < size; ++m) {
                        int ch = h_off + i * stride + n;
                        int cw = w_off + j * stride + m;
                        if (ch >= 0 && ch < h && cw >= 0 && cw < w)
                            max_val = std::max(max_val, input[cw + w * (ch + h * k)]);
                    }
                }
                output[out_idx] = max_val;
            }
        }
    }
}

static void forward_yolo_cpu(layer_def_t *def, float *input, float *output,
                              int num_anchors, int num_classes) {
    int w = def->out_w, h = def->out_h;
    int outputs = w * h * def->out_c;
    std::memcpy(output, input, outputs * sizeof(float));

    for (int n = 0; n < num_anchors; ++n) {
        int base = n * (4 + 1 + num_classes) * w * h;
        for (int i = 0; i < 2 * w * h; ++i)
            output[base + i] = logistic_activate_cpu(output[base + i]);
        int obj_base = base + 4 * w * h;
        for (int i = 0; i < (1 + num_classes) * w * h; ++i)
            output[obj_base + i] = logistic_activate_cpu(output[obj_base + i]);
    }
}

static void init_weights_xavier(std::vector<itype_t> &weights, int fan_in) {
    float scale = std::sqrt(2.0f / (float)fan_in);
    for (auto &v : weights)
        v = to_itype((static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale);
}

static bool compare_float(float a, float b, int index, int errors) {
    // Multi-layer sparse WMMA inference accumulates rounding differences
    // between the tile-ordered GPU computation and the sequential CPU reference.
    // Use relative-error tolerance that is tight enough to catch wrong results
    // while tolerating normal fp32/fp16/bf16 quantisation noise.
    //   fp32 weights → 1 % relative tolerance
    //   fp16/bf16    → 5 % relative tolerance
    constexpr float rel_tol =
        std::is_same_v<itype_t, float> ? 0.01f : 0.05f;
    float abs_err = std::abs(a - b);
    float tol = std::abs(b) * rel_tol + 1e-6f;
    if (abs_err > tol) {
        if (errors < 100)
            printf("*** error: [%d] expected=%f, actual=%f (rel_err=%.2f%%)\n",
                   index, (double)b, (double)a,
                   100.0 * (double)abs_err / ((double)std::abs(b) + 1e-10));
        return false;
    }
    return true;
}

const char *kernel_file = "kernel.vxbin";

vx_device_h device            = nullptr;
vx_buffer_h krnl_buffer       = nullptr;
vx_buffer_h args_buffer       = nullptr;
vx_buffer_h input_buffer      = nullptr;
vx_buffer_h workspace_buffer  = nullptr;
vx_buffer_h layer_configs_buf = nullptr;

struct layer_buffers_t {
    vx_buffer_h weights;         
    vx_buffer_h biases;          
    vx_buffer_h scales;          
    vx_buffer_h rolling_mean;    
    vx_buffer_h rolling_variance;
    vx_buffer_h output;          
};

layer_buffers_t layer_bufs[NUM_LAYERS];
kernel_arg_t    kernel_arg = {};

void cleanup() {
    if (device) {
        vx_mem_free(input_buffer);
        vx_mem_free(workspace_buffer);
        vx_mem_free(layer_configs_buf);
        for (int i = 0; i < NUM_LAYERS; ++i) {
            if (layer_bufs[i].weights)          vx_mem_free(layer_bufs[i].weights);
            if (layer_bufs[i].biases)           vx_mem_free(layer_bufs[i].biases);
            if (layer_bufs[i].scales)           vx_mem_free(layer_bufs[i].scales);
            if (layer_bufs[i].rolling_mean)     vx_mem_free(layer_bufs[i].rolling_mean);
            if (layer_bufs[i].rolling_variance) vx_mem_free(layer_bufs[i].rolling_variance);
            if (layer_bufs[i].output)           vx_mem_free(layer_bufs[i].output);
        }
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

static void show_usage() {
    std::cout << "Vortex YOLOv3-Tiny Sparse Inference Test." << std::endl;
    std::cout << "Usage: [-k kernel_file] [-p sparsity] [-h]" << std::endl;
    std::cout << "  -p  Sparsity degree: 1 (1:4) or 2 (2:4) [default: 2]" << std::endl;
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "k:p:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'p':
            g_sparsity_degree = atoi(optarg);
            if (g_sparsity_degree != 1 && g_sparsity_degree != 2) {
                std::cerr << "Error: sparsity must be 1 or 2" << std::endl;
                show_usage(); exit(-1);
            }
            break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    std::srand(42);

    std::cout << "=== Vortex YOLOv3-Tiny Sparse Inference Test ===" << std::endl;
    std::cout << "Network  : YOLOv3-Tiny (scaled-down)" << std::endl;
    std::cout << "Input    : " << NET_WIDTH << "x" << NET_HEIGHT << "x" << NET_CHANNELS << std::endl;
    std::cout << "Classes  : " << NUM_CLASSES << ", Anchors: " << NUM_ANCHORS << std::endl;
    std::cout << "Sparsity : " << g_sparsity_degree << ":4" << std::endl;
    std::cout << "WMMA tile: M=" << sparse_cfg_t::tileM
              << " N=" << sparse_cfg_t::tileN
              << " K=" << sparse_cfg_t::tileK << std::endl;
    std::cout << "dtype    : " << vt::ITYPE::name << " -> " << vt::OTYPE::name << std::endl;

    // ── Open device ─────────────────────────────────────────────────────────
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));

    // Validate warp size
    uint64_t num_threads_dev;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads_dev));
    if (num_threads_dev != NUM_THREADS) {
        std::cout << "Error: device warp size (" << num_threads_dev
                  << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
        cleanup();
        return -1;
    }

    // ── Prepare host-side data ───────────────────────────────────────────────

    std::vector<uint32_t> M_pad_arr(NUM_LAYERS, 0);
    std::vector<uint32_t> K_pad_arr(NUM_LAYERS, 0);
    std::vector<uint32_t> N_pad_arr(NUM_LAYERS, 0);

    std::vector<SparseMat> h_weights_sparse(NUM_LAYERS);

    std::vector<std::vector<float>> h_biases(NUM_LAYERS);
    std::vector<std::vector<float>> h_scales(NUM_LAYERS);
    std::vector<std::vector<float>> h_mean(NUM_LAYERS);
    std::vector<std::vector<float>> h_variance(NUM_LAYERS);
    std::vector<std::vector<float>> h_outputs(NUM_LAYERS);

    size_t max_workspace_bytes = 0; 

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];

        uint32_t M = (uint32_t)def->n;
        uint32_t N = (uint32_t)(def->out_h * def->out_w);

        if (def->type == LAYER_CONVOLUTIONAL) {
            uint32_t K = (uint32_t)(def->c * def->size * def->size);

            uint32_t Mp = align_up32(M, sparse_cfg_t::tileM);
            uint32_t Kp = align_up32(K, sparse_cfg_t::tileK);
            uint32_t Np = align_up32(N, sparse_cfg_t::tileN); 
            M_pad_arr[i] = Mp;
            K_pad_arr[i] = Kp;
            N_pad_arr[i] = Np;

            size_t ws_bytes = (size_t)Kp * Np * sizeof(itype_t);
            if (ws_bytes > max_workspace_bytes)
                max_workspace_bytes = ws_bytes;

            if (Kp % 4 != 0) {
                std::cout << "Error: layer " << i << " K_pad=" << Kp
                          << " is not a multiple of 4!" << std::endl;
                cleanup();
                return -1;
            }
            if (Mp % sparse_cfg_t::tileM || Np % sparse_cfg_t::tileN) {
                std::cout << "Error: layer " << i << " M_pad=" << Mp
                          << " or N_pad=" << Np << " misaligned with tile!" << std::endl;
                cleanup();
                return -1;
            }

            std::vector<itype_t> h_weights_dense(M * K);
            init_weights_xavier(h_weights_dense, K);

            std::vector<itype_t> dense_padded((size_t)Mp * Kp, itype_t(0));
            for (uint32_t row = 0; row < M; ++row)
                for (uint32_t col = 0; col < K; ++col)
                    dense_padded[row * Kp + col] = h_weights_dense[row * K + col];

            h_weights_sparse[i] = pruneAndPack(dense_padded, Mp, Kp, g_sparsity_degree);

            h_biases[i].resize(M, 0.0f);

            if (def->batch_normalize) {
                h_scales[i].resize(M);
                h_mean[i].resize(M);
                h_variance[i].resize(M);
                for (uint32_t j = 0; j < M; ++j) {
                    h_mean[i][j]     = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                    h_variance[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.5f + 0.1f;
                    h_scales[i][j]   = 0.5f + static_cast<float>(rand()) / RAND_MAX;
                    h_biases[i][j]   = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                }
            }

            h_outputs[i].resize((size_t)Mp * N, 0.0f);
        } else {
            int out_size = def->out_h * def->out_w * def->out_c;
            h_outputs[i].resize(out_size, 0.0f);
        }
    }

    // Random input image
    int input_size = NET_HEIGHT * NET_WIDTH * NET_CHANNELS;
    std::vector<float> h_input(input_size);
    for (int i = 0; i < input_size; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    // ── Allocate device memory ───────────────────────────────────────────────
    std::cout << "Allocating device memory..." << std::endl;

    RT_CHECK(vx_mem_alloc(device, input_size * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));

    RT_CHECK(vx_mem_alloc(device, max_workspace_bytes, VX_MEM_READ_WRITE, &workspace_buffer));
    RT_CHECK(vx_mem_address(workspace_buffer, &kernel_arg.workspace_addr));
    
    std::vector<float> h_workspace_zeros(max_workspace_bytes / sizeof(float), 0.0f);
    RT_CHECK(vx_copy_to_dev(workspace_buffer, h_workspace_zeros.data(), 0, max_workspace_bytes));

    std::vector<layer_config_t> layer_configs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t   *def = &network_def[i];
        layer_config_t *cfg = &layer_configs[i];

        cfg->type            = def->type;
        cfg->h = def->h;  cfg->w = def->w;  cfg->c = def->c;
        cfg->out_h = def->out_h;  cfg->out_w = def->out_w;  cfg->out_c = def->out_c;
        cfg->n               = def->n;
        cfg->size            = def->size;
        cfg->stride          = def->stride;
        cfg->pad             = def->pad;
        cfg->batch_normalize = def->batch_normalize;
        cfg->activation      = def->activation;

        if (def->type == LAYER_CONVOLUTIONAL) {
            uint32_t Mp = M_pad_arr[i];
            uint32_t Np = N_pad_arr[i];
            uint32_t N  = (uint32_t)(def->out_h * def->out_w);
            const SparseMat &sw = h_weights_sparse[i];

            uint32_t out_bytes = Mp * Np * (uint32_t)sizeof(float);
            RT_CHECK(vx_mem_alloc(device, out_bytes, VX_MEM_READ_WRITE, &layer_bufs[i].output));
            RT_CHECK(vx_mem_address(layer_bufs[i].output, &cfg->output_addr));

            size_t values_bytes = sw.values.size() * sizeof(itype_t);
            size_t meta_offset  = align_up(values_bytes, sizeof(uint32_t));
            size_t meta_bytes   = sw.meta.size() * sizeof(uint32_t);
            size_t sparse_bytes = meta_offset + meta_bytes;

            std::vector<uint8_t> packed(sparse_bytes, 0);
            memcpy(packed.data(), sw.values.data(), values_bytes);
            auto meta_words = reinterpret_cast<uint32_t *>(packed.data() + meta_offset);
            for (size_t j = 0; j < sw.meta.size(); ++j)
                meta_words[j] = static_cast<uint32_t>(sw.meta[j]);

            RT_CHECK(vx_mem_alloc(device, sparse_bytes, VX_MEM_READ, &layer_bufs[i].weights));
            RT_CHECK(vx_mem_address(layer_bufs[i].weights, &cfg->weights_addr));
            RT_CHECK(vx_copy_to_dev(layer_bufs[i].weights, packed.data(), 0, sparse_bytes));

            std::cout << "  Layer " << i << ": " << def->c << "x" << def->size << "x" << def->size
                      << " -> " << def->n << " filters"
                      << " (M_pad=" << Mp << " K_pad=" << K_pad_arr[i]
                      << " sparse " << g_sparsity_degree << ":4, "
                      << sparse_bytes << " bytes)" << std::endl;

            uint32_t b_bytes = def->n * (uint32_t)sizeof(float);
            RT_CHECK(vx_mem_alloc(device, b_bytes, VX_MEM_READ, &layer_bufs[i].biases));
            RT_CHECK(vx_mem_address(layer_bufs[i].biases, &cfg->biases_addr));
            RT_CHECK(vx_copy_to_dev(layer_bufs[i].biases, h_biases[i].data(), 0, b_bytes));

            if (def->batch_normalize) {
                uint32_t bn_bytes = def->n * (uint32_t)sizeof(float);

                RT_CHECK(vx_mem_alloc(device, bn_bytes, VX_MEM_READ, &layer_bufs[i].scales));
                RT_CHECK(vx_mem_address(layer_bufs[i].scales, &cfg->scales_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].scales, h_scales[i].data(), 0, bn_bytes));

                RT_CHECK(vx_mem_alloc(device, bn_bytes, VX_MEM_READ, &layer_bufs[i].rolling_mean));
                RT_CHECK(vx_mem_address(layer_bufs[i].rolling_mean, &cfg->rolling_mean_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].rolling_mean, h_mean[i].data(), 0, bn_bytes));

                RT_CHECK(vx_mem_alloc(device, bn_bytes, VX_MEM_READ, &layer_bufs[i].rolling_variance));
                RT_CHECK(vx_mem_address(layer_bufs[i].rolling_variance, &cfg->rolling_variance_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].rolling_variance, h_variance[i].data(), 0, bn_bytes));
            }

            (void)N;
        } else {
            int out_size = def->out_h * def->out_w * def->out_c;
            RT_CHECK(vx_mem_alloc(device, out_size * sizeof(float), VX_MEM_READ_WRITE, &layer_bufs[i].output));
            RT_CHECK(vx_mem_address(layer_bufs[i].output, &cfg->output_addr));
        }
    }

    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_configs_buf));
    RT_CHECK(vx_mem_address(layer_configs_buf, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_configs_buf, layer_configs.data(), 0,
                            NUM_LAYERS * sizeof(layer_config_t)));
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, input_size * sizeof(float)));

    kernel_arg.num_layers      = NUM_LAYERS;
    kernel_arg.net_w           = NET_WIDTH;
    kernel_arg.net_h           = NET_HEIGHT;
    kernel_arg.num_classes     = NUM_CLASSES;
    kernel_arg.num_anchors     = NUM_ANCHORS;
    kernel_arg.sparsity_degree = g_sparsity_degree;
    kernel_arg.anchors[0]  = ANCHOR_W0; kernel_arg.anchors[1] = ANCHOR_H0;
    kernel_arg.anchors[2]  = ANCHOR_W1; kernel_arg.anchors[3] = ANCHOR_H1;
    kernel_arg.anchors[4]  = ANCHOR_W2; kernel_arg.anchors[5] = ANCHOR_H2;

    // ── Upload kernel binary + args ──────────────────────────────────────────
    std::cout << "Uploading kernel binary..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    std::cout << "Uploading kernel arguments..." << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // ── Run device ───────────────────────────────────────────────────────────
    std::cout << "Starting YOLO-Sparse inference on device..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    // ── Download YOLO output ─────────────────────────────────────────────────
    layer_def_t *yolo_def = &network_def[NUM_LAYERS - 1];
    int yolo_out_size = yolo_def->out_h * yolo_def->out_w * yolo_def->out_c;
    std::vector<float> h_dev_output(yolo_out_size);
    RT_CHECK(vx_copy_from_dev(h_dev_output.data(), layer_bufs[NUM_LAYERS - 1].output,
                              0, yolo_out_size * sizeof(float)));

    // ── CPU reference forward pass ───────────────────────────────────────────
    std::cout << "Computing CPU reference (sparse fp32)..." << std::endl;

    std::vector<std::vector<float>> cpu_outputs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];
        uint32_t Mp = M_pad_arr[i];
        uint32_t N  = (uint32_t)(def->out_h * def->out_w);
        int out_size = def->out_h * def->out_w * def->out_c;

        if (def->type == LAYER_CONVOLUTIONAL) {
            cpu_outputs[i].assign((size_t)Mp * N, 0.0f);
        } else {
            cpu_outputs[i].assign(out_size, 0.0f);
        }

        float *layer_input = (i == 0) ? h_input.data() : cpu_outputs[i - 1].data();

        switch (def->type) {
        case LAYER_CONVOLUTIONAL:
            forward_conv_cpu_sparse(
                def, layer_input, cpu_outputs[i].data(),
                h_weights_sparse[i],
                h_biases[i].data(),
                def->batch_normalize ? h_scales[i].data()    : nullptr,
                def->batch_normalize ? h_mean[i].data()      : nullptr,
                def->batch_normalize ? h_variance[i].data()  : nullptr,
                g_sparsity_degree);
            break;

        case LAYER_MAXPOOL:
            forward_maxpool_cpu(def, layer_input, cpu_outputs[i].data());
            break;

        case LAYER_YOLO:
            forward_yolo_cpu(def, layer_input, cpu_outputs[i].data(),
                             NUM_ANCHORS, NUM_CLASSES);
            break;
        }
    }

    // ── Verify results ───────────────────────────────────────────────────────
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float *cpu_yolo = cpu_outputs[NUM_LAYERS - 1].data();

    for (int i = 0; i < yolo_out_size; ++i) {
        if (!compare_float(h_dev_output[i], cpu_yolo[i], i, errors))
            ++errors;
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    std::cout << "Cleaning up..." << std::endl;
    cleanup();

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
