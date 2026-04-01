// ---------------------------------------------------------------------------
// YOLOv3-Tiny TCU Host Driver
// https://github.com/AlexeyAB/darknet/tree/master/src
// ---------------------------------------------------------------------------

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <vortex.h>
#include <tensor_cfg.h>
#include <rvfloats.h>
#include <util.h>
#include "common.h"

#define FLOAT_ULP 6000

#define RT_CHECK(_expr)                                          \
  do {                                                           \
    int _ret = (_expr);                                          \
    if (0 == _ret) break;                                        \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                   \
    exit(-1);                                                    \
  } while (false)

using namespace vortex;
namespace vt = vortex::tensor;

using tcu_cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;

static inline uint32_t align_up(uint32_t x, uint32_t a) {
    return (x + a - 1u) / a * a;
}

static inline uint16_t f32_to_fp16(float f) {
    return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr);
}
static inline float fp16_to_f32(uint16_t h) {
    return bit_cast<float>(rv_htof_s(h, 0, nullptr));
}

const char *kernel_file = "kernel.vxbin";

// ---------------------------------------------------------------------------
// Device handles
// ---------------------------------------------------------------------------
vx_device_h device             = nullptr;
vx_buffer_h krnl_buffer        = nullptr;
vx_buffer_h args_buffer        = nullptr;
vx_buffer_h input_buffer       = nullptr;
vx_buffer_h workspace_buffer   = nullptr;
vx_buffer_h layer_configs_buf  = nullptr;

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

// ---------------------------------------------------------------------------
// Network architecture // @mitul: same as yolov3tiny
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CPU reference 
// ---------------------------------------------------------------------------

// im2col
static inline float im2col_get_pixel_cpu(float *im, int height, int width,
                                         int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0.0f;
    return im[col + width * (row + height * channel)];
}

static void im2col_cpu_fp16rt(float *data_im,
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
                val = fp16_to_f32(f32_to_fp16(val));
                data_col[c * N + h * width_col + w] = val;
            }
        }
    }
}

// GEMM 
static void gemm_nn_cpu_tcu(int M, int N, int K, int A_stride,
                             const std::vector<uint16_t> &A_fp16,  
                             const std::vector<float>    &B,        
                             float *C) {                           
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = fp16_to_f32(A_fp16[i * A_stride + k]); 
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

static inline float leaky_activate_cpu(float x)    { return (x > 0) ? x : 0.1f * x; }
static inline float logistic_activate_cpu(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Fused BN + scale + bias + activate – // @mitul: same as yolov3tiny
static void bn_activate_cpu(float *output, int M, int N,
                             float *biases, float *scales,
                             float *mean, float *variance,
                             int batch_normalize, ACTIVATION activation) {
    for (int f = 0; f < M; ++f) {
        int base = f * N;
        if (batch_normalize) {
            float mn     = mean[f];
            float v      = variance[f];
            float s      = scales[f];
            float b      = biases[f];
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

// Conv layer CPU reference
static void forward_conv_cpu_tcu(layer_def_t *def, float *input, float *output,
                                  const std::vector<uint16_t> &weights_fp16,
                                  uint32_t K_pad,
                                  float *biases, float *scales,
                                  float *mean, float *variance,
                                  float *) {
    int M       = def->n;
    int K       = def->c * def->size * def->size;
    int N       = def->out_h * def->out_w;

    std::vector<float> col;
    im2col_cpu_fp16rt(input, def->c, def->h, def->w,
                      def->size, def->stride, def->pad,
                      def->out_h, def->out_w, col);

    std::fill(output, output + M * N, 0.0f);
    gemm_nn_cpu_tcu(M, N, K, (int)K_pad, weights_fp16, col, output);

    bn_activate_cpu(output, M, N, biases, scales, mean, variance,
                    def->batch_normalize, def->activation);
}

// MaxPool CPU reference // @mitul: same as yolov3tiny
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

static int entry_index_cpu(int w, int h, int n_anchors, int classes,
                            int batch, int location, int entry) {
    int n   = location / (w * h);
    int loc = location % (w * h);
    return batch * (n_anchors * w * h * (4 + classes + 1))
         + n * w * h * (4 + classes + 1)
         + entry * w * h + loc;
}

// YOLO decode CPU reference // @mitul: same as yolov3tiny
static void forward_yolo_cpu(layer_def_t *def, float *input, float *output,
                              int num_anchors, int num_classes) {
    int w = def->out_w, h = def->out_h;
    int outputs = w * h * def->out_c;
    std::memcpy(output, input, outputs * sizeof(float));

    for (int n = 0; n < num_anchors; ++n) {
        int bbox_idx = entry_index_cpu(w, h, num_anchors, num_classes, 0, n * w * h, 0);
        for (int i = 0; i < 2 * w * h; ++i)
            output[bbox_idx + i] = logistic_activate_cpu(output[bbox_idx + i]);

        int obj_idx  = entry_index_cpu(w, h, num_anchors, num_classes, 0, n * w * h, 4);
        for (int i = 0; i < (1 + num_classes) * w * h; ++i)
            output[obj_idx + i] = logistic_activate_cpu(output[obj_idx + i]);
    }
}

// ---------------------------------------------------------------------------
// Weight initialisation helpers
// ---------------------------------------------------------------------------
static void init_weights_xavier(std::vector<float> &weights, int fan_in) {
    float scale = std::sqrt(2.0f / (float)fan_in);
    for (float &v : weights)
        v = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
}

// ---------------------------------------------------------------------------
// ULP comparison
// ---------------------------------------------------------------------------
static bool compare_float(float a, float b, int index, int errors) {
    auto ordered = [](float f) -> int64_t {
        union { float f; uint32_t u; } tmp;
        tmp.f = f;
        return (tmp.u & 0x80000000u) ? -(int64_t)(tmp.u & 0x7FFFFFFFu) : (int64_t)tmp.u;
    };
    auto d = std::abs(ordered(a) - ordered(b));
    if (d > FLOAT_ULP) {
        if (errors < 100)
            printf("*** error: [%d] expected=%f, actual=%f (diff=%d ULP)\n",
                   index, b, a, (int)d);
        return false;
    }
    return true;
}

static void show_usage() {
    std::cout << "Vortex YOLOv3-Tiny TCU Inference Test." << std::endl;
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "k:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    std::srand(42);

    std::cout << "=== Vortex YOLOv3-Tiny TCU Inference Test ===" << std::endl;
    std::cout << "Network: YOLOv3-Tiny (scaled-down)" << std::endl;
    std::cout << "Input  : " << NET_WIDTH << "x" << NET_HEIGHT << "x" << NET_CHANNELS << std::endl;
    std::cout << "Classes: " << NUM_CLASSES << ", Anchors: " << NUM_ANCHORS << std::endl;
    std::cout << "TCU tiles: M=" << tcu_cfg::tileM
              << " N=" << tcu_cfg::tileN
              << " K=" << tcu_cfg::tileK << std::endl;

    // ── Open device ─────────────────────────────────────────────────────────
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));

    // Check TCU support
    uint64_t isa_flags;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if (!(isa_flags & VX_ISA_EXT_TCU)) {
        std::cout << "TCU extension not supported on this device!" << std::endl;
        cleanup();
        return -1;
    }

    // Validate warp size matches NUM_THREADS
    uint64_t num_threads_dev;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads_dev));
    if (num_threads_dev != NUM_THREADS) {
        std::cout << "Error: device warp size (" << num_threads_dev
                  << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
        cleanup();
        return -1;
    }

    std::vector<std::vector<float>>    h_weights(NUM_LAYERS);
    std::vector<std::vector<float>>    h_biases(NUM_LAYERS);
    std::vector<std::vector<float>>    h_scales(NUM_LAYERS);
    std::vector<std::vector<float>>    h_mean(NUM_LAYERS);
    std::vector<std::vector<float>>    h_variance(NUM_LAYERS);
    std::vector<std::vector<float>>    h_outputs(NUM_LAYERS);

    std::vector<std::vector<uint16_t>> h_weights_fp16(NUM_LAYERS);

    std::vector<uint32_t> M_pad_arr(NUM_LAYERS, 0);
    std::vector<uint32_t> K_pad_arr(NUM_LAYERS, 0);
    std::vector<uint32_t> N_pad_arr(NUM_LAYERS, 0);

    uint32_t max_workspace_bytes = 0; 

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];

        uint32_t M = def->n;
        uint32_t K = (def->type == LAYER_CONVOLUTIONAL)
                     ? (uint32_t)(def->c * def->size * def->size) : 0u;
        uint32_t N = (uint32_t)(def->out_h * def->out_w);

        if (def->type == LAYER_CONVOLUTIONAL) {
            uint32_t Mp = align_up(M, tcu_cfg::tileM);
            uint32_t Kp = align_up(K, tcu_cfg::tileK);
            uint32_t Np = align_up(N, tcu_cfg::tileN);   
            M_pad_arr[i] = Mp;
            K_pad_arr[i] = Kp;
            N_pad_arr[i] = Np;

            uint32_t ws_bytes = Kp * Np * (uint32_t)sizeof(uint16_t);
            if (ws_bytes > max_workspace_bytes)
                max_workspace_bytes = ws_bytes;

            int nweights = M * K;
            h_weights[i].resize(nweights);
            init_weights_xavier(h_weights[i], def->c * def->size * def->size);

            h_weights_fp16[i].assign(Mp * Kp, (uint16_t)0);
            for (uint32_t row = 0; row < M; ++row) {
                for (uint32_t col = 0; col < K; ++col) {
                    h_weights_fp16[i][row * Kp + col] =
                        f32_to_fp16(h_weights[i][row * K + col]);
                }
            }

            h_biases[i].resize(M, 0.0f);

            if (def->batch_normalize) {
                h_scales[i].resize(M);
                h_mean[i].resize(M);
                h_variance[i].resize(M);
                for (int j = 0; j < (int)M; ++j) {
                    h_mean[i][j]     = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                    h_variance[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.5f + 0.1f;
                    h_scales[i][j]   = 0.5f + static_cast<float>(rand()) / RAND_MAX;
                    h_biases[i][j]   = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                }
            }

            h_outputs[i].resize(Mp * N, 0.0f);
        } else {
            int out_size = def->out_h * def->out_w * def->out_c;
            h_outputs[i].resize(out_size, 0.0f);
        }
    }

    int input_size = NET_HEIGHT * NET_WIDTH * NET_CHANNELS;
    std::vector<float> h_input(input_size);
    for (int i = 0; i < input_size; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    std::cout << "Allocating device memory..." << std::endl;

    RT_CHECK(vx_mem_alloc(device, input_size * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));

    RT_CHECK(vx_mem_alloc(device, max_workspace_bytes, VX_MEM_READ_WRITE, &workspace_buffer));
    RT_CHECK(vx_mem_address(workspace_buffer, &kernel_arg.workspace_addr));

    std::vector<layer_config_t> layer_configs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t   *def = &network_def[i];
        layer_config_t *cfg = &layer_configs[i];

        cfg->type           = def->type;
        cfg->h = def->h;  cfg->w = def->w;  cfg->c = def->c;
        cfg->out_h = def->out_h;  cfg->out_w = def->out_w;  cfg->out_c = def->out_c;
        cfg->n              = def->n;
        cfg->size           = def->size;
        cfg->stride         = def->stride;
        cfg->pad            = def->pad;
        cfg->batch_normalize = def->batch_normalize;
        cfg->activation     = def->activation;

        if (def->type == LAYER_CONVOLUTIONAL) {
            uint32_t Mp = M_pad_arr[i];
            uint32_t Kp = K_pad_arr[i];
            uint32_t N  = (uint32_t)(def->out_h * def->out_w);

            // Output: M_pad × N floats
            uint32_t out_bytes = Mp * N * (uint32_t)sizeof(float);
            RT_CHECK(vx_mem_alloc(device, out_bytes, VX_MEM_READ_WRITE, &layer_bufs[i].output));
            RT_CHECK(vx_mem_address(layer_bufs[i].output, &cfg->output_addr));

            // Weights: fp16[M_pad × K_pad]
            uint32_t w_bytes = Mp * Kp * (uint32_t)sizeof(uint16_t);
            RT_CHECK(vx_mem_alloc(device, w_bytes, VX_MEM_READ, &layer_bufs[i].weights));
            RT_CHECK(vx_mem_address(layer_bufs[i].weights, &cfg->weights_addr));
            RT_CHECK(vx_copy_to_dev(layer_bufs[i].weights, h_weights_fp16[i].data(), 0, w_bytes));

            // Biases: float[n]
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
        } else {
            // MaxPool
            int out_size = def->out_h * def->out_w * def->out_c;
            RT_CHECK(vx_mem_alloc(device, out_size * sizeof(float), VX_MEM_READ_WRITE, &layer_bufs[i].output));
            RT_CHECK(vx_mem_address(layer_bufs[i].output, &cfg->output_addr));
        }
    }

    // Upload layer config array
    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_configs_buf));
    RT_CHECK(vx_mem_address(layer_configs_buf, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_configs_buf, layer_configs.data(), 0,
                            NUM_LAYERS * sizeof(layer_config_t)));

    // Upload input image
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, input_size * sizeof(float)));

    // Set remaining kernel args
    kernel_arg.num_layers  = NUM_LAYERS;
    kernel_arg.net_w       = NET_WIDTH;
    kernel_arg.net_h       = NET_HEIGHT;
    kernel_arg.num_classes = NUM_CLASSES;
    kernel_arg.num_anchors = NUM_ANCHORS;
    kernel_arg.anchors[0]  = ANCHOR_W0; kernel_arg.anchors[1] = ANCHOR_H0;
    kernel_arg.anchors[2]  = ANCHOR_W1; kernel_arg.anchors[3] = ANCHOR_H1;
    kernel_arg.anchors[4]  = ANCHOR_W2; kernel_arg.anchors[5] = ANCHOR_H2;

    // ── Upload kernel binary + args ──────────────────────────────────────────
    std::cout << "Uploading kernel binary..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    std::cout << "Uploading kernel arguments..." << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // ── Run device ───────────────────────────────────────────────────────────
    std::cout << "Starting YOLO-TCU inference on device..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    // ── Download YOLO output ─────────────────────────────────────────────────
    layer_def_t *yolo_def  = &network_def[NUM_LAYERS - 1];
    int yolo_out_size = yolo_def->out_h * yolo_def->out_w * yolo_def->out_c;
    std::vector<float> h_dev_output(yolo_out_size);
    RT_CHECK(vx_copy_from_dev(h_dev_output.data(), layer_bufs[NUM_LAYERS - 1].output,
                              0, yolo_out_size * sizeof(float)));

    // ── CPU reference forward pass ───────────────────────────────────────────
    std::cout << "Computing CPU reference (fp16 TCU precision)..." << std::endl;

    // Scratch workspace for CPU im2col
    std::vector<float> cpu_ws_unused(max_workspace_bytes / sizeof(uint16_t), 0.0f);
    std::vector<std::vector<float>> cpu_outputs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];
        uint32_t Mp = M_pad_arr[i];   
        uint32_t N  = (uint32_t)(def->out_h * def->out_w);
        int out_size = def->out_h * def->out_w * def->out_c;

        if (def->type == LAYER_CONVOLUTIONAL) {
            cpu_outputs[i].assign(Mp * N, 0.0f);
        } else {
            cpu_outputs[i].assign(out_size, 0.0f);
        }

        float *layer_input = (i == 0) ? h_input.data() : cpu_outputs[i - 1].data();

        switch (def->type) {
        case LAYER_CONVOLUTIONAL:
            forward_conv_cpu_tcu(
                def, layer_input, cpu_outputs[i].data(),
                h_weights_fp16[i],
                K_pad_arr[i],
                h_biases[i].data(),
                def->batch_normalize ? h_scales[i].data()   : nullptr,
                def->batch_normalize ? h_mean[i].data()     : nullptr,
                def->batch_normalize ? h_variance[i].data() : nullptr,
                cpu_ws_unused.data());
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
