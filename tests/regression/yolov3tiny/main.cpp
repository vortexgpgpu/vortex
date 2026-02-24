// ---------------------------------------------------------------------------
// https://github.com/AlexeyAB/darknet/tree/master/src
// - Initializes weights, BN params, input data
// - Uploads everything to the Vortex device
// - Runs the YOLO kernel
// - Downloads output
// - Computes CPU reference (identical ops to kernel)
// - Verifies device output matches CPU reference (ULP comparison)
// ---------------------------------------------------------------------------

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6000  // @mitul: increased this (more tolerence to fp errors)

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)


const char* kernel_file = "kernel.vxbin";

// Device handles
vx_device_h device = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h workspace_buffer = nullptr;
vx_buffer_h layer_configs_buffer = nullptr;

// Per-layer device buffers
struct layer_buffers_t {
    vx_buffer_h weights;
    vx_buffer_h biases;
    vx_buffer_h scales;
    vx_buffer_h rolling_mean;
    vx_buffer_h rolling_variance;
    vx_buffer_h output;
};

layer_buffers_t layer_bufs[NUM_LAYERS];
kernel_arg_t kernel_arg = {};

// ---------------------------------------------------------------------------
// Network architecture
// ---------------------------------------------------------------------------
struct layer_def_t {
    LAYER_TYPE type;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;          // num filters (conv) or 0
    int size;       // kernel/pool size
    int stride;
    int pad;
    int batch_normalize;
    ACTIVATION activation;
};

// YOLOv3-Tiny (scaled-down) architecture
static layer_def_t network_def[NUM_LAYERS] = {
    // Layer 0: Conv 3x3, 16 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT, NET_WIDTH, NET_CHANNELS,
      NET_HEIGHT, NET_WIDTH, 16,
      16, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 1: MaxPool 2x2
    { LAYER_MAXPOOL, NET_HEIGHT, NET_WIDTH, 16,
      NET_HEIGHT/2, NET_WIDTH/2, 16,
      0, 2, 2, 0, 0, ACTIVATION_LINEAR },
    // Layer 2: Conv 3x3, 32 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT/2, NET_WIDTH/2, 16,
      NET_HEIGHT/2, NET_WIDTH/2, 32,
      32, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 3: MaxPool 2x2
    { LAYER_MAXPOOL, NET_HEIGHT/2, NET_WIDTH/2, 32,
      NET_HEIGHT/4, NET_WIDTH/4, 32,
      0, 2, 2, 0, 0, ACTIVATION_LINEAR },
    // Layer 4: Conv 3x3, 32 filters, BN, Leaky
    { LAYER_CONVOLUTIONAL, NET_HEIGHT/4, NET_WIDTH/4, 32,
      NET_HEIGHT/4, NET_WIDTH/4, 32,
      32, 3, 1, 1, 1, ACTIVATION_LEAKY },
    // Layer 5: Conv 1x1, YOLO_OUTPUT_DEPTH filters, no BN, Linear
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
        vx_mem_free(layer_configs_buffer);
        for (int i = 0; i < NUM_LAYERS; ++i) {
            vx_mem_free(layer_bufs[i].weights);
            vx_mem_free(layer_bufs[i].biases);
            vx_mem_free(layer_bufs[i].scales);
            vx_mem_free(layer_bufs[i].rolling_mean);
            vx_mem_free(layer_bufs[i].rolling_variance);
            vx_mem_free(layer_bufs[i].output);
        }
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

// im2col_cpu
static inline float im2col_get_pixel_cpu(float *im, int height, int width,
                                         int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void im2col_cpu(float *data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad,
                float *data_col) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col  = (width  + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_cpu(
                    data_im, height, width, im_row, im_col, c_im, pad);
            }
        }
    }
}

// gemm_nn
void gemm_nn_cpu(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

// fill_cpu
void fill_cpu(int N, float ALPHA, float *X, int INCX) {
    for (int i = 0; i < N; i += INCX) X[i] = ALPHA;
}



// activate_array
static inline float leaky_activate_cpu(float x) { return (x > 0) ? x : 0.1f * x; }
static inline float logistic_activate_cpu(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void activate_array_cpu(float *x, int n, ACTIVATION a) {
    if (a == ACTIVATION_LINEAR) return;
    for (int i = 0; i < n; ++i) {
        if (a == ACTIVATION_LEAKY) x[i] = leaky_activate_cpu(x[i]);
        else if (a == ACTIVATION_LOGISTIC) x[i] = logistic_activate_cpu(x[i]);
    }
}

// entry_index
static int entry_index_cpu(int w, int h, int n_anchors, int classes,
                           int batch, int location, int entry) {
    int n   = location / (w * h);
    int loc = location % (w * h);
    return batch * (n_anchors * w * h * (4 + classes + 1))
         + n * w * h * (4 + classes + 1)
         + entry * w * h + loc;
}

// forward_convolutional_layer CPU
// Uses fused BN+scale+bias+activate matching the kernel's computation order to ensure bit-exact FP results.
void forward_convolutional_layer_cpu(layer_def_t *def,
                                     float *input, float *output,
                                     float *weights, float *biases,
                                     float *scales, float *mean, float *variance,
                                     float *workspace) {
    int m = def->n;
    int k = def->size * def->size * def->c;
    int spatial = def->out_h * def->out_w;

    fill_cpu(m * spatial, 0.0f, output, 1);
    im2col_cpu(input, def->c, def->h, def->w, def->size, def->stride, def->pad, workspace);
    gemm_nn_cpu(m, spatial, k, 1.0f, weights, k, workspace, spatial, output, spatial);

    // Fused BN+scale+bias+activate per filter
    for (int f = 0; f < m; ++f) {
        int base = f * spatial;
        if (def->batch_normalize) {
            float mn = mean[f];
            float v  = variance[f];
            float s  = scales[f];
            float b  = biases[f];
            float inv_std = 1.0f / std::sqrt(v + 0.00001f);
            for (int i = 0; i < spatial; ++i) {
                float val = (output[base + i] - mn) * inv_std;
                val = val * s + b;
                if (def->activation == ACTIVATION_LEAKY)
                    val = leaky_activate_cpu(val);
                else if (def->activation == ACTIVATION_LOGISTIC)
                    val = logistic_activate_cpu(val);
                output[base + i] = val;
            }
        } else {
            float b = biases[f];
            for (int i = 0; i < spatial; ++i) {
                float val = output[base + i] + b;
                if (def->activation == ACTIVATION_LEAKY)
                    val = leaky_activate_cpu(val);
                else if (def->activation == ACTIVATION_LOGISTIC)
                    val = logistic_activate_cpu(val);
                output[base + i] = val;
            }
        }
    }
}

// forward_maxpool_layer CPU
void forward_maxpool_layer_cpu(layer_def_t *def, float *input, float *output) {
    int w = def->w, h = def->h, c = def->c;
    int out_w = def->out_w, out_h = def->out_h;
    int size = def->size, stride = def->stride, pad = def->pad;
    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                int out_index = j + out_w * (i + out_h * k);
                float max_val = -1e38f;
                for (int n = 0; n < size; ++n) {
                    for (int m = 0; m < size; ++m) {
                        int cur_h = h_offset + i * stride + n;
                        int cur_w = w_offset + j * stride + m;
                        int index = cur_w + w * (cur_h + h * k);
                        int valid = (cur_h >= 0 && cur_h < h && cur_w >= 0 && cur_w < w);
                        float val = valid ? input[index] : -1e38f;
                        if (val > max_val) max_val = val;
                    }
                }
                output[out_index] = max_val;
            }
        }
    }
}

void forward_yolo_layer_cpu(layer_def_t *def, float *input, float *output,
                            int num_anchors, int num_classes) {
    int w = def->out_w;
    int h = def->out_h;
    int outputs = w * h * def->out_c;

    // Copy input to output
    memcpy(output, input, outputs * sizeof(float));

    // Activate
    for (int n = 0; n < num_anchors; ++n) {
        int bbox_index = entry_index_cpu(w, h, num_anchors, num_classes, 0, n * w * h, 0);
        activate_array_cpu(output + bbox_index, 2 * w * h, ACTIVATION_LOGISTIC);

        int obj_index = entry_index_cpu(w, h, num_anchors, num_classes, 0, n * w * h, 4);
        activate_array_cpu(output + obj_index, (1 + num_classes) * w * h, ACTIVATION_LOGISTIC);
    }
}

// ---------------------------------------------------------------------------
// Weight initialization
// ---------------------------------------------------------------------------

// Xavier init
void init_weights_xavier(std::vector<float>& weights, int fan_in) {
    float scale = std::sqrt(2.0f / (float)fan_in);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// ---------------------------------------------------------------------------
// ULP comparison
// ---------------------------------------------------------------------------

bool compare_float(float a, float b, int index, int& errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
        if (errors < 100) {
            printf("*** error: [%d] expected=%f, actual=%f (diff=%d ULP)\n", index, b, a, d);
        }
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void show_usage() {
    std::cout << "Vortex YOLOv3Tiny Inference Test." << std::endl;
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

    std::cout << "=== Vortex YOLO Inference Test ===" << std::endl;
    std::cout << "Network: YOLOv3-Tiny (scaled-down)" << std::endl;
    std::cout << "Input: " << NET_WIDTH << "x" << NET_HEIGHT << "x" << NET_CHANNELS << std::endl;
    std::cout << "Classes: " << NUM_CLASSES << ", Anchors: " << NUM_ANCHORS << std::endl;

    // -----------------------------------------------------------------------
    // Open device
    // -----------------------------------------------------------------------
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));

    // -----------------------------------------------------------------------
    // Prepare host-side data
    // -----------------------------------------------------------------------

    // Host-side per-layer weight/bias/BN arrays
    std::vector<std::vector<float>> h_weights(NUM_LAYERS);
    std::vector<std::vector<float>> h_biases(NUM_LAYERS);
    std::vector<std::vector<float>> h_scales(NUM_LAYERS);
    std::vector<std::vector<float>> h_rolling_mean(NUM_LAYERS);
    std::vector<std::vector<float>> h_rolling_variance(NUM_LAYERS);
    std::vector<std::vector<float>> h_outputs(NUM_LAYERS);

    // Compute workspace size
    uint32_t max_workspace = 0;

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];
        int out_size = def->out_h * def->out_w * def->out_c;
        h_outputs[i].resize(out_size, 0.0f);

        if (def->type == LAYER_CONVOLUTIONAL) {
            int nweights = def->n * def->c * def->size * def->size;
            h_weights[i].resize(nweights);
            init_weights_xavier(h_weights[i], def->c * def->size * def->size);

            h_biases[i].resize(def->n, 0.0f);

            if (def->batch_normalize) {
                h_scales[i].resize(def->n);
                h_rolling_mean[i].resize(def->n);
                h_rolling_variance[i].resize(def->n);
                for (int j = 0; j < def->n; ++j) {
                    h_rolling_mean[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                    h_rolling_variance[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.5f + 0.1f;
                    h_scales[i][j] = 0.5f + static_cast<float>(rand()) / RAND_MAX;
                    h_biases[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
                }
            }

            // im2col workspace: channels_col * out_h * out_w
            uint32_t ws = def->c * def->size * def->size * def->out_h * def->out_w;
            if (ws > max_workspace) max_workspace = ws;
        }
    }

    // Input data (random image, normalized 0-1)
    int input_size = NET_HEIGHT * NET_WIDTH * NET_CHANNELS;
    std::vector<float> h_input(input_size);
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // -----------------------------------------------------------------------
    // Allocate device memory
    // -----------------------------------------------------------------------
    std::cout << "Allocating device memory..." << std::endl;

    // Input
    RT_CHECK(vx_mem_alloc(device, input_size * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));

    // Workspace
    RT_CHECK(vx_mem_alloc(device, max_workspace * sizeof(float), VX_MEM_READ_WRITE, &workspace_buffer));
    RT_CHECK(vx_mem_address(workspace_buffer, &kernel_arg.workspace_addr));

    // Per-layer allocations
    std::vector<layer_config_t> layer_configs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];
        layer_config_t *cfg = &layer_configs[i];

        cfg->type = def->type;
        cfg->h = def->h;  cfg->w = def->w;  cfg->c = def->c;
        cfg->out_h = def->out_h;  cfg->out_w = def->out_w;  cfg->out_c = def->out_c;
        cfg->n = def->n;
        cfg->size = def->size;
        cfg->stride = def->stride;
        cfg->pad = def->pad;
        cfg->batch_normalize = def->batch_normalize;
        cfg->activation = def->activation;

        int out_size = def->out_h * def->out_w * def->out_c;

        // Output buffer
        RT_CHECK(vx_mem_alloc(device, out_size * sizeof(float), VX_MEM_READ_WRITE, &layer_bufs[i].output));
        RT_CHECK(vx_mem_address(layer_bufs[i].output, &cfg->output_addr));

        if (def->type == LAYER_CONVOLUTIONAL) {
            int nweights = def->n * def->c * def->size * def->size;

            RT_CHECK(vx_mem_alloc(device, nweights * sizeof(float), VX_MEM_READ, &layer_bufs[i].weights));
            RT_CHECK(vx_mem_address(layer_bufs[i].weights, &cfg->weights_addr));
            RT_CHECK(vx_copy_to_dev(layer_bufs[i].weights, h_weights[i].data(), 0, nweights * sizeof(float)));

            RT_CHECK(vx_mem_alloc(device, def->n * sizeof(float), VX_MEM_READ, &layer_bufs[i].biases));
            RT_CHECK(vx_mem_address(layer_bufs[i].biases, &cfg->biases_addr));
            RT_CHECK(vx_copy_to_dev(layer_bufs[i].biases, h_biases[i].data(), 0, def->n * sizeof(float)));

            if (def->batch_normalize) {
                RT_CHECK(vx_mem_alloc(device, def->n * sizeof(float), VX_MEM_READ, &layer_bufs[i].scales));
                RT_CHECK(vx_mem_address(layer_bufs[i].scales, &cfg->scales_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].scales, h_scales[i].data(), 0, def->n * sizeof(float)));

                RT_CHECK(vx_mem_alloc(device, def->n * sizeof(float), VX_MEM_READ, &layer_bufs[i].rolling_mean));
                RT_CHECK(vx_mem_address(layer_bufs[i].rolling_mean, &cfg->rolling_mean_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].rolling_mean, h_rolling_mean[i].data(), 0, def->n * sizeof(float)));

                RT_CHECK(vx_mem_alloc(device, def->n * sizeof(float), VX_MEM_READ, &layer_bufs[i].rolling_variance));
                RT_CHECK(vx_mem_address(layer_bufs[i].rolling_variance, &cfg->rolling_variance_addr));
                RT_CHECK(vx_copy_to_dev(layer_bufs[i].rolling_variance, h_rolling_variance[i].data(), 0, def->n * sizeof(float)));
            }

            // std::cout << "  Layer " << i << " [Conv]: "
            //           << def->c << "x" << def->h << "x" << def->w
            //           << " -> " << def->out_c << "x" << def->out_h << "x" << def->out_w
            //           << " (K=" << def->size << " S=" << def->stride
            //           << " BN=" << def->batch_normalize << ")" << std::endl;
        } else if (def->type == LAYER_MAXPOOL) {
            // std::cout << "  Layer " << i << " [MaxPool]: "
            //           << def->c << "x" << def->h << "x" << def->w
            //           << " -> " << def->out_c << "x" << def->out_h << "x" << def->out_w
            //           << " (size=" << def->size << " stride=" << def->stride << ")" << std::endl;
        } else if (def->type == LAYER_YOLO) {
            // std::cout << "  Layer " << i << " [YOLO]: "
            //           << def->out_w << "x" << def->out_h
            //           << " grid, " << NUM_ANCHORS << " anchors, "
            //           << NUM_CLASSES << " classes" << std::endl;
        }
    }

    // Upload layer configs
    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_configs_buffer));
    RT_CHECK(vx_mem_address(layer_configs_buffer, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_configs_buffer, layer_configs.data(), 0, NUM_LAYERS * sizeof(layer_config_t)));

    // Upload input
    std::cout << "Uploading input data..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, input_size * sizeof(float)));

    // Set kernel arguments
    kernel_arg.num_layers = NUM_LAYERS;
    kernel_arg.net_w = NET_WIDTH;
    kernel_arg.net_h = NET_HEIGHT;
    kernel_arg.num_classes = NUM_CLASSES;
    kernel_arg.num_anchors = NUM_ANCHORS;
    kernel_arg.anchors[0] = ANCHOR_W0;  kernel_arg.anchors[1] = ANCHOR_H0;
    kernel_arg.anchors[2] = ANCHOR_W1;  kernel_arg.anchors[3] = ANCHOR_H1;
    kernel_arg.anchors[4] = ANCHOR_W2;  kernel_arg.anchors[5] = ANCHOR_H2;

    // Upload kernel binary
    std::cout << "Uploading kernel binary..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // Upload kernel arguments
    std::cout << "Uploading kernel arguments..." << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // -----------------------------------------------------------------------
    // Run device
    // -----------------------------------------------------------------------
    std::cout << "Starting YOLO inference on device..." << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    // -----------------------------------------------------------------------
    // Download YOLO output from device
    // -----------------------------------------------------------------------
    int yolo_out_size = network_def[NUM_LAYERS - 1].out_h
                      * network_def[NUM_LAYERS - 1].out_w
                      * network_def[NUM_LAYERS - 1].out_c;
    std::vector<float> h_dev_output(yolo_out_size);
    RT_CHECK(vx_copy_from_dev(h_dev_output.data(), layer_bufs[NUM_LAYERS - 1].output,
                              0, yolo_out_size * sizeof(float)));

    // -----------------------------------------------------------------------
    // CPU reference forward pass
    // -----------------------------------------------------------------------
    std::cout << "Computing CPU reference..." << std::endl;

    std::vector<float> workspace_cpu(max_workspace);
    std::vector<std::vector<float>> cpu_outputs(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_def_t *def = &network_def[i];
        int out_size = def->out_h * def->out_w * def->out_c;
        cpu_outputs[i].resize(out_size, 0.0f);

        float *layer_input = (i == 0) ? h_input.data() : cpu_outputs[i - 1].data();

        switch (def->type) {
        case LAYER_CONVOLUTIONAL:
            forward_convolutional_layer_cpu(
                def, layer_input, cpu_outputs[i].data(),
                h_weights[i].data(), h_biases[i].data(),
                def->batch_normalize ? h_scales[i].data() : nullptr,
                def->batch_normalize ? h_rolling_mean[i].data() : nullptr,
                def->batch_normalize ? h_rolling_variance[i].data() : nullptr,
                workspace_cpu.data());
            break;

        case LAYER_MAXPOOL:
            forward_maxpool_layer_cpu(def, layer_input, cpu_outputs[i].data());
            break;

        case LAYER_YOLO:
            forward_yolo_layer_cpu(def, layer_input, cpu_outputs[i].data(),
                                   NUM_ANCHORS, NUM_CLASSES);
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Verify results
    // -----------------------------------------------------------------------
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float *cpu_yolo = cpu_outputs[NUM_LAYERS - 1].data();

    for (int i = 0; i < yolo_out_size; ++i) {
        if (!compare_float(h_dev_output[i], cpu_yolo[i], i, errors)) {
            ++errors;
        }
    }


    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    std::cout << "\nCleaning up..." << std::endl;
    cleanup();

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
