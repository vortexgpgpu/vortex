// ---------------------------------------------------------------------------
// https://github.com/AlexeyAB/darknet/tree/master/src
//   - GEMM: one spawn per conv layer (1 thread per filter row)
//   - BN+Bias+Activate: one spawn per conv layer (1 thread per filter)
//   - MaxPool: one spawn per pool layer (1 thread per output element)
//   - YOLO decode: one spawn for all activations
//   - Fill: merged into GEMM thread (each thread zeros its own row)
//   - Copy: merged into YOLO activation thread
// ---------------------------------------------------------------------------

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// ---------------------------------------------------------------------------
// im2col
// Rearranges image patches into columns for GEMM-based convolution.
// One thread per output channel (channels_col = C × ksize × ksize).
// ---------------------------------------------------------------------------

static inline float im2col_get_pixel(float *im, int height, int width,
                                     int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

typedef struct {
    float *data_im;
    float *data_col;
    int height, width;
    int ksize, stride, pad;
    int height_col, width_col;
    int channels_col;
} im2col_args_t;

void im2col_thread(im2col_args_t * __UNIFORM__ args) {
    int c = blockIdx.x;
    if (c >= args->channels_col) return;

    int ksize = args->ksize;
    int stride = args->stride;
    int pad = args->pad;
    int height_col = args->height_col;
    int width_col = args->width_col;

    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;

    for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
            int im_row = h_offset + h * stride;
            int im_col = w_offset + w * stride;
            int col_index = (c * height_col + h) * width_col + w;
            args->data_col[col_index] = im2col_get_pixel(
                args->data_im, args->height, args->width,
                im_row, im_col, c_im, pad);
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_nn_thread
// Each thread: zeros its output row, then computes C[i,:] = A[i,:] * B
// (Fill + GEMM merged into a single spawn to halve overhead)
// ---------------------------------------------------------------------------

typedef struct {
    int M, N, K;
    float ALPHA;
    float *A; int lda;
    float *B; int ldb;
    float *C; int ldc;
} gemm_args_t;

void gemm_nn_thread(gemm_args_t * __UNIFORM__ args) {
    int i = blockIdx.x;
    if (i >= args->M) return;

    int N = args->N;
    int K = args->K;
    float ALPHA = args->ALPHA;
    float *A = args->A;
    float *B = args->B;
    float *C = args->C;
    int lda = args->lda;
    int ldb = args->ldb;
    int ldc = args->ldc;

    // Zero this row first (merged fill_cpu)
    for (int j = 0; j < N; ++j) {
        C[i * ldc + j] = 0.0f;
    }

    // GEMM: C[i,:] = ALPHA * A[i,:] * B
    for (int k = 0; k < K; ++k) {
        float A_PART = ALPHA * A[i * lda + k];
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += A_PART * B[k * ldb + j];
        }
    }
}

// ---------------------------------------------------------------------------
// bn_activate_thread - fused BN + scale + bias + activation
// From darknet: normalize_cpu + scale_bias + add_bias + activate_array
// One thread per filter channel. Fusing avoids 4 separate memory passes.
// ---------------------------------------------------------------------------

typedef struct {
    float *output;
    float *biases;
    float *scales;
    float *mean;
    float *variance;
    int n;          // number of filters
    int spatial;    // out_h * out_w
    int batch_normalize;
    int activation; // ACTIVATION enum
} bn_activate_args_t;

static inline float leaky_activate(float x) {
    return (x > 0) ? x : 0.1f * x;
}

static inline float logistic_activate(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void bn_activate_thread(bn_activate_args_t * __UNIFORM__ args) {
    int f = blockIdx.x;
    if (f >= args->n) return;

    float *output  = args->output;
    float *biases  = args->biases;
    int spatial    = args->spatial;
    int activation = args->activation;
    int base = f * spatial;

    if (args->batch_normalize) {
        float m = args->mean[f];
        float v = args->variance[f];
        float s = args->scales[f];
        float b = biases[f];
        float inv_std = 1.0f / sqrtf(v + 0.00001f);

        for (int i = 0; i < spatial; ++i) {
            float val = (output[base + i] - m) * inv_std;
            val = val * s + b;
            if (activation == ACTIVATION_LEAKY) {
                val = leaky_activate(val);
            } else if (activation == ACTIVATION_LOGISTIC) {
                val = logistic_activate(val);
            }
            output[base + i] = val;
        }
    } else {
        float b = biases[f];
        for (int i = 0; i < spatial; ++i) {
            float val = output[base + i] + b;
            if (activation == ACTIVATION_LEAKY) {
                val = leaky_activate(val);
            } else if (activation == ACTIVATION_LOGISTIC) {
                val = logistic_activate(val);
            }
            output[base + i] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// maxpool_thread
// One thread per output element (channel × spatial).
// ---------------------------------------------------------------------------

typedef struct {
    float *input;
    float *output;
    int w, h, c;
    int out_w, out_h;
    int size, stride, pad;
} maxpool_args_t;

void maxpool_thread(maxpool_args_t * __UNIFORM__ args) {
    int idx = blockIdx.x;
    int total = args->out_h * args->out_w * args->c;
    if (idx >= total) return;

    int out_w = args->out_w;
    int out_h = args->out_h;
    int w = args->w;
    int h = args->h;
    int size = args->size;
    int stride = args->stride;
    int w_offset = -(args->pad) / 2;
    int h_offset = -(args->pad) / 2;

    int j = idx % out_w;
    int i = (idx / out_w) % out_h;
    int k = idx / (out_w * out_h);

    float max_val = -1e38f;
    for (int n = 0; n < size; ++n) {
        for (int m = 0; m < size; ++m) {
            int cur_h = h_offset + i * stride + n;
            int cur_w = w_offset + j * stride + m;
            if (cur_h >= 0 && cur_h < h && cur_w >= 0 && cur_w < w) {
                float val = args->input[cur_w + w * (cur_h + h * k)];
                if (val > max_val) max_val = val;
            }
        }
    }
    args->output[idx] = max_val;
}

// ---------------------------------------------------------------------------
// yolo_activate_thread
// Processes one output element: copies input to output, then applies
// sigmoid where needed (x,y offsets, objectness, class probabilities).
// This merges copy + all per-element activations into a single spawn.
// ---------------------------------------------------------------------------

typedef struct {
    float *input;
    float *output;
    int w, h;          // grid dimensions
    int num_anchors;
    int num_classes;
    int total_outputs;  // w * h * out_c
} yolo_args_t;

void yolo_activate_thread(yolo_args_t * __UNIFORM__ args) {
    int idx = blockIdx.x;
    if (idx >= args->total_outputs) return;

    int w = args->w;
    int h = args->h;
    int num_anchors = args->num_anchors;
    int num_classes = args->num_classes;
    int entries = 5 + num_classes;   // entries per anchor: x,y,w,h,obj,cls...
    int spatial = w * h;

    // Copy input to output
    float val = args->input[idx];

    // Determine which anchor and which entry this element belongs to
    // Layout: [anchor][entry][spatial]
    int anchor = idx / (entries * spatial);
    int remainder = idx - anchor * entries * spatial;
    int entry = remainder / spatial;

    // Apply sigmoid to x(0), y(1), objectness(4), class probs(5+)
    // Leave w(2) and h(3) as-is (log-space)
    if (anchor < num_anchors) {
        if (entry == 0 || entry == 1 || entry >= 4) {
            val = logistic_activate(val);
        }
    }

    args->output[idx] = val;
}


// ---------------------------------------------------------------------------
// forward_convolutional_layer
// 3 spawns per layer: im2col + GEMM (with merged fill) + BN/bias/activate
// ---------------------------------------------------------------------------

void forward_convolutional_layer(layer_config_t *l, float *input, float *workspace) {
    int out_h = l->out_h;
    int out_w = l->out_w;
    int n = l->n;
    int size = l->size;
    int stride = l->stride;
    int pad = l->pad;
    int c = l->c;
    int h = l->h;
    int w = l->w;

    float *output  = reinterpret_cast<float*>(l->output_addr);
    float *weights = reinterpret_cast<float*>(l->weights_addr);
    float *biases  = reinterpret_cast<float*>(l->biases_addr);

    int m = n;                        // number of filters
    int k = size * size * c;          // filter volume
    int spatial = out_h * out_w;

    // Step 1: im2col
    im2col_args_t im2col_args;
    im2col_args.data_im = input;
    im2col_args.data_col = workspace;
    im2col_args.height = h;
    im2col_args.width = w;
    im2col_args.ksize = size;
    im2col_args.stride = stride;
    im2col_args.pad = pad;
    im2col_args.height_col = out_h;
    im2col_args.width_col = out_w;
    im2col_args.channels_col = k;  // channels_col = c * size * size = k
    uint32_t im2col_dim = (uint32_t)k;
    vx_spawn_threads(1, &im2col_dim, nullptr, (vx_kernel_func_cb)im2col_thread, &im2col_args);

    // Step 2: Parallel GEMM (along with merged zero-fill)
    gemm_args_t gemm_args = { m, spatial, k, 1.0f,
                              weights, k, workspace, spatial, output, spatial };
    uint32_t gemm_dim = (uint32_t)m;
    vx_spawn_threads(1, &gemm_dim, nullptr, (vx_kernel_func_cb)gemm_nn_thread, &gemm_args);

    // Step 3: Parallel fused BN + bias + activation
    bn_activate_args_t bn_args;
    bn_args.output  = output;
    bn_args.biases  = biases;
    bn_args.n       = n;
    bn_args.spatial = spatial;
    bn_args.batch_normalize = l->batch_normalize;
    bn_args.activation = l->activation;
    if (l->batch_normalize) {
        bn_args.scales   = reinterpret_cast<float*>(l->scales_addr);
        bn_args.mean     = reinterpret_cast<float*>(l->rolling_mean_addr);
        bn_args.variance = reinterpret_cast<float*>(l->rolling_variance_addr);
    }
    uint32_t bn_dim = (uint32_t)n;
    vx_spawn_threads(1, &bn_dim, nullptr, (vx_kernel_func_cb)bn_activate_thread, &bn_args);
}

// ---------------------------------------------------------------------------
// forward_maxpool_layer
// 1 spawn per layer.
// ---------------------------------------------------------------------------

void forward_maxpool_layer(layer_config_t *l, float *input) {
    float *output = reinterpret_cast<float*>(l->output_addr);

    maxpool_args_t mp_args;
    mp_args.input  = input;
    mp_args.output = output;
    mp_args.w = l->w;  mp_args.h = l->h;  mp_args.c = l->c;
    mp_args.out_w = l->out_w;  mp_args.out_h = l->out_h;
    mp_args.size = l->size;  mp_args.stride = l->stride;  mp_args.pad = l->pad;

    uint32_t mp_dim = (uint32_t)(l->out_h * l->out_w * l->c);
    vx_spawn_threads(1, &mp_dim, nullptr, (vx_kernel_func_cb)maxpool_thread, &mp_args);
}

// ---------------------------------------------------------------------------
// forward_yolo_layer
// 1 spawn total (copy + sigmoid merged per element).
// ---------------------------------------------------------------------------

void forward_yolo_layer(layer_config_t *l, float *input,
                        int num_anchors, int num_classes) {
    int w = l->out_w;
    int h = l->out_h;
    int outputs = w * h * l->out_c;

    float *output = reinterpret_cast<float*>(l->output_addr);

    yolo_args_t yolo_args;
    yolo_args.input = input;
    yolo_args.output = output;
    yolo_args.w = w;
    yolo_args.h = h;
    yolo_args.num_anchors = num_anchors;
    yolo_args.num_classes = num_classes;
    yolo_args.total_outputs = outputs;

    uint32_t yolo_dim = (uint32_t)outputs;
    vx_spawn_threads(1, &yolo_dim, nullptr, (vx_kernel_func_cb)yolo_activate_thread, &yolo_args);
}

// ---------------------------------------------------------------------------
// kernel_body
// Runs the full YOLOv3-Tiny forward pass layer by layer.
// Total spawn calls: 4 conv × 3 + 2 maxpool × 1 + 1 YOLO = 15 spawns
// ---------------------------------------------------------------------------

void kernel_body(kernel_arg_t * __UNIFORM__ arg) {
    auto layer_configs = reinterpret_cast<layer_config_t*>(arg->layer_configs_addr);
    auto workspace     = reinterpret_cast<float*>(arg->workspace_addr);
    float *input       = reinterpret_cast<float*>(arg->input_addr);

    for (uint32_t i = 0; i < arg->num_layers; ++i) {
        layer_config_t *l = &layer_configs[i];

        float *layer_input = (i == 0) ? input
                           : reinterpret_cast<float*>(layer_configs[i - 1].output_addr);

        switch ((LAYER_TYPE)l->type) {
        case LAYER_CONVOLUTIONAL:
            forward_convolutional_layer(l, layer_input, workspace);
            break;

        case LAYER_MAXPOOL:
            forward_maxpool_layer(l, layer_input);
            break;

        case LAYER_YOLO:
            forward_yolo_layer(l, layer_input,
                               arg->num_anchors, arg->num_classes);
            break;
        }
    }
}

int main() {
    kernel_arg_t *arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim = 1;
    return vx_spawn_threads(1, &grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
