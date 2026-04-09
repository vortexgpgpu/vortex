// ---------------------------------------------------------------------------
// YOLOv3-Tiny Sparse Kernel
// https://github.com/AlexeyAB/darknet/tree/master/src
//
// This is the sparse WMMA-accelerated variant of yolov3tiny/kernel.cpp.
// ---------------------------------------------------------------------------

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_sparse.h>
#include <math.h>
#include "common.h"

namespace vt = vortex::sparse;
using ctx = vt::wmma_context<NUM_THREADS, vt::fp32, vt::fp32>;

static inline uint32_t align_up(uint32_t x, uint32_t a) {
    return (x + a - 1) / a * a;
}


static inline size_t align_up_size(size_t value, size_t align) {
    return (value + align - 1) & ~(align - 1);
}

// ---------------------------------------------------------------------------
// im2col_thread 
// ---------------------------------------------------------------------------

static inline float im2col_get_pixel(float *im, int height, int width,
                                     int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0.0f;
    return im[col + width * (row + height * channel)];
}

typedef struct {
    float    *data_im;   
    float    *data_col;  
    int height, width;
    int ksize, stride, pad;
    int height_col, width_col;
    int channels_col;  
    int ldB;          
} im2col_args_t;

void im2col_thread(im2col_args_t * __UNIFORM__ args) {
    int idx = blockIdx.x;
    int channels_col = args->channels_col; 
    int height_col = args->height_col;
    int width_col  = args->width_col;
    int N = height_col * width_col;

    if (idx >= channels_col * N) return;

    int ldB = args->ldB;        
    
    int c           = idx / N;
    int spatial_idx = idx % N;

    int ksize  = args->ksize;
    int stride = args->stride;
    int pad    = args->pad;

    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im     = c / ksize / ksize;

    int h = spatial_idx / width_col;
    int w = spatial_idx % width_col;

    int im_row = h_offset + h * stride;
    int im_col = w_offset + w * stride;

    float val = im2col_get_pixel(
        args->data_im, args->height, args->width,
        im_row, im_col, c_im, pad);
    args->data_col[c * ldB + spatial_idx] = val;
}

// ---------------------------------------------------------------------------
// sparse_gemm_thread
// ---------------------------------------------------------------------------

typedef struct {
    ctx::input_t  *W_buf;          
    ctx::input_t  *B;              
    ctx::output_t *C;             
    uint32_t       M;              
    uint32_t       N;             
    uint32_t       K;              
    uint32_t       sparsity_degree;
} sparse_gemm_args_t;

void sparse_gemm_thread(sparse_gemm_args_t * __UNIFORM__ args) {
    uint32_t M = args->M;
    uint32_t N = args->N;
    uint32_t K = args->K;
    uint32_t sparsity_degree = args->sparsity_degree;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;
    if (tile_row >= M || tile_col >= N)
        return;

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
    auto pB        = args->B;
    auto pC        = args->C;

    uint32_t kblocks = K / 4;

    // Hoist base pointers out of the loop
    auto pTileA_base = pA_values + tile_row * values_per_row;
    const uint32_t *pMeta_base = meta_base + tile_row * kblocks;
    auto pB_base = pB + tile_col;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += ctx::tileK) {
        ctx::load_matrix_sync(fragB, pB_base + k_tile * N, N);

        uint32_t k_block = k_tile / 4;
        ctx::load_matrix_sync(fragA,
                              pTileA_base + k_block * sparsity_degree,
                              values_per_row,
                              reinterpret_cast<const void *>(pMeta_base + k_block),
                              kblocks, 0, sparsity_degree, 0);

        ctx::mma_sync(fragC, fragA, fragB, fragC, sparsity_degree);
    }

    ctx::store_matrix_sync(pC + tile_row * N + tile_col, fragC, N);
}

// ---------------------------------------------------------------------------
// bn_activate_thread 
// ---------------------------------------------------------------------------

typedef struct {
    float *output;
    float *biases;
    float *scales;
    float *mean;
    float *variance;
    int n;           
    int spatial;      
    int batch_normalize;
    int activation;  
} bn_activate_args_t;

static inline float leaky_activate(float x)    { return (x > 0.0f) ? x : 0.1f * x; }
static inline float logistic_activate(float x) { return 1.0f / (1.0f + expf(-x)); }

void bn_activate_thread(bn_activate_args_t * __UNIFORM__ args) {
    int idx = blockIdx.x;
    int n       = args->n;
    int spatial = args->spatial;

    if (idx >= n * spatial) return;

    int f           = idx / spatial;
    int spatial_idx = idx % spatial;
    int base        = f * spatial;

    float *output  = args->output;
    int activation = args->activation;

    if (args->batch_normalize) {
        float m       = args->mean[f];
        float v       = args->variance[f];
        float s       = args->scales[f];
        float b       = args->biases[f];
        float inv_std = 1.0f / sqrtf(v + 0.00001f);

        float val = (output[base + spatial_idx] - m) * inv_std * s + b;
        if (activation == ACTIVATION_LEAKY)        val = leaky_activate(val);
        else if (activation == ACTIVATION_LOGISTIC) val = logistic_activate(val);
        output[base + spatial_idx] = val;
    } else {
        float b   = args->biases[f];
        float val = output[base + spatial_idx] + b;
        if (activation == ACTIVATION_LEAKY)        val = leaky_activate(val);
        else if (activation == ACTIVATION_LOGISTIC) val = logistic_activate(val);
        output[base + spatial_idx] = val;
    }
}

// ---------------------------------------------------------------------------
// maxpool_thread
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
    if (idx >= args->out_h * args->out_w * args->c) return;

    int out_w  = args->out_w;
    int out_h  = args->out_h;
    int w      = args->w;
    int h      = args->h;
    int size   = args->size;
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
// ---------------------------------------------------------------------------

typedef struct {
    float *input;
    float *output;
    int w, h;
    int num_anchors;
    int num_classes;
    int total_outputs;  
} yolo_args_t;

void yolo_activate_thread(yolo_args_t * __UNIFORM__ args) {
    int idx = blockIdx.x;
    if (idx >= args->total_outputs) return;

    int w           = args->w;
    int h           = args->h;
    int num_anchors = args->num_anchors;
    int num_classes = args->num_classes;
    int entries     = 5 + num_classes;
    int spatial     = w * h;

    float val = args->input[idx];

    int anchor    = idx / (entries * spatial);
    int remainder = idx - anchor * entries * spatial;
    int entry     = remainder / spatial;

    if (anchor < num_anchors) {
        if (entry == 0 || entry == 1 || entry >= 4)
            val = logistic_activate(val);
    }

    args->output[idx] = val;
}

// ---------------------------------------------------------------------------
// forward_convolutional_layer // mitul: changed
// ---------------------------------------------------------------------------

void forward_convolutional_layer(layer_config_t *l, float *input, void *workspace,
                                 uint32_t sparsity_degree) {
    uint32_t out_h = l->out_h, out_w = l->out_w;
    uint32_t n    = l->n;                        
    uint32_t size = l->size;
    uint32_t c    = l->c;
    uint32_t h    = l->h, w = l->w;

    uint32_t M = n;
    uint32_t K = size * size * c;               
    uint32_t N = out_h * out_w;                

    uint32_t M_pad = align_up(M, ctx::tileM);
    uint32_t K_pad = align_up(K, ctx::tileK);
    uint32_t N_pad = align_up(N, ctx::tileN);   

    auto output  = reinterpret_cast<float*>(l->output_addr);
    auto weights = reinterpret_cast<ctx::input_t*>(l->weights_addr);  
    auto ws      = reinterpret_cast<float*>(workspace);


    im2col_args_t im_args;
    im_args.data_im      = input;
    im_args.data_col     = ws;
    im_args.height       = h;
    im_args.width        = w;
    im_args.ksize        = size;
    im_args.stride       = l->stride;
    im_args.pad          = l->pad;
    im_args.height_col   = out_h;
    im_args.width_col    = out_w;
    im_args.channels_col = K;
    im_args.ldB          = N_pad;

    uint32_t im2col_dim = K * N;
    vx_spawn_threads(1, &im2col_dim, nullptr, (vx_kernel_func_cb)im2col_thread, &im_args);


    sparse_gemm_args_t gemm_args;
    gemm_args.W_buf           = weights;
    gemm_args.B               = ws;
    gemm_args.C               = output;
    gemm_args.M               = M_pad;
    gemm_args.N               = N_pad;
    gemm_args.K               = K_pad;
    gemm_args.sparsity_degree = sparsity_degree;

    uint32_t block_dim[2] = { NUM_THREADS, 1 };
    uint32_t grid_dim[2]  = { N_pad / ctx::tileN, M_pad / ctx::tileM };
    vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)sparse_gemm_thread, &gemm_args);

    
    bn_activate_args_t bn_args;
    bn_args.output          = output;
    bn_args.biases          = reinterpret_cast<float*>(l->biases_addr);
    bn_args.n               = n;       
    bn_args.spatial         = N;      
    bn_args.batch_normalize = l->batch_normalize;
    bn_args.activation      = l->activation;
    if (l->batch_normalize) {
        bn_args.scales   = reinterpret_cast<float*>(l->scales_addr);
        bn_args.mean     = reinterpret_cast<float*>(l->rolling_mean_addr);
        bn_args.variance = reinterpret_cast<float*>(l->rolling_variance_addr);
    }
    uint32_t bn_dim = n * N;
    vx_spawn_threads(1, &bn_dim, nullptr, (vx_kernel_func_cb)bn_activate_thread, &bn_args);
}

// ---------------------------------------------------------------------------
// forward_maxpool_layer 
// ---------------------------------------------------------------------------
void forward_maxpool_layer(layer_config_t *l, float *input) {
    float *output = reinterpret_cast<float*>(l->output_addr);

    maxpool_args_t mp;
    mp.input  = input;
    mp.output = output;
    mp.w = l->w;     mp.h = l->h;     mp.c    = l->c;
    mp.out_w = l->out_w; mp.out_h = l->out_h;
    mp.size   = l->size; mp.stride = l->stride; mp.pad = l->pad;

    uint32_t mp_dim = l->out_h * l->out_w * l->c;
    vx_spawn_threads(1, &mp_dim, nullptr, (vx_kernel_func_cb)maxpool_thread, &mp);
}

// ---------------------------------------------------------------------------
// forward_yolo_layer 
// ---------------------------------------------------------------------------
void forward_yolo_layer(layer_config_t *l, float *input,
                        int num_anchors, int num_classes) {
    int outputs   = l->out_w * l->out_h * l->out_c;
    float *output = reinterpret_cast<float*>(l->output_addr);

    yolo_args_t yolo;
    yolo.input         = input;
    yolo.output        = output;
    yolo.w             = l->out_w;
    yolo.h             = l->out_h;
    yolo.num_anchors   = num_anchors;
    yolo.num_classes   = num_classes;
    yolo.total_outputs = outputs;

    uint32_t yolo_dim = (uint32_t)outputs;
    vx_spawn_threads(1, &yolo_dim, nullptr, (vx_kernel_func_cb)yolo_activate_thread, &yolo);
}

// ---------------------------------------------------------------------------
// kernel_body
// ---------------------------------------------------------------------------
void kernel_body(kernel_arg_t * __UNIFORM__ arg) {
    auto layer_configs    = reinterpret_cast<layer_config_t*>(arg->layer_configs_addr);
    auto workspace        = reinterpret_cast<void*>(arg->workspace_addr);
    float *net_input      = reinterpret_cast<float*>(arg->input_addr);
    uint32_t sparsity_deg = arg->sparsity_degree;

    for (uint32_t i = 0; i < arg->num_layers; ++i) {
        layer_config_t *l = &layer_configs[i];

        float *layer_input = (i == 0)
            ? net_input
            : reinterpret_cast<float*>(layer_configs[i - 1].output_addr);

        switch ((LAYER_TYPE)l->type) {
        case LAYER_CONVOLUTIONAL:
            forward_convolutional_layer(l, layer_input, workspace, sparsity_deg);
            break;

        case LAYER_MAXPOOL:
            forward_maxpool_layer(l, layer_input);
            break;

        case LAYER_YOLO:
            forward_yolo_layer(l, layer_input, arg->num_anchors, arg->num_classes);
            break;
        }
    }
}

int main() {
    kernel_arg_t *arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim = 1;
    return vx_spawn_threads(1, &grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
