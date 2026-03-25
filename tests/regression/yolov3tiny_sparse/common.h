#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// ---------------------------------------------------------------------------
// YOLOv3-Tiny Sparse 
// ---------------------------------------------------------------------------

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp32
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// ---------------------------------------------------------------------------
// Network dimensions
// ---------------------------------------------------------------------------
#define NET_WIDTH    16
#define NET_HEIGHT   16
#define NET_CHANNELS  3

#define NUM_CLASSES  2
#define NUM_ANCHORS  3

#define ANCHOR_W0 1.0f
#define ANCHOR_H0 1.5f
#define ANCHOR_W1 2.0f
#define ANCHOR_H1 2.5f
#define ANCHOR_W2 3.0f
#define ANCHOR_H2 4.0f


#define YOLO_OUTPUT_DEPTH  (NUM_ANCHORS * (5 + NUM_CLASSES))

// ---------------------------------------------------------------------------
// Layer types and activations
// ---------------------------------------------------------------------------
typedef enum {
    LAYER_CONVOLUTIONAL = 0,
    LAYER_MAXPOOL       = 1,
    LAYER_YOLO          = 2
} LAYER_TYPE;

typedef enum {
    ACTIVATION_LINEAR   = 0,
    ACTIVATION_LEAKY    = 1,
    ACTIVATION_LOGISTIC = 2
} ACTIVATION;

// ---------------------------------------------------------------------------
// Network: 7-layer YOLOv3-Tiny (scaled down, same as yolov3tiny)
// ---------------------------------------------------------------------------
#define NUM_LAYERS  7


typedef struct {
    uint32_t type;                   // LAYER_TYPE
    uint32_t h, w, c;               // input  height, width, channels
    uint32_t out_h, out_w, out_c;   // output height, width, channels
    uint32_t n;                      // number of filters (conv) or 0
    uint32_t size;                   // kernel size (size×size)
    uint32_t stride;
    uint32_t pad;                    // spatial padding per side
    uint32_t batch_normalize;        // 1 if BN is used
    uint32_t activation;             // ACTIVATION enum

    uint64_t weights_addr;           // packed sparse buffer for conv layers, empty for maxpool and yolo
    uint64_t biases_addr;            // float[n]
    uint64_t scales_addr;            // float[n] 
    uint64_t rolling_mean_addr;      // float[n]  
    uint64_t rolling_variance_addr;  // float[n]  
    uint64_t output_addr;            // float[M_pad × out_h × out_w]
} layer_config_t;

typedef struct {
    uint32_t num_layers;
    uint32_t net_w;           // NET_WIDTH
    uint32_t net_h;           // NET_HEIGHT
    uint32_t num_classes;
    uint32_t num_anchors;
    uint32_t sparsity_degree; // 1 for 1:4, 2 for 2:4
    uint64_t input_addr;             // float[NET_H × NET_W × NET_C]
    uint64_t layer_configs_addr;     // layer_config_t[NUM_LAYERS]
    uint64_t workspace_addr;         // fp32 scratch for im2col
                                     //   size = max over conv layers of
                                     //          K_pad × N_pad × sizeof(float)
    float anchors[NUM_ANCHORS * 2];  // flat: w0,h0, w1,h1, w2,h2
} kernel_arg_t;

#endif // _COMMON_H_
