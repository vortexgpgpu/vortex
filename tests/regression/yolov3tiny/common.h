#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// ---------------------------------------------------------------------------
// YOLOv3-Tiny: https://github.com/AlexeyAB/darknet
// ---------------------------------------------------------------------------

// Input image dimensions
#define NET_WIDTH   16
#define NET_HEIGHT  16
#define NET_CHANNELS 3

// YOLO detection parameters
#define NUM_CLASSES  2
#define NUM_ANCHORS  3

// Anchors (width, height pairs) — scaled down proportionally from YOLOv3-Tiny
#define ANCHOR_W0 1.0f
#define ANCHOR_H0 1.5f
#define ANCHOR_W1 2.0f
#define ANCHOR_H1 2.5f
#define ANCHOR_W2 3.0f
#define ANCHOR_H2 4.0f

// YOLO output:
#define YOLO_OUTPUT_DEPTH  (NUM_ANCHORS * (5 + NUM_CLASSES))

// ---------------------------------------------------------------------------
// Layer type
// ---------------------------------------------------------------------------
typedef enum {
    LAYER_CONVOLUTIONAL = 0,
    LAYER_MAXPOOL       = 1,
    LAYER_YOLO          = 2
} LAYER_TYPE;

// ---------------------------------------------------------------------------
// Activation type
// ---------------------------------------------------------------------------
typedef enum {
    ACTIVATION_LINEAR   = 0,
    ACTIVATION_LEAKY    = 1,
    ACTIVATION_LOGISTIC = 2
} ACTIVATION;

// ---------------------------------------------------------------------------
// Network architecture — 7 layers modelling YOLOv3-Tiny
//
//  Layer 0: Conv 3x3, 16 filters, stride=1, pad=1, BN, Leaky  -> 16x16x16
//  Layer 1: MaxPool 2x2, stride=2                              ->  8x8x16
//  Layer 2: Conv 3x3, 32 filters, stride=1, pad=1, BN, Leaky  ->  8x8x32
//  Layer 3: MaxPool 2x2, stride=2                              ->  4x4x32
//  Layer 4: Conv 3x3, 32 filters, stride=1, pad=1, BN, Leaky  ->  4x4x32
//  Layer 5: Conv 1x1, 21 filters, stride=1, pad=0, no BN, Linear -> 4x4x21
//  Layer 6: YOLO detection decode                               -> detections
// ---------------------------------------------------------------------------
#define NUM_LAYERS  7

// Conv layer configuration
typedef struct {
    // --- topology ---
    uint32_t type;          // LAYER_TYPE
    uint32_t h, w, c;      // input height, width, channels
    uint32_t out_h, out_w, out_c;  // output dimensions
    uint32_t n;             // number of filters (== out_c for conv)
    uint32_t size;          // kernel size (size x size)
    uint32_t stride;        // stride
    uint32_t pad;           // padding (each side)
    uint32_t batch_normalize; // 1 if batch-norm is used
    uint32_t activation;    // ACTIVATION enum

    // --- device pointers ---
    uint64_t weights_addr;      // float[n * c * size * size]
    uint64_t biases_addr;       // float[n]
    uint64_t scales_addr;       // float[n]  (BN scale, gamma)
    uint64_t rolling_mean_addr; // float[n]  (BN running mean)
    uint64_t rolling_variance_addr; // float[n] (BN running variance)
    uint64_t output_addr;       // float[out_h * out_w * out_c]
} layer_config_t;

// Top-level kernel argument structure
typedef struct {
    uint32_t num_layers;
    uint32_t net_w;           // network width  (NET_WIDTH)
    uint32_t net_h;           // network height (NET_HEIGHT)
    uint32_t num_classes;
    uint32_t num_anchors;
    uint64_t input_addr;      // input image   float[NET_HEIGHT * NET_WIDTH * NET_CHANNELS]
    uint64_t layer_configs_addr; // layer_config_t[NUM_LAYERS]
    uint64_t workspace_addr;  // scratch buffer for im2col
    // Anchor boxes (flat: w0,h0,w1,h1,w2,h2)
    TYPE anchors[NUM_ANCHORS * 2];
} kernel_arg_t;

#endif // _COMMON_H_
