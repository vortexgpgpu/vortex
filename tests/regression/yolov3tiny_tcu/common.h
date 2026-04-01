#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// ---------------------------------------------------------------------------
// Network dimensions (identical to yolov3tiny)
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
//
//  Layer 0: Conv 3×3, 16 filters, stride=1, pad=1, BN, Leaky  → 16×16×16
//  Layer 1: MaxPool 2×2, stride=2                              →  8×8×16
//  Layer 2: Conv 3×3, 32 filters, stride=1, pad=1, BN, Leaky  →  8×8×32
//  Layer 3: MaxPool 2×2, stride=2                              →  4×4×32
//  Layer 4: Conv 3×3, 32 filters, stride=1, pad=1, BN, Leaky  →  4×4×32
//  Layer 5: Conv 1×1, 21 filters, stride=1, pad=0, no BN, Lin →  4×4×21
//  Layer 6: YOLO decode                                        → detections
// ---------------------------------------------------------------------------
#define NUM_LAYERS  7



typedef struct {
    uint32_t type;                   
    uint32_t h, w, c;               
    uint32_t out_h, out_w, out_c;   
    uint32_t n;                      
    uint32_t size;                   
    uint32_t stride;
    uint32_t pad;                    
    uint32_t batch_normalize;        
    uint32_t activation;             

    uint64_t weights_addr;           
    uint64_t biases_addr;            
    uint64_t scales_addr;            
    uint64_t rolling_mean_addr;     
    uint64_t rolling_variance_addr;  
    uint64_t output_addr;            
} layer_config_t;

// Top-level kernel arguments
typedef struct {
    uint32_t num_layers;
    uint32_t net_w;           
    uint32_t net_h;           
    uint32_t num_classes;
    uint32_t num_anchors;
    uint64_t input_addr;            
    uint64_t layer_configs_addr;     
    uint64_t workspace_addr;         
    float anchors[NUM_ANCHORS * 2];  
} kernel_arg_t;

#endif // _COMMON_H_
