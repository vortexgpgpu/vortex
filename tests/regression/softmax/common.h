#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;
  
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

inline float exp(float x){
    float y = 1 + x * 0.25;
    y = 1 + x * y * 0.333;
    y = 1 + x * y * 0.5;
    y = 1 + x * y;
    return y;
}


#endif
