#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
/*#define TYPE float*/
#define TYPE int
#endif

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;
  uint32_t num_points;

  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
} kernel_arg_t;


#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

int *run(int *wall, int *result_s, int *src, uint32_t cols, uint32_t rows,
         uint32_t num_runs) {
  int *dst = nullptr;
  for (uint32_t j = 0; j < num_runs; j++) {
    for (uint32_t x = 0; x < cols; x++) {
      result_s[x] = wall[x];
    }
    dst = result_s;
    int* src_bak = src;;
    for (uint32_t t = 0; t < rows - 1; t++) {
      int* temp = src;
      src = dst;
      dst = temp;
      for (uint32_t n = 0; n < cols; n++) {
        int min = src[n];
        if (n > 0)
          min = MIN(min, src[n - 1]);
        if (n < cols - 1)
          min = MIN(min, src[n + 1]);
        dst[n] = wall[(t + 1) * cols + n] + min;
      }
    }
    // Reset the pointer not to lose it
    src = src_bak;
  }
  return dst;
}


#endif
