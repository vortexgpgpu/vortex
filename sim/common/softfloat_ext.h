#include <stdint.h>
#include <softfloat_types.h>

#ifdef __cplusplus
extern "C" {
#endif

uint_fast16_t f16_classify(float16_t);
float16_t f16_rsqrte7(float16_t);
float16_t f16_recip7(float16_t);

uint_fast16_t f32_classify(float32_t);
float32_t f32_rsqrte7(float32_t);
float32_t f32_recip7(float32_t);

uint_fast16_t f64_classify(float64_t);
float64_t f64_rsqrte7(float64_t);
float64_t f64_recip7(float64_t);

typedef struct { uint8_t v; } float8_t;  // e4m3
typedef struct { uint8_t v; } bfloat8_t; // e5m2

// fp8_e4m3 <--> fp32 conversions
float8_t f32_to_f8(float32_t);
float32_t f8_to_f32(float8_t);

// fp8_e5m2 <--> fp32 conversions
bfloat8_t f32_to_bf8(float32_t);
float32_t bf8_to_f32(bfloat8_t);

#ifdef __cplusplus
}
#endif