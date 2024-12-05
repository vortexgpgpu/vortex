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

#ifdef __cplusplus
}
#endif