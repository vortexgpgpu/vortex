#include <stdint.h>

typedef struct { uint16_t v; } float16_t;
typedef struct { uint32_t v; } float32_t;
typedef struct { uint64_t v; } float64_t;

float16_t f16_rsqrte7( float16_t );
float16_t f16_recip7( float16_t );
float32_t f32_rsqrte7( float32_t );
float32_t f32_recip7( float32_t );
float64_t f64_rsqrte7( float64_t );
float64_t f64_recip7( float64_t );