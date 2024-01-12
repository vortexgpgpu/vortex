#include "core.h"

using namespace vortex;

union reg_data_t {
  Word     u;
  WordI    i;
  WordF    f;
  float    f32;
  double   f64;
  uint32_t u32;
  uint64_t u64; 
  int32_t  i32;
  int64_t  i64;
};