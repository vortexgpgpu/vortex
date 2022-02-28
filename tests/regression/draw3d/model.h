#pragma once

#include "common.h"
#include <vector>

typedef struct {
  uint32_t i0;
  uint32_t i1;
  uint32_t i2;
} primitive_t;

struct model_t {
  std::vector<vertex_t>    vertives;
  std::vector<primitive_t> primitives;
};