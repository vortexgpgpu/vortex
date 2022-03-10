#pragma once

#include "common.h"
#include <vector>
#include <string>

typedef struct {
  float     x;
  float     y;
  float     z;
  float     w;
  uint32_t  c;
  float     u;
  float     v;
} vertex_t;

typedef struct {
  uint32_t i0;
  uint32_t i1;
  uint32_t i2;
} primitive_t;

struct model_t {
  bool depth_enabled;
  bool color_enabled;
  bool tex_enabled;
  std::vector<vertex_t>    vertives;
  std::vector<primitive_t> primitives;
  std::string              texture;
};