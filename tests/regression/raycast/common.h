#pragma once

#include "geometry.h"
#include <stdint.h>
#include <cmath>

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    return _ret;                                             \
  } while (false)

#define INVALID_IDX 0xFFFFFFFF

// additional triangle data, for texturing and shading
struct tri_ex_t {
  float3_t N0, N1, N2;
  float2_t uv0, uv1, uv2;
};

struct ray_hit_t {
  float    dist = LARGE_FLOAT; // intersection distance along ray
  float3_t bcoords = {0, 0, 0}; // triangle barycentric coordinates
  uint32_t blasIdx = 0; // BLAS index
  uint32_t triIdx = 0;  // triangle index
};

// BVH node struct
struct bvh_node_t {
  float3_t aabbMin;
  uint32_t leftFirst;

  float3_t aabbMax;
  uint32_t triCount;

  bool isLeaf() const { return triCount != 0; }

  float calculateNodeCost() const {
    return surfaceArea(aabbMin, aabbMax) * triCount;
  }
};

// bottom-level acceleration structure
struct blas_node_t {
  mat4_t   transform; // transformation matrix
  mat4_t   invTransform; // inverse transformation matrix
  uint32_t bvh_offset = 0; // offset in bvh buffer
  uint64_t tex_offset = 0; // offset in texture buffer
  uint32_t tex_width = 0;
  uint32_t tex_height = 0;
  float    reflectivity = 0; // reflectivity factor

  void applyTransform(const mat4_t &T) {
    this->transform = T * this->transform;
    this->invTransform = this->transform.inverted();
  }
};

// top-level acceleration structure
struct tlas_node_t {
  float3_t aabbMin;
  uint32_t leftRight;

  float3_t aabbMax;
  uint32_t blasIdx; // bottom level acceleration structure index

  bool isLeaf() const { return leftRight == 0; }

  void setLeftRight(uint32_t left, uint32_t right) {
    leftRight = (right << 16) | left;
  }

  uint32_t left() const {
    return leftRight & 0xFFFF;
  }

  uint32_t right() const {
    return leftRight >> 16;
  }
};

inline uint32_t WangHash(uint32_t s) {
  s = (s ^ 61) ^ (s >> 16);
  s *= 9, s = s ^ (s >> 4);
  s *= 0x27d4eb2d;
  s = s ^ (s >> 15);
  return s;
}

inline uint32_t RandomInt(uint32_t *s) {
  // Marsaglia's XOR32 RNG
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

inline float RandomFloat(uint32_t *s) {
  return RandomInt(s) * 2.3283064365387e-10f; // = 1 / (2^32-1)
}

inline uint32_t RGB32FtoRGB8(float3_t c) {
  int r = (int)(std::min(c.x, 1.f) * 255);
  int g = (int)(std::min(c.y, 1.f) * 255);
  int b = (int)(std::min(c.z, 1.f) * 255);
  return (r << 16) + (g << 8) + b;
}

inline float3_t RGB8toRGB32F(uint32_t c) {
  float s = 1 / 256.0f;
  int r = (c >> 16) & 255;
  int g = (c >> 8) & 255;
  int b = c & 255;
  return float3_t(r * s, g * s, b * s);
}

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t dst_addr;

	uint64_t tri_addr;
	uint64_t triEx_addr;
	uint64_t triIdx_addr;
  uint64_t tex_addr;
	uint64_t bvh_addr;
	uint64_t blas_addr;
  uint64_t tlas_addr;
  uint32_t tlas_root;

  float3_t camera_pos;
	float3_t camera_forward;
  float3_t camera_right;
  float3_t camera_up;
  float2_t viewplane;

  uint32_t samples_per_pixel;
  uint32_t max_depth;

  float3_t light_pos;
  float3_t light_color;
  float3_t ambient_color;
  float3_t background_color;
} kernel_arg_t;



