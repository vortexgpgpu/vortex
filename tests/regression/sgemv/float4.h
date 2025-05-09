// float4.h
#ifndef FLOAT4_H
#define FLOAT4_H

#include <stdint.h>
#include <math.h>

typedef union {
  struct { float x, y, z, w; };
  struct { float r, g, b, a; };
  struct { float s0, s1, s2, s3; };
  float data[4];
} float4;

// Load a float4 from unaligned memory
static inline float4 float4_load(const float* ptr) {
  float4 v;
  v.x = ptr[0];
  v.y = ptr[1];
  v.z = ptr[2];
  v.w = ptr[3];
  return v;
}

// Store a float4 to unaligned memory
static inline void float4_store(float* ptr, float4 v) {
  ptr[0] = v.x;
  ptr[1] = v.y;
  ptr[2] = v.z;
  ptr[3] = v.w;
}

// Element-wise operations
static inline float4 float4_add(float4 a, float4 b) {
  return (float4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

static inline float4 float4_mul(float4 a, float4 b) {
  return (float4){a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

// Dot product (returns sum of a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)
static inline float float4_dot(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Broadcast a scalar to all lanes
static inline float4 float4_broadcast(float val) {
  return (float4){val, val, val, val};
}

#endif // FLOAT4_H