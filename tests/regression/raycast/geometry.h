#pragma once

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define PI 3.14159265358979323846264f
#define INVPI 0.31830988618379067153777f
#define INV2PI 0.15915494309189533576888f
#define TWOPI 6.28318530717958647692528f
#define SQRT_PI_INV 0.56418958355f
#define LARGE_FLOAT 1e30f
#define DEG2RAD 0.017453292519943295769236907684886f
#define EPSILON 1e-6f

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

struct int2_t {
  int2_t() = default;
  int2_t(const int a, const int b) : x(a), y(b) {}
  int2_t(const int a) : x(a), y(a) {}
  union {
    struct {
      int x, y;
    };
    int cell[2];
  };
  int &operator[](const int n) { return cell[n]; }
  int operator[](const int n) const { return cell[n]; }
};

struct uint2_t {
  uint2_t() = default;
  uint2_t(const int a, const int b) : x(a), y(b) {}
  uint2_t(const uint32_t a) : x(a), y(a) {}
  union {
    struct {
      uint32_t x, y;
    };
    uint32_t cell[2];
  };
  uint32_t &operator[](const int n) { return cell[n]; }
  uint32_t operator[](const int n) const { return cell[n]; }
};

struct float2_t {
  float2_t() = default;
  float2_t(const float a, const float b) : x(a), y(b) {}
  float2_t(const float a) : x(a), y(a) {}
  union {
    struct {
      float x, y;
    };
    float cell[2];
  };
  float &operator[](const int n) { return cell[n]; }
  float operator[](const int n) const { return cell[n]; }
};

struct int3_t;

struct int4_t {
  int4_t() = default;
  int4_t(const int a, const int b, const int c, const int d) : x(a), y(b), z(c), w(d) {}
  int4_t(const int a) : x(a), y(a), z(a), w(a) {}
  int4_t(const int3_t &a, const int d);
  union {
    struct {
      int x, y, z, w;
    };
    int cell[4];
  };
  int &operator[](const int n) { return cell[n]; }
  int operator[](const int n) const { return cell[n]; }
};

struct int3_t {
  int3_t() = default;
  int3_t(const int a, const int b, const int c) : x(a), y(b), z(c) {}
  int3_t(const int a) : x(a), y(a), z(a) {}
  int3_t(const int4_t a) : x(a.x), y(a.y), z(a.z) {}
  union {
    struct {
      int x, y, z;
    };
    int cell[3];
  };
  int &operator[](const int n) { return cell[n]; }
  int operator[](const int n) const { return cell[n]; }
};

struct uint3_t;

struct uint4_t {
  uint4_t() = default;
  uint4_t(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d) : x(a), y(b), z(c), w(d) {}
  uint4_t(const uint32_t a) : x(a), y(a), z(a), w(a) {}
  uint4_t(const uint3_t &a, const uint32_t d);
  union {
    struct {
      uint32_t x, y, z, w;
    };
    uint32_t cell[4];
  };
  uint32_t &operator[](const int n) { return cell[n]; }
  uint32_t operator[](const int n) const { return cell[n]; }
};

struct uint3_t {
  uint3_t() = default;
  uint3_t(const uint32_t a, const uint32_t b, const uint32_t c) : x(a), y(b), z(c) {}
  uint3_t(const uint32_t a) : x(a), y(a), z(a) {}
  uint3_t(const uint4_t a) : x(a.x), y(a.y), z(a.z) {}
  union {
    struct {
      uint32_t x, y, z;
    };
    uint32_t cell[3];
  };
  uint32_t &operator[](const int n) { return cell[n]; }
  uint32_t operator[](const int n) const { return cell[n]; }
};

struct float3_t;

struct float4_t {
  float4_t() = default;
  float4_t(const float a, const float b, const float c, const float d) : x(a), y(b), z(c), w(d) {}
  float4_t(const float a) : x(a), y(a), z(a), w(a) {}
  float4_t(const float3_t &a, const float d);
  float4_t(const float3_t &a);
  union {
    struct {
      float x, y, z, w;
    };
    float cell[4];
  };
  float &operator[](const int n) { return cell[n]; }
  float operator[](const int n) const { return cell[n]; }
};

struct float3_t {
  float3_t() = default;
  float3_t(const float a, const float b, const float c) : x(a), y(b), z(c) {}
  float3_t(const float a) : x(a), y(a), z(a) {}
  float3_t(const float4_t a) : x(a.x), y(a.y), z(a.z) {}
  float3_t(const uint3_t a) : x((float)a.x), y((float)a.y), z((float)a.z) {}
  union {
    struct {
      float x, y, z;
    };
    float cell[3];
  };
  float &operator[](const int n) { return cell[n]; }
  float operator[](const int n) const { return cell[n]; }
};

struct uchar4_t {
  uchar4_t() = default;
  uchar4_t(const uint8_t a, const uint8_t b, const uint8_t c, const uint8_t d) : x(a), y(b), z(c), w(d) {}
  uchar4_t(const uint8_t a) : x(a), y(a), z(a), w(a) {}
  union {
    struct {
      uint8_t x, y, z, w;
    };
    uint8_t cell[4];
  };
  uint8_t &operator[](const int n) { return cell[n]; }
  uint8_t operator[](const int n) const { return cell[n]; }
};

#pragma GCC diagnostic pop

inline float fminf(float a, float b) { return a < b ? a : b; }
inline float fmaxf(float a, float b) { return a > b ? a : b; }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float sqrf(float x) { return x * x; }
inline int sqr(int x) { return x * x; }

inline float2_t make_float2(const float a, float b) {
  float2_t f2;
  f2.x = a, f2.y = b;
  return f2;
}
inline float2_t make_float2(const float s) { return make_float2(s, s); }
inline float2_t make_float2(const float3_t &a) { return make_float2(a.x, a.y); }
inline float2_t make_float2(const int2_t &a) { return make_float2(float(a.x), float(a.y)); } // explicit casts prevent gcc warnings
inline float2_t make_float2(const uint2_t &a) { return make_float2(float(a.x), float(a.y)); }
inline int2_t make_int2(const int a, const int b) {
  int2_t i2;
  i2.x = a, i2.y = b;
  return i2;
}
inline int2_t make_int2(const int s) { return make_int2(s, s); }
inline int2_t make_int2(const int3_t &a) { return make_int2(a.x, a.y); }
inline int2_t make_int2(const uint2_t &a) { return make_int2(int(a.x), int(a.y)); }
inline int2_t make_int2(const float2_t &a) { return make_int2(int(a.x), int(a.y)); }
inline uint2_t make_uint2(const uint32_t a, const uint32_t b) {
  uint2_t u2;
  u2.x = a, u2.y = b;
  return u2;
}
inline uint2_t make_uint2(const uint32_t s) { return make_uint2(s, s); }
inline uint2_t make_uint2(const uint3_t &a) { return make_uint2(a.x, a.y); }
inline uint2_t make_uint2(const int2_t &a) { return make_uint2(uint32_t(a.x), uint32_t(a.y)); }
inline float3_t make_float3(const float &a, const float &b, const float &c) {
  float3_t f3;
  f3.x = a, f3.y = b, f3.z = c;
  return f3;
}
inline float3_t make_float3(const float &s) { return make_float3(s, s, s); }
inline float3_t make_float3(const float2_t &a) { return make_float3(a.x, a.y, 0.0f); }
inline float3_t make_float3(const float2_t &a, const float &s) { return make_float3(a.x, a.y, s); }
inline float3_t make_float3(const float4_t &a) { return make_float3(a.x, a.y, a.z); }
inline float3_t make_float3(const int3_t &a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline float3_t make_float3(const uint3_t &a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline int3_t make_int3(const int &a, const int &b, const int &c) {
  int3_t i3;
  i3.x = a, i3.y = b, i3.z = c;
  return i3;
}
inline int3_t make_int3(const int &s) { return make_int3(s, s, s); }
inline int3_t make_int3(const int2_t &a) { return make_int3(a.x, a.y, 0); }
inline int3_t make_int3(const int2_t &a, const int &s) { return make_int3(a.x, a.y, s); }
inline int3_t make_int3(const uint3_t &a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline int3_t make_int3(const float3_t &a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline int3_t make_int3(const float4_t &a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline uint3_t make_uint3(const uint32_t a, uint32_t b, uint32_t c) {
  uint3_t u3;
  u3.x = a, u3.y = b, u3.z = c;
  return u3;
}
inline uint3_t make_uint3(const uint32_t s) { return make_uint3(s, s, s); }
inline uint3_t make_uint3(const uint2_t &a) { return make_uint3(a.x, a.y, 0); }
inline uint3_t make_uint3(const uint2_t &a, const uint32_t s) { return make_uint3(a.x, a.y, s); }
inline uint3_t make_uint3(const uint4_t &a) { return make_uint3(a.x, a.y, a.z); }
inline uint3_t make_uint3(const int3_t &a) { return make_uint3(uint32_t(a.x), uint32_t(a.y), uint32_t(a.z)); }
inline float4_t make_float4(const float a, const float b, const float c, const float d) {
  float4_t f4;
  f4.x = a, f4.y = b, f4.z = c, f4.w = d;
  return f4;
}
inline float4_t make_float4(const float s) { return make_float4(s, s, s, s); }
inline float4_t make_float4(const float3_t &a) { return make_float4(a.x, a.y, a.z, 0.0f); }
inline float4_t make_float4(const float3_t &a, const float w) { return make_float4(a.x, a.y, a.z, w); }
inline float4_t make_float4(const int3_t &a, const float w) { return make_float4((float)a.x, (float)a.y, (float)a.z, w); }
inline float4_t make_float4(const int4_t &a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }
inline float4_t make_float4(const uint4_t &a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }
inline int4_t make_int4(const int a, const int b, const int c, const int d) {
  int4_t i4;
  i4.x = a, i4.y = b, i4.z = c, i4.w = d;
  return i4;
}
inline int4_t make_int4(const int s) { return make_int4(s, s, s, s); }
inline int4_t make_int4(const int3_t &a) { return make_int4(a.x, a.y, a.z, 0); }
inline int4_t make_int4(const int3_t &a, const int w) { return make_int4(a.x, a.y, a.z, w); }
inline int4_t make_int4(const uint4_t &a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline int4_t make_int4(const float4_t &a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline uint4_t make_uint4(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d) {
  uint4_t u4;
  u4.x = a, u4.y = b, u4.z = c, u4.w = d;
  return u4;
}
inline uint4_t make_uint4(const uint32_t s) { return make_uint4(s, s, s, s); }
inline uint4_t make_uint4(const uint3_t &a) { return make_uint4(a.x, a.y, a.z, 0); }
inline uint4_t make_uint4(const uint3_t &a, const uint32_t w) { return make_uint4(a.x, a.y, a.z, w); }
inline uint4_t make_uint4(const int4_t &a) { return make_uint4(uint32_t(a.x), uint32_t(a.y), uint32_t(a.z), uint32_t(a.w)); }
inline uchar4_t make_uchar4(const uint8_t a, const uint8_t b, const uint8_t c, const uint8_t d) {
  uchar4_t c4;
  c4.x = a, c4.y = b, c4.z = c, c4.w = d;
  return c4;
}

inline float2_t operator-(const float2_t &a) { return make_float2(-a.x, -a.y); }
inline int2_t operator-(const int2_t &a) { return make_int2(-a.x, -a.y); }
inline float3_t operator-(const float3_t &a) { return make_float3(-a.x, -a.y, -a.z); }
inline int3_t operator-(const int3_t &a) { return make_int3(-a.x, -a.y, -a.z); }
inline float4_t operator-(const float4_t &a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }
inline int4_t operator-(const int4_t &a) { return make_int4(-a.x, -a.y, -a.z, -a.w); }
inline int2_t operator<<(const int2_t &a, int b) { return make_int2(a.x << b, a.y << b); }
inline int2_t operator>>(const int2_t &a, int b) { return make_int2(a.x >> b, a.y >> b); }
inline int3_t operator<<(const int3_t &a, int b) { return make_int3(a.x << b, a.y << b, a.z << b); }
inline int3_t operator>>(const int3_t &a, int b) { return make_int3(a.x >> b, a.y >> b, a.z >> b); }
inline int4_t operator<<(const int4_t &a, int b) { return make_int4(a.x << b, a.y << b, a.z << b, a.w << b); }
inline int4_t operator>>(const int4_t &a, int b) { return make_int4(a.x >> b, a.y >> b, a.z >> b, a.w >> b); }

inline float2_t operator+(const float2_t &a, const float2_t &b) { return make_float2(a.x + b.x, a.y + b.y); }
inline float2_t operator+(const float2_t &a, const int2_t &b) { return make_float2(a.x + (float)b.x, a.y + (float)b.y); }
inline float2_t operator+(const float2_t &a, const uint2_t &b) { return make_float2(a.x + (float)b.x, a.y + (float)b.y); }
inline float2_t operator+(const int2_t &a, const float2_t &b) { return make_float2((float)a.x + b.x, (float)a.y + b.y); }
inline float2_t operator+(const uint2_t &a, const float2_t &b) { return make_float2((float)a.x + b.x, (float)a.y + b.y); }
inline void operator+=(float2_t &a, const float2_t &b) {
  a.x += b.x;
  a.y += b.y;
}
inline void operator+=(float2_t &a, const int2_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
}
inline void operator+=(float2_t &a, const uint2_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
}
inline float2_t operator+(const float2_t &a, float b) { return make_float2(a.x + b, a.y + b); }
inline float2_t operator+(const float2_t &a, int b) { return make_float2(a.x + (float)b, a.y + (float)b); }
inline float2_t operator+(const float2_t &a, uint32_t b) { return make_float2(a.x + (float)b, a.y + (float)b); }
inline float2_t operator+(float b, const float2_t &a) { return make_float2(a.x + b, a.y + b); }
inline void operator+=(float2_t &a, float b) {
  a.x += b;
  a.y += b;
}
inline void operator+=(float2_t &a, int b) {
  a.x += (float)b;
  a.y += (float)b;
}
inline void operator+=(float2_t &a, uint32_t b) {
  a.x += (float)b;
  a.y += (float)b;
}
inline int2_t operator+(const int2_t &a, const int2_t &b) { return make_int2(a.x + b.x, a.y + b.y); }
inline void operator+=(int2_t &a, const int2_t &b) {
  a.x += b.x;
  a.y += b.y;
}
inline int2_t operator+(const int2_t &a, int b) { return make_int2(a.x + b, a.y + b); }
inline int2_t operator+(int b, const int2_t &a) { return make_int2(a.x + b, a.y + b); }
inline void operator+=(int2_t &a, int b) {
  a.x += b;
  a.y += b;
}
inline uint2_t operator+(const uint2_t &a, const uint2_t &b) { return make_uint2(a.x + b.x, a.y + b.y); }
inline void operator+=(uint2_t &a, const uint2_t &b) {
  a.x += b.x;
  a.y += b.y;
}
inline uint2_t operator+(const uint2_t &a, uint32_t b) { return make_uint2(a.x + b, a.y + b); }
inline uint2_t operator+(uint32_t b, const uint2_t &a) { return make_uint2(a.x + b, a.y + b); }
inline void operator+=(uint2_t &a, uint32_t b) {
  a.x += b;
  a.y += b;
}
inline float3_t operator+(const float3_t &a, const float3_t &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3_t operator+(const float3_t &a, const int3_t &b) { return make_float3(a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z); }
inline float3_t operator+(const float3_t &a, const uint3_t &b) { return make_float3(a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z); }
inline float3_t operator+(const int3_t &a, const float3_t &b) { return make_float3((float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z); }
inline float3_t operator+(const uint3_t &a, const float3_t &b) { return make_float3((float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z); }
inline void operator+=(float3_t &a, const float3_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline void operator+=(float3_t &a, const int3_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
  a.z += (float)b.z;
}
inline void operator+=(float3_t &a, const uint3_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
  a.z += (float)b.z;
}
inline float3_t operator+(const float3_t &a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
inline float3_t operator+(const float3_t &a, int b) { return make_float3(a.x + (float)b, a.y + (float)b, a.z + (float)b); }
inline float3_t operator+(const float3_t &a, uint32_t b) { return make_float3(a.x + (float)b, a.y + (float)b, a.z + (float)b); }
inline void operator+=(float3_t &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
}
inline void operator+=(float3_t &a, int b) {
  a.x += (float)b;
  a.y += (float)b;
  a.z += (float)b;
}
inline void operator+=(float3_t &a, uint32_t b) {
  a.x += (float)b;
  a.y += (float)b;
  a.z += (float)b;
}
inline int3_t operator+(const int3_t &a, const int3_t &b) { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline void operator+=(int3_t &a, const int3_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline int3_t operator+(const int3_t &a, int b) { return make_int3(a.x + b, a.y + b, a.z + b); }
inline void operator+=(int3_t &a, int b) {
  a.x += b;
  a.y += b;
  a.z += b;
}
inline uint3_t operator+(const uint3_t &a, const uint3_t &b) { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline void operator+=(uint3_t &a, const uint3_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline uint3_t operator+(const uint3_t &a, uint32_t b) { return make_uint3(a.x + b, a.y + b, a.z + b); }
inline void operator+=(uint3_t &a, uint32_t b) {
  a.x += b;
  a.y += b;
  a.z += b;
}
inline int3_t operator+(int b, const int3_t &a) { return make_int3(a.x + b, a.y + b, a.z + b); }
inline uint3_t operator+(uint32_t b, const uint3_t &a) { return make_uint3(a.x + b, a.y + b, a.z + b); }
inline float3_t operator+(float b, const float3_t &a) { return make_float3(a.x + b, a.y + b, a.z + b); }
inline float4_t operator+(const float4_t &a, const float4_t &b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline float4_t operator+(const float4_t &a, const int4_t &b) { return make_float4(a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z, a.w + (float)b.w); }
inline float4_t operator+(const float4_t &a, const uint4_t &b) { return make_float4(a.x + (float)b.x, a.y + (float)b.y, a.z + (float)b.z, a.w + (float)b.w); }
inline float4_t operator+(const int4_t &a, const float4_t &b) { return make_float4((float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z, (float)a.w + b.w); }
inline float4_t operator+(const uint4_t &a, const float4_t &b) { return make_float4((float)a.x + b.x, (float)a.y + b.y, (float)a.z + b.z, (float)a.w + b.w); }
inline void operator+=(float4_t &a, const float4_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline void operator+=(float4_t &a, const int4_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
  a.z += (float)b.z;
  a.w += (float)b.w;
}
inline void operator+=(float4_t &a, const uint4_t &b) {
  a.x += (float)b.x;
  a.y += (float)b.y;
  a.z += (float)b.z;
  a.w += (float)b.w;
}
inline float4_t operator+(const float4_t &a, float b) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline float4_t operator+(const float4_t &a, int b) { return make_float4(a.x + (float)b, a.y + (float)b, a.z + (float)b, a.w + (float)b); }
inline float4_t operator+(const float4_t &a, uint32_t b) { return make_float4(a.x + (float)b, a.y + (float)b, a.z + (float)b, a.w + (float)b); }
inline float4_t operator+(float b, const float4_t &a) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline void operator+=(float4_t &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}
inline void operator+=(float4_t &a, int b) {
  a.x += (float)b;
  a.y += (float)b;
  a.z += (float)b;
  a.w += (float)b;
}
inline void operator+=(float4_t &a, uint32_t b) {
  a.x += (float)b;
  a.y += (float)b;
  a.z += (float)b;
  a.w += (float)b;
}
inline int4_t operator+(const int4_t &a, const int4_t &b) { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline void operator+=(int4_t &a, const int4_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline int4_t operator+(const int4_t &a, int b) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline int4_t operator+(int b, const int4_t &a) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline void operator+=(int4_t &a, int b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}
inline uint4_t operator+(const uint4_t &a, const uint4_t &b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline void operator+=(uint4_t &a, const uint4_t &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline uint4_t operator+(const uint4_t &a, uint32_t b) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline uint4_t operator+(uint32_t b, const uint4_t &a) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline void operator+=(uint4_t &a, uint32_t b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

inline float2_t operator-(const float2_t &a, const float2_t &b) { return make_float2(a.x - b.x, a.y - b.y); }
inline float2_t operator-(const float2_t &a, const int2_t &b) { return make_float2(a.x - (float)b.x, a.y - (float)b.y); }
inline float2_t operator-(const float2_t &a, const uint2_t &b) { return make_float2(a.x - (float)b.x, a.y - (float)b.y); }
inline float2_t operator-(const int2_t &a, const float2_t &b) { return make_float2((float)a.x - b.x, (float)a.y - b.y); }
inline float2_t operator-(const uint2_t &a, const float2_t &b) { return make_float2((float)a.x - b.x, (float)a.y - b.y); }
inline void operator-=(float2_t &a, const float2_t &b) {
  a.x -= b.x;
  a.y -= b.y;
}
inline void operator-=(float2_t &a, const int2_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
}
inline void operator-=(float2_t &a, const uint2_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
}
inline float2_t operator-(const float2_t &a, float b) { return make_float2(a.x - b, a.y - b); }
inline float2_t operator-(const float2_t &a, int b) { return make_float2(a.x - (float)b, a.y - (float)b); }
inline float2_t operator-(const float2_t &a, uint32_t b) { return make_float2(a.x - (float)b, a.y - (float)b); }
inline float2_t operator-(float b, const float2_t &a) { return make_float2(b - a.x, b - a.y); }
inline void operator-=(float2_t &a, float b) {
  a.x -= b;
  a.y -= b;
}
inline void operator-=(float2_t &a, int b) {
  a.x -= (float)b;
  a.y -= (float)b;
}
inline void operator-=(float2_t &a, uint32_t b) {
  a.x -= (float)b;
  a.y -= (float)b;
}
inline int2_t operator-(const int2_t &a, const int2_t &b) { return make_int2(a.x - b.x, a.y - b.y); }
inline void operator-=(int2_t &a, const int2_t &b) {
  a.x -= b.x;
  a.y -= b.y;
}
inline int2_t operator-(const int2_t &a, int b) { return make_int2(a.x - b, a.y - b); }
inline int2_t operator-(int b, const int2_t &a) { return make_int2(b - a.x, b - a.y); }
inline void operator-=(int2_t &a, int b) {
  a.x -= b;
  a.y -= b;
}
inline uint2_t operator-(const uint2_t &a, const uint2_t &b) { return make_uint2(a.x - b.x, a.y - b.y); }
inline void operator-=(uint2_t &a, const uint2_t &b) {
  a.x -= b.x;
  a.y -= b.y;
}
inline uint2_t operator-(const uint2_t &a, uint32_t b) { return make_uint2(a.x - b, a.y - b); }
inline uint2_t operator-(uint32_t b, const uint2_t &a) { return make_uint2(b - a.x, b - a.y); }
inline void operator-=(uint2_t &a, uint32_t b) {
  a.x -= b;
  a.y -= b;
}
inline float3_t operator-(const float3_t &a, const float3_t &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3_t operator-(const float3_t &a, const int3_t &b) { return make_float3(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z); }
inline float3_t operator-(const float3_t &a, const uint3_t &b) { return make_float3(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z); }
inline float3_t operator-(const int3_t &a, const float3_t &b) { return make_float3((float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z); }
inline float3_t operator-(const uint3_t &a, const float3_t &b) { return make_float3((float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z); }
inline void operator-=(float3_t &a, const float3_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
inline void operator-=(float3_t &a, const int3_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
  a.z -= (float)b.z;
}
inline void operator-=(float3_t &a, const uint3_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
  a.z -= (float)b.z;
}
inline float3_t operator-(const float3_t &a, float b) { return make_float3(a.x - b, a.y - b, a.z - b); }
inline float3_t operator-(const float3_t &a, int b) { return make_float3(a.x - (float)b, a.y - (float)b, a.z - (float)b); }
inline float3_t operator-(const float3_t &a, uint32_t b) { return make_float3(a.x - (float)b, a.y - (float)b, a.z - (float)b); }
inline float3_t operator-(float b, const float3_t &a) { return make_float3(b - a.x, b - a.y, b - a.z); }
inline void operator-=(float3_t &a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}
inline void operator-=(float3_t &a, int b) {
  a.x -= (float)b;
  a.y -= (float)b;
  a.z -= (float)b;
}
inline void operator-=(float3_t &a, uint32_t b) {
  a.x -= (float)b;
  a.y -= (float)b;
  a.z -= (float)b;
}
inline int3_t operator-(const int3_t &a, const int3_t &b) { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline void operator-=(int3_t &a, const int3_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
inline int3_t operator-(const int3_t &a, int b) { return make_int3(a.x - b, a.y - b, a.z - b); }
inline int3_t operator-(int b, const int3_t &a) { return make_int3(b - a.x, b - a.y, b - a.z); }
inline void operator-=(int3_t &a, int b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}
inline uint3_t operator-(const uint3_t &a, const uint3_t &b) { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline void operator-=(uint3_t &a, const uint3_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
inline uint3_t operator-(const uint3_t &a, uint32_t b) { return make_uint3(a.x - b, a.y - b, a.z - b); }
inline uint3_t operator-(uint32_t b, const uint3_t &a) { return make_uint3(b - a.x, b - a.y, b - a.z); }
inline void operator-=(uint3_t &a, uint32_t b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}
inline float4_t operator-(const float4_t &a, const float4_t &b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline float4_t operator-(const float4_t &a, const int4_t &b) { return make_float4(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z, a.w - (float)b.w); }
inline float4_t operator-(const float4_t &a, const uint4_t &b) { return make_float4(a.x - (float)b.x, a.y - (float)b.y, a.z - (float)b.z, a.w - (float)b.w); }
inline float4_t operator-(const int4_t &a, const float4_t &b) { return make_float4((float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z, (float)a.w - b.w); }
inline float4_t operator-(const uint4_t &a, const float4_t &b) { return make_float4((float)a.x - b.x, (float)a.y - b.y, (float)a.z - b.z, (float)a.w - b.w); }
inline void operator-=(float4_t &a, const float4_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
inline void operator-=(float4_t &a, const int4_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
  a.z -= (float)b.z;
  a.w -= (float)b.w;
}
inline void operator-=(float4_t &a, const uint4_t &b) {
  a.x -= (float)b.x;
  a.y -= (float)b.y;
  a.z -= (float)b.z;
  a.w -= (float)b.w;
}
inline float4_t operator-(const float4_t &a, float b) { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline float4_t operator-(const float4_t &a, int b) { return make_float4(a.x - (float)b, a.y - (float)b, a.z - (float)b, a.w - (float)b); }
inline float4_t operator-(const float4_t &a, uint32_t b) { return make_float4(a.x - (float)b, a.y - (float)b, a.z - (float)b, a.w - (float)b); }
inline void operator-=(float4_t &a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}
inline void operator-=(float4_t &a, int b) {
  a.x -= (float)b;
  a.y -= (float)b;
  a.z -= (float)b;
  a.w -= (float)b;
}
inline void operator-=(float4_t &a, uint32_t b) {
  a.x -= (float)b;
  a.y -= (float)b;
  a.z -= (float)b;
  a.w -= (float)b;
}
inline int4_t operator-(const int4_t &a, const int4_t &b) { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline void operator-=(int4_t &a, const int4_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
inline int4_t operator-(const int4_t &a, int b) { return make_int4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline int4_t operator-(int b, const int4_t &a) { return make_int4(b - a.x, b - a.y, b - a.z, b - a.w); }
inline void operator-=(int4_t &a, int b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}
inline uint4_t operator-(const uint4_t &a, const uint4_t &b) { return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline void operator-=(uint4_t &a, const uint4_t &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
inline uint4_t operator-(const uint4_t &a, uint32_t b) { return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline uint4_t operator-(uint32_t b, const uint4_t &a) { return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w); }
inline void operator-=(uint4_t &a, uint32_t b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}

inline float2_t operator*(const float2_t &a, const float2_t &b) { return make_float2(a.x * b.x, a.y * b.y); }
inline void operator*=(float2_t &a, const float2_t &b) {
  a.x *= b.x;
  a.y *= b.y;
}
inline float2_t operator*(const float2_t &a, float b) { return make_float2(a.x * b, a.y * b); }
inline float2_t operator*(float b, const float2_t &a) { return make_float2(b * a.x, b * a.y); }
inline void operator*=(float2_t &a, float b) {
  a.x *= b;
  a.y *= b;
}
inline int2_t operator*(const int2_t &a, const int2_t &b) { return make_int2(a.x * b.x, a.y * b.y); }
inline void operator*=(int2_t &a, const int2_t &b) {
  a.x *= b.x;
  a.y *= b.y;
}
inline int2_t operator*(const int2_t &a, int b) { return make_int2(a.x * b, a.y * b); }
inline int2_t operator*(int b, const int2_t &a) { return make_int2(b * a.x, b * a.y); }
inline void operator*=(int2_t &a, int b) {
  a.x *= b;
  a.y *= b;
}
inline uint2_t operator*(const uint2_t &a, const uint2_t &b) { return make_uint2(a.x * b.x, a.y * b.y); }
inline void operator*=(uint2_t &a, const uint2_t &b) {
  a.x *= b.x;
  a.y *= b.y;
}
inline uint2_t operator*(const uint2_t &a, uint32_t b) { return make_uint2(a.x * b, a.y * b); }
inline uint2_t operator*(uint32_t b, const uint2_t &a) { return make_uint2(b * a.x, b * a.y); }
inline void operator*=(uint2_t &a, uint32_t b) {
  a.x *= b;
  a.y *= b;
}
inline float3_t operator*(const float3_t &a, const float3_t &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline void operator*=(float3_t &a, const float3_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
inline float3_t operator*(const float3_t &a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline float3_t operator*(float b, const float3_t &a) { return make_float3(b * a.x, b * a.y, b * a.z); }
inline void operator*=(float3_t &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}
inline int3_t operator*(const int3_t &a, const int3_t &b) { return make_int3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline void operator*=(int3_t &a, const int3_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
inline int3_t operator*(const int3_t &a, int b) { return make_int3(a.x * b, a.y * b, a.z * b); }
inline int3_t operator*(int b, const int3_t &a) { return make_int3(b * a.x, b * a.y, b * a.z); }
inline void operator*=(int3_t &a, int b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}
inline uint3_t operator*(const uint3_t &a, const uint3_t &b) { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline void operator*=(uint3_t &a, const uint3_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
inline uint3_t operator*(const uint3_t &a, uint32_t b) { return make_uint3(a.x * b, a.y * b, a.z * b); }
inline uint3_t operator*(uint32_t b, const uint3_t &a) { return make_uint3(b * a.x, b * a.y, b * a.z); }
inline void operator*=(uint3_t &a, uint32_t b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}
inline float4_t operator*(const float4_t &a, const float4_t &b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline void operator*=(float4_t &a, const float4_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
inline float4_t operator*(const float4_t &a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline float4_t operator*(float b, const float4_t &a) { return make_float4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline void operator*=(float4_t &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
inline int4_t operator*(const int4_t &a, const int4_t &b) { return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline void operator*=(int4_t &a, const int4_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
inline int4_t operator*(const int4_t &a, int b) { return make_int4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline int4_t operator*(int b, const int4_t &a) { return make_int4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline void operator*=(int4_t &a, int b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
inline uint4_t operator*(const uint4_t &a, const uint4_t &b) { return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline void operator*=(uint4_t &a, const uint4_t &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
inline uint4_t operator*(const uint4_t &a, uint32_t b) { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline uint4_t operator*(uint32_t b, const uint4_t &a) { return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline void operator*=(uint4_t &a, uint32_t b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

inline float2_t operator/(const float2_t &a, const float2_t &b) { return make_float2(a.x / b.x, a.y / b.y); }
inline void operator/=(float2_t &a, const float2_t &b) {
  a.x /= b.x;
  a.y /= b.y;
}
inline float2_t operator/(const float2_t &a, float b) { return make_float2(a.x / b, a.y / b); }
inline void operator/=(float2_t &a, float b) {
  a.x /= b;
  a.y /= b;
}
inline float2_t operator/(float b, const float2_t &a) { return make_float2(b / a.x, b / a.y); }
inline float3_t operator/(const float3_t &a, const float3_t &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline void operator/=(float3_t &a, const float3_t &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}
inline float3_t operator/(const float3_t &a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline void operator/=(float3_t &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}
inline float3_t operator/(float b, const float3_t &a) { return make_float3(b / a.x, b / a.y, b / a.z); }
inline float4_t operator/(const float4_t &a, const float4_t &b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
inline void operator/=(float4_t &a, const float4_t &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}
inline float4_t operator/(const float4_t &a, float b) { return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); }
inline void operator/=(float4_t &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}
inline float4_t operator/(float b, const float4_t &a) { return make_float4(b / a.x, b / a.y, b / a.z, b / a.w); }

inline float2_t fminf(const float2_t &a, const float2_t &b) { return make_float2(fminf(a.x, b.x), fminf(a.y, b.y)); }
inline float3_t fminf(const float3_t &a, const float3_t &b) { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
inline float4_t fminf(const float4_t &a, const float4_t &b) { return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w)); }
inline int2_t min(const int2_t &a, const int2_t &b) { return make_int2(std::min(a.x, b.x), std::min(a.y, b.y)); }
inline int3_t min(const int3_t &a, const int3_t &b) { return make_int3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
inline int4_t min(const int4_t &a, const int4_t &b) { return make_int4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)); }
inline uint2_t min(const uint2_t &a, const uint2_t &b) { return make_uint2(std::min(a.x, b.x), std::min(a.y, b.y)); }
inline uint3_t min(const uint3_t &a, const uint3_t &b) { return make_uint3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
inline uint4_t min(const uint4_t &a, const uint4_t &b) { return make_uint4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)); }

inline float2_t fmaxf(const float2_t &a, const float2_t &b) { return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y)); }
inline float3_t fmaxf(const float3_t &a, const float3_t &b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
inline float4_t fmaxf(const float4_t &a, const float4_t &b) { return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w)); }
inline int2_t max(const int2_t &a, const int2_t &b) { return make_int2(std::max(a.x, b.x), std::max(a.y, b.y)); }
inline int3_t max(const int3_t &a, const int3_t &b) { return make_int3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
inline int4_t max(const int4_t &a, const int4_t &b) { return make_int4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)); }
inline uint2_t max(const uint2_t &a, const uint2_t &b) { return make_uint2(std::max(a.x, b.x), std::max(a.y, b.y)); }
inline uint3_t max(const uint3_t &a, const uint3_t &b) { return make_uint3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
inline uint4_t max(const uint4_t &a, const uint4_t &b) { return make_uint4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)); }

inline float lerp(float a, float b, float t) { return a + t * (b - a); }
inline float2_t lerp(const float2_t &a, const float2_t &b, float t) { return a + t * (b - a); }
inline float3_t lerp(const float3_t &a, const float3_t &b, float t) { return a + t * (b - a); }
inline float4_t lerp(const float4_t &a, const float4_t &b, float t) { return a + t * (b - a); }

inline float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }
inline int clamp(int f, int a, int b) { return std::max(a, std::min(f, b)); }
inline uint32_t clamp(uint32_t f, uint32_t a, uint32_t b) { return std::max(a, std::min(f, b)); }
inline float2_t clamp(const float2_t &v, float a, float b) { return make_float2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline float2_t clamp(const float2_t &v, const float2_t &a, const float2_t &b) { return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline float3_t clamp(const float3_t &v, float a, float b) { return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline float3_t clamp(const float3_t &v, const float3_t &a, const float3_t &b) { return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
inline float4_t clamp(const float4_t &v, float a, float b) { return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline float4_t clamp(const float4_t &v, const float4_t &a, const float4_t &b) { return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }
inline int2_t clamp(const int2_t &v, int a, int b) { return make_int2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline int2_t clamp(const int2_t &v, const int2_t &a, const int2_t &b) { return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline int3_t clamp(const int3_t &v, int a, int b) { return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline int3_t clamp(const int3_t &v, const int3_t &a, const int3_t &b) { return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
inline int4_t clamp(const int4_t &v, int a, int b) { return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline int4_t clamp(const int4_t &v, const int4_t &a, const int4_t &b) { return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }
inline uint2_t clamp(const uint2_t &v, uint32_t a, uint32_t b) { return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline uint2_t clamp(const uint2_t &v, const uint2_t &a, const uint2_t &b) { return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
inline uint3_t clamp(const uint3_t &v, uint32_t a, uint32_t b) { return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline uint3_t clamp(const uint3_t &v, const uint3_t &a, const uint3_t &b) { return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
inline uint4_t clamp(const uint4_t &v, uint32_t a, uint32_t b) { return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline uint4_t clamp(const uint4_t &v, const uint4_t &a, const uint4_t &b) { return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

inline float dot(const float2_t &a, const float2_t &b) { return a.x * b.x + a.y * b.y; }
inline float dot(const float3_t &a, const float3_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(const float4_t &a, const float4_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline int dot(const int2_t &a, const int2_t &b) { return a.x * b.x + a.y * b.y; }
inline int dot(const int3_t &a, const int3_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline int dot(const int4_t &a, const int4_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline uint32_t dot(const uint2_t &a, const uint2_t &b) { return a.x * b.x + a.y * b.y; }
inline uint32_t dot(const uint3_t &a, const uint3_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline uint32_t dot(const uint4_t &a, const uint4_t &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline float sqrLength(const float2_t &v) { return dot(v, v); }
inline float sqrLength(const float3_t &v) { return dot(v, v); }
inline float sqrLength(const float4_t &v) { return dot(v, v); }

inline float length(const float2_t &v) { return sqrtf(dot(v, v)); }
inline float length(const float3_t &v) { return sqrtf(dot(v, v)); }
inline float length(const float4_t &v) { return sqrtf(dot(v, v)); }

inline float length(const int2_t &v) { return sqrtf((float)dot(v, v)); }
inline float length(const int3_t &v) { return sqrtf((float)dot(v, v)); }
inline float length(const int4_t &v) { return sqrtf((float)dot(v, v)); }

inline float2_t normalize(const float2_t &v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}
inline float3_t normalize(const float3_t &v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}
inline float4_t normalize(const float4_t &v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

inline uint32_t dominantAxis(const float2_t &v) {
  float x = fabs(v.x), y = fabs(v.y);
  return x > y ? 0 : 1;
} // for coherent grid traversal
inline uint32_t dominantAxis(const float3_t &v) {
  float x = fabs(v.x), y = fabs(v.y), z = fabs(v.z);
  float m = std::max(std::max(x, y), z);
  return m == x ? 0 : (m == y ? 1 : 2);
}

inline float2_t floorf(const float2_t &v) { return make_float2(floorf(v.x), floorf(v.y)); }
inline float3_t floorf(const float3_t &v) { return make_float3(floorf(v.x), floorf(v.y), floorf(v.z)); }
inline float4_t floorf(const float4_t &v) { return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)); }

inline float fracf(float v) { return v - floorf(v); }
inline float2_t fracf(const float2_t &v) { return make_float2(fracf(v.x), fracf(v.y)); }
inline float3_t fracf(const float3_t &v) { return make_float3(fracf(v.x), fracf(v.y), fracf(v.z)); }
inline float4_t fracf(const float4_t &v) { return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w)); }

inline float2_t fmodf(const float2_t &a, const float2_t &b) { return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
inline float3_t fmodf(const float3_t &a, const float3_t &b) { return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
inline float4_t fmodf(const float4_t &a, const float4_t &b) { return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }

inline float2_t fabs(const float2_t &v) { return make_float2(fabs(v.x), fabs(v.y)); }
inline float3_t fabs(const float3_t &v) { return make_float3(fabs(v.x), fabs(v.y), fabs(v.z)); }
inline float4_t fabs(const float4_t &v) { return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w)); }
inline int2_t abs(const int2_t &v) { return make_int2(abs(v.x), abs(v.y)); }
inline int3_t abs(const int3_t &v) { return make_int3(abs(v.x), abs(v.y), abs(v.z)); }
inline int4_t abs(const int4_t &v) { return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w)); }

inline float3_t cross(const float3_t &a, const float3_t &b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

inline float smoothstep(float a, float b, float x) {
  float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (3.0f - (2.0f * y)));
}
inline float2_t smoothstep(float2_t a, float2_t b, float2_t x) {
  float2_t y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}
inline float3_t smoothstep(float3_t a, float3_t b, float3_t x) {
  float3_t y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}
inline float4_t smoothstep(float4_t a, float4_t b, float4_t x) {
  float4_t y = clamp((x - a) / (b - a), 0.0f, 1.0f);
  return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}

inline float surfaceArea(const float3_t& min, const float3_t& max) {
  float3_t e = max - min;
  return 2 * (e.x * e.y + e.y * e.z + e.z * e.x);
}

// matrix class
class mat4_t {
public:
  union {
    float cell[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float m[4][4];
  };

  mat4_t() = default;

  float operator[](const int idx) const { return cell[idx]; }
  float &operator[](const int idx) { return cell[idx]; }
  float operator()(const int i, const int j) const { return cell[i * 4 + j]; }
  float &operator()(const int i, const int j) { return cell[i * 4 + j]; }

  mat4_t &operator+=(const mat4_t &rhs) {
    for (int i = 0; i < 16; i++)
      cell[i] += rhs.cell[i];
    return *this;
  }

  bool operator==(const mat4_t &rhs) {
    for (int i = 0; i < 16; i++)
      if (cell[i] != rhs.cell[i])
        return false;
    return true;
  }

  float3_t getTranslation() const { return make_float3(cell[3], cell[7], cell[11]); }

  static mat4_t FromColumnMajor(const mat4_t &T) {
    mat4_t M;
    M.cell[0] = T.cell[0], M.cell[1] = T.cell[4], M.cell[2] = T.cell[8], M.cell[3] = T.cell[12];
    M.cell[4] = T.cell[1], M.cell[5] = T.cell[5], M.cell[6] = T.cell[9], M.cell[7] = T.cell[13];
    M.cell[8] = T.cell[2], M.cell[9] = T.cell[6], M.cell[10] = T.cell[10], M.cell[11] = T.cell[14];
    M.cell[12] = T.cell[3], M.cell[13] = T.cell[7], M.cell[14] = T.cell[11], M.cell[15] = T.cell[15];
    return M;
  }

  constexpr static mat4_t Identity() { return mat4_t{}; }

  static mat4_t ZeroMatrix() {
    mat4_t r;
    memset(r.cell, 0, 64);
    return r;
  }

  static mat4_t RotateX(const float a) {
    mat4_t r;
    r.cell[5] = cosf(a);
    r.cell[6] = -sinf(a);
    r.cell[9] = sinf(a);
    r.cell[10] = cosf(a);
    return r;
  };

  static mat4_t RotateY(const float a) {
    mat4_t r;
    r.cell[0] = cosf(a);
    r.cell[2] = sinf(a);
    r.cell[8] = -sinf(a);
    r.cell[10] = cosf(a);
    return r;
  };

  static mat4_t RotateZ(const float a) {
    mat4_t r;
    r.cell[0] = cosf(a);
    r.cell[1] = -sinf(a);
    r.cell[4] = sinf(a);
    r.cell[5] = cosf(a);
    return r;
  };

  static mat4_t Scale(const float s) {
    mat4_t r;
    r.cell[0] = r.cell[5] = r.cell[10] = s;
    return r;
  }

  static mat4_t Scale(const float3_t s) {
    mat4_t r;
    r.cell[0] = s.x, r.cell[5] = s.y, r.cell[10] = s.z;
    return r;
  }

  static mat4_t Scale(const float4_t s) {
    mat4_t r;
    r.cell[0] = s.x, r.cell[5] = s.y, r.cell[10] = s.z, r.cell[15] = s.w;
    return r;
  }

  static mat4_t Rotate(const float3_t &u, const float a) { return Rotate(u.x, u.y, u.z, a); }

  static mat4_t Rotate(const float x, const float y, const float z, const float a) {
    const float c = cosf(a), l_c = 1 - c, s = sinf(a);
    // row major
    mat4_t m;
    m[0] = x * x + (1 - x * x) * c, m[1] = x * y * l_c + z * s, m[2] = x * z * l_c - y * s, m[3] = 0;
    m[4] = x * y * l_c - z * s, m[5] = y * y + (1 - y * y) * c, m[6] = y * z * l_c + x * s, m[7] = 0;
    m[8] = x * z * l_c + y * s, m[9] = y * z * l_c - x * s, m[10] = z * z + (1 - z * z) * c, m[11] = 0;
    m[12] = m[13] = m[14] = 0, m[15] = 1;
    return m;
  }

  static mat4_t LookAt(const float3_t P, const float3_t T) {
    const float3_t z = normalize(T - P);
    const float3_t x = normalize(cross(z, make_float3(0, 1, 0)));
    const float3_t y = cross(x, z);
    mat4_t M = Translate(P);
    M[0] = x.x, M[4] = x.y, M[8] = x.z;
    M[1] = y.x, M[5] = y.y, M[9] = y.z;
    M[2] = z.x, M[6] = z.y, M[10] = z.z;
    return M;
  }

  static mat4_t LookAt(const float3_t &pos, const float3_t &look, const float3_t &up) {
    // PBRT's lookat
    mat4_t cameraToWorld;
    // initialize fourth column of viewing matrix
    cameraToWorld(0, 3) = pos.x;
    cameraToWorld(1, 3) = pos.y;
    cameraToWorld(2, 3) = pos.z;
    cameraToWorld(3, 3) = 1;

    // initialize first three columns of viewing matrix
    float3_t dir = normalize(look - pos);
    float3_t right = cross(normalize(up), dir);
    if (dot(right, right) == 0) {
      printf(
          "\"up\" vector (%f, %f, %f) and viewing direction (%f, %f, %f) "
          "passed to LookAt are pointing in the same direction.  Using "
          "the identity transformation.\n",
          up.x, up.y, up.z, dir.x, dir.y, dir.z);
      return mat4_t();
    }
    right = normalize(right);
    float3_t newUp = cross(dir, right);
    cameraToWorld(0, 0) = right.x, cameraToWorld(1, 0) = right.y;
    cameraToWorld(2, 0) = right.z, cameraToWorld(3, 0) = 0.;
    cameraToWorld(0, 1) = newUp.x, cameraToWorld(1, 1) = newUp.y;
    cameraToWorld(2, 1) = newUp.z, cameraToWorld(3, 1) = 0.;
    cameraToWorld(0, 2) = dir.x, cameraToWorld(1, 2) = dir.y;
    cameraToWorld(2, 2) = dir.z, cameraToWorld(3, 2) = 0.;
    return cameraToWorld.inverted();
  }

  static mat4_t Translate(const float x, const float y, const float z) {
    mat4_t r;
    r.cell[3] = x;
    r.cell[7] = y;
    r.cell[11] = z;
    return r;
  };

  static mat4_t Translate(const float3_t P) {
    mat4_t r;
    r.cell[3] = P.x;
    r.cell[7] = P.y;
    r.cell[11] = P.z;
    return r;
  };

  float trace3() const { return cell[0] + cell[5] + cell[10]; }

  mat4_t transposed() const {
    mat4_t M;
    M[0] = cell[0], M[1] = cell[4], M[2] = cell[8];
    M[4] = cell[1], M[5] = cell[5], M[6] = cell[9];
    M[8] = cell[2], M[9] = cell[6], M[10] = cell[10];
    return M;
  }

  mat4_t inverted() const {
    // from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    const float inv[16] = {
        cell[5] * cell[10] * cell[15] - cell[5] * cell[11] * cell[14] - cell[9] * cell[6] * cell[15] +
            cell[9] * cell[7] * cell[14] + cell[13] * cell[6] * cell[11] - cell[13] * cell[7] * cell[10],
        -cell[1] * cell[10] * cell[15] + cell[1] * cell[11] * cell[14] + cell[9] * cell[2] * cell[15] -
            cell[9] * cell[3] * cell[14] - cell[13] * cell[2] * cell[11] + cell[13] * cell[3] * cell[10],
        cell[1] * cell[6] * cell[15] - cell[1] * cell[7] * cell[14] - cell[5] * cell[2] * cell[15] +
            cell[5] * cell[3] * cell[14] + cell[13] * cell[2] * cell[7] - cell[13] * cell[3] * cell[6],
        -cell[1] * cell[6] * cell[11] + cell[1] * cell[7] * cell[10] + cell[5] * cell[2] * cell[11] -
            cell[5] * cell[3] * cell[10] - cell[9] * cell[2] * cell[7] + cell[9] * cell[3] * cell[6],
        -cell[4] * cell[10] * cell[15] + cell[4] * cell[11] * cell[14] + cell[8] * cell[6] * cell[15] -
            cell[8] * cell[7] * cell[14] - cell[12] * cell[6] * cell[11] + cell[12] * cell[7] * cell[10],
        cell[0] * cell[10] * cell[15] - cell[0] * cell[11] * cell[14] - cell[8] * cell[2] * cell[15] +
            cell[8] * cell[3] * cell[14] + cell[12] * cell[2] * cell[11] - cell[12] * cell[3] * cell[10],
        -cell[0] * cell[6] * cell[15] + cell[0] * cell[7] * cell[14] + cell[4] * cell[2] * cell[15] -
            cell[4] * cell[3] * cell[14] - cell[12] * cell[2] * cell[7] + cell[12] * cell[3] * cell[6],
        cell[0] * cell[6] * cell[11] - cell[0] * cell[7] * cell[10] - cell[4] * cell[2] * cell[11] +
            cell[4] * cell[3] * cell[10] + cell[8] * cell[2] * cell[7] - cell[8] * cell[3] * cell[6],
        cell[4] * cell[9] * cell[15] - cell[4] * cell[11] * cell[13] - cell[8] * cell[5] * cell[15] +
            cell[8] * cell[7] * cell[13] + cell[12] * cell[5] * cell[11] - cell[12] * cell[7] * cell[9],
        -cell[0] * cell[9] * cell[15] + cell[0] * cell[11] * cell[13] + cell[8] * cell[1] * cell[15] -
            cell[8] * cell[3] * cell[13] - cell[12] * cell[1] * cell[11] + cell[12] * cell[3] * cell[9],
        cell[0] * cell[5] * cell[15] - cell[0] * cell[7] * cell[13] - cell[4] * cell[1] * cell[15] +
            cell[4] * cell[3] * cell[13] + cell[12] * cell[1] * cell[7] - cell[12] * cell[3] * cell[5],
        -cell[0] * cell[5] * cell[11] + cell[0] * cell[7] * cell[9] + cell[4] * cell[1] * cell[11] -
            cell[4] * cell[3] * cell[9] - cell[8] * cell[1] * cell[7] + cell[8] * cell[3] * cell[5],
        -cell[4] * cell[9] * cell[14] + cell[4] * cell[10] * cell[13] + cell[8] * cell[5] * cell[14] -
            cell[8] * cell[6] * cell[13] - cell[12] * cell[5] * cell[10] + cell[12] * cell[6] * cell[9],
        cell[0] * cell[9] * cell[14] - cell[0] * cell[10] * cell[13] - cell[8] * cell[1] * cell[14] +
            cell[8] * cell[2] * cell[13] + cell[12] * cell[1] * cell[10] - cell[12] * cell[2] * cell[9],
        -cell[0] * cell[5] * cell[14] + cell[0] * cell[6] * cell[13] + cell[4] * cell[1] * cell[14] -
            cell[4] * cell[2] * cell[13] - cell[12] * cell[1] * cell[6] + cell[12] * cell[2] * cell[5],
        cell[0] * cell[5] * cell[10] - cell[0] * cell[6] * cell[9] - cell[4] * cell[1] * cell[10] +
            cell[4] * cell[2] * cell[9] + cell[8] * cell[1] * cell[6] - cell[8] * cell[2] * cell[5]};
    const float det = cell[0] * inv[0] + cell[1] * inv[4] + cell[2] * inv[8] + cell[3] * inv[12];
    mat4_t retVal;
    if (det != 0) {
      const float invdet = 1.0f / det;
      for (int i = 0; i < 16; i++)
        retVal.cell[i] = inv[i] * invdet;
    }
    return retVal;
  }

  mat4_t inverted3x3() const {
    // via https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c
    const float invdet = 1.0f / (cell[0] * (cell[5] * cell[10] - cell[6] * cell[9]) -
                                 cell[4] * (cell[1] * cell[10] - cell[9] * cell[2]) +
                                 cell[8] * (cell[1] * cell[6] - cell[5] * cell[2]));
    mat4_t R;
    R.cell[0] = (cell[5] * cell[10] - cell[6] * cell[9]) * invdet;
    R.cell[4] = (cell[8] * cell[6] - cell[4] * cell[10]) * invdet;
    R.cell[8] = (cell[4] * cell[9] - cell[8] * cell[5]) * invdet;
    R.cell[1] = (cell[9] * cell[2] - cell[1] * cell[10]) * invdet;
    R.cell[5] = (cell[0] * cell[10] - cell[8] * cell[2]) * invdet;
    R.cell[9] = (cell[1] * cell[8] - cell[0] * cell[9]) * invdet;
    R.cell[2] = (cell[1] * cell[6] - cell[2] * cell[5]) * invdet;
    R.cell[6] = (cell[2] * cell[4] - cell[0] * cell[6]) * invdet;
    R.cell[10] = (cell[0] * cell[5] - cell[1] * cell[4]) * invdet;
    return R;
  }

  inline float3_t transformVector(const float3_t &v) const {
    return make_float3(cell[0] * v.x + cell[1] * v.y + cell[2] * v.z,
                       cell[4] * v.x + cell[5] * v.y + cell[6] * v.z,
                       cell[8] * v.x + cell[9] * v.y + cell[10] * v.z);
  }

  inline float3_t transformPoint(const float3_t &v) const {
    const float3_t res = make_float3(
        cell[0] * v.x + cell[1] * v.y + cell[2] * v.z + cell[3],
        cell[4] * v.x + cell[5] * v.y + cell[6] * v.z + cell[7],
        cell[8] * v.x + cell[9] * v.y + cell[10] * v.z + cell[11]);
    const float w = cell[12] * v.x + cell[13] * v.y + cell[14] * v.z + cell[15];
    if (w == 1)
      return res;
    return res * (1.f / w);
  }
};

inline mat4_t operator*(const mat4_t &a, const mat4_t &b) {
  mat4_t r;
  for (uint32_t i = 0; i < 16; i += 4)
    for (uint32_t j = 0; j < 4; ++j) {
      r[i + j] =
          (a.cell[i + 0] * b.cell[j + 0]) +
          (a.cell[i + 1] * b.cell[j + 4]) +
          (a.cell[i + 2] * b.cell[j + 8]) +
          (a.cell[i + 3] * b.cell[j + 12]);
    }
  return r;
}

inline mat4_t operator+(const mat4_t &a, const mat4_t &b) {
  mat4_t r;
  for (uint32_t i = 0; i < 16; i += 4)
    r.cell[i] = a.cell[i] + b.cell[i];
  return r;
}

inline mat4_t operator*(const mat4_t &a, const float s) {
  mat4_t r;
  for (uint32_t i = 0; i < 16; i += 4)
    r.cell[i] = a.cell[i] * s;
  return r;
}

inline mat4_t operator*(const float s, const mat4_t &a) {
  mat4_t r;
  for (uint32_t i = 0; i < 16; i++)
    r.cell[i] = a.cell[i] * s;
  return r;
}

inline bool operator==(const mat4_t &a, const mat4_t &b) {
  for (uint32_t i = 0; i < 16; i++)
    if (a.cell[i] != b.cell[i])
      return false;
  return true;
}

inline bool operator!=(const mat4_t &a, const mat4_t &b) { return !(a == b); }

inline float4_t operator*(const mat4_t &a, const float4_t &b) {
  return make_float4(a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3] * b.w,
                     a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7] * b.w,
                     a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11] * b.w,
                     a.cell[12] * b.x + a.cell[13] * b.y + a.cell[14] * b.z + a.cell[15] * b.w);
}

inline float4_t operator*(const float4_t &a, const mat4_t &b) {
  return make_float4(b.cell[0] * a.x + b.cell[1] * a.y + b.cell[2] * a.z + b.cell[3] * a.w,
                     b.cell[4] * a.x + b.cell[5] * a.y + b.cell[6] * a.z + b.cell[7] * a.w,
                     b.cell[8] * a.x + b.cell[9] * a.y + b.cell[10] * a.z + b.cell[11] * a.w,
                     b.cell[12] * a.x + b.cell[13] * a.y + b.cell[14] * a.z + b.cell[15] * a.w);
}

inline float3_t TransformPosition(const float3_t &a, const mat4_t &M) {
  return make_float3(make_float4(a, 1) * M);
}

inline float3_t TransformVector(const float3_t &a, const mat4_t &M) {
  return make_float3(make_float4(a, 0) * M);
}

// based on https://github.com/adafruit
class quat_t {
public:
  quat_t() = default;
  quat_t(float _w, float _x, float _y, float _z) : w(_w), x(_x), y(_y), z(_z) {}
  quat_t(float _w, float3_t v) : w(_w), x(v.x), y(v.y), z(v.z) {}
  float magnitude() const { return sqrtf(w * w + x * x + y * y + z * z); }
  void normalize() {
    float m = magnitude();
    *this = this->scale(1 / m);
  }
  quat_t conjugate() const { return quat_t(w, -x, -y, -z); }
  void fromAxisAngle(const float3_t &axis, float theta) {
    w = cosf(theta / 2);
    const float s = sinf(theta / 2);
    x = axis.x * s, y = axis.y * s, z = axis.z * s;
  }
  void fromMatrix(const mat4_t &m) {
    float tr = m.trace3(), S;
    if (tr > 0) {
      S = sqrtf(tr + 1.0f) * 2, w = 0.25f * S;
      x = (m(2, 1) - m(1, 2)) / S, y = (m(0, 2) - m(2, 0)) / S;
      z = (m(1, 0) - m(0, 1)) / S;
    } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
      S = sqrt(1.0f + m(0, 0) - m(1, 1) - m(2, 2)) * 2;
      w = (m(2, 1) - m(1, 2)) / S, x = 0.25f * S;
      y = (m(0, 1) + m(1, 0)) / S, z = (m(0, 2) + m(2, 0)) / S;
    } else if (m(1, 1) > m(2, 2)) {
      S = sqrt(1.0f + m(1, 1) - m(0, 0) - m(2, 2)) * 2;
      w = (m(0, 2) - m(2, 0)) / S;
      x = (m(0, 1) + m(1, 0)) / S, y = 0.25f * S;
      z = (m(1, 2) + m(2, 1)) / S;
    } else {
      S = sqrt(1.0f + m(2, 2) - m(0, 0) - m(1, 1)) * 2;
      w = (m(1, 0) - m(0, 1)) / S, x = (m(0, 2) + m(2, 0)) / S;
      y = (m(1, 2) + m(2, 1)) / S, z = 0.25f * S;
    }
  }
  void toAxisAngle(float3_t &axis, float &angle) const {
    float s = sqrtf(1 - w * w);
    if (s == 0)
      return;
    angle = 2 * acosf(w);
    axis.x = x / s, axis.y = y / s, axis.z = z / s;
  }
  mat4_t toMatrix() const {
    mat4_t ret;
    ret.cell[0] = 1 - 2 * y * y - 2 * z * z;
    ret.cell[1] = 2 * x * y - 2 * w * z, ret.cell[2] = 2 * x * z + 2 * w * y, ret.cell[4] = 2 * x * y + 2 * w * z;
    ret.cell[5] = 1 - 2 * x * x - 2 * z * z;
    ret.cell[6] = 2 * y * z - 2 * w * x, ret.cell[8] = 2 * x * z - 2 * w * y, ret.cell[9] = 2 * y * z + 2 * w * x;
    ret.cell[10] = 1 - 2 * x * x - 2 * y * y;
    return ret;
  }
  float3_t toEuler() const {
    float3_t ret;
    float sqw = w * w, sqx = x * x, sqy = y * y, sqz = z * z;
    ret.x = atan2f(2.0f * (x * y + z * w), (sqx - sqy - sqz + sqw));
    ret.y = asinf(-2.0f * (x * z - y * w) / (sqx + sqy + sqz + sqw));
    ret.z = atan2f(2.0f * (y * z + x * w), (-sqx - sqy + sqz + sqw));
    return ret;
  }
  float3_t toAngularVelocity(float dt) const {
    float3_t ret;
    quat_t one(1, 0, 0, 0), delta = one - *this, r = (delta / dt);
    r = r * 2, r = r * one, ret.x = r.x, ret.y = r.y, ret.z = r.z;
    return ret;
  }
  float3_t rotateVector(const float3_t &v) const {
    float3_t qv = make_float3(x, y, z), t = cross(qv, v) * 2.0f;
    return v + t * w + cross(qv, t);
  }
  quat_t operator*(const quat_t &q) const {
    return quat_t(
        w * q.w - x * q.x - y * q.y - z * q.z, w * q.x + x * q.w + y * q.z - z * q.y,
        w * q.y - x * q.z + y * q.w + z * q.x, w * q.z + x * q.y - y * q.x + z * q.w);
  }
  static quat_t slerp(const quat_t &a, const quat_t &b, const float t) {
    // from https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
    quat_t qm;
    float cosHalfTheta = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (abs(cosHalfTheta) >= 1.0) {
      qm.w = a.w, qm.x = a.x, qm.y = a.y, qm.z = a.z;
      return qm;
    }
    float halfTheta = acosf(cosHalfTheta);
    float sinHalfTheta = sqrtf(1.0f - cosHalfTheta * cosHalfTheta);
    if (fabs(sinHalfTheta) < 0.001f) {
      qm.w = a.w * 0.5f + b.w * 0.5f, qm.x = a.x * 0.5f + b.x * 0.5f;
      qm.y = a.y * 0.5f + b.y * 0.5f, qm.z = a.z * 0.5f + b.z * 0.5f;
      return qm;
    }
    float ratioA = sinf((1 - t) * halfTheta) / sinHalfTheta;
    float ratioB = sinf(t * halfTheta) / sinHalfTheta;
    qm.w = (a.w * ratioA + b.w * ratioB), qm.x = (a.x * ratioA + b.x * ratioB);
    qm.y = (a.y * ratioA + b.y * ratioB), qm.z = (a.z * ratioA + b.z * ratioB);
    return qm;
  }
  quat_t operator+(const quat_t &q) const { return quat_t(w + q.w, x + q.x, y + q.y, z + q.z); }
  quat_t operator-(const quat_t &q) const { return quat_t(w - q.w, x - q.x, y - q.y, z - q.z); }
  quat_t operator/(float s) const { return quat_t(w / s, x / s, y / s, z / s); }
  quat_t operator*(float s) const { return scale(s); }
  quat_t scale(float s) const { return quat_t(w * s, x * s, y * s, z * s); }
  float w = 1, x = 0, y = 0, z = 0;
};

struct tri_t {
  float3_t v0;
  float3_t v1;
  float3_t v2;
};

struct ray_t {
  float3_t orig; // origin
  float3_t dir;  // direction

  void transform(const mat4_t& mat) {
    this->dir = TransformVector(this->dir, mat);
    this->orig = TransformPosition(this->orig, mat);
  }

  bool intersect(const tri_t& tri, float* dist, float3_t* bcoords) const {
    float3_t edge1 = tri.v1 - tri.v0;
    float3_t edge2 = tri.v2 - tri.v0;
    float3_t h = cross(this->dir, edge2);
    float a = dot(edge1, h);
    if (fabs(a) < EPSILON)
      return false; // ray parallel to triangle

    float f = 1 / a;
    float3_t s = this->orig - tri.v0;
    float w1 = f * dot(s, h);
    if (w1 < 0 || w1 > 1)
      return false; // intersection outside triangle

    const float3_t q = cross(s, edge1);
    const float w2 = f * dot(this->dir, q);
    if (w2 < 0 || w1 + w2 > 1)
      return false;

    const float t = f * dot(edge2, q);
    if (t <= EPSILON)
      return false; // intersection behind ray origin

    // return intersection data
    *dist = t;
    bcoords->x = w1;
    bcoords->y = w2;
    bcoords->z = 1 - w1 - w2;
    return true;
  }

  float intersect(const float3_t& aabbMin, const float3_t& aabbMax) const {
    float idir_x = 1.0f / this->dir.x;
    float idir_y = 1.0f / this->dir.y;
    float idir_z = 1.0f / this->dir.z;
    float tx1 = (aabbMin.x - this->orig.x) * idir_x;
    float tx2 = (aabbMax.x - this->orig.x) * idir_x;
    float tmin = std::min(tx1, tx2);
    float tmax = std::max(tx1, tx2);
    float ty1 = (aabbMin.y - this->orig.y) * idir_y;
    float ty2 = (aabbMax.y - this->orig.y) * idir_y;
    tmin = std::max(tmin, std::min(ty1, ty2)),
    tmax = std::min(tmax, std::max(ty1, ty2));
    float tz1 = (aabbMin.z - this->orig.z) * idir_z;
    float tz2 = (aabbMax.z - this->orig.z) * idir_z;
    tmin = std::max(tmin, std::min(tz1, tz2)),
    tmax = std::min(tmax, std::max(tz1, tz2));
    if (tmax < tmin || tmax <= 0)
      return LARGE_FLOAT; // no intersection

    // return min distance
    return tmin;
  }
};
