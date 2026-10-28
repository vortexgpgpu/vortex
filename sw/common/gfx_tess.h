#pragma once

// The tessellator, shared by the device kernels and the host. It is the D3D11
// reference tessellator llvmpipe runs on the CPU (CHWTessellator in mesa's
// src/gallium/auxiliary/tessellator), kept in its 16.16 fixed point with the
// same tables and rounding, so every domain point and every primitive comes out
// bit-identical to llvmpipe's -- and therefore meets the invariance rules the
// same way. What changes is the shape: the reference walks a patch serially,
// while here each piece of a patch is a closed form one thread evaluates on its
// own.
//
//   process     once per patch: clamp and round the levels, build the per-edge
//               and inside contexts, and count the patch's domain points,
//               primitives and stitch units (gfx_tess_patch_t)
//   point       domain point i of a patch, in the reference's order, so its
//               index is the reference's index
//   unit        stitch unit s of a patch -- a (ring, edge) pair, a quad's centre
//               strip, a triangle's centre triangle, or an isoline -- writes its
//               primitives at their place in the reference's index order
//
// The C part (the descriptor and the mode encoding) is what the driver and the
// kernels exchange; the C++ part is the algorithm.

#include <stdint.h>

#define GFX_TESS_DOMAIN_ISOLINE 0u
#define GFX_TESS_DOMAIN_TRI     1u
#define GFX_TESS_DOMAIN_QUAD    2u

#define GFX_TESS_SPACING_EQUAL  0u   // integer partitioning
#define GFX_TESS_SPACING_ODD    1u   // fractional odd
#define GFX_TESS_SPACING_EVEN   2u   // fractional even

// The primitives a patch becomes. Triangle "CW" and "CCW" are the reference's
// names for the two index orders; llvmpipe selects CW for a shader whose
// vertex order is counter-clockwise.
#define GFX_TESS_OUT_POINT      0u
#define GFX_TESS_OUT_LINE       1u
#define GFX_TESS_OUT_TRI_CW     2u
#define GFX_TESS_OUT_TRI_CCW    3u

#define GFX_TESS_MODE(domain, spacing, out) ((domain) | ((spacing) << 2) | ((out) << 4))
#define GFX_TESS_MODE_DOMAIN(m)  ((m) & 3u)
#define GFX_TESS_MODE_SPACING(m) (((m) >> 2) & 3u)
#define GFX_TESS_MODE_OUT(m)     (((m) >> 4) & 3u)

#define GFX_TESS_CULLED  (1u << 0)   // an outer level <= 0, denormal or NaN: nothing made
#define GFX_TESS_MINIMUM (1u << 1)   // every level 1: the reference's fixed minimum patch

#define GFX_TESS_MAX_PRIMS  8192u    // 64 * 64 * 2

// One level's tessellation context (the reference's TESS_FACTOR_CONTEXT).
typedef struct {
  uint32_t frac;       // fraction of the half level, 16.16
  int32_t  nhalf;      // points on half an edge
  int32_t  split;      // the point at which the floor and ceiling halves part
  uint32_t inv_floor;  // 1 / segments at the floor level, 16.16
  uint32_t inv_ceil;   // 1 / segments at the ceiling level, 16.16
} gfx_tess_ctx_t;

// A processed patch. For an isoline, in_pts holds the points per line and the
// line count, and inner[0] and inner[1] the detail and density contexts.
typedef struct {
  uint32_t mode;         // GFX_TESS_MODE
  uint32_t flags;        // GFX_TESS_CULLED, GFX_TESS_MINIMUM
  uint32_t num_points;   // domain points
  uint32_t num_prims;    // primitives: points, lines or triangles
  uint32_t num_units;    // stitch units
  uint32_t parity;       // bit e: outer edge e odd; bit 4 + a: inside axis a odd
  int32_t  out_pts[4];   // points on each outer edge
  int32_t  in_pts[2];    // points across the inside, per axis
  int32_t  inside_base;  // the first inside point
  uint32_t prim_id;      // gl_PrimitiveID, in a batch's own descriptors (desc_stride)
  gfx_tess_ctx_t outer[4];
  gfx_tess_ctx_t inner[2];
} gfx_tess_patch_t;

// A batch of tessellated patches, as the stitch kernel (tess_stitch_k) and the
// evaluation stage read it. Patch p's points, primitives and stitch units start
// at the exclusive prefixes scan[3p], scan[3p + 1] and scan[3p + 2] of the
// patches' counts when the counts come from the device; with scan_addr 0 every
// patch has `points`, `prims` and `units` of them and patch p's start at p times
// those. A range of a scanned batch's patches runs as a batch of its own: its
// scan and descriptors start at its first patch, and first_* are where its
// points, primitives and units start in that scan, so its records and list
// start at zero.
typedef struct {
  uint64_t desc_addr;     // gfx_tess_patch_t of each patch, or one all share
  uint64_t scan_addr;     // uint32[3 * patches], 0 = every patch alike
  uint64_t list_addr;     // the primitive list (pipe_list_entry_t[]) stitched into
  uint64_t units_addr;    // the batch's stitch units on the device (uint32), 0 = units
  uint32_t patches;
  uint32_t desc_stride;   // bytes between descriptors, 0 = one shared
  uint32_t points;        // per patch, with scan_addr 0
  uint32_t prims;
  uint32_t units;         // per patch with scan_addr 0, else a bound on the batch's
  uint32_t patch_base;    // the batch's first patch in the draw
  uint32_t inst_patches;  // with one shared descriptor: patches per instance (gl_PrimitiveID
                          // wraps), 0 = one instance
  uint32_t first_point;   // with scan_addr, the range's first point, primitive and unit
  uint32_t first_prim;
  uint32_t first_unit;
} gfx_tess_draw_t;

#ifdef __cplusplus

namespace gfx_tess {

typedef uint32_t fxp_t;

static constexpr fxp_t FXP_ONE          = 0x00010000u;
static constexpr fxp_t FXP_ONE_HALF     = 0x00008000u;
static constexpr fxp_t FXP_ONE_THIRD    = 0x00005555u;
static constexpr fxp_t FXP_TWO_THIRDS   = 0x0000aaaau;
static constexpr fxp_t FXP_FRACTION     = 0x0000ffffu;
static constexpr fxp_t FXP_INTEGER      = 0x7fff0000u;
static constexpr int   FXP_BITS         = 16;

static constexpr float MIN_ODD  = 1.0f;
static constexpr float MAX_ODD  = 63.0f;
static constexpr float MIN_EVEN = 2.0f;
static constexpr float MAX_EVEN = 64.0f;
// 2^-16, the smallest positive fixed-point fraction.
static constexpr float EPSILON  = 0.0000152587890625f;

// 1 / n in 16.16 for n segments (entry 0 unused).
static const fxp_t RECIPROCAL[65] = {
  0xffffffffu,
  0x10000, 0x8000, 0x5555, 0x4000, 0x3333, 0x2aab, 0x2492, 0x2000,
  0x1c72, 0x199a, 0x1746, 0x1555, 0x13b1, 0x1249, 0x1111, 0x1000,
  0xf0f, 0xe39, 0xd79, 0xccd, 0xc31, 0xba3, 0xb21, 0xaab,
  0xa3d, 0x9d9, 0x97b, 0x925, 0x8d4, 0x889, 0x842, 0x800,
  0x7c2, 0x788, 0x750, 0x71c, 0x6eb, 0x6bd, 0x690, 0x666,
  0x63e, 0x618, 0x5f4, 0x5d1, 0x5b0, 0x591, 0x572, 0x555,
  0x539, 0x51f, 0x505, 0x4ec, 0x4d5, 0x4be, 0x4a8, 0x492,
  0x47e, 0x46a, 0x457, 0x444, 0x432, 0x421, 0x410, 0x400,
};

// Where vertex i of a half edge ends up at the largest level, in the ruler
// function's split order, and the loop bounds over it for each half level.
static const int8_t FINAL_POSITION[33] = {
  0, 32, 16, 8, 17, 4, 18, 9, 19, 2, 20, 10, 21, 5, 22, 11, 23,
  1, 24, 12, 25, 6, 26, 13, 27, 3, 28, 14, 29, 7, 30, 15, 31,
};
static const int8_t LOOP_START[33] = {
  1, 1, 17, 9, 9, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};
static const int8_t LOOP_END[33] = {
  0, 0, 17, 17, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 29, 29, 31,
  31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32,
};

static inline uint32_t f2u(float f) {
  union { float f; uint32_t u; } x;
  x.f = f;
  return x.u;
}

static inline float u2f(uint32_t u) {
  union { float f; uint32_t u; } x;
  x.u = u;
  return x.f;
}

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

static inline bool is_nan(float a) {
  const uint32_t i = f2u(a);
  return (i & 0x7f800000u) == 0x7f800000u && (i & 0x007fffffu);
}

// x > 0 as llvmpipe decides it, tessellating with denormals flushed: a
// positive denormal is zero. Decided on the bits, so the device's FPU mode
// does not matter.
static inline bool gt_zero(float a) {
  const int32_t i = (int32_t)f2u(a);
  return i >= 0x00800000 && i <= 0x7f800000;
}

static inline float flush(float a) {
  const uint32_t i = f2u(a);
  if ((int32_t)(i & 0x7fffffffu) < 0x00800000) {
    return u2f(i & 0x80000000u);
  }
  return a;
}

// The reference's IEEE 754R min and max, NaN in b giving a.
static inline float fmin_r(float a, float b) {
  const float fa = flush(a);
  const float fb = flush(b);
  if (is_nan(fb)) {
    return a;
  }
  if (fa == 0 && fb == 0) {
    return (f2u(fa) & 0x80000000u) ? a : b;
  }
  return fa < fb ? a : b;
}

static inline float fmax_r(float a, float b) {
  const float fa = flush(a);
  const float fb = flush(b);
  if (is_nan(fb)) {
    return a;
  }
  if (fa == 0 && fb == 0) {
    return (f2u(fb) & 0x80000000u) ? a : b;
  }
  return fa >= fb ? a : b;
}

static inline float clampf(float lo, float hi, float x) {
  return fmin_r(hi, fmax_r(lo, x));
}

// ceil of a clamped level (1..64), without libm.
static inline float ceil_level(float x) {
  const int i = (int)x;
  return (float)i == x ? x : (float)(i + 1);
}

static inline bool is_even(float x) {
  return !(((int)x) & 1);
}

// float to unsigned 15.16, the reference's floatToIDotF<15, 16, false>: NaN and
// negatives to 0, too large to all ones, round to nearest even.
static inline fxp_t to_fixed(float input) {
  int32_t out = (int32_t)f2u(input);
  const int32_t uexp = ((out & 0x7f800000) >> 23) - 127;
  const int32_t neg = out & (int32_t)0x80000000u;
  if (uexp == 128 && (out & 0x007fffff)) {
    out = 0;
  } else if (!neg && out >= 0x47000000) {
    out = -1;
  } else if (neg) {
    out = 0;
  } else if (uexp < -17) {
    out = 0;
  } else {
    out = (out & 0x007fffff) | 0x00800000;
    const int32_t extra = 7 - uexp;
    if (extra >= 0) {
      const int32_t lsb = 1 << extra;
      const int32_t mask = lsb - 1;
      const int32_t half = lsb >> 1;
      if ((out & lsb) || (out & mask) > half) {
        out += half;
      }
      out >>= extra;
    } else {
      out <<= -extra;
    }
  }
  return (fxp_t)out;
}

static inline float to_float(fxp_t x) {
  return (float)(x >> FXP_BITS) + (float)(x & FXP_FRACTION) / (float)(1 << FXP_BITS);
}

static inline fxp_t fxp_ceil(fxp_t x) {
  return (x & FXP_FRACTION) ? (x & FXP_INTEGER) + FXP_ONE : x;
}

static inline fxp_t fxp_floor(fxp_t x) {
  return x & FXP_INTEGER;
}

static inline int remove_msb(int val) {
  int check;
  if (val <= 0x0000ffff) {
    check = (val <= 0x000000ff) ? 0x00000080 : 0x00008000;
  } else {
    check = (val <= 0x00ffffff) ? 0x00800000 : (int)0x80000000u;
  }
  for (int i = 0; i < 8; i++, check >>= 1) {
    if (val & check) {
      return val & ~check;
    }
  }
  return 0;
}

static inline int num_points(fxp_t f, bool odd) {
  if (odd) {
    return (int)((fxp_ceil(FXP_ONE_HALF + (f + 1) / 2) * 2) >> FXP_BITS);
  }
  return (int)((fxp_ceil((f + 1) / 2) * 2) >> FXP_BITS) + 1;
}

static inline void make_ctx(fxp_t f, bool odd, gfx_tess_ctx_t* c) {
  fxp_t half = (f + 1) / 2;
  if (odd || half == FXP_ONE_HALF) {
    half += FXP_ONE_HALF;
  }
  const fxp_t fl = fxp_floor(half);
  const fxp_t ce = fxp_ceil(half);
  c->frac = half - fl;
  c->nhalf = (int32_t)(ce >> FXP_BITS);
  if (ce == fl) {
    c->split = c->nhalf + 1;
  } else if (odd) {
    c->split = (fl == FXP_ONE) ? 0 : (remove_msb((int)(fl >> FXP_BITS) - 1) << 1) + 1;
  } else {
    c->split = (remove_msb((int)(fl >> FXP_BITS)) << 1) + 1;
  }
  int nfloor = (int)((fl * 2) >> FXP_BITS);
  int nceil = (int)((ce * 2) >> FXP_BITS);
  if (odd) {
    nfloor -= 1;
    nceil -= 1;
  }
  c->inv_floor = RECIPROCAL[nfloor];
  c->inv_ceil = RECIPROCAL[nceil];
}

// Point `point` of an edge with context c (the reference's PlacePointIn1D).
static inline fxp_t place(const gfx_tess_ctx_t& c, int point, bool odd) {
  bool flip = false;
  if (point >= c.nhalf) {
    point = (c.nhalf << 1) - point;
    if (odd) {
      point -= 1;
    }
    flip = true;
  }
  if (point == c.nhalf) {
    return FXP_ONE_HALF;
  }
  const uint32_t on_ceil = (uint32_t)point;
  uint32_t on_floor = on_ceil;
  if (point > c.split) {
    on_floor -= 1;
  }
  const fxp_t at_floor = on_floor * c.inv_floor;
  const fxp_t at_ceil = on_ceil * c.inv_ceil;
  fxp_t loc = at_floor * (FXP_ONE - c.frac) + at_ceil * c.frac;
  loc = (loc + FXP_ONE_HALF) >> FXP_BITS;
  return flip ? FXP_ONE - loc : loc;
}

static inline bool odd_outer(const gfx_tess_patch_t& p, int e) { return (p.parity >> e) & 1u; }
static inline bool odd_inner(const gfx_tess_patch_t& p, int a) { return (p.parity >> (4 + a)) & 1u; }

static inline void level_bounds(uint32_t spacing, float* lo, float* hi) {
  if (spacing == GFX_TESS_SPACING_EVEN) {
    *lo = MIN_EVEN;
    *hi = MAX_EVEN;
  } else if (spacing == GFX_TESS_SPACING_ODD) {
    *lo = MIN_ODD;
    *hi = MAX_ODD;
  } else {
    *lo = MIN_ODD;
    *hi = MAX_EVEN;
  }
}

// ---- quads ------------------------------------------------------------------

// Rings of points inside the outer one, and the rings the connectivity walks.
static inline int quad_point_rings(const gfx_tess_patch_t& p) {
  return imin(p.in_pts[0], p.in_pts[1]) >> 1;
}

static inline int quad_stitch_rings(const gfx_tess_patch_t& p) {
  return imin((p.in_pts[0] + 1) >> 1, (p.in_pts[1] + 1) >> 1);
}

// The first point of inside ring k >= 1: rings run 2 (nu + nv) - 4 - 8 k points.
static inline int quad_ring_start(const gfx_tess_patch_t& p, int k) {
  const int c = 2 * (p.in_pts[0] + p.in_pts[1]) - 4;
  return p.inside_base + (k - 1) * c - 4 * k * (k - 1);
}

// The centre strip of a patch whose inside levels differ in parity or size:
// its quads (0 if none), and whether it runs along U.
static inline int quad_strip(const gfx_tess_patch_t& p, bool* along_u) {
  const int nu = p.in_pts[0], nv = p.in_pts[1];
  if (nu > nv && odd_inner(p, 1)) {
    *along_u = true;
    return (((nu >> 1) - (nv >> 1)) << 1) + (odd_inner(p, 0) ? 1 : 2);
  }
  if (nv >= nu && odd_inner(p, 0)) {
    *along_u = false;
    return (((nv >> 1) - (nu >> 1)) << 1) + (odd_inner(p, 1) ? 1 : 2);
  }
  *along_u = false;
  return 0;
}

// Triangles of ring r: ring 1 stitches the outer edges to the first inside
// ring, every later one runs 4 (nu + nv) - 16 r. The first triangle of ring r.
static inline int quad_ring1_tris(const gfx_tess_patch_t& p) {
  int n = 0;
  for (int e = 0; e < 4; e++) {
    n += p.in_pts[(e + 1) & 1] - 2 + p.out_pts[e] - 2;
  }
  return n;
}

static inline int quad_prim_start(const gfx_tess_patch_t& p, int r) {
  if (r <= 1) {
    return 0;
  }
  const int s = 4 * (p.in_pts[0] + p.in_pts[1]);
  // ring 1, then rings 2 .. r - 1
  return quad_ring1_tris(p) + (r - 2) * s - 16 * ((r - 1) * r / 2 - 1);
}

static inline void quad_counts(gfx_tess_patch_t* p) {
  const int rings = quad_stitch_rings(*p);
  bool along_u;
  const int strip = quad_strip(*p, &along_u);
  p->num_prims = (uint32_t)(quad_prim_start(*p, rings) + 2 * strip);
  p->num_units = (uint32_t)(4 * (rings - 1) + (strip ? 1 : 0));
}

static inline void quad_process(uint32_t spacing, const float* outer, const float* inner,
                                gfx_tess_patch_t* p) {
  float lo, hi;
  level_bounds(spacing, &lo, &hi);
  const bool integer = spacing == GFX_TESS_SPACING_EQUAL;
  float o[4], in[2];
  for (int e = 0; e < 4; e++) {
    o[e] = clampf(lo, hi, outer[e]);
    if (integer) {
      o[e] = ceil_level(o[e]);
    }
  }
  in[0] = inner[0];
  in[1] = inner[1];
  if (spacing == GFX_TESS_SPACING_ODD) {
    const float t = MIN_ODD + EPSILON / 2;
    if (o[0] > t || o[1] > t || o[2] > t || o[3] > t || in[0] > t || in[1] > t) {
      lo = MIN_ODD + EPSILON;
    }
  }
  for (int a = 0; a < 2; a++) {
    in[a] = clampf(lo, hi, in[a]);
    if (integer) {
      in[a] = ceil_level(in[a]);
    }
  }
  uint32_t parity = 0;
  for (int e = 0; e < 4; e++) {
    const bool odd = integer ? !is_even(o[e]) : spacing == GFX_TESS_SPACING_ODD;
    parity |= (odd ? 1u : 0u) << e;
  }
  for (int a = 0; a < 2; a++) {
    const bool odd = integer ? !(is_even(in[a]) || in[a] == 1.0f) : spacing == GFX_TESS_SPACING_ODD;
    parity |= (odd ? 1u : 0u) << (4 + a);
  }
  p->parity = parity;
  fxp_t fo[4], fi[2];
  for (int e = 0; e < 4; e++) {
    fo[e] = to_fixed(o[e]);
  }
  for (int a = 0; a < 2; a++) {
    fi[a] = to_fixed(in[a]);
  }
  if ((integer || spacing == GFX_TESS_SPACING_ODD) && fi[0] == FXP_ONE && fi[1] == FXP_ONE &&
      fo[0] == FXP_ONE && fo[1] == FXP_ONE && fo[2] == FXP_ONE && fo[3] == FXP_ONE) {
    p->flags = GFX_TESS_MINIMUM;
    p->num_points = 4;
    return;
  }
  int npts = 0;
  for (int e = 0; e < 4; e++) {
    make_ctx(fo[e], odd_outer(*p, e), &p->outer[e]);
    p->out_pts[e] = num_points(fo[e], odd_outer(*p, e));
    npts += p->out_pts[e];
  }
  npts -= 4;
  for (int a = 0; a < 2; a++) {
    make_ctx(fi[a], odd_inner(*p, a), &p->inner[a]);
    const int n = num_points(fi[a], odd_inner(*p, a));
    p->in_pts[a] = imax(odd_inner(*p, a) ? 4 : 3, n);
  }
  p->inside_base = npts;
  npts += (p->in_pts[0] - 2) * (p->in_pts[1] - 2);
  p->num_points = (uint32_t)npts;
}

static inline void quad_point(const gfx_tess_patch_t& p, int i, fxp_t* u, fxp_t* v) {
  if (i < p.inside_base) {
    int e = 0;
    while (i >= p.out_pts[e] - 1) {
      i -= p.out_pts[e] - 1;
      e++;
    }
    const int end = p.out_pts[e] - 1;
    const int q = (e == 1 || e == 2) ? i : end - i;
    const fxp_t t = place(p.outer[e], q, odd_outer(p, e));
    if (e & 1) {
      *u = t;
      *v = (e == 3) ? FXP_ONE : 0;
    } else {
      *u = (e == 2) ? FXP_ONE : 0;
      *v = t;
    }
    return;
  }
  const int rings = quad_point_rings(p);
  if (i < quad_ring_start(p, rings)) {
    // the inside ring k holding point i, by bisection over the ring starts
    int lo = 1, hi = rings - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (quad_ring_start(p, mid) <= i) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const int k = lo;
    int j = i - quad_ring_start(p, k);
    const int end[2] = { p.in_pts[0] - 1 - k, p.in_pts[1] - 1 - k };
    int e = 0;
    while (j >= end[(e + 1) & 1] - k) {
      j -= end[(e + 1) & 1] - k;
      e++;
    }
    const int along = (e + 1) & 1, perp = e & 1;
    const int at = (e < 2) ? k : end[perp];
    const fxp_t tp = place(p.inner[perp], at, odd_inner(p, perp));
    const int pt = k + j;
    const int q = (e == 1 || e == 2) ? pt : end[along] - (pt - k);
    const fxp_t t = place(p.inner[along], q, odd_inner(p, along));
    if (along) {
      *u = tp;
      *v = t;
    } else {
      *u = t;
      *v = tp;
    }
    return;
  }
  // even inside: the degenerate innermost "ring" is a row of points
  const int j = i - quad_ring_start(p, rings);
  if (p.in_pts[0] > p.in_pts[1] && !odd_inner(p, 1)) {
    *u = place(p.inner[0], rings + j, odd_inner(p, 0));
    *v = FXP_ONE_HALF;
  } else {
    const int end = p.in_pts[1] - 1 - rings;
    *u = FXP_ONE_HALF;
    *v = place(p.inner[1], end - j, odd_inner(p, 1));
  }
}

// ---- triangles -------------------------------------------------------------

static inline int tri_point_rings(const gfx_tess_patch_t& p) {
  return p.in_pts[0] >> 1;
}

static inline int tri_stitch_rings(const gfx_tess_patch_t& p) {
  return (p.in_pts[0] + 1) >> 1;
}

// The first point of inside ring k >= 1: rings run 3 (n - 1 - 2 k) points.
static inline int tri_ring_start(const gfx_tess_patch_t& p, int k) {
  return p.inside_base + 3 * ((k - 1) * (p.in_pts[0] - 1) - k * (k - 1));
}

static inline int tri_ring1_tris(const gfx_tess_patch_t& p) {
  int n = 0;
  for (int e = 0; e < 3; e++) {
    n += p.in_pts[0] - 2 + p.out_pts[e] - 2;
  }
  return n;
}

// Ring 1, then rings 2 .. r - 1 of 6 n - 12 k triangles.
static inline int tri_prim_start(const gfx_tess_patch_t& p, int r) {
  if (r <= 1) {
    return 0;
  }
  return tri_ring1_tris(p) + (r - 2) * 6 * p.in_pts[0] - 12 * ((r - 1) * r / 2 - 1);
}

static inline void tri_counts(gfx_tess_patch_t* p) {
  const int rings = tri_stitch_rings(*p);
  const bool centre = odd_inner(*p, 0);
  p->num_prims = (uint32_t)(tri_prim_start(*p, rings) + (centre ? 1 : 0));
  p->num_units = (uint32_t)(3 * (rings - 1) + (centre ? 1 : 0));
}

static inline void tri_process(uint32_t spacing, const float* outer, float inner,
                               gfx_tess_patch_t* p) {
  float lo, hi;
  level_bounds(spacing, &lo, &hi);
  const bool integer = spacing == GFX_TESS_SPACING_EQUAL;
  float o[3];
  for (int e = 0; e < 3; e++) {
    o[e] = clampf(lo, hi, outer[e]);
    if (integer) {
      o[e] = ceil_level(o[e]);
    }
  }
  if (spacing == GFX_TESS_SPACING_ODD) {
    const float t = MIN_ODD + EPSILON / 2;
    if (o[0] > t || o[1] > t || o[2] > t) {
      lo = MIN_ODD + EPSILON;
    }
  }
  float in = clampf(lo, hi, inner);
  if (integer) {
    in = ceil_level(in);
  }
  uint32_t parity = 0;
  for (int e = 0; e < 3; e++) {
    const bool odd = integer ? !is_even(o[e]) : spacing == GFX_TESS_SPACING_ODD;
    parity |= (odd ? 1u : 0u) << e;
  }
  {
    const bool odd = integer ? !(is_even(in) || in == 1.0f) : spacing == GFX_TESS_SPACING_ODD;
    parity |= (odd ? 1u : 0u) << 4;
  }
  p->parity = parity;
  fxp_t fo[3];
  for (int e = 0; e < 3; e++) {
    fo[e] = to_fixed(o[e]);
  }
  const fxp_t fi = to_fixed(in);
  if ((integer || spacing == GFX_TESS_SPACING_ODD) && fi == FXP_ONE && fo[0] == FXP_ONE &&
      fo[1] == FXP_ONE && fo[2] == FXP_ONE) {
    p->flags = GFX_TESS_MINIMUM;
    p->num_points = 3;
    return;
  }
  int npts = 0;
  for (int e = 0; e < 3; e++) {
    make_ctx(fo[e], odd_outer(*p, e), &p->outer[e]);
    p->out_pts[e] = num_points(fo[e], odd_outer(*p, e));
    npts += p->out_pts[e];
  }
  npts -= 3;
  const bool iodd = odd_inner(*p, 0);
  make_ctx(fi, iodd, &p->inner[0]);
  p->in_pts[0] = imax(iodd ? 4 : 3, num_points(fi, iodd));
  p->inside_base = npts;
  const int rings = (p->in_pts[0] >> 1) - 1;
  npts += iodd ? 3 * (rings * (rings + 1) - rings) : 3 * (rings * (rings + 1)) + 1;
  p->num_points = (uint32_t)npts;
}

static inline void tri_point(const gfx_tess_patch_t& p, int i, fxp_t* u, fxp_t* v) {
  if (i < p.inside_base) {
    int e = 0;
    while (i >= p.out_pts[e] - 1) {
      i -= p.out_pts[e] - 1;
      e++;
    }
    const int end = p.out_pts[e] - 1;
    const int q = (e & 1) ? i : end - i;
    const fxp_t t = place(p.outer[e], q, odd_outer(p, e));
    if (e == 0) {
      *u = 0;
      *v = t;
    } else {
      *u = t;
      *v = (e == 2) ? FXP_ONE - t : 0;
    }
    return;
  }
  const bool iodd = odd_inner(p, 0);
  const int rings = tri_point_rings(p);
  if (i >= tri_ring_start(p, rings)) {
    *u = FXP_ONE_THIRD;
    *v = FXP_ONE_THIRD;
    return;
  }
  int lo = 1, hi = rings - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (tri_ring_start(p, mid) <= i) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int k = lo;
  const int end = p.in_pts[0] - 1 - k;
  const int n = end - k;
  int j = i - tri_ring_start(p, k);
  const int e = j / n;
  j -= e * n;
  fxp_t tp = place(p.inner[0], k, iodd);
  tp *= FXP_TWO_THIRDS;
  tp = (tp + FXP_ONE_HALF) >> FXP_BITS;
  const int pt = k + j;
  const int q = (e & 1) ? pt : end - (pt - k);
  const fxp_t t = place(p.inner[0], q, iodd);
  const fxp_t s = t - (tp + 1) / 2u;
  if (e == 0) {
    *u = tp;
    *v = s;
  } else if (e == 1) {
    *u = s;
    *v = tp;
  } else {
    *u = s;
    *v = FXP_ONE - s - tp;
  }
}

// ---- isolines --------------------------------------------------------------

static inline void iso_process(uint32_t spacing, uint32_t out, float density, float detail,
                               gfx_tess_patch_t* p) {
  float lo, hi;
  level_bounds(spacing, &lo, &hi);
  const bool integer = spacing == GFX_TESS_SPACING_EQUAL;
  density = clampf(1.0f, 64.0f, density);
  detail = clampf(lo, hi, detail);
  bool dodd;
  if (integer) {
    detail = ceil_level(detail);
    dodd = !is_even(detail);
  } else {
    dodd = spacing == GFX_TESS_SPACING_ODD;
  }
  const fxp_t fdetail = to_fixed(detail);
  make_ctx(fdetail, dodd, &p->inner[0]);
  p->in_pts[0] = num_points(fdetail, dodd);
  density = ceil_level(density);
  const bool lodd = !is_even(density);
  const fxp_t fdensity = to_fixed(density);
  make_ctx(fdensity, lodd, &p->inner[1]);
  p->in_pts[1] = num_points(fdensity, lodd) - 1;
  p->parity = ((dodd ? 1u : 0u) << 4) | ((lodd ? 1u : 0u) << 5);
  p->num_points = (uint32_t)(p->in_pts[0] * p->in_pts[1]);
  if (out == GFX_TESS_OUT_POINT) {
    p->num_prims = p->num_points;
    p->num_units = 0;
  } else {
    p->num_prims = (uint32_t)(p->in_pts[1] * (p->in_pts[0] - 1));
    p->num_units = (uint32_t)p->in_pts[1];
  }
}

// ---- the patch --------------------------------------------------------------

// Process a patch's levels (outer[4], inner[2], as the control stage wrote
// them) for `mode`.
static inline void process(uint32_t mode, const float* outer, const float* inner,
                           gfx_tess_patch_t* p) {
  const uint32_t domain = GFX_TESS_MODE_DOMAIN(mode);
  const uint32_t spacing = GFX_TESS_MODE_SPACING(mode);
  const uint32_t out = GFX_TESS_MODE_OUT(mode);
  *p = gfx_tess_patch_t{};
  p->mode = mode;
  const int nedges = domain == GFX_TESS_DOMAIN_QUAD ? 4 : domain == GFX_TESS_DOMAIN_TRI ? 3 : 2;
  for (int e = 0; e < nedges; e++) {
    if (!gt_zero(outer[e])) {
      p->flags = GFX_TESS_CULLED;
      return;
    }
  }
  if (domain == GFX_TESS_DOMAIN_ISOLINE) {
    iso_process(spacing, out, outer[0], outer[1], p);
    return;
  }
  if (domain == GFX_TESS_DOMAIN_QUAD) {
    quad_process(spacing, outer, inner, p);
  } else {
    tri_process(spacing, outer, inner[0], p);
  }
  if (out == GFX_TESS_OUT_POINT) {
    p->num_prims = p->num_points;
    p->num_units = 0;
  } else if (p->flags & GFX_TESS_MINIMUM) {
    p->num_prims = domain == GFX_TESS_DOMAIN_QUAD ? 2 : 1;
    p->num_units = 1;
  } else if (domain == GFX_TESS_DOMAIN_QUAD) {
    quad_counts(p);
  } else {
    tri_counts(p);
  }
}

// Domain point i of a processed patch, (u, v) in 16.16.
static inline void point(const gfx_tess_patch_t& p, uint32_t i, fxp_t* u, fxp_t* v) {
  const uint32_t domain = GFX_TESS_MODE_DOMAIN(p.mode);
  if (domain == GFX_TESS_DOMAIN_ISOLINE) {
    const uint32_t ppl = (uint32_t)p.in_pts[0];
    const uint32_t line = i / ppl;
    *u = place(p.inner[0], (int)(i - line * ppl), odd_inner(p, 0));
    *v = place(p.inner[1], (int)line, odd_inner(p, 1));
    return;
  }
  if (p.flags & GFX_TESS_MINIMUM) {
    if (domain == GFX_TESS_DOMAIN_QUAD) {
      *u = (i == 1 || i == 2) ? FXP_ONE : 0;
      *v = (i >= 2) ? FXP_ONE : 0;
    } else {
      *u = (i == 2) ? FXP_ONE : 0;
      *v = (i == 0) ? FXP_ONE : 0;
    }
    return;
  }
  if (domain == GFX_TESS_DOMAIN_QUAD) {
    quad_point(p, (int)i, u, v);
  } else {
    tri_point(p, (int)i, u, v);
  }
}

// ---- stitching ---------------------------------------------------------------

// Where stitching writes a patch's primitives: entry k of the patch at
// list + k * stride, its point indices offset by base, and, in an entry of four
// words, `aux` in the last.
struct sink {
  uint32_t* list;
  uint32_t  stride;
  uint32_t  base;
  uint32_t  aux;
};

// Writes triangles of point indices in the patch's winding, through the
// reference's index patching: a ring's last edge wraps to its first points, and
// an even inside level's degenerate row is walked backwards.
struct emitter {
  sink     dst;
  uint32_t* out;
  uint32_t n;
  bool cw;
  int mode;   // 0 none, 1 wrap (IndexPatchContext), 2 invert (IndexPatchContext2)
  // wrap
  int in_delta, in_bad, in_repl, out_base, out_delta, out_bad, out_repl;
  // invert
  int inv_base, inv_bad, inv_repl, inv_end;

  void begin(uint32_t first) {
    out = dst.list + first * dst.stride;
  }

  int patch(int index) const {
    if (mode == 1) {
      if (index >= out_base) {
        return index == out_bad ? out_repl : index + out_delta;
      }
      return index == in_bad ? in_repl : index + in_delta;
    }
    if (mode == 2) {
      if (index >= inv_base) {
        return index == inv_bad ? inv_repl : inv_end - index;
      }
      if (index == inv_bad) {
        return inv_repl;
      }
    }
    return index;
  }

  void put(uint32_t i0, uint32_t i1, uint32_t i2) {
    out[0] = dst.base + i0;
    out[1] = dst.base + i1;
    if (dst.stride > 2) {
      out[2] = dst.base + i2;
    }
    if (dst.stride > 3) {
      out[3] = dst.aux;
    }
    out += dst.stride;
    n++;
  }

  void tri(int i0, int i1, int i2) {
    if (cw) {
      put((uint32_t)patch(i0), (uint32_t)patch(i1), (uint32_t)patch(i2));
    } else {
      put((uint32_t)patch(i0), (uint32_t)patch(i2), (uint32_t)patch(i1));
    }
  }
};

enum { DIAG_INSIDE_TO_OUTSIDE, DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE, DIAG_MIRRORED };

static inline void stitch_regular(emitter& em, bool trapezoid, int diag, int n_inside,
                                  int inside, int outside) {
  if (trapezoid) {
    em.tri(outside, outside + 1, inside);
    outside++;
  }
  int p;
  switch (diag) {
  case DIAG_INSIDE_TO_OUTSIDE:
    for (p = 0; p < n_inside - 1; p++) {
      em.tri(inside, outside, outside + 1);
      em.tri(inside, outside + 1, inside + 1);
      inside++;
      outside++;
    }
    break;
  case DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE:
    for (p = 0; p < n_inside / 2 - 1; p++) {
      em.tri(outside, outside + 1, inside);
      em.tri(inside, outside + 1, inside + 1);
      inside++;
      outside++;
    }
    em.tri(outside, inside + 1, inside);
    em.tri(outside, outside + 1, inside + 1);
    inside++;
    outside++;
    p += 2;
    for (; p < n_inside; p++) {
      em.tri(outside, outside + 1, inside);
      em.tri(inside, outside + 1, inside + 1);
      inside++;
      outside++;
    }
    break;
  default:
    for (p = 0; p < n_inside / 2; p++) {
      em.tri(outside, inside + 1, inside);
      em.tri(outside, outside + 1, inside + 1);
      inside++;
      outside++;
    }
    for (; p < n_inside - 1; p++) {
      em.tri(inside, outside, outside + 1);
      em.tri(inside, outside + 1, inside + 1);
      inside++;
      outside++;
    }
    break;
  }
  if (trapezoid) {
    em.tri(outside, outside + 1, inside);
  }
}

static inline void stitch_transition(emitter& em, int inside, int in_half, bool in_odd,
                                     int outside, int out_half, bool out_odd) {
  if (in_odd) {
    in_half -= 1;
  }
  if (out_odd) {
    out_half -= 1;
  }
  const int start = imin(LOOP_START[in_half], LOOP_START[out_half]);
  const int end = imax(LOOP_END[in_half], LOOP_END[out_half]);
  if (FINAL_POSITION[0] < out_half) {
    em.tri(outside, outside + 1, inside);
    outside++;
  }
  for (int i = start; i <= end; i++) {
    if (FINAL_POSITION[i] < in_half) {
      em.tri(inside, outside, inside + 1);
      inside++;
    }
    if (FINAL_POSITION[i] < out_half) {
      em.tri(outside, outside + 1, inside);
      outside++;
    }
  }
  if (in_odd != out_odd || in_odd) {
    if (in_odd == out_odd) {
      em.tri(inside, outside, inside + 1);
      em.tri(inside + 1, outside, outside + 1);
      inside++;
      outside++;
    } else if (!in_odd) {
      em.tri(inside, outside, outside + 1);
      outside++;
    } else {
      em.tri(inside, outside, inside + 1);
      inside++;
    }
  }
  for (int i = end; i >= start; i--) {
    if (FINAL_POSITION[i] < out_half) {
      em.tri(outside, outside + 1, inside);
      outside++;
    }
    if (FINAL_POSITION[i] < in_half) {
      em.tri(inside, outside, inside + 1);
      inside++;
    }
  }
  if (FINAL_POSITION[0] < out_half) {
    em.tri(outside, outside + 1, inside);
  }
}

// Points on edge e's outside row in stitch ring r: the outer edge's, then the
// previous ring's inside row.
static inline int quad_outside_pts(const gfx_tess_patch_t& p, int r, int e) {
  return r == 1 ? p.out_pts[e] : p.in_pts[(e + 1) & 1] - 2 * (r - 1);
}

static inline uint32_t quad_unit(const gfx_tess_patch_t& p, uint32_t s, emitter& em) {
  const int rings = quad_stitch_rings(p);
  const int nu = p.in_pts[0], nv = p.in_pts[1];
  if ((int)s >= 4 * (rings - 1)) {
    // the centre strip, after every ring
    bool along_u;
    const int strip = quad_strip(p, &along_u);
    const int outside = rings > 1 ? quad_ring_start(p, rings - 1) : 0;
    em.begin((uint32_t)quad_prim_start(p, rings));
    em.mode = 2;
    if (along_u) {
      em.inv_base = outside + strip + 2;
      em.inv_bad = em.inv_base;
      em.inv_repl = outside;
      em.inv_end = em.inv_base + em.inv_base + strip;
      stitch_regular(em, false, DIAG_INSIDE_TO_OUTSIDE, strip + 1, em.inv_base, outside + 1);
    } else {
      em.inv_base = outside + strip + 1;
      em.inv_bad = -1;
      em.inv_repl = 0;
      em.inv_end = em.inv_base + em.inv_base + strip;
      const int diag = odd_inner(p, 1) ? DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE
                                       : DIAG_INSIDE_TO_OUTSIDE;
      stitch_regular(em, false, diag, strip + 1, em.inv_base, outside);
    }
    return em.n;
  }
  const int r = 1 + (int)s / 4;
  const int edge = (int)s % 4;
  const int rows[2] = { (nu + 1) >> 1, (nv + 1) >> 1 };
  const int degenerate[2] = { odd_inner(p, 1) ? -1 : rows[1] - 1, odd_inner(p, 0) ? -1 : rows[0] - 1 };
  const int n_in[2] = { nu - 2 * r, nv - 2 * r };
  const int edge0_inside = quad_ring_start(p, r);
  const int edge0_outside = r == 1 ? 0 : quad_ring_start(p, r - 1);
  int inside_at = edge0_inside, outside_at = edge0_outside;
  int tri0 = quad_prim_start(p, r);
  for (int e = 0; e < edge; e++) {
    const int par = (e + 1) & 1;
    tri0 += n_in[par] + quad_outside_pts(p, r, e) - 2;
    outside_at += quad_outside_pts(p, r, e) - 1;
    if (e == 2 && r == degenerate[par]) {
      inside_at -= n_in[par] - 1;
    } else {
      inside_at += n_in[par] - 1;
    }
  }
  em.begin((uint32_t)tri0);
  const int par = (edge + 1) & 1;
  const int n_out = quad_outside_pts(p, r, edge);
  int inside_base, outside_base;
  if (edge == 3) {
    if (r == degenerate[par]) {
      em.mode = 2;
      em.inv_base = inside_at + 1;
      em.inv_bad = outside_at + n_out - 1;
      em.inv_repl = edge0_outside;
      em.inv_end = (em.inv_base << 1) - 1;
      inside_base = em.inv_base;
      outside_base = outside_at;
    } else {
      em.mode = 1;
      em.in_delta = inside_at;
      em.in_bad = n_in[par] - 1;
      em.in_repl = edge0_inside;
      em.out_base = em.in_bad + 1;
      em.out_delta = outside_at - em.out_base;
      em.out_bad = em.out_base + n_out - 1;
      em.out_repl = edge0_outside;
      inside_base = 0;
      outside_base = em.out_base;
    }
  } else if (edge == 2 && r == degenerate[par]) {
    em.mode = 2;
    em.inv_base = inside_at;
    em.inv_bad = -1;
    em.inv_repl = -1;
    em.inv_end = em.inv_base << 1;
    inside_base = em.inv_base;
    outside_base = outside_at;
  } else {
    inside_base = inside_at;
    outside_base = outside_at;
  }
  if (r == 1) {
    stitch_transition(em, inside_base, p.inner[par].nhalf, odd_inner(p, par), outside_base,
                      p.outer[edge].nhalf, odd_outer(p, edge));
  } else {
    stitch_regular(em, true, DIAG_MIRRORED, n_in[par], inside_base, outside_base);
  }
  return em.n;
}

static inline int tri_outside_pts(const gfx_tess_patch_t& p, int r, int e) {
  return r == 1 ? p.out_pts[e] : p.in_pts[0] - 2 * (r - 1);
}

static inline uint32_t tri_unit(const gfx_tess_patch_t& p, uint32_t s, emitter& em) {
  const int rings = tri_stitch_rings(p);
  if ((int)s >= 3 * (rings - 1)) {
    const int o = rings > 1 ? tri_ring_start(p, rings - 1) : 0;
    em.begin((uint32_t)tri_prim_start(p, rings));
    em.tri(o, o + 1, o + 2);
    return 1;
  }
  const int r = 1 + (int)s / 3;
  const int edge = (int)s % 3;
  const int n_in = p.in_pts[0] - 2 * r;
  const int edge0_inside = tri_ring_start(p, r);
  const int edge0_outside = r == 1 ? 0 : tri_ring_start(p, r - 1);
  int inside_at = edge0_inside, outside_at = edge0_outside;
  int tri0 = tri_prim_start(p, r);
  for (int e = 0; e < edge; e++) {
    tri0 += n_in + tri_outside_pts(p, r, e) - 2;
    outside_at += tri_outside_pts(p, r, e) - 1;
    inside_at += n_in - 1;
  }
  em.begin((uint32_t)tri0);
  const int n_out = tri_outside_pts(p, r, edge);
  int inside_base = inside_at, outside_base = outside_at;
  if (edge == 2) {
    em.mode = 1;
    em.in_delta = inside_at;
    em.in_bad = n_in - 1;
    em.in_repl = edge0_inside;
    em.out_base = em.in_bad + 1;
    em.out_delta = outside_at - em.out_base;
    em.out_bad = em.out_base + n_out - 1;
    em.out_repl = edge0_outside;
    inside_base = 0;
    outside_base = em.out_base;
  }
  if (r == 1) {
    stitch_transition(em, inside_base, p.inner[0].nhalf, odd_inner(p, 0), outside_base,
                      p.outer[edge].nhalf, odd_outer(p, edge));
  } else {
    stitch_regular(em, true, DIAG_MIRRORED, n_in, inside_base, outside_base);
  }
  return em.n;
}

// Stitch unit s of a processed patch (s < num_units): writes its primitives at
// their place among the patch's in `dst` -- a triangle as three point indices,
// a line as two (with the second repeated in a third word, if any) -- and
// returns how many it wrote: at most 2 * 64 + 2 triangles, or 63 lines.
static inline uint32_t unit(const gfx_tess_patch_t& p, uint32_t s, const sink& dst) {
  emitter em = {};
  em.dst = dst;
  const uint32_t domain = GFX_TESS_MODE_DOMAIN(p.mode);
  if (domain == GFX_TESS_DOMAIN_ISOLINE) {
    const uint32_t ppl = (uint32_t)p.in_pts[0];
    em.begin(s * (ppl - 1));
    for (uint32_t k = 0; k + 1 < ppl; k++) {
      em.put(s * ppl + k, s * ppl + k + 1, s * ppl + k + 1);
    }
    return em.n;
  }
  em.cw = GFX_TESS_MODE_OUT(p.mode) == GFX_TESS_OUT_TRI_CW;
  if (p.flags & GFX_TESS_MINIMUM) {
    em.begin(0);
    if (domain == GFX_TESS_DOMAIN_QUAD) {
      em.tri(0, 1, 3);
      em.tri(1, 2, 3);
    } else {
      em.tri(0, 1, 2);
    }
    return em.n;
  }
  if (domain == GFX_TESS_DOMAIN_QUAD) {
    return quad_unit(p, s, em);
  }
  return tri_unit(p, s, em);
}

// ---- batches --------------------------------------------------------------------

// The patch whose range of lane `lane` holds element `idx`: the last with its
// exclusive prefix at or below idx, which owns at least one element.
static inline uint32_t locate(const uint32_t* scan, uint32_t patches, uint32_t lane,
                              uint32_t idx) {
  uint32_t lo = 0, hi = patches - 1;
  while (lo < hi) {
    const uint32_t mid = (lo + hi + 1) >> 1;
    if (scan[3 * mid + lane] <= idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

static inline const gfx_tess_patch_t* descriptor(const gfx_tess_draw_t& d, uint32_t p) {
  return reinterpret_cast<const gfx_tess_patch_t*>(
      (uintptr_t)d.desc_addr + (uintptr_t)p * d.desc_stride);
}

// gl_PrimitiveID of batch patch p when the patches share one descriptor; a
// batch's own descriptors carry theirs.
static inline uint32_t primitive_id(const gfx_tess_draw_t& d, uint32_t p) {
  const uint32_t q = d.patch_base + p;
  return d.inst_patches ? q % d.inst_patches : q;
}

} // namespace gfx_tess

#endif // __cplusplus
