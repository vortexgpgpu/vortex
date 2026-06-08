#ifndef _SETUP_MATH_H_
#define _SETUP_MATH_H_

// gfx_v2 on-device triangle setup — shared host/device math.
//
// A freestanding port of host Binning()'s per-triangle setup
// (sw/runtime/graphics.cpp): clip->HDC edge equations, half-pixel offset,
// Q15.16 edge fixed-point, Q7.24 affine attribute deltas, and the screen-space
// bbox. Identical source on both sides, so the kernel test isolates exactly
// the codegen/FP-rounding axis; the host reference is in turn anchored against
// the real Binning() (main.cpp), proving this math faithful to the gfx-v1
// oracle. No <cmath>/<algorithm> so the device (baremetal) can include it.

#include <stdint.h>
#include <vx_gfx_abi.h>   // vortex::graphics::{rast_prim_t, vec3e_t, FloatE, FloatA}
#include "common.h"       // setup_vertex_t, setup_bbox_t

namespace gfx_setup {

using vortex::graphics::rast_prim_t;
using vortex::graphics::rast_attrib_t;
using vortex::graphics::vec3e_t;
using vortex::graphics::FloatE;
using vortex::graphics::FloatA;

struct vec4f { float x, y, z, w; };
struct vec3f { float x, y, z; };

// Near-plane clip distance: GL near plane is z_clip + w_clip >= 0. A kept
// vertex projects to screen-z = 0 (near) with finite x/y for w > 0 — unlike a
// w>=eps cut, which sits at the projection singularity.
static inline float near_dist(const setup_vertex_t& v) { return v.pos[2] + v.pos[3]; }

// Linear (clip-space) interpolation of a vertex's position and varyings.
static inline setup_vertex_t vertex_lerp(const setup_vertex_t& a,
                                         const setup_vertex_t& b, float t) {
  setup_vertex_t r;
  for (int i = 0; i < 4; ++i) r.pos[i]      = a.pos[i]      + t * (b.pos[i]      - a.pos[i]);
  for (int i = 0; i < 4; ++i) r.color[i]    = a.color[i]    + t * (b.color[i]    - a.color[i]);
  for (int i = 0; i < 2; ++i) r.texcoord[i] = a.texcoord[i] + t * (b.texcoord[i] - a.texcoord[i]);
  return r;
}

// Sutherland-Hodgman clip of triangle (a,b,c) against the near plane, then
// fan-triangulate the (3- or 4-vertex) result. Writes up to SETUP_MAX_SUB
// subtriangles to out[] and returns the count (0/1/2). Vertex order is
// preserved so winding is consistent (setup_triangle normalizes it anyway).
static inline int clip_near(const setup_vertex_t& a, const setup_vertex_t& b,
                            const setup_vertex_t& c, clip_tri_t out[SETUP_MAX_SUB]) {
  const setup_vertex_t in[3] = { a, b, c };
  setup_vertex_t poly[4];
  int pn = 0;
  for (int i = 0; i < 3; ++i) {
    const setup_vertex_t& cur = in[i];
    const setup_vertex_t& nxt = in[(i + 1) % 3];
    float fc = near_dist(cur), fn = near_dist(nxt);
    bool cur_in = fc >= 0.0f, nxt_in = fn >= 0.0f;
    if (cur_in) poly[pn++] = cur;
    if (cur_in != nxt_in)
      poly[pn++] = vertex_lerp(cur, nxt, fc / (fc - fn));  // crossing: near_dist == 0
  }
  int k = 0;
  for (int i = 1; i + 1 < pn; ++i) {  // fan from poly[0]: pn=3 -> 1, pn=4 -> 2
    out[k].v[0] = poly[0]; out[k].v[1] = poly[i]; out[k].v[2] = poly[i + 1];
    ++k;
  }
  return k;
}

static inline float fmin2(float a, float b) { return a < b ? a : b; }
static inline float fmax2(float a, float b) { return a > b ? a : b; }
static inline int   imin2(int a, int b)     { return a < b ? a : b; }
static inline int   imax2(int a, int b)     { return a > b ? a : b; }

// Clip-space (x,y,z,w) -> homogeneous-device-coordinate; W preserved so edge
// equations stay perspective-correct (graphics.cpp ClipToHDC).
static inline vec4f ClipToHDC(const vec4f& in, int left, int right,
                              int top, int bottom, float near, float far) {
  float minX   = (float)(left + right) * 0.5f;
  float scaleX = (float)(right - left) * 0.5f;
  float minY   = (float)(top + bottom) * 0.5f;
  float scaleY = (float)(bottom - top) * 0.5f;
  float minZ   = (near + far) * 0.5f;
  float scaleZ = (far - near) * 0.5f;
  return { in.x * scaleX + in.w * minX,
           in.y * scaleY + in.w * minY,
           in.z * scaleZ + in.w * minZ,
           in.w };
}

// Clip-space -> screen space (perspective divide then viewport).
static inline vec4f ClipToScreen(const vec4f& in, int left, int right,
                                 int top, int bottom, float near, float far) {
  float rhw = (in.w != 0.0f) ? (1.0f / in.w) : 0.0f;
  float nx = in.x * rhw, ny = in.y * rhw, nz = in.z * rhw;
  float minX   = (float)(left + right) * 0.5f;
  float scaleX = (float)(right - left) * 0.5f;
  float minY   = (float)(top + bottom) * 0.5f;
  float scaleY = (float)(bottom - top) * 0.5f;
  float minZ   = (near + far) * 0.5f;
  float scaleZ = (far - near) * 0.5f;
  return { nx * scaleX + minX, ny * scaleY + minY, nz * scaleZ + minZ, rhw };
}

// Edge functions (a,b,c) per edge in HDC; winding flipped so interior is
// positive; returns false for the degenerate (det==0) triangle.
static inline bool EdgeEquation(vec3f edges[3], const vec4f& v0,
                                const vec4f& v1, const vec4f& v2) {
  float a0 = (v1.y * v2.w) - (v2.y * v1.w);
  float a1 = (v2.y * v0.w) - (v0.y * v2.w);
  float a2 = (v0.y * v1.w) - (v1.y * v0.w);
  float b0 = (v2.x * v1.w) - (v1.x * v2.w);
  float b1 = (v0.x * v2.w) - (v2.x * v0.w);
  float b2 = (v1.x * v0.w) - (v0.x * v1.w);
  float c0 = (v1.x * v2.y) - (v2.x * v1.y);
  float c1 = (v2.x * v0.y) - (v0.x * v2.y);
  float c2 = (v0.x * v1.y) - (v1.x * v0.y);
  edges[0] = {a0, b0, c0};
  edges[1] = {a1, b1, c1};
  edges[2] = {a2, b2, c2};
  float det = c0 * v0.w + c1 * v1.w + c2 * v2.w;
  if (det < 0) {
    for (int i = 0; i < 3; ++i) {
      edges[i].x *= -1.0f; edges[i].y *= -1.0f; edges[i].z *= -1.0f;
    }
  }
  return (det != 0.0f);
}

// Normalize the edge matrix by 1/maxVal and convert to Q15.16.
static inline void EdgeToFixed(vec3e_t out[3], const vec3f in[3]) {
  float maxVal = __builtin_fabsf(in[0].x);
  maxVal = fmax2(maxVal, __builtin_fabsf(in[1].x));
  maxVal = fmax2(maxVal, __builtin_fabsf(in[2].x));
  maxVal = fmax2(maxVal, __builtin_fabsf(in[0].y));
  maxVal = fmax2(maxVal, __builtin_fabsf(in[1].y));
  maxVal = fmax2(maxVal, __builtin_fabsf(in[2].y));
  float scale = (maxVal != 0.0f) ? (1.0f / maxVal) : 1.0f;
  for (int i = 0; i < 3; ++i) {
    out[i].x = FloatE(in[i].x * scale);
    out[i].y = FloatE(in[i].y * scale);
    out[i].z = FloatE(in[i].z * scale);
  }
}

// Full per-triangle setup. Fills out_prim (edges + attrib deltas) and out_bbox
// (screen bbox clamped to the render target). Returns false when the triangle
// is culled (degenerate or empty bbox) — matching Binning()'s two `continue`s.
static inline bool setup_triangle(const setup_vertex_t& v0,
                                  const setup_vertex_t& v1,
                                  const setup_vertex_t& v2,
                                  int width, int height, float near, float far,
                                  rast_prim_t& out_prim, setup_bbox_t& out_bbox) {
  vec4f p0 = { v0.pos[0], v0.pos[1], v0.pos[2], v0.pos[3] };
  vec4f p1 = { v1.pos[0], v1.pos[1], v1.pos[2], v1.pos[3] };
  vec4f p2 = { v2.pos[0], v2.pos[1], v2.pos[2], v2.pos[3] };

  // Edge equations in HDC.
  vec3f edges[3];
  {
    vec4f ph0 = ClipToHDC(p0, 0, width, 0, height, near, far);
    vec4f ph1 = ClipToHDC(p1, 0, width, 0, height, near, far);
    vec4f ph2 = ClipToHDC(p2, 0, width, 0, height, near, far);
    if (!EdgeEquation(edges, ph0, ph1, ph2))
      return false;  // degenerate
  }

  // Screen-space bbox (clamped to render target).
  vec4f ps0 = ClipToScreen(p0, 0, width, 0, height, near, far);
  vec4f ps1 = ClipToScreen(p1, 0, width, 0, height, near, far);
  vec4f ps2 = ClipToScreen(p2, 0, width, 0, height, near, far);
  {
    float left   = fmin2(ps0.x, fmin2(ps1.x, ps2.x));
    float right  = fmax2(ps0.x, fmax2(ps1.x, ps2.x));
    float top    = fmin2(ps0.y, fmin2(ps1.y, ps2.y));
    float bottom = fmax2(ps0.y, fmax2(ps1.y, ps2.y));
    int l = (int)__builtin_floorf(left);
    int r = (int)__builtin_ceilf(right);
    int t = (int)__builtin_floorf(top);
    int b = (int)__builtin_ceilf(bottom);
    out_bbox.bbL = (uint32_t)imax2(l, 0);
    out_bbox.bbR = (uint32_t)imin2(r, width);
    out_bbox.bbT = (uint32_t)imax2(t, 0);
    out_bbox.bbB = (uint32_t)imin2(b, height);
    if ((int)out_bbox.bbR <= (int)out_bbox.bbL ||
        (int)out_bbox.bbB <= (int)out_bbox.bbT)
      return false;  // empty
  }

  // Half-pixel sampling offset, then Q15.16 edges.
  edges[0].z += edges[0].x * 0.5f + edges[0].y * 0.5f;
  edges[1].z += edges[1].x * 0.5f + edges[1].y * 0.5f;
  edges[2].z += edges[2].x * 0.5f + edges[2].y * 0.5f;
  EdgeToFixed(out_prim.edges, edges);

  // Affine attribute deltas (a0-a2, a1-a2, a2) in Q7.24.
  auto delta = [](rast_attrib_t& d, float a0, float a1, float a2) {
    d.x = FloatA(a0 - a2);
    d.y = FloatA(a1 - a2);
    d.z = FloatA(a2);
  };
  delta(out_prim.attribs.z, ps0.z,          ps1.z,          ps2.z);
  delta(out_prim.attribs.r, v0.color[0],    v1.color[0],    v2.color[0]);
  delta(out_prim.attribs.g, v0.color[1],    v1.color[1],    v2.color[1]);
  delta(out_prim.attribs.b, v0.color[2],    v1.color[2],    v2.color[2]);
  delta(out_prim.attribs.a, v0.color[3],    v1.color[3],    v2.color[3]);
  delta(out_prim.attribs.u, v0.texcoord[0], v1.texcoord[0], v2.texcoord[0]);
  delta(out_prim.attribs.v, v0.texcoord[1], v1.texcoord[1], v2.texcoord[1]);
  return true;
}

} // namespace gfx_setup

#endif
