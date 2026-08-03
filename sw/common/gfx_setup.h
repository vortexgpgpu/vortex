#pragma once

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
#include <vx_gfx_abi.h>        // vortex::graphics::{rast_prim_t, vec3e_t, FloatE, FloatA}
#include <gfx_frontend_abi.h>  // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

namespace gfx_setup {

using vortex::graphics::rast_prim_t;
using vortex::graphics::rast_attrib_t;
using vortex::graphics::vec3e_t;
using vortex::graphics::FloatE;
using vortex::graphics::FloatA;

struct vec4f { float x, y, z, w; };
struct vec3f { float x, y, z; };

// Near-plane clip distance. GL's near plane is z_clip + w_clip >= 0; Vulkan's,
// whose clip z already starts at 0, is z_clip >= 0. A kept vertex projects to
// screen-z = near with finite x/y for w > 0 — unlike a w>=eps cut, which sits at
// the projection singularity.
static inline float near_dist(const setup_vertex_t& v, bool halfz = false) {
  return halfz ? v.pos[2] : (v.pos[2] + v.pos[3]);
}

// Linear (clip-space) interpolation of a vertex's position and varyings.
static inline setup_vertex_t vertex_lerp(const setup_vertex_t& a,
                                         const setup_vertex_t& b, float t) {
  setup_vertex_t r;
  for (int i = 0; i < 4; ++i) r.pos[i]      = a.pos[i]      + t * (b.pos[i]      - a.pos[i]);
  for (int i = 0; i < 4; ++i) r.color[i]    = a.color[i]    + t * (b.color[i]    - a.color[i]);
  for (int i = 0; i < 2; ++i) r.texcoord[i] = a.texcoord[i] + t * (b.texcoord[i] - a.texcoord[i]);
  for (int i = 0; i < 6; ++i) r.varying2[i] = a.varying2[i] + t * (b.varying2[i] - a.varying2[i]);
  return r;
}

// Sutherland-Hodgman clip of triangle (a,b,c) against the near plane, then
// fan-triangulate the (3- or 4-vertex) result. Writes up to SETUP_MAX_SUB
// subtriangles to out[] and returns the count (0/1/2). Vertex order is
// preserved so winding is consistent (setup_triangle normalizes it anyway).
static inline int clip_near(const setup_vertex_t& a, const setup_vertex_t& b,
                            const setup_vertex_t& c, clip_tri_t out[SETUP_MAX_SUB],
                            bool halfz = false) {
  const setup_vertex_t in[3] = { a, b, c };
  setup_vertex_t poly[4];
  int pn = 0;
  for (int i = 0; i < 3; ++i) {
    const setup_vertex_t& cur = in[i];
    const setup_vertex_t& nxt = in[(i + 1) % 3];
    float fc = near_dist(cur, halfz), fn = near_dist(nxt, halfz);
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
// Viewport bounds are float so a fractional / y-flipped (top>bottom) app
// viewport transforms exactly; a full-framebuffer viewport passes integer
// bounds unchanged.
static inline vec4f ClipToHDC(const vec4f& in, float left, float right,
                              float top, float bottom, float near, float far,
                              bool halfz = false) {
  float minX   = (left + right) * 0.5f;
  float scaleX = (right - left) * 0.5f;
  float minY   = (top + bottom) * 0.5f;
  float scaleY = (bottom - top) * 0.5f;
  // GL maps [-1,1] onto [near,far]; Vulkan's clip z is already 0-based.
  float minZ   = halfz ? near : (near + far) * 0.5f;
  float scaleZ = halfz ? (far - near) : (far - near) * 0.5f;
  return { in.x * scaleX + in.w * minX,
           in.y * scaleY + in.w * minY,
           in.z * scaleZ + in.w * minZ,
           in.w };
}

// Clip-space -> screen space (perspective divide then viewport).
static inline vec4f ClipToScreen(const vec4f& in, float left, float right,
                                 float top, float bottom, float near, float far,
                                 bool halfz = false) {
  float rhw = (in.w != 0.0f) ? (1.0f / in.w) : 0.0f;
  float nx = in.x * rhw, ny = in.y * rhw, nz = in.z * rhw;
  float minX   = (left + right) * 0.5f;
  float scaleX = (right - left) * 0.5f;
  float minY   = (top + bottom) * 0.5f;
  float scaleY = (bottom - top) * 0.5f;
  float minZ   = halfz ? near : (near + far) * 0.5f;
  float scaleZ = halfz ? (far - near) : (far - near) * 0.5f;
  return { nx * scaleX + minX, ny * scaleY + minY, nz * scaleZ + minZ, rhw };
}

// Edge functions (a,b,c) per edge in HDC; winding flipped so interior is
// positive. Returns false for the degenerate (det==0) triangle, and — under a
// face-cull mode (SETUP_CULL_*) — for the culled winding (BACK = negative
// area, FRONT = positive). cull_mode defaults to two-sided (no cull).
static inline bool EdgeEquation(vec3f edges[3], const vec4f& v0,
                                const vec4f& v1, const vec4f& v2,
                                uint32_t cull_mode = SETUP_CULL_NONE) {
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
  if (det == 0.0f)
    return false;                          // degenerate (zero area)
  const bool back = (det < 0.0f);          // winding that needs a flip
  if ((cull_mode == SETUP_CULL_BACK  &&  back) ||
      (cull_mode == SETUP_CULL_FRONT && !back))
    return false;                          // face-culled
  if (back) {
    for (int i = 0; i < 3; ++i) {
      edges[i].x *= -1.0f; edges[i].y *= -1.0f; edges[i].z *= -1.0f;
    }
  }
  return true;
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
// vp carries the app viewport transform; nullptr => the default full-framebuffer
// y-down viewport (screen = ndc mapped onto [0,width] x [0,height], y-down).
static inline bool setup_triangle(const setup_vertex_t& v0,
                                  const setup_vertex_t& v1,
                                  const setup_vertex_t& v2,
                                  int width, int height, float near, float far,
                                  rast_prim_t& out_prim, setup_bbox_t& out_bbox,
                                  uint32_t cull_mode = SETUP_CULL_NONE,
                                  const setup_viewport_t* vp = nullptr) {
  vec4f p0 = { v0.pos[0], v0.pos[1], v0.pos[2], v0.pos[3] };
  vec4f p1 = { v1.pos[0], v1.pos[1], v1.pos[2], v1.pos[3] };
  vec4f p2 = { v2.pos[0], v2.pos[1], v2.pos[2], v2.pos[3] };

  // Viewport rect in screen pixels (left,right,top,bottom). From the app
  // transform (screen = ndc*scale + bias => bounds = bias ± scale) when bound;
  // else the default full-framebuffer y-down mapping. A y-flip (sy<0) yields
  // top>bottom, which flips screen-space Y — and hence the signed-area sign the
  // face cull reads — exactly as the app's negative-height viewport intends.
  float L = vp ? (vp->tx - vp->sx) : 0.0f;
  float R = vp ? (vp->tx + vp->sx) : (float)width;
  float T = vp ? (vp->ty - vp->sy) : 0.0f;
  float B = vp ? (vp->ty + vp->sy) : (float)height;

  // The app viewport also carries the depth range and the clip-z convention.
  // Without one bound, the near/far arguments apply under the GL convention —
  // the standalone setup tests and the native runtime both rely on that.
  const bool  halfz = vp && vp->halfz;
  // A viewport whose depth range is all zero never had one set, so fall back to
  // the near/far arguments rather than collapsing every fragment onto one plane.
  const bool  vp_has_z = vp && (vp->minz != 0.0f || vp->maxz != 0.0f);
  const float zn = vp_has_z ? vp->minz : near;
  const float zf = vp_has_z ? vp->maxz : far;

  // Edge equations in HDC.
  vec3f edges[3];
  {
    vec4f ph0 = ClipToHDC(p0, L, R, T, B, zn, zf, halfz);
    vec4f ph1 = ClipToHDC(p1, L, R, T, B, zn, zf, halfz);
    vec4f ph2 = ClipToHDC(p2, L, R, T, B, zn, zf, halfz);
    if (!EdgeEquation(edges, ph0, ph1, ph2, cull_mode))
      return false;  // degenerate or face-culled
  }

  // Screen-space bbox (clamped to render target).
  vec4f ps0 = ClipToScreen(p0, L, R, T, B, zn, zf, halfz);
  vec4f ps1 = ClipToScreen(p1, L, R, T, B, zn, zf, halfz);
  vec4f ps2 = ClipToScreen(p2, L, R, T, B, zn, zf, halfz);
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
  // Depth is a fixed-function screen-space plane: post-projection Z is affine in
  // the screen (x,y) of a planar triangle, so the FF path interpolates it as a
  // plane Z = A'*x + B'*y + C' rather than a per-pixel perspective barycentric
  // (which is not affine when w varies). The plane is solved directly from the
  // three screen vertices; the raster early-Z and the FS evaluate the identical
  // fixed-point MAC, so early-Z and late-Z agree bit-for-bit. attribs.z carries
  // {x:A', y:B', z:C'} (not the {a0-a2,a1-a2,a2} deltas used by the other attrs).
  {
    float twice_area = (ps1.x - ps0.x) * (ps2.y - ps0.y)
                     - (ps2.x - ps0.x) * (ps1.y - ps0.y);
    float inv = (twice_area != 0.0f) ? (1.0f / twice_area) : 0.0f;
    float za = ((ps1.z - ps0.z) * (ps2.y - ps0.y)
              - (ps2.z - ps0.z) * (ps1.y - ps0.y)) * inv;   // dz/dx
    float zb = ((ps2.z - ps0.z) * (ps1.x - ps0.x)
              - (ps1.z - ps0.z) * (ps2.x - ps0.x)) * inv;   // dz/dy
    // C' evaluated so an integer-pixel MAC samples Z at the pixel center
    // (X+0.5, Y+0.5) — matching the +0.5 half-pixel offset the coverage edges use.
    float zc = ps0.z - za * ps0.x - zb * ps0.y + 0.5f * (za + zb);
    out_prim.attribs.z.x = FloatA(za);
    out_prim.attribs.z.y = FloatA(zb);
    out_prim.attribs.z.z = FloatA(zc);
  }
  // Perspective-correct varyings: premultiply each colour/texcoord by the
  // vertex's 1/w and carry a separate 1/w plane; the FS interpolates every plane
  // affinely in screen space then divides the colour/uv planes by the
  // interpolated 1/w to recover the perspective-correct attribute. 1/w is
  // constant across a screen-aligned triangle, so this reduces exactly to affine
  // interpolation there. Depth (attribs.z) stays a pure screen-space plane.
  float rhw0 = (p0.w != 0.0f) ? (1.0f / p0.w) : 0.0f;
  float rhw1 = (p1.w != 0.0f) ? (1.0f / p1.w) : 0.0f;
  float rhw2 = (p2.w != 0.0f) ? (1.0f / p2.w) : 0.0f;
  // Normalize the three 1/w by their max magnitude before storing: 1/w is
  // unbounded as w->0 (the near clip bounds z+w, not w), so raw a·(1/w) / (1/w)
  // would overflow the Q7.24 range for near geometry. The common scale cancels
  // in the FS divide interp(a/w)/interp(1/w), so this is exact — not lossy.
  float rhw_max = fmax2(fmax2(__builtin_fabsf(rhw0), __builtin_fabsf(rhw1)),
                        __builtin_fabsf(rhw2));
  float rhw_scale = (rhw_max != 0.0f) ? (1.0f / rhw_max) : 0.0f;
  rhw0 *= rhw_scale; rhw1 *= rhw_scale; rhw2 *= rhw_scale;
  // Guard the fixed-point range of the premultiplied planes. Colours are <=1, but
  // texcoords (tiling/wrap UV well beyond 1.0) can drive a*(1/w) past FloatA's
  // ~+-128 Q7.24 range, silently wrapping to garbage UV. Every perspective plane
  // (r,g,b,a,u,v,rhw) carries the SAME rhw scale and the FS recovers the attribute
  // as interp(a*rhw)/interp(rhw), so an extra common power-of-2 downscale on
  // rhw0/1/2 cancels exactly in that divide — exact, not lossy. Fold in the
  // smallest power of 2 that keeps the emitted deltas (up to 2x a vertex value) in
  // range; when the UV is already in range no scaling is applied (byte-identical to
  // the un-guarded path, so in-range colour/UV and the perspective goldens are
  // untouched). Only the texcoords are probed: colour is [0,1], and the current
  // varying2 users (cube textureGrad coord/gradients) stay well inside range, so
  // probing texcoords alone keeps native/host draws (which may leave varying2
  // unset) bit-identical — the unread w0..w5 planes never reach the framebuffer.
  {
    float attr_max = 0.0f;
    attr_max = fmax2(attr_max, __builtin_fabsf(v0.texcoord[0] * rhw0));
    attr_max = fmax2(attr_max, __builtin_fabsf(v1.texcoord[0] * rhw1));
    attr_max = fmax2(attr_max, __builtin_fabsf(v2.texcoord[0] * rhw2));
    attr_max = fmax2(attr_max, __builtin_fabsf(v0.texcoord[1] * rhw0));
    attr_max = fmax2(attr_max, __builtin_fabsf(v1.texcoord[1] * rhw1));
    attr_max = fmax2(attr_max, __builtin_fabsf(v2.texcoord[1] * rhw2));
    // Keep a single vertex value <= 32 so the delta a0-a2 (<= 2x) stays <= 64,
    // half of FloatA's +-128 range — headroom for rounding and the direct a2 plane.
    for (int g = 0; g < 24 && attr_max > 32.0f; ++g) {
      rhw0 *= 0.5f; rhw1 *= 0.5f; rhw2 *= 0.5f; attr_max *= 0.5f;
    }
  }
  delta(out_prim.attribs.r,   v0.color[0]    * rhw0, v1.color[0]    * rhw1, v2.color[0]    * rhw2);
  delta(out_prim.attribs.g,   v0.color[1]    * rhw0, v1.color[1]    * rhw1, v2.color[1]    * rhw2);
  delta(out_prim.attribs.b,   v0.color[2]    * rhw0, v1.color[2]    * rhw1, v2.color[2]    * rhw2);
  delta(out_prim.attribs.a,   v0.color[3]    * rhw0, v1.color[3]    * rhw1, v2.color[3]    * rhw2);
  delta(out_prim.attribs.u,   v0.texcoord[0] * rhw0, v1.texcoord[0] * rhw1, v2.texcoord[0] * rhw2);
  delta(out_prim.attribs.v,   v0.texcoord[1] * rhw0, v1.texcoord[1] * rhw1, v2.texcoord[1] * rhw2);
  delta(out_prim.attribs.w0,  v0.varying2[0] * rhw0, v1.varying2[0] * rhw1, v2.varying2[0] * rhw2);
  delta(out_prim.attribs.w1,  v0.varying2[1] * rhw0, v1.varying2[1] * rhw1, v2.varying2[1] * rhw2);
  delta(out_prim.attribs.w2,  v0.varying2[2] * rhw0, v1.varying2[2] * rhw1, v2.varying2[2] * rhw2);
  delta(out_prim.attribs.w3,  v0.varying2[3] * rhw0, v1.varying2[3] * rhw1, v2.varying2[3] * rhw2);
  delta(out_prim.attribs.w4,  v0.varying2[4] * rhw0, v1.varying2[4] * rhw1, v2.varying2[4] * rhw2);
  delta(out_prim.attribs.w5,  v0.varying2[5] * rhw0, v1.varying2[5] * rhw1, v2.varying2[5] * rhw2);
  delta(out_prim.attribs.rhw, rhw0, rhw1, rhw2);
  return true;
}

} // namespace gfx_setup
