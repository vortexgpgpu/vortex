#pragma once

// gfx_v2 fused setup -> binning front end — device kernel stages.
//
// Defines the two kernel entries that produce RASTER's gfx-v1 buffers: setup_k
// (clip + triangle setup -> dense primbuf + bbox + P) and binning_k (bin-sort
// over the dense tile grid -> tilebuf). Include from a test's kernel.cpp; the
// test adds its own fragment/consumer kernel. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // pipe_arg_t, PIPE_STAGE_*, PIPE_PRIM_*, setup types
#include <gfx_setup.h>          // gfx_setup::{clip_near, setup_triangle}

// Coarse bin granularity the front end bins at (= VX_CFG_RASTER_BIN_LOG_SIZE,
// supplied as a -D in the kernel build; fallback for standalone include).
#ifndef VX_CFG_RASTER_BIN_LOG_SIZE
#define VX_CFG_RASTER_BIN_LOG_SIZE 7
#endif
#ifndef PIPE_BIN_LOG
#define PIPE_BIN_LOG   VX_CFG_RASTER_BIN_LOG_SIZE
#endif

namespace gfx_pipe {

using gfx_setup::rast_prim_t;
using gfx_setup::setup_triangle;
using gfx_setup::clip_near;
using vortex::graphics::FloatA;
using vortex::graphics::rast_bin_header_t;

static inline uint32_t pipe_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

static inline void pipe_bin_range(const setup_bbox_t& p, int& bL, int& bR, int& bT, int& bB) {
  bL = p.bbL >> PIPE_BIN_LOG; bR = (p.bbR - 1) >> PIPE_BIN_LOG;
  bT = p.bbT >> PIPE_BIN_LOG; bB = (p.bbB - 1) >> PIPE_BIN_LOG;
}

// The VS records draw primitive t reads, by prim_type (see pipe_arg_t): a
// triangle's three vertices, a line's two ends (the second repeated), a point's
// one vertex three times, or a polygon's triangle with the edge or vertex
// t % 3 it stands for.
struct pipe_prim_ref {
  uint32_t rec[3];
  uint32_t sub;
};

static inline pipe_prim_ref pipe_prim_records(const pipe_arg_t* arg, uint32_t t) {
  pipe_prim_ref r{};
  uint32_t base = 0;
  if (arg->inst_prims) {
    const uint32_t i = t / arg->inst_prims;
    t -= i * arg->inst_prims;
    base = i * arg->inst_verts;
  }
  switch (arg->prim_type) {
  case SETUP_PRIM_LINE:
    r.rec[0] = 2 * t; r.rec[1] = r.rec[2] = 2 * t + 1;
    break;
  case SETUP_PRIM_POINT:
    r.rec[0] = r.rec[1] = r.rec[2] = t;
    break;
  case SETUP_PRIM_POLYGON: {
    const uint32_t base = 3 * (t / 3);
    r.rec[0] = base; r.rec[1] = base + 1; r.rec[2] = base + 2;
    r.sub = t % 3;
  } break;
  default:
    r.rec[0] = 3 * t; r.rec[1] = 3 * t + 1; r.rec[2] = 3 * t + 2;
    break;
  }
  for (uint32_t k = 0; k < 3; ++k) {
    r.rec[k] += base;
  }
  return r;
}

// Clip + setup of one triangle into at most SETUP_MAX_SUB records.
static uint32_t
pipe_setup_triangle(const setup_vertex_t& a, const setup_vertex_t& b, const setup_vertex_t& c,
                    int W, int H, const pipe_arg_t* arg, const setup_viewport_t* vp,
                    bool coverage_only, rast_prim_t* prim_out, setup_bbox_t* bbox_out) {
  clip_tri_t sub[SETUP_MAX_SUB];
  // The near plane the clip uses and the depth range setup maps onto have to
  // come from the same convention, so both read it off the bound viewport.
  int ns = clip_near(a, b, c, sub, vp && vp->halfz, arg->near_clip_off != 0);
  uint32_t kept = 0;
  for (int s = 0; s < ns; ++s) {
    rast_prim_t prim{};
    setup_bbox_t bb{};
    if (setup_triangle(sub[s].v[0], sub[s].v[1], sub[s].v[2], W, H,
                       SETUP_NEAR, SETUP_FAR, prim, bb, arg->cull_mode, vp, arg->front_face,
                       coverage_only, &arg->depth_bias)) {
      prim_out[kept] = prim;
      bbox_out[kept] = bb;
      ++kept;
    }
  }
  return kept;
}

// A quad (a0, a1, b0, b1) as the two triangles the rasterizer takes, every one
// with the given facing and depth bias; a line or point is never culled by
// winding.
static uint32_t
pipe_setup_quad(const setup_vertex_t& a0, const setup_vertex_t& a1,
                const setup_vertex_t& b0, const setup_vertex_t& b1, uint32_t facing, float zbias,
                int W, int H, const setup_viewport_t* vp,
                rast_prim_t* prim_out, setup_bbox_t* bbox_out) {
  uint32_t kept = 0;
  const setup_vertex_t* tri[2][3] = { { &a0, &a1, &b0 }, { &b0, &a1, &b1 } };
  for (int s = 0; s < 2; ++s) {
    rast_prim_t prim{};
    setup_bbox_t bb{};
    if (setup_triangle(*tri[s][0], *tri[s][1], *tri[s][2], W, H, SETUP_NEAR, SETUP_FAR,
                       prim, bb, SETUP_CULL_NONE, vp, SETUP_FRONT_POSITIVE, true, nullptr,
                       zbias)) {
      prim.facing = facing;
      prim_out[kept] = prim;
      bbox_out[kept] = bb;
      ++kept;
    }
  }
  return kept;
}

// Three floats in a record word triple a coverage-only primitive leaves unused.
static inline void pipe_plane_set(vortex::graphics::rast_attrib_t& p, float x, float y, float z) {
  p.x = FloatA::make(__builtin_bit_cast(int32_t, x));
  p.y = FloatA::make(__builtin_bit_cast(int32_t, y));
  p.z = FloatA::make(__builtin_bit_cast(int32_t, z));
}

// Varying planes that weight record slot `at` by plane `f` alone.
static inline void pipe_vary_set(rast_vary_t& vr, int at, const float f[3]) {
  for (int i = 0; i < 3; ++i) {
    vr.bary[at][i] = f[i];
  }
}

// A line's length in stipple steps: Euclidean for a rectangular line, along the
// major axis for a Bresenham one.
static inline float pipe_stipple_length(const gfx_setup::vec4f& sa, const gfx_setup::vec4f& sb,
                                        uint32_t line_mode) {
  const float dx = __builtin_fabsf(sb.x - sa.x), dy = __builtin_fabsf(sb.y - sa.y);
  if (line_mode == SETUP_LINE_BRESENHAM) {
    return dx > dy ? dx : dy;
  }
  return __builtin_sqrtf(dx * dx + dy * dy);
}

// Whether draw segment i of a stippled strip begins a run: each instance starts
// one, and `heads` (may be null) marks the restarts within an instance.
#define PIPE_SCAN_HEAD 0x80000000u
static inline bool pipe_strip_head(const pipe_arg_t* arg, const uint32_t* heads, uint32_t i) {
  if (arg->inst_prims) {
    i %= arg->inst_prims;
    if (i == 0) {
      return true;
    }
  }
  return heads && ((heads[i >> 5] >> (i & 31u)) & 1u);
}

// Line a-b's length in whole stipple steps, as drawn after the near clip: how
// far the counter runs on past it to the next segment.
static inline uint32_t pipe_stipple_steps(setup_vertex_t a, setup_vertex_t b, int W, int H,
                                          const pipe_arg_t* arg, const setup_viewport_t* vp) {
  const gfx_setup::setup_rect_t r = gfx_setup::setup_rect(vp, W, H, SETUP_NEAR, SETUP_FAR);
  if (!gfx_setup::clip_line_near(a, b, r.halfz, arg->near_clip_off != 0)) {
    return 0;
  }
  const float len = pipe_stipple_length(gfx_setup::clip_to_screen(a, r),
                                        gfx_setup::clip_to_screen(b, r), arg->line_mode);
  return (uint32_t)(int32_t)__builtin_ceilf(len);   // ceil through the conversion, as floor_w
}

// A line from a to b, `width` pixels wide, as SETUP_LINE_* shapes it. Its
// varying planes come from the unclipped ends: attributes run from a to b
// along it and stay constant across it. `lo` and `hi` are the record slots
// that take the start and end planes; the third stays zero.
static uint32_t
pipe_setup_line(setup_vertex_t a, setup_vertex_t b, float width, uint32_t facing, float zbias,
                float stipple,
                int W, int H, const pipe_arg_t* arg, const setup_viewport_t* vp,
                rast_prim_t* prim_out, setup_bbox_t* bbox_out, rast_vary_t* vr, int lo, int hi) {
  const gfx_setup::setup_rect_t r = gfx_setup::setup_rect(vp, W, H, SETUP_NEAR, SETUP_FAR);
  const setup_vertex_t a_src = a, b_src = b;
  if (!gfx_setup::clip_line_near(a, b, r.halfz, arg->near_clip_off != 0)) {
    return 0;
  }
  const gfx_setup::vec4f sa = gfx_setup::clip_to_screen(a, r);
  const gfx_setup::vec4f sb = gfx_setup::clip_to_screen(b, r);
  const float dx = sb.x - sa.x, dy = sb.y - sa.y;
  const float len = __builtin_sqrtf(dx * dx + dy * dy);
  if (len == 0.0f) {
    return 0;
  }
  float ox, oy;
  uint32_t kept;
  if (arg->line_mode == SETUP_LINE_BRESENHAM) {
    // A parallelogram a whole number of pixels across the minor axis, its ends
    // moved along the major axis by the diamond-exit rule.
    const gfx_setup::line_ends_t e = gfx_setup::bresenham_line_ends(sa.x, sa.y, sb.x, sb.y);
    const float hw = 0.5f * (float)(int32_t)(width + 0.5f);
    ox = e.xmajor ? 0.0f : hw;
    oy = e.xmajor ? hw : 0.0f;
    const float adx = e.first_is_a ? e.first_dx : e.last_dx;
    const float ady = e.first_is_a ? e.first_dy : e.last_dy;
    const float bdx = e.first_is_a ? e.last_dx : e.first_dx;
    const float bdy = e.first_is_a ? e.last_dy : e.first_dy;
    kept = pipe_setup_quad(gfx_setup::clip_offset(a, adx + ox, ady + oy, r),
                           gfx_setup::clip_offset(a, adx - ox, ady - oy, r),
                           gfx_setup::clip_offset(b, bdx + ox, bdy + oy, r),
                           gfx_setup::clip_offset(b, bdx - ox, bdy - oy, r),
                           facing, zbias, W, H, vp, prim_out, bbox_out);
  } else {
    // A smooth line is drawn half a pixel larger all round, at least a pixel
    // each side of its centre, and fades out over that margin.
    const bool smooth = arg->line_mode == SETUP_LINE_SMOOTH;
    const float ux = dx / len, uy = dy / len;
    const float hw = smooth ? (width > 1.0f ? 0.5f * width + 0.5f : 1.0f) : 0.5f * width;
    const float ex = smooth ? 0.5f * ux : 0.0f, ey = smooth ? 0.5f * uy : 0.0f;
    ox = -uy * hw;
    oy = ux * hw;
    kept = pipe_setup_quad(gfx_setup::clip_offset(a, ox - ex, oy - ey, r),
                           gfx_setup::clip_offset(a, -ox - ex, -oy - ey, r),
                           gfx_setup::clip_offset(b, ox + ex, oy + ey, r),
                           gfx_setup::clip_offset(b, -ox + ex, -oy + ey, r),
                           facing, zbias, W, H, vp, prim_out, bbox_out);
    if (smooth) {
      const float mx = 0.5f * (sa.x + sb.x), my = 0.5f * (sa.y + sb.y);
      for (uint32_t s = 0; s < kept; ++s) {
        pipe_plane_set(SETUP_PRIM_SMOOTH_ACROSS(prim_out[s]), -uy, ux, uy * mx - ux * my);
        pipe_plane_set(SETUP_PRIM_SMOOTH_ALONG(prim_out[s]), ux, uy, -(ux * mx + uy * my));
        pipe_plane_set(SETUP_PRIM_SMOOTH_SIZE(prim_out[s]), hw, 0.5f * len + 0.5f, len);
      }
    }
  }
  // The stipple counter as a plane over the window, starting at the drawn start.
  float sx, sy;
  if (arg->line_mode == SETUP_LINE_BRESENHAM) {
    const bool xmajor = __builtin_fabsf(dx) >= __builtin_fabsf(dy);
    sx = xmajor ? (dx >= 0.0f ? 1.0f : -1.0f) : 0.0f;
    sy = xmajor ? 0.0f : (dy >= 0.0f ? 1.0f : -1.0f);
  } else {
    sx = dx / len;
    sy = dy / len;
  }
  const float sc = stipple - (sx * sa.x + sy * sa.y);
  for (uint32_t s = 0; s < kept; ++s) {
    pipe_plane_set(SETUP_PRIM_STIPPLE_PLANE(prim_out[s]), sx, sy, sc);
  }
  if (kept && vr) {
    // A virtual triangle (a, b, b + offset): its first barycentric is the line
    // parameter's complement, the other two together its value.
    float planes[3][3];
    gfx_setup::setup_vary_planes(a_src, b_src, gfx_setup::clip_offset(b_src, ox, oy, r),
                                 r, planes);
    const float end[3] = { planes[1][0] + planes[2][0], planes[1][1] + planes[2][1],
                           planes[1][2] + planes[2][2] };
    pipe_vary_set(*vr, lo, planes[0]);
    pipe_vary_set(*vr, hi, end);
  }
  return kept;
}

// A point at v, `size` pixels square; dropped when its centre lies outside the
// clip volume, as Vulkan clips points. Its record carries the centre and size
// for gl_PointCoord. The varying plane in slot `at` is constant.
static uint32_t
pipe_setup_point(const setup_vertex_t& v, float size, uint32_t facing, float zbias,
                 int W, int H, const pipe_arg_t* arg, const setup_viewport_t* vp,
                 rast_prim_t* prim_out, setup_bbox_t* bbox_out, rast_vary_t* vr, int at) {
  const gfx_setup::setup_rect_t r = gfx_setup::setup_rect(vp, W, H, SETUP_NEAR, SETUP_FAR);
  const float w = v.pos[3];
  if (gfx_setup::near_dist(v, r.halfz, arg->near_clip_off != 0) < 0.0f ||
      v.pos[0] < -w || v.pos[0] > w || v.pos[1] < -w || v.pos[1] > w || size <= 0.0f) {
    return 0;
  }
  const float hs = 0.5f * size;
  const uint32_t kept = pipe_setup_quad(gfx_setup::clip_offset(v, -hs, -hs, r),
                                        gfx_setup::clip_offset(v, hs, -hs, r),
                                        gfx_setup::clip_offset(v, -hs, hs, r),
                                        gfx_setup::clip_offset(v, hs, hs, r),
                                        facing, zbias, W, H, vp, prim_out, bbox_out);
  const gfx_setup::vec4f c = gfx_setup::clip_to_screen(v, r);
  for (uint32_t s = 0; s < kept; ++s) {
    pipe_plane_set(SETUP_PRIM_POINT_PLANE(prim_out[s]), c.x, c.y, size);
    // Full coverage, for a draw whose other face draws smooth lines.
    if (arg->line_mode == SETUP_LINE_SMOOTH) {
      pipe_plane_set(SETUP_PRIM_SMOOTH_SIZE(prim_out[s]), 1.0f, 1.0f, 1.0f);
    }
  }
  if (kept && vr) {
    const float one[3] = { 0.0f, 0.0f, 1.0f };
    pipe_vary_set(*vr, at, one);
  }
  return kept;
}

// Clip + setup of primitive t, out-of-line so the FP math does not bloat the
// entry (the merged 9-stage function otherwise overruns the uniform pass).
// Fills `vr` (when non-null) with the varying record every emitted
// subprimitive shares. A line's stipple counter starts at `stipple`.
static uint32_t __attribute__((noinline))
pipe_prim_setup(const setup_vertex_t* v, const pipe_prim_ref& ref, float psize, float stipple,
                int W, int H, const pipe_arg_t* arg, const setup_viewport_t* vp,
                bool coverage_only, rast_prim_t* prim_out, setup_bbox_t* bbox_out,
                rast_vary_t* vr) {
  if (vr) {
    *vr = rast_vary_t{};
  }
  uint32_t type = arg->prim_type;
  if (type == SETUP_PRIM_POLYGON) {
    const gfx_setup::setup_rect_t r = gfx_setup::setup_rect(vp, W, H, SETUP_NEAR, SETUP_FAR);
    const int facing = gfx_setup::triangle_facing(v[0], v[1], v[2], r, arg->cull_mode,
                                                  arg->front_face);
    if (facing < 0) {
      return 0;
    }
    const uint32_t mode = facing ? arg->poly_back : arg->poly_front;
    if (mode == SETUP_POLY_FILL) {
      if (ref.sub != 0) {
        return 0;   // the triangle is drawn once, by its first primitive
      }
      type = SETUP_PRIM_TRIANGLE;
    } else {
      // Its edges and vertices keep the bias of the triangle they come from.
      float zbias = 0.0f;
      if (arg->depth_bias.enable) {
        const gfx_setup::vec4f p0 = gfx_setup::clip_to_screen(v[0], r);
        const gfx_setup::vec4f p1 = gfx_setup::clip_to_screen(v[1], r);
        const gfx_setup::vec4f p2 = gfx_setup::clip_to_screen(v[2], r);
        float za, zb;
        gfx_setup::depth_plane_slopes(p0, p1, p2, za, zb);
        zbias = gfx_setup::depth_bias_offset(arg->depth_bias, za, zb, p0.z, p1.z, p2.z);
      }
      // Flat inputs take the triangle's provoking vertex, which sits in the
      // slot the fragment stage reads and gets a zero plane.
      const uint32_t flat = ref.rec[arg->provoke_last ? 2 : 0];
      const int fs = arg->provoke_last ? 2 : 0;
      const int s0 = arg->provoke_last ? 0 : 1, s1 = arg->provoke_last ? 1 : 2;
      const uint32_t e0 = ref.sub, e1 = (ref.sub + 1) % 3;
      if (vr) {
        vr->vrec[fs] = flat;
        vr->vrec[s0] = ref.rec[e0];
        vr->vrec[s1] = ref.rec[e1];
      }
      if (mode == SETUP_POLY_LINE) {
        // The stipple counter starts once per triangle and runs on over its
        // edges in the order v2-v0, v0-v1, v1-v2.
        float start = 0.0f;
        if (ref.sub != 2) {
          uint32_t steps = pipe_stipple_steps(v[2], v[0], W, H, arg, vp);
          if (ref.sub == 1) {
            steps += pipe_stipple_steps(v[0], v[1], W, H, arg, vp);
          }
          start = (float)steps;
        }
        return pipe_setup_line(v[e0], v[e1], arg->line_width, (uint32_t)facing, zbias, start,
                               W, H, arg, vp, prim_out, bbox_out, vr, s0, s1);
      }
      if (vr) {
        vr->vrec[s1] = ref.rec[e0];
      }
      return pipe_setup_point(v[e0], psize, (uint32_t)facing, zbias, W, H, arg, vp,
                              prim_out, bbox_out, vr, s0);
    }
  }
  if (vr) {
    vr->vrec[0] = ref.rec[0];
    vr->vrec[1] = ref.rec[1];
    vr->vrec[2] = ref.rec[2];
  }
  switch (type) {
  case SETUP_PRIM_LINE:
    return pipe_setup_line(v[0], v[1], arg->line_width, 0, 0.0f, stipple, W, H, arg, vp,
                           prim_out, bbox_out, vr, 0, 1);
  case SETUP_PRIM_POINT:
    return pipe_setup_point(v[0], psize, 0, 0.0f, W, H, arg, vp, prim_out, bbox_out, vr, 0);
  default: {
    const uint32_t kept = pipe_setup_triangle(v[0], v[1], v[2], W, H, arg, vp, coverage_only,
                                              prim_out, bbox_out);
    // Every subtriangle of a clipped triangle shares the source triangle's
    // varying record: its barycentrics are those of the source triangle.
    if (kept && vr) {
      gfx_setup::setup_vary_planes(v[0], v[1], v[2],
                                   gfx_setup::setup_rect(vp, W, H, SETUP_NEAR, SETUP_FAR),
                                   vr->bary);
    }
    return kept;
  }
  }
}

// Draw primitive t's vertices: from the VS output records when the draw
// supplies them (positions only -- varyings stay in the records for the fragment
// stage), else from the setup_vertex_t stream (triangles only). Returns the
// point size a point primitive is drawn at.
static inline float pipe_load_prim(const pipe_arg_t* arg, uint32_t t, const pipe_prim_ref& ref,
                                   setup_vertex_t v[3]) {
  float psize = arg->point_size;
  if (arg->vsrec_addr) {
    auto recs = reinterpret_cast<const uint8_t*>(arg->vsrec_addr);
    for (uint32_t k = 0; k < 3; ++k) {
      const float* pos = reinterpret_cast<const float*>(recs + (size_t)ref.rec[k] * arg->vstride);
      setup_vertex_t& o = v[k];
      o = setup_vertex_t{};
      o.pos[0] = pos[0]; o.pos[1] = pos[1]; o.pos[2] = pos[2]; o.pos[3] = pos[3];
    }
    if (arg->psize_off &&
        (arg->prim_type == SETUP_PRIM_POINT || arg->prim_type == SETUP_PRIM_POLYGON)) {
      // A polygon's point takes the size its own vertex wrote.
      const uint32_t at = (arg->prim_type == SETUP_PRIM_POLYGON) ? ref.rec[ref.sub] : ref.rec[0];
      psize = *reinterpret_cast<const float*>(recs + (size_t)at * arg->vstride + arg->psize_off);
    }
    if (arg->point_size_max > 0.0f) {
      psize = psize > arg->point_size_max ? arg->point_size_max : psize;
    }
  } else {
    auto verts = reinterpret_cast<const setup_vertex_t*>(arg->verts_addr);
    v[0] = verts[3 * t]; v[1] = verts[3 * t + 1]; v[2] = verts[3 * t + 2];
  }
  return psize;
}

// Whether a primitive's VS records all have one cull distance negative.
static inline bool pipe_cull_distance(const pipe_arg_t* arg, const pipe_prim_ref& ref) {
  const uint32_t n = SETUP_CULL_DIST_COUNT(arg->cull_dist);
  if (n == 0 || !arg->vsrec_addr) {
    return false;
  }
  auto recs = reinterpret_cast<const uint8_t*>(arg->vsrec_addr) + SETUP_CULL_DIST_OFF(arg->cull_dist);
  const float* d[3];
  for (uint32_t k = 0; k < 3; ++k) {
    d[k] = reinterpret_cast<const float*>(recs + (size_t)ref.rec[k] * arg->vstride);
  }
  for (uint32_t i = 0; i < n; ++i) {
    if (d[0][i] < 0.0f && d[1][i] < 0.0f && d[2][i] < 0.0f) {
      return true;
    }
  }
  return false;
}

// The viewport slot and layer primitive t is drawn with: the ViewportIndex and
// Layer its provoking vertex wrote, an index past the array taking slot 0 and a
// layer past the framebuffer its last layer.
struct pipe_prim_view {
  const setup_viewport_t* vp;
  uint32_t scissor_x, scissor_y;
  uint32_t index, layer;
};

static inline pipe_prim_view pipe_prim_viewport(const pipe_arg_t* arg, const pipe_prim_ref& ref,
                                                const setup_viewport_t* vp) {
  pipe_prim_view v{ vp, RAST_SCISSOR_PACK(0, 0xffff), RAST_SCISSOR_PACK(0, 0xffff), 0, 0 };
  const uint8_t* pv = nullptr;
  if (arg->vsrec_addr && (arg->vpi_off || arg->layer_off)) {
    pv = reinterpret_cast<const uint8_t*>(arg->vsrec_addr)
       + (size_t)ref.rec[arg->provoke_last ? 2 : 0] * arg->vstride;
  }
  if (arg->vp_table_addr) {
    const auto* table = reinterpret_cast<const setup_vp_slot_t*>(arg->vp_table_addr);
    if (pv && arg->vpi_off) {
      v.index = *reinterpret_cast<const uint32_t*>(pv + arg->vpi_off);
      v.index = v.index < arg->vp_count ? v.index : 0u;
    }
    v.vp = &table[v.index].vp;
    v.scissor_x = table[v.index].scissor_x;
    v.scissor_y = table[v.index].scissor_y;
  }
  if (pv && arg->layer_off) {
    v.layer = *reinterpret_cast<const uint32_t*>(pv + arg->layer_off);
    v.layer = v.layer < arg->layer_max ? v.layer : arg->layer_max;
  }
  return v;
}

// Confine the kept subprimitives to the primitive's viewport rectangle,
// compacting out those it excludes.
static inline uint32_t pipe_prim_scissor(rast_prim_t* pr, setup_bbox_t* bb, uint32_t kept,
                                         const pipe_prim_view& view) {
  uint32_t n = 0;
  for (uint32_t s = 0; s < kept; ++s) {
    if (!gfx_setup::setup_scissor(pr[s], bb[s], view.scissor_x, view.scissor_y)) {
      continue;
    }
    if (n != s) {
      pr[n] = pr[s];
      bb[n] = bb[s];
    }
    ++n;
  }
  return n;
}

} // namespace gfx_pipe

// ---- setup front end: clip + setup -> dense primbuf + bbox + P (meta[0]) ----
__kernel void setup_k(pipe_arg_t* __UNIFORM__ arg) {
  using namespace gfx_pipe;
  auto keep      = reinterpret_cast<uint32_t*>(arg->keep_addr);
  auto offset    = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto tsum      = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  auto stipple   = reinterpret_cast<uint32_t*>(arg->stipple_addr);
  auto prim      = reinterpret_cast<rast_prim_t*>(arg->prim_addr);
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);
  auto vary      = reinterpret_cast<rast_vary_t*>(arg->vary_addr);
  // A draw fed from VS records takes its barycentrics from rast_vary_t, so its
  // primitive record carries coverage edges and depth only.
  const bool coverage_only = (arg->vsrec_addr != 0);

  const uint32_t ntri = arg->num_tris;
  const int      W = (int)arg->width;
  const int      H = (int)arg->height;
  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;

  // App viewport transform, or the default full-framebuffer y-down mapping when
  // unset (all four zero — what the standalone setup tests, which zero-init the
  // arg block, feed). Passing &vp / nullptr selects the branch inside setup.
  setup_viewport_t vp;
  const bool have_vp = (arg->vp_sx != 0.0f) || (arg->vp_tx != 0.0f)
                    || (arg->vp_sy != 0.0f) || (arg->vp_ty != 0.0f)
                    || (arg->vp_halfz != 0u);
  vp.sx = arg->vp_sx; vp.tx = arg->vp_tx;
  vp.sy = arg->vp_sy; vp.ty = arg->vp_ty;
  vp.minz  = arg->vp_minz; vp.maxz = arg->vp_maxz;
  vp.halfz = arg->vp_halfz;
  const setup_viewport_t* vpp = have_vp ? &vp : nullptr;

  switch (arg->stage) {

  case PIPE_STAGE_SETUP: {
    // Count survivors only. The dense scatter (EMIT) recomputes clip+setup and
    // writes each rast_prim_t straight to its compacted slot, so the 120-byte
    // record is never staged to a scratch buffer and read back — clip+setup is
    // a pure function of the (unchanged) input vertices, so the count here and
    // the per-triangle output in EMIT match exactly. Halves primbuf write
    // traffic and removes the slot_prim/slot_bbox scratch entirely.
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      setup_vertex_t v[3];
      rast_prim_t pr[SETUP_MAX_SUB];
      setup_bbox_t bb[SETUP_MAX_SUB];
      const pipe_prim_ref ref = pipe_prim_records(arg, arg->prim_base + t);
      const float psize = pipe_load_prim(arg, arg->prim_base + t, ref, v);
      const pipe_prim_view view = pipe_prim_viewport(arg, ref, vpp);
      uint32_t kept = 0;
      if (!pipe_cull_distance(arg, ref)) {
        kept = pipe_prim_setup(v, ref, psize, 0.0f, W, H, arg, view.vp, coverage_only, pr, bb,
                               nullptr);
        kept = pipe_prim_scissor(pr, bb, kept, view);
      }
      keep[t] = kept;
      if (arg->stipple_strip) {
        stipple[t] = pipe_stipple_steps(v[0], v[1], W, H, arg, view.vp);
      }
    }
  } break;

  case PIPE_STAGE_SCAN: {
    uint32_t chunk = (ntri + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, ntri), hi = pipe_umin(lo + chunk, ntri);
    { uint32_t s = 0; for (uint32_t i = lo; i < hi; ++i) s += keep[i]; tsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = tsum[t]; tsum[t] = acc; acc += v; }
      meta[0] = acc; offset[ntri] = acc;
    }
    __syncthreads();
    { uint32_t acc = tsum[tid]; for (uint32_t i = lo; i < hi; ++i) { offset[i] = acc; acc += keep[i]; } }
    if (arg->stipple_strip) {
      // Each segment's steps become the counter it starts at, in place: a scan
      // segmented at the strip's run heads. A chunk's partial carries in bit 31
      // whether it holds a head, which stops the carry from earlier chunks. A
      // pass after the draw's first starts from the counter the previous pass
      // left in meta[3], and leaves its own there for the next.
      auto heads = reinterpret_cast<const uint32_t*>(arg->strip_head_addr);
      __syncthreads();
      {
        uint32_t s = 0, h = 0;
        for (uint32_t i = lo; i < hi; ++i) {
          if (pipe_strip_head(arg, heads, arg->prim_base + i)) { s = 0; h = PIPE_SCAN_HEAD; }
          s += stipple[i];
        }
        tsum[tid] = s | h;
      }
      __syncthreads();
      if (tid == 0) {
        uint32_t acc = arg->prim_base ? meta[3] : 0;
        for (uint32_t t = 0; t < T; ++t) {
          const uint32_t v = tsum[t];
          tsum[t] = acc;
          acc = (v & PIPE_SCAN_HEAD) ? (v & ~PIPE_SCAN_HEAD) : acc + v;
        }
        meta[3] = acc;
      }
      __syncthreads();
      {
        uint32_t acc = tsum[tid];
        for (uint32_t i = lo; i < hi; ++i) {
          if (pipe_strip_head(arg, heads, arg->prim_base + i)) {
            acc = 0;
          }
          const uint32_t v = stipple[i];
          stipple[i] = acc;
          acc += v;
        }
      }
    }
  } break;

  case PIPE_STAGE_EMIT: {
    // Recompute clip+setup and scatter directly into the dense primbuf at the
    // scanned offset — no round-trip through a per-triangle scratch slot. kept
    // here equals keep[t] from SETUP (same pure function, same inputs).
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      setup_vertex_t v[3];
      rast_prim_t pr[SETUP_MAX_SUB];
      setup_bbox_t bb[SETUP_MAX_SUB];
      const pipe_prim_ref ref = pipe_prim_records(arg, arg->prim_base + t);
      const float psize = pipe_load_prim(arg, arg->prim_base + t, ref, v);
      rast_vary_t vr;
      const float start = arg->stipple_strip ? (float)stipple[t] : 0.0f;
      const pipe_prim_view view = pipe_prim_viewport(arg, ref, vpp);
      uint32_t kept = 0;
      if (!pipe_cull_distance(arg, ref)) {
        kept = pipe_prim_setup(v, ref, psize, start, W, H, arg, view.vp, coverage_only, pr, bb,
                               vary ? &vr : nullptr);
        kept = pipe_prim_scissor(pr, bb, kept, view);
        vr.vp_layer = RAST_VARY_VP_LAYER(view.index, view.layer);
      }
      uint32_t w = offset[t];
      for (uint32_t s = 0; s < kept; ++s) {
        prim[w] = pr[s];
        bbox[w] = bb[s];
        if (vary) {
          vary[w] = vr;
        }
        ++w;
      }
    }
  } break;

  }
}

// ---- binning back end: bin-sort dense bbox[] (P=meta[0]) -> dense tilebuf ----
__kernel void binning_k(pipe_arg_t* __UNIFORM__ arg) {
  using namespace gfx_pipe;
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto bcount    = reinterpret_cast<uint32_t*>(arg->bcount_addr);
  auto boffset   = reinterpret_cast<uint32_t*>(arg->boffset_addr);
  auto keys      = reinterpret_cast<uint32_t*>(arg->keys_addr);
  auto btsum     = reinterpret_cast<uint32_t*>(arg->btsum_addr);
  auto thist     = reinterpret_cast<uint32_t*>(arg->thist_addr);
  auto bincount  = reinterpret_cast<uint32_t*>(arg->bincount_addr);
  auto binbase   = reinterpret_cast<uint32_t*>(arg->binbase_addr);
  auto tilebuf   = reinterpret_cast<uint8_t*>(arg->tilebuf_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;
  const uint32_t B   = arg->num_bins;     // dense tile grid sized to the render target

  switch (arg->stage) {

  case PIPE_STAGE_BCOUNT: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; pipe_bin_range(bbox[i], bL, bR, bT, bB);
      bcount[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
    }
  } break;

  case PIPE_STAGE_BSCAN: {
    uint32_t P = meta[0];
    uint32_t chunk = (P + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, P), hi = pipe_umin(lo + chunk, P);
    { uint32_t s = 0; for (uint32_t i = lo; i < hi; ++i) s += bcount[i]; btsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = btsum[t]; btsum[t] = acc; acc += v; }
      meta[1] = acc; boffset[P] = acc;
    }
    __syncthreads();
    { uint32_t acc = btsum[tid]; for (uint32_t i = lo; i < hi; ++i) { boffset[i] = acc; acc += bcount[i]; } }
  } break;

  case PIPE_STAGE_BEMIT: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; pipe_bin_range(bbox[i], bL, bR, bT, bB);
      uint32_t w = boffset[i];
      for (int by = bT; by <= bB; ++by)
        for (int bx = bL; bx <= bR; ++bx)
          keys[w++] = ((uint32_t)(by * arg->bin_cols + bx) << PIPE_PRIM_BITS) | i;
    }
  } break;

  case PIPE_STAGE_BHIST: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = pipe_umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = pipe_umin(tid * kchunk, K), khi = pipe_umin(klo + kchunk, K);
    for (uint32_t b = lo; b < hi; ++b) thist[tid * B + b] = 0;
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) { uint32_t b = keys[k] >> PIPE_PRIM_BITS; if (b >= lo && b < hi) thist[tid * B + b]++; }
    __syncthreads();
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b]; bincount[b] = s; }
  } break;

  case PIPE_STAGE_BBASE: {
    // Dense bin grid: one rast_bin_header_t per bin (empty bins get
    // pids_count=0 and RASTER skips them), so the bin count is B =
    // bin_cols*bin_rows. pids_offset is the absolute index into the sorted_pids
    // array that follows the header block. Cooperative exclusive
    // prefix-scan of bincount[] across the CTA's T threads (mirrors BSCAN),
    // replacing the single-lane O(B) loop so a large bin grid parallelises.
    uint32_t chunk = (B + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, B), hi = pipe_umin(lo + chunk, B);
    { uint32_t s = 0; for (uint32_t b = lo; b < hi; ++b) s += bincount[b]; btsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = btsum[t]; btsum[t] = acc; acc += v; }
    }
    __syncthreads();
    auto hdr = reinterpret_cast<rast_bin_header_t*>(tilebuf);
    uint32_t nb_local = 0;
    { uint32_t acc = btsum[tid];
      for (uint32_t b = lo; b < hi; ++b) {
        hdr[b].bin_x       = (uint16_t)(b % arg->bin_cols);
        hdr[b].bin_y       = (uint16_t)(b / arg->bin_cols);
        hdr[b].pids_offset = acc;
        hdr[b].pids_count  = bincount[b];
        binbase[b]         = acc;
        if (bincount[b]) ++nb_local;
        acc += bincount[b];
      }
    }
    __syncthreads();
    btsum[tid] = nb_local;   // offsets consumed; reuse btsum for the nb reduction
    __syncthreads();
    if (tid == 0) {
      uint32_t nb = 0;
      for (uint32_t t = 0; t < T; ++t) nb += btsum[t];
      meta[2] = nb;  // non-empty bin count (informational)
    }
  } break;

  case PIPE_STAGE_BSCATTER: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = pipe_umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    auto pids = reinterpret_cast<uint32_t*>(tilebuf + B * sizeof(rast_bin_header_t));
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = pipe_umin(tid * kchunk, K), khi = pipe_umin(klo + kchunk, K);
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t run = binbase[b]; for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; } }
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) {
      uint32_t key = keys[k], b = key >> PIPE_PRIM_BITS;
      if (b >= lo && b < hi) pids[thist[tid * B + b]++] = key & PIPE_PRIM_MASK;
    }
  } break;

  }
}
