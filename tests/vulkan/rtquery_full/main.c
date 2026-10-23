/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * rtquery_full -- every VK_KHR_ray_query value, checked per invocation.
 *
 * A two-level scene of three instances (a triangle BLAS twice, once scaled,
 * translated and face-flipped, and a procedural BLAS) is traced by a table of
 * rays. Each 32-invocation warp shares one action -- confirm, ignore,
 * terminate, generate, leave the loop and re-trace, leave it on some lanes
 * only, abandon a query and trace another, two queries of an array, a null
 * acceleration structure -- and one set of ray flags, while its lanes spread
 * over the primitive. Built with RTQ_INTERLEAVE it traces two queries
 * proceeded in turn instead; built with RTQ_BARRIER, every query of a
 * two-warp workgroup holds its first candidate across a workgroup barrier. The host traverses the same scene itself and
 * compares every committed, candidate and ray value the shader recorded.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

#define WARP       32u
#define REC_WORDS  80u

/* Mirrors the ACT_* constants in rtquery_full.comp. */
#define ACT_CONFIRM   0x01u
#define ACT_TERMINATE 0x02u
#define ACT_GENERATE  0x04u
#define ACT_EXIT      0x08u
#define ACT_TWO       0x10u
#define ACT_NULL_AS   0x20u
#define ACT_QSEL      0x100u
/* Host only: the even lanes of the warp confirm instead of leaving the loop. */
#define ACT_SPLIT     0x200u
#define ACT_ABANDON   0x400u
#define ACT_INTERLEAVE 0x800u
#define ACT_BARRIER   0x1000u

/* SpvRayFlags */
#define RF_NONE           0x000u
#define RF_OPAQUE         0x001u
#define RF_NO_OPAQUE      0x002u
#define RF_CULL_BACK      0x010u
#define RF_CULL_FRONT     0x020u
#define RF_CULL_OPAQUE    0x040u
#define RF_CULL_NO_OPAQUE 0x080u
#define RF_SKIP_TRIANGLES 0x100u
#define RF_SKIP_AABBS     0x200u

/* Record word indices (rtquery_full.comp). */
enum {
   R_TAG = 0, R_CTYPE = 1, R_CT = 2, R_CU = 3, R_CV = 4, R_CPRIM = 5,
   R_CGEOM = 6, R_CFRONT = 7, R_CINST = 8, R_CCUSTOM = 9, R_CSBT = 10,
   R_COBJ = 11, R_O2W = 17, R_W2O = 29, R_COUNT = 41, R_KTYPE = 42,
   R_KT = 43, R_KPRIM = 44, R_KGEOM = 45, R_KFRONT = 46, R_KINST = 47,
   R_KCUSTOM = 48, R_KSBT = 49, R_KAABBOPQ = 50, R_KU = 51, R_KV = 52,
   R_KOBJ = 53, R_TMIN = 59, R_FLAGS = 60, R_WO = 61, R_WD = 64,
   R_PROCEED = 67, R_Q2 = 68, R_Q2T = 69,
};

static VkDevice dev = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties memprops;
static PFN_vkCreateAccelerationStructureKHR         p_CreateAccelStruct;
static PFN_vkDestroyAccelerationStructureKHR        p_DestroyAccelStruct;
static PFN_vkGetAccelerationStructureBuildSizesKHR  p_GetAccelBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR      p_CmdBuildAccelStructs;
static PFN_vkGetAccelerationStructureDeviceAddressKHR p_GetAccelAddress;

/* ---------------------------------------------------------------- scene */

#define TRI_GEOMS  2
#define AABB_GEOMS 2

/* Triangle BLAS: geometry 0 opaque at z = 0, geometry 1 non-opaque above it
 * (z = 1 over the lower-left half of the square, z = 0.5 over the upper-right
 * half), so a ray down -Z meets one non-opaque candidate before one opaque
 * hit. */
static const float tri_verts[TRI_GEOMS][2][3][3] = {
   { { { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 } },
     { {  1, -1, 0 }, { 1,  1, 0 }, { -1, 1, 0 } } },
   { { { -1, -1, 1 }, { 1, -1, 1 }, { -1, 1, 1 } },
     { {  1, -1, 0.5f }, { 1, 1, 0.5f }, { -1, 1, 0.5f } } },
};
static const bool tri_opaque[TRI_GEOMS] = { true, false };

/* Procedural BLAS: geometry 0 one opaque box over x < 0, geometry 1 two
 * non-opaque boxes over x > 0 (primitive 0 below y = 0, primitive 1 above). */
static const VkAabbPositionsKHR aabb_g0[1] = {
   { -1, -1, -0.5f, 0, 1, 0.5f },
};
static const VkAabbPositionsKHR aabb_g1[2] = {
   { 0, -1, -0.5f, 1, 0, 0.5f },
   { 0,  0, -0.5f, 1, 1, 0.5f },
};
static const bool aabb_opaque[AABB_GEOMS] = { true, false };

struct inst_desc {
   bool  procedural;
   float o2w[3][4];
   uint32_t custom, mask, sbt, flags;
};
#define NUM_INST 3
static const struct inst_desc insts[NUM_INST] = {
   { false, { {1,0,0,0}, {0,1,0,0}, {0,0,1,0} }, 0x11, 0x01, 0, 0 },
   { false, { {2,0,0,10}, {0,1,0,0}, {0,0,1,0} }, 0x55, 0x02, 5,
     VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR },
   { true,  { {1,0,0,20}, {0,1,0,0}, {0,0,1,0} }, 0x77, 0x04, 9, 0 },
};

/* ----------------------------------------------------------- ray table */

struct ray {
   float origin[3], tmin;
   float dir[3], tmax;
   uint32_t flags, cull, act, gen_t;
   float origin2[3], pad;
};

struct warp_desc {
   const char *name;
   float x, y, x2, y2;
   float tmin, tmax;
   uint32_t flags, cull, act;
   float gen_t;
};

/* Each entry is one warp; lane j shifts the ray by a sixteenth per step over
 * an 8 x 4 grid, so every lane stays on the same primitive. */
#if defined(RTQ_BARRIER)
/* Two warps to a workgroup (local size 64): pairs of these share a barrier. */
static const struct warp_desc warps[] = {
   { "barrier_confirm", -0.75f, -0.75f, 0, 0, 0, 100, RF_NONE, 0xff,
     ACT_BARRIER | ACT_CONFIRM, 0 },
   { "barrier_inst",     8.5f,  -0.75f, 0, 0, 0, 100, RF_NONE, 0xff,
     ACT_BARRIER | ACT_CONFIRM, 0 },
   { "barrier_opaque",   0.25f,  0.25f, 0, 0, 0, 100, RF_NONE, 0xff,
     ACT_BARRIER | ACT_CONFIRM, 0 },
   { "barrier_miss",     5.0f,   0.0f,  0, 0, 0, 100, RF_NONE, 0xff,
     ACT_BARRIER | ACT_CONFIRM, 0 },
};
#define LOCAL_SIZE 64u
#elif defined(RTQ_INTERLEAVE)
static const struct warp_desc warps[] = {
   { "interleave",     -0.75f, -0.75f, 9.5f, 0.5f, 0, 100, RF_NONE, 0xff,
     ACT_INTERLEAVE | ACT_CONFIRM, 0 },
   { "interleave_inst", 8.5f,  -0.75f, 9.5f, 0.5f, 0, 100, RF_NONE, 0xff,
     ACT_INTERLEAVE | ACT_CONFIRM, 0 },
};
#else
static const struct warp_desc warps[] = {
   { "confirm",        -0.75f, -0.75f, 0, 0, 0, 100, RF_NONE, 0xff, ACT_CONFIRM, 0 },
   { "ignore",         -0.75f, -0.75f, 0, 0, 0, 100, RF_NONE, 0xff, 0, 0 },
   { "ray_opaque",      0.25f,  0.25f, 0, 0, 0, 100, RF_OPAQUE, 0xff, ACT_CONFIRM, 0 },
   { "terminate",      -0.75f, -0.75f, 0, 0, 0, 100, RF_CULL_OPAQUE, 0xff, ACT_TERMINATE, 0 },
   { "confirm_term",   -0.75f, -0.75f, 0, 0, 0, 100, RF_CULL_OPAQUE, 0xff,
     ACT_CONFIRM | ACT_TERMINATE, 0 },
   { "inst_xform",      8.5f,  -0.75f, 0, 0, 0, 100, RF_NONE, 0xff, ACT_CONFIRM, 0 },
   { "cull_back",       8.5f,  -0.75f, 0, 0, 0, 100, RF_CULL_BACK, 0xff, ACT_CONFIRM, 0 },
   { "cull_front",     -0.75f, -0.75f, 0, 0, 0, 100, RF_CULL_FRONT, 0xff, ACT_CONFIRM, 0 },
   { "gen_opaque_box", 19.25f,  0.25f, 0, 0, 0, 100, RF_NONE, 0xff, ACT_GENERATE, 1.75f },
   { "gen_box",        20.25f,  0.25f, 0, 0, 0, 100, RF_NONE, 0xff, ACT_GENERATE, 2.25f },
   { "ignore_box",     20.25f, -0.75f, 0, 0, 0, 100, RF_NONE, 0xff, 0, 0 },
   { "cull_opq_box",   19.25f,  0.25f, 0, 0, 0, 100, RF_CULL_OPAQUE, 0xff, ACT_GENERATE, 1.75f },
   { "skip_aabbs",     20.25f,  0.25f, 0, 0, 0, 100, RF_SKIP_AABBS, 0xff, ACT_GENERATE, 2.25f },
   { "skip_tris",      -0.75f, -0.75f, 0, 0, 0, 100, RF_SKIP_TRIANGLES, 0xff, ACT_CONFIRM, 0 },
   { "cull_mask",      -0.75f, -0.75f, 0, 0, 0, 100, RF_NONE, 0x02, ACT_CONFIRM, 0 },
   { "miss",            5.0f,   0.0f,  0, 0, 0, 100, RF_NONE, 0xff, ACT_CONFIRM, 0 },
   { "exit_retrace",   -0.75f, -0.75f, 0.25f, 0.25f, 0, 100, RF_NONE, 0xff, ACT_EXIT, 0 },
   { "two_queries",    -0.75f, -0.75f, 9.5f, 0.5f, 0, 100, RF_NONE, 0xff,
     ACT_TWO | ACT_CONFIRM, 0 },
   { "two_queries_sel",-0.75f, -0.75f, 9.5f, 0.5f, 0, 100, RF_NONE, 0xff,
     ACT_TWO | ACT_CONFIRM | ACT_QSEL, 0 },
   { "null_as",        -0.75f, -0.75f, 0, 0, 0, 100, RF_NONE, 0xff,
     ACT_NULL_AS | ACT_CONFIRM, 0 },
   { "tmin",           -0.75f, -0.75f, 0, 0, 1.25f, 100, RF_NONE, 0xff, ACT_CONFIRM, 0 },
   { "tmax",           -0.75f, -0.75f, 0, 0, 0, 1.25f, RF_NONE, 0xff, 0, 0 },
   { "no_opaque",       0.25f,  0.25f, 0, 0, 0, 100, RF_NO_OPAQUE, 0xff, ACT_CONFIRM, 0 },
   { "cull_no_opaque", -0.75f, -0.75f, 0, 0, 0, 100, RF_CULL_NO_OPAQUE, 0xff, ACT_CONFIRM, 0 },
   { "exit_split",     -0.75f, -0.75f, 0.25f, 0.25f, 0, 100, RF_NONE, 0xff,
     ACT_EXIT | ACT_SPLIT, 0 },
   { "abandon",        -0.75f, -0.75f, 0.25f, 0.25f, 0, 100, RF_NONE, 0xff, ACT_ABANDON, 0 },
};
#endif
#ifndef LOCAL_SIZE
#define LOCAL_SIZE WARP
#endif
#define NUM_WARPS (sizeof(warps) / sizeof(warps[0]))
#define NUM_RAYS  (NUM_WARPS * WARP)

/* ---------------------------------------------------- host reference */

struct hit {
   bool  valid, procedural, opaque, front;
   double t, u, v;
   uint32_t inst, geom, prim;
};

static void
xform_point(const float m[3][4], const double p[3], double w, double out[3])
{
   for (int r = 0; r < 3; r++)
      out[r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3] * w;
}

/* The inverse of an affine 3x4 whose linear part is diagonal, as every
 * instance here is. */
static void
invert_diag(const float m[3][4], float inv[3][4])
{
   memset(inv, 0, sizeof(float) * 12);
   for (int r = 0; r < 3; r++) {
      inv[r][r] = 1.0f / m[r][r];
      inv[r][3] = -m[r][3] / m[r][r];
   }
}

/* Möller-Trumbore in double. `front` is the facing before the instance's
 * flip: counter-clockwise winding seen from the ray origin (det > 0). */
static bool
ray_tri(const double o[3], const double d[3], const float tv[3][3],
        double *t, double *u, double *v, bool *front)
{
   double e1[3], e2[3], s[3], p[3], q[3];
   for (int i = 0; i < 3; i++) {
      e1[i] = tv[1][i] - tv[0][i];
      e2[i] = tv[2][i] - tv[0][i];
      s[i]  = o[i] - tv[0][i];
   }
   p[0] = d[1] * e2[2] - d[2] * e2[1];
   p[1] = d[2] * e2[0] - d[0] * e2[2];
   p[2] = d[0] * e2[1] - d[1] * e2[0];
   double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
   if (fabs(det) < 1e-12)
      return false;
   double inv = 1.0 / det;
   *u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
   q[0] = s[1] * e1[2] - s[2] * e1[1];
   q[1] = s[2] * e1[0] - s[0] * e1[2];
   q[2] = s[0] * e1[1] - s[1] * e1[0];
   *v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
   *t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
   *front = det > 0;
   return *u >= 0 && *v >= 0 && *u + *v <= 1;
}

static bool
ray_box(const double o[3], const double d[3], const VkAabbPositionsKHR *b,
        double tmin, double tmax)
{
   const double lo[3] = { b->minX, b->minY, b->minZ };
   const double hi[3] = { b->maxX, b->maxY, b->maxZ };
   for (int a = 0; a < 3; a++) {
      if (d[a] == 0) {
         if (o[a] < lo[a] || o[a] > hi[a])
            return false;
         continue;
      }
      double t0 = (lo[a] - o[a]) / d[a], t1 = (hi[a] - o[a]) / d[a];
      if (t0 > t1) { double x = t0; t0 = t1; t1 = x; }
      if (t0 > tmin) tmin = t0;
      if (t1 < tmax) tmax = t1;
   }
   return tmin <= tmax;
}

/* Every primitive the ray meets after culling, sorted by distance. */
static unsigned
trace_host(const double o[3], const double d[3], double tmin, double tmax,
           uint32_t flags, uint32_t cull, struct hit *hits)
{
   unsigned n = 0;
   for (uint32_t i = 0; i < NUM_INST; i++) {
      const struct inst_desc *in = &insts[i];
      if ((in->mask & cull) == 0)
         continue;
      float w2o[3][4];
      invert_diag(in->o2w, w2o);
      double oo[3], od[3];
      xform_point(w2o, o, 1, oo);
      xform_point(w2o, d, 0, od);
      unsigned ngeom = in->procedural ? AABB_GEOMS : TRI_GEOMS;
      for (uint32_t g = 0; g < ngeom; g++) {
         bool opaque = in->procedural ? aabb_opaque[g] : tri_opaque[g];
         if (in->flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR) opaque = true;
         if (in->flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) opaque = false;
         if (flags & RF_OPAQUE) opaque = true;
         else if (flags & RF_NO_OPAQUE) opaque = false;
         if ((flags & RF_CULL_OPAQUE) && opaque) continue;
         if ((flags & RF_CULL_NO_OPAQUE) && !opaque) continue;
         if (in->procedural) {
            if (flags & RF_SKIP_AABBS) continue;
            const VkAabbPositionsKHR *boxes = g ? aabb_g1 : aabb_g0;
            unsigned nb = g ? 2 : 1;
            for (uint32_t p = 0; p < nb; p++) {
               if (!ray_box(oo, od, &boxes[p], tmin, tmax))
                  continue;
               hits[n++] = (struct hit){ true, true, opaque, false, 0, 0, 0, i, g, p };
            }
         } else {
            if (flags & RF_SKIP_TRIANGLES) continue;
            for (uint32_t p = 0; p < 2; p++) {
               double t, u, v; bool front;
               if (!ray_tri(oo, od, tri_verts[g][p], &t, &u, &v, &front))
                  continue;
               if (t <= tmin || t >= tmax)
                  continue;
               if (in->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR)
                  front = !front;
               if (!(in->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)) {
                  if ((flags & RF_CULL_BACK) && !front) continue;
                  if ((flags & RF_CULL_FRONT) && front) continue;
               }
               hits[n++] = (struct hit){ true, false, opaque, front, t, u, v, i, g, p };
            }
         }
      }
   }
   for (unsigned a = 1; a < n; a++)
      for (unsigned b = a; b > 0 && hits[b - 1].t > hits[b].t; b--) {
         struct hit x = hits[b]; hits[b] = hits[b - 1]; hits[b - 1] = x;
      }
   return n;
}

/* Expected record plus, per word, whether it is checked and whether it is a
 * float compared to a tolerance. */
struct expect {
   uint32_t w[REC_WORDS];
   bool     check[REC_WORDS];
   bool     isf[REC_WORDS];
};

static void eu(struct expect *e, unsigned i, uint32_t v) { e->w[i] = v; e->check[i] = true; e->isf[i] = false; }
static void ef(struct expect *e, unsigned i, double v)
{
   float f = (float)v;
   memcpy(&e->w[i], &f, 4); e->check[i] = true; e->isf[i] = true;
}

/* The instance-dependent values a committed or candidate record carries. */
static void
expect_instance(struct expect *e, const struct hit *h, const double o[3],
                const double d[3], unsigned inst_w, unsigned custom_w,
                unsigned sbt_w, unsigned obj_w, bool matrices)
{
   const struct inst_desc *in = &insts[h->inst];
   float w2o[3][4];
   invert_diag(in->o2w, w2o);
   double oo[3], od[3];
   xform_point(w2o, o, 1, oo);
   xform_point(w2o, d, 0, od);
   eu(e, inst_w, h->inst);
   eu(e, custom_w, in->custom);
   eu(e, sbt_w, in->sbt);
   for (int k = 0; k < 3; k++) {
      ef(e, obj_w + k, oo[k]);
      ef(e, obj_w + 3 + k, od[k]);
   }
   if (matrices)
      for (int c = 0; c < 4; c++)
         for (int r = 0; r < 3; r++) {
            ef(e, R_O2W + c * 3 + r, in->o2w[r][c]);
            ef(e, R_W2O + c * 3 + r, w2o[r][c]);
         }
}

static void
expect_committed(struct expect *e, const struct hit *h, bool generated,
                 double gen_t, const double o[3], const double d[3])
{
   if (!h) {
      eu(e, R_CTYPE, 0);
      return;
   }
   eu(e, R_CTYPE, generated ? 2 : 1);
   ef(e, R_CT, generated ? gen_t : h->t);
   eu(e, R_CPRIM, h->prim);
   eu(e, R_CGEOM, h->geom);
   if (!generated) {
      ef(e, R_CU, h->u);
      ef(e, R_CV, h->v);
      eu(e, R_CFRONT, h->front);
   }
   expect_instance(e, h, o, d, R_CINST, R_CCUSTOM, R_CSBT, R_COBJ, true);
}

/* The committed hit of a full traversal that confirms every triangle
 * candidate and generates every box at gen_t, per the action bits. */
static const struct hit *
resolve(const struct hit *hits, unsigned n, uint32_t act, double gen_t,
        bool *generated)
{
   const struct hit *best = NULL;
   double best_t = INFINITY;
   *generated = false;
   for (unsigned i = 0; i < n; i++) {
      const struct hit *h = &hits[i];
      double t;
      bool gen = false;
      if (h->procedural) {
         if (!(act & ACT_GENERATE)) continue;
         t = gen_t; gen = true;
      } else if (h->opaque || (act & ACT_CONFIRM)) {
         t = h->t;
      } else {
         continue;
      }
      if (t < best_t) { best = h; best_t = t; *generated = gen; }
   }
   return best;
}

static bool
build_expect(const struct warp_desc *wd, const struct ray *r, struct expect *e)
{
   memset(e, 0, sizeof(*e));
   double o[3] = { r->origin[0], r->origin[1], r->origin[2] };
   double d[3] = { r->dir[0], r->dir[1], r->dir[2] };
   eu(e, R_TAG, 0xC0DE0000u | r->act);
   ef(e, R_TMIN, r->tmin);
   eu(e, R_FLAGS, r->flags);
   for (int k = 0; k < 3; k++) {
      ef(e, R_WO + k, o[k]);
      ef(e, R_WD + k, d[k]);
   }

   struct hit hits[16];
   unsigned n = 0;
   if (!(r->act & ACT_NULL_AS))
      n = trace_host(o, d, r->tmin, r->tmax, r->flags, r->cull, hits);

   /* Candidates: every non-opaque primitive and every box. The scene puts each
    * ray in front of at most one of them before any opaque hit, which is what
    * makes the presented candidate and the count exact. */
   const struct hit *cands[16];
   unsigned nc = 0;
   for (unsigned i = 0; i < n; i++)
      if (hits[i].procedural || !hits[i].opaque)
         cands[nc++] = &hits[i];
   for (unsigned i = 0; i < n; i++)
      if (hits[i].opaque && !hits[i].procedural)
         for (unsigned c = 0; c < nc; c++)
            if (cands[c]->t > hits[i].t && !cands[c]->procedural) {
               fprintf(stderr, "scene error: %s has a candidate behind an opaque hit\n",
                       wd->name);
               return false;
            }

   if (r->act & (ACT_TWO | ACT_INTERLEAVE)) {
      bool gen;
      const struct hit *h = resolve(hits, n, r->act, wd->gen_t, &gen);
      expect_committed(e, h, gen, wd->gen_t, o, d);
      if (nc <= 1)
         eu(e, R_COUNT, nc);
      double o2[3] = { r->origin2[0], r->origin2[1], r->origin2[2] };
      struct hit hits2[16];
      unsigned n2 = trace_host(o2, d, r->tmin, r->tmax, r->flags, r->cull, hits2);
      const struct hit *h2 = resolve(hits2, n2, r->act, wd->gen_t, &gen);
      if (!h2 || gen) {
         fprintf(stderr, "scene error: %s second ray must hit a triangle\n", wd->name);
         return false;
      }
      eu(e, R_Q2, 1u | (h2->inst << 4) | (h2->prim << 8));
      ef(e, R_Q2T, h2->t);
      return true;
   }

   eu(e, R_PROCEED, nc > 0);
   if (nc <= 1)
      eu(e, R_COUNT, nc);
   if (r->act & (ACT_EXIT | ACT_ABANDON))
      eu(e, R_COUNT, nc > 0 ? 1 : 0);
   if (nc == 1) {
      const struct hit *k = cands[0];
      eu(e, R_KTYPE, k->procedural ? 1 : 0);
      eu(e, R_KPRIM, k->prim);
      eu(e, R_KGEOM, k->geom);
      if (k->procedural) {
         eu(e, R_KAABBOPQ, k->opaque);
      } else {
         ef(e, R_KT, k->t);
         ef(e, R_KU, k->u);
         ef(e, R_KV, k->v);
         eu(e, R_KFRONT, k->front);
      }
      expect_instance(e, k, o, d, R_KINST, R_KCUSTOM, R_KSBT, R_KOBJ, false);
   }

   if (r->act & (ACT_EXIT | ACT_ABANDON)) {
      double o2[3] = { r->origin2[0], r->origin2[1], r->origin2[2] };
      struct hit hits2[16];
      unsigned n2 = trace_host(o2, d, r->tmin, r->tmax, r->flags, r->cull, hits2);
      bool gen;
      const struct hit *h2 = resolve(hits2, n2, ACT_CONFIRM, 0, &gen);
      expect_committed(e, h2, gen, 0, o2, d);
      for (int k = 0; k < 3; k++)
         ef(e, R_WO + k, o2[k]);
      return true;
   }

   if (r->act & ACT_TERMINATE) {
      for (unsigned i = 0; i < n; i++)
         if (hits[i].opaque && !hits[i].procedural) {
            fprintf(stderr, "scene error: %s terminates beside an opaque hit\n",
                    wd->name);
            return false;
         }
      const struct hit *h = NULL;
      bool gen = false;
      if (nc > 0) {
         const struct hit *k = cands[0];
         if (!k->procedural && (r->act & ACT_CONFIRM)) h = k;
         if (k->procedural && (r->act & ACT_GENERATE)) { h = k; gen = true; }
      }
      expect_committed(e, h, gen, wd->gen_t, o, d);
      return true;
   }

   bool gen;
   const struct hit *h = resolve(hits, n, r->act, wd->gen_t, &gen);
   expect_committed(e, h, gen, wd->gen_t, o, d);
   return true;
}

/* ------------------------------------------------------- vulkan helpers */

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static bool
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void *src,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
   };
   if (vkCreateBuffer(dev, &bci, NULL, buf) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mr.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX)
      return false;
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &mafi, .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS)
      return false;
   vkBindBufferMemory(dev, *buf, *mem, 0);
   if (src) {
      void *p;
      if (vkMapMemory(dev, *mem, 0, size, 0, &p) != VK_SUCCESS)
         return false;
      memcpy(p, src, size);
      vkUnmapMemory(dev, *mem);
   }
   return true;
}

static VkDeviceAddress
buffer_addr(VkBuffer buf)
{
   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf,
   };
   return vkGetBufferDeviceAddress(dev, &bdai);
}

static VkAccelerationStructureKHR
build_as(VkCommandBuffer cmd, VkAccelerationStructureTypeKHR type,
         const VkAccelerationStructureGeometryKHR *geoms, unsigned ngeom,
         const uint32_t *prim_counts)
{
   VkAccelerationStructureBuildGeometryInfoKHR bgi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = type,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = ngeom, .pGeometries = geoms,
   };
   VkAccelerationStructureBuildSizesInfoKHR sizes = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
   };
   p_GetAccelBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                        &bgi, prim_counts, &sizes);

   VkBuffer asbuf; VkDeviceMemory asmem;
   if (!make_buffer(sizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                    NULL, &asbuf, &asmem))
      return VK_NULL_HANDLE;
   VkAccelerationStructureCreateInfoKHR aci = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = asbuf, .size = sizes.accelerationStructureSize, .type = type,
   };
   VkAccelerationStructureKHR as;
   if (p_CreateAccelStruct(dev, &aci, NULL, &as) != VK_SUCCESS)
      return VK_NULL_HANDLE;

   VkBuffer scratch; VkDeviceMemory scratchmem;
   if (!make_buffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    NULL, &scratch, &scratchmem))
      return VK_NULL_HANDLE;
   bgi.dstAccelerationStructure  = as;
   bgi.scratchData.deviceAddress = buffer_addr(scratch);

   VkAccelerationStructureBuildRangeInfoKHR ranges[4] = {{0}};
   for (unsigned g = 0; g < ngeom; g++)
      ranges[g].primitiveCount = prim_counts[g];
   const VkAccelerationStructureBuildRangeInfoKHR *pranges = ranges;
   p_CmdBuildAccelStructs(cmd, 1, &bgi, &pranges);

   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
   };
   vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &mb, 0, NULL, 0, NULL);
   return as;
}

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "FAILED: cannot open %s\n", path); return NULL; }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *buf = malloc((size_t)sz);
   if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); buf = NULL; }
   fclose(f);
   if (buf) *out_size = (size_t)sz;
   return buf;
}

static VkAccelerationStructureGeometryKHR
tri_geometry(VkBuffer vbuf, bool opaque)
{
   return (VkAccelerationStructureGeometryKHR){
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0,
      .geometry.triangles = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
         .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
         .vertexData.deviceAddress = buffer_addr(vbuf),
         .vertexStride = 3 * sizeof(float),
         .maxVertex = 5,
         .indexType = VK_INDEX_TYPE_NONE_KHR,
      },
   };
}

static VkAccelerationStructureGeometryKHR
aabb_geometry(VkBuffer abuf, bool opaque)
{
   return (VkAccelerationStructureGeometryKHR){
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
      .flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0,
      .geometry.aabbs = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
         .data.deviceAddress = buffer_addr(abuf),
         .stride = sizeof(VkAabbPositionsKHR),
      },
   };
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "rtquery_full.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-rtquery_full",
      .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(pd, &memprops);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   const char *exts[] = {
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
   };
   /* rayTraversalPrimitiveCulling gates the skip-triangles / skip-AABBs ray
    * flags for ray queries too. */
   VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .rayTraversalPrimitiveCulling = VK_TRUE,
   };
   VkPhysicalDeviceRayQueryFeaturesKHR rqf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
      .pNext = &rtpf, .rayQuery = VK_TRUE,
   };
   VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rqf, .accelerationStructure = VK_TRUE,
   };
   VkPhysicalDeviceBufferDeviceAddressFeatures bdaf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &asf, .bufferDeviceAddress = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &bdaf, .features.shaderInt64 = VK_TRUE,
   };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 4, .ppEnabledExtensionNames = exts,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   p_CreateAccelStruct = (PFN_vkCreateAccelerationStructureKHR)
      vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
   p_DestroyAccelStruct = (PFN_vkDestroyAccelerationStructureKHR)
      vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");
   p_GetAccelBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)
      vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
   p_CmdBuildAccelStructs = (PFN_vkCmdBuildAccelerationStructuresKHR)
      vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");
   p_GetAccelAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
      vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR");
   if (!p_CreateAccelStruct || !p_DestroyAccelStruct || !p_GetAccelBuildSizes ||
       !p_CmdBuildAccelStructs || !p_GetAccelAddress) {
      fprintf(stderr, "FAILED: acceleration-structure entry points unavailable\n");
      return 1;
   }

   /* --- geometry ---------------------------------------------------- */
   VkBuffer tbuf[TRI_GEOMS], abuf[AABB_GEOMS];
   VkDeviceMemory tmem[TRI_GEOMS], amem[AABB_GEOMS];
   const VkBufferUsageFlags in_usage =
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   for (unsigned g = 0; g < TRI_GEOMS; g++)
      if (!make_buffer(sizeof(tri_verts[g]), in_usage, tri_verts[g], &tbuf[g], &tmem[g])) {
         fprintf(stderr, "FAILED: vertex buffer\n"); return 1;
      }
   if (!make_buffer(sizeof(aabb_g0), in_usage, aabb_g0, &abuf[0], &amem[0]) ||
       !make_buffer(sizeof(aabb_g1), in_usage, aabb_g1, &abuf[1], &amem[1])) {
      fprintf(stderr, "FAILED: aabb buffer\n"); return 1;
   }

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   VkAccelerationStructureGeometryKHR tg[TRI_GEOMS], ag[AABB_GEOMS];
   for (unsigned g = 0; g < TRI_GEOMS; g++)
      tg[g] = tri_geometry(tbuf[g], tri_opaque[g]);
   for (unsigned g = 0; g < AABB_GEOMS; g++)
      ag[g] = aabb_geometry(abuf[g], aabb_opaque[g]);
   const uint32_t tri_counts[TRI_GEOMS] = { 2, 2 };
   const uint32_t aabb_counts[AABB_GEOMS] = { 1, 2 };
   VkAccelerationStructureKHR blas_t =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, tg, TRI_GEOMS, tri_counts);
   VkAccelerationStructureKHR blas_p =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, ag, AABB_GEOMS, aabb_counts);
   if (!blas_t || !blas_p) { fprintf(stderr, "FAILED: BLAS\n"); return 1; }

   VkAccelerationStructureInstanceKHR vinst[NUM_INST];
   for (unsigned i = 0; i < NUM_INST; i++) {
      VkAccelerationStructureDeviceAddressInfoKHR adai = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
         .accelerationStructure = insts[i].procedural ? blas_p : blas_t,
      };
      memset(&vinst[i], 0, sizeof(vinst[i]));
      memcpy(vinst[i].transform.matrix, insts[i].o2w, sizeof(insts[i].o2w));
      vinst[i].instanceCustomIndex = insts[i].custom;
      vinst[i].mask = insts[i].mask;
      vinst[i].instanceShaderBindingTableRecordOffset = insts[i].sbt;
      vinst[i].flags = insts[i].flags;
      vinst[i].accelerationStructureReference = p_GetAccelAddress(dev, &adai);
   }
   VkBuffer instbuf; VkDeviceMemory instmem;
   if (!make_buffer(sizeof(vinst), in_usage, vinst, &instbuf, &instmem)) {
      fprintf(stderr, "FAILED: instance buffer\n"); return 1;
   }
   VkAccelerationStructureGeometryKHR ig = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
         .data.deviceAddress = buffer_addr(instbuf),
      },
   };
   const uint32_t inst_count = NUM_INST;
   VkAccelerationStructureKHR tlas =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &ig, 1, &inst_count);
   if (!tlas) { fprintf(stderr, "FAILED: TLAS\n"); return 1; }

   /* --- rays --------------------------------------------------------- */
   struct ray *rays = calloc(NUM_RAYS, sizeof(*rays));
   for (unsigned w = 0; w < NUM_WARPS; w++)
      for (unsigned j = 0; j < WARP; j++) {
         const struct warp_desc *wd = &warps[w];
         struct ray *r = &rays[w * WARP + j];
         float dx = (float)(j % 8) * 0.0625f, dy = (float)(j / 8) * 0.0625f;
         r->origin[0] = wd->x + dx; r->origin[1] = wd->y + dy; r->origin[2] = 2.0f;
         r->dir[2] = -1.0f;
         r->tmin = wd->tmin; r->tmax = wd->tmax;
         r->flags = wd->flags; r->cull = wd->cull;
         r->act = wd->act & ~ACT_SPLIT;
         if ((wd->act & ACT_SPLIT) && (j % 2) == 0)
            r->act = (r->act & ~ACT_EXIT) | ACT_CONFIRM;
         memcpy(&r->gen_t, &wd->gen_t, 4);
         r->origin2[0] = wd->x2 + dx; r->origin2[1] = wd->y2 + dy; r->origin2[2] = 2.0f;
      }
   VkBuffer rbuf, obuf; VkDeviceMemory rmem, omem;
   const VkDeviceSize obytes = (VkDeviceSize)NUM_RAYS * REC_WORDS * 4;
   if (!make_buffer(NUM_RAYS * sizeof(*rays), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    rays, &rbuf, &rmem) ||
       !make_buffer(obytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, &obuf, &omem)) {
      fprintf(stderr, "FAILED: ray/record buffers\n"); return 1;
   }

   /* --- pipeline ----------------------------------------------------- */
   size_t spv_size = 0;
   uint32_t *spv = read_spirv(spv_path, &spv_size);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(spv);

   VkDescriptorSetLayoutBinding dslb[3] = {
      { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkComputePipelineCreateInfo cpc = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpc, NULL, &pipe));

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkWriteDescriptorSetAccelerationStructureKHR was = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 1, .pAccelerationStructures = &tlas,
   };
   VkDescriptorBufferInfo rdbi = { rbuf, 0, VK_WHOLE_SIZE };
   VkDescriptorBufferInfo odbi = { obuf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &was,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &rdbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 2, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &odbi },
   };
   vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cmd, NUM_RAYS / LOCAL_SIZE, 1, 1);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify ------------------------------------------------------- */
   uint32_t *rec;
   CHECK(vkMapMemory(dev, omem, 0, obytes, 0, (void **)&rec));
   unsigned bad_rays = 0, bad_warps = 0, checked = 0;
   for (unsigned w = 0; w < NUM_WARPS; w++) {
      unsigned bad_here = 0;
      for (unsigned j = 0; j < WARP; j++) {
         unsigned id = w * WARP + j;
         struct expect e;
         if (!build_expect(&warps[w], &rays[id], &e)) {
            printf("FAILED (host scene description)\n");
            return 1;
         }
         const uint32_t *got = rec + (size_t)id * REC_WORDS;
         bool ok = true;
         for (unsigned i = 0; i < REC_WORDS; i++) {
            if (!e.check[i])
               continue;
            checked++;
            bool same;
            if (e.isf[i]) {
               float a, b;
               memcpy(&a, &got[i], 4); memcpy(&b, &e.w[i], 4);
               same = fabsf(a - b) <= 1e-5f * fmaxf(1.0f, fabsf(b));
            } else {
               same = got[i] == e.w[i];
            }
            if (!same) {
               if (bad_here < 4) {
                  float a, b;
                  memcpy(&a, &got[i], 4); memcpy(&b, &e.w[i], 4);
                  printf("  %s lane %u word %u: got 0x%08x (%g) want 0x%08x (%g)\n",
                         warps[w].name, j, i, got[i], a, e.w[i], b);
               }
               bad_here++;
               ok = false;
            }
         }
         if (!ok) bad_rays++;
      }
      if (bad_here) bad_warps++;
   }
   vkUnmapMemory(dev, omem);

   p_DestroyAccelStruct(dev, tlas, NULL);
   p_DestroyAccelStruct(dev, blas_t, NULL);
   p_DestroyAccelStruct(dev, blas_p, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad_rays) {
      printf("FAILED (%u of %u rays wrong in %u of %u cases)\n",
             bad_rays, (unsigned)NUM_RAYS, bad_warps, (unsigned)NUM_WARPS);
      return 1;
   }
   printf("PASSED (ray-query values: %u cases x %u rays, %u values checked)\n",
          (unsigned)NUM_WARPS, WARP, checked);
   return 0;
}
