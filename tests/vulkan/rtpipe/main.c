/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * rtpipe -- vkCmdTraceRaysKHR with every shader stage.
 *
 * A raygen shader traces one ray per launch cell into a two-instance scene (a
 * triangle BLAS with an opaque and a non-opaque geometry, and a procedural
 * BLAS with an opaque and a non-opaque box geometry), then calls one of two
 * callables. Launch rows select the scenario: an any-hit alpha test, an
 * any-hit terminate, intersection shaders with and without any-hit, both miss
 * shaders, closest-hit shaders that trace again (recursion depth 3), a callable
 * that calls another, an sbtRecordOffset into a second set of hit records, and
 * an opaque and an accepted non-opaque triangle at one distance, where the
 * record must be wholly one of the two. Rows sharing a warp trace with
 * different ray flags; the first of them hits, so its lanes hold their results
 * while the later configurations trace. Every closest-hit, miss and callable reads its own
 * shader-record word, so the record each ray landed on is checked as well as
 * the hit. The host traverses the scene itself and compares every word of
 * every cell.
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

#define LAUNCH_W 32u
#define LAUNCH_H 9u

/* Mirrors rtpipe.glsl. */
enum {
   P_KIND, P_HITKIND, P_T, P_PRIM, P_INST, P_CUSTOM, P_GEOM, P_ATTR_U,
   P_ATTR_V, P_RECORD, P_OBJ_OX, P_O2W_TX, P_W2O_XX, P_TMIN, P_TMAX,
   P_FLAGS, P_LAUNCH, P_SEC_KIND, P_SEC_T, P_SEC_CUST, P_CALL, P_AHS,
   P_WDIR_Z, P_RGEN_REC, P_THIRD, P_WORDS
};
enum { K_NONE, K_CHIT0, K_CHIT1, K_CHIT2, K_MISS0, K_MISS1 };
enum { ROW_ALPHA, ROW_TERMINATE, ROW_PROC_OPQ, ROW_PROC, ROW_RECURSE,
       ROW_MISS0, ROW_MISS1, ROW_SBT_OFF, ROW_TIE };

#define RF_OPAQUE         0x01u
#define RF_SKIP_CHIT      0x08u
#define RF_CULL_BACK      0x10u
#define RF_CULL_NO_OPAQUE 0x80u

/* Shader stages, in the order the SPIR-V files are passed. */
enum { S_RGEN, S_MISS0, S_MISS1, S_CHIT0, S_CHIT1, S_CHIT2, S_AHIT1,
       S_AHIT2, S_RINT, S_CALL0, S_CALL1, NUM_STAGES };

/* Shader groups. */
enum { G_RGEN, G_MISS0, G_MISS1, G_TRI_OPAQUE, G_TRI_ALPHA, G_PROC,
       G_CALL0, G_CALL1, NUM_GROUPS };

/* Hit records: the group each one points at and the word it carries. */
static const struct { uint32_t group, data, kind; } hit_records[] = {
   { G_TRI_OPAQUE, 100, K_CHIT0 },
   { G_TRI_ALPHA,  101, K_CHIT1 },
   { G_PROC,       102, K_CHIT2 },
   { G_PROC,       103, K_CHIT2 },
   { G_TRI_OPAQUE, 104, K_CHIT0 },
   { G_TRI_ALPHA,  105, K_CHIT1 },
};
#define NUM_HIT_RECORDS (sizeof(hit_records) / sizeof(hit_records[0]))
static const uint32_t miss_data[2] = { 200, 201 };
static const uint32_t call_data[2] = { 300, 301 };
static const uint32_t rgen_data    = 400;

static VkDevice dev = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties memprops;
static PFN_vkCreateAccelerationStructureKHR         p_CreateAccelStruct;
static PFN_vkDestroyAccelerationStructureKHR        p_DestroyAccelStruct;
static PFN_vkGetAccelerationStructureBuildSizesKHR  p_GetAccelBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR      p_CmdBuildAccelStructs;
static PFN_vkGetAccelerationStructureDeviceAddressKHR p_GetAccelAddress;
static PFN_vkCreateRayTracingPipelinesKHR           p_CreateRtPipelines;
static PFN_vkGetRayTracingShaderGroupHandlesKHR     p_GetGroupHandles;
static PFN_vkCmdTraceRaysKHR                        p_CmdTraceRays;

/* ---------------------------------------------------------------- scene */

/* Triangle BLAS: geometry 0 opaque at z = 0, geometry 1 non-opaque above it
 * (z = 1 over the lower-left half, z = 0.5 over the upper-right half). Each
 * also has a primitive 2 over x in [2, 3], the two coincident at z = 0: a
 * ray there meets an opaque and a non-opaque triangle at the same distance. */
#define TRI_PRIMS 3
static const float tri_verts[2][TRI_PRIMS][3][3] = {
   { { { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 } },
     { {  1, -1, 0 }, { 1,  1, 0 }, { -1, 1, 0 } },
     { {  2, -1, 0 }, { 3, -1, 0 }, {  2, 1, 0 } } },
   { { { -1, -1, 1 }, { 1, -1, 1 }, { -1, 1, 1 } },
     { {  1, -1, 0.5f }, { 1, 1, 0.5f }, { -1, 1, 0.5f } },
     { {  2, -1, 0 }, { 3, -1, 0 }, {  2, 1, 0 } } },
};

/* On the tie row either triangle is a valid closest hit; the host checks the
 * record against both orders and requires it to be one of them, whole. */
static bool tie_non_opaque_first;
static const bool tri_opaque[2] = { true, false };

/* Procedural BLAS: geometry 0 one opaque box over x < 0, geometry 1 two
 * non-opaque boxes over x > 0 (primitive 0 below y = 0, primitive 1 above). */
static const VkAabbPositionsKHR aabb_g0[1] = { { -1, -1, -0.5f, 0, 1, 0.5f } };
static const VkAabbPositionsKHR aabb_g1[2] = {
   { 0, -1, -0.5f, 1, 0, 0.5f },
   { 0,  0, -0.5f, 1, 1, 0.5f },
};
static const bool aabb_opaque[2] = { true, false };

struct inst_desc { bool procedural; float tx; uint32_t custom, sbt; };
static const struct inst_desc insts[2] = {
   { false, 0.0f,  0x11, 0 },
   { true,  20.0f, 0x22, 2 },
};

/* ---------------------------------------------------- host reference */

struct hit {
   bool procedural, opaque, front;
   float t;
   float u, v;          /* barycentrics, or the intersection's attribute */
   uint32_t inst, geom, prim, hitkind;
   float obj_ox;
};

/* Möller-Trumbore in double; front facing is counter-clockwise winding seen
 * from the ray origin (det > 0). */
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
in_box(double x, double y, const VkAabbPositionsKHR *b)
{
   return x >= b->minX && x <= b->maxX && y >= b->minY && y <= b->maxY;
}

/* Every primitive a vertical ray down -Z meets, with the intersection
 * shader's reported distance for boxes, sorted by distance. */
static unsigned
trace_host(const float o[3], float tmin, float tmax, uint32_t flags,
           struct hit *hits)
{
   const double d[3] = { 0, 0, -1 };
   unsigned n = 0;
   for (uint32_t i = 0; i < 2; i++) {
      double oo[3] = { (double)o[0] - insts[i].tx, o[1], o[2] };
      for (uint32_t g = 0; g < 2; g++) {
         bool opaque = insts[i].procedural ? aabb_opaque[g] : tri_opaque[g];
         if (flags & RF_OPAQUE) opaque = true;
         if ((flags & RF_CULL_NO_OPAQUE) && !opaque) continue;
         if (insts[i].procedural) {
            const VkAabbPositionsKHR *boxes = g ? aabb_g1 : aabb_g0;
            for (uint32_t p = 0; p < (g ? 2u : 1u); p++) {
               if (!in_box(oo[0], oo[1], &boxes[p]))
                  continue;
               float t = o[2] - 0.25f - 0.5f * (float)g;
               if (t < tmin || t > tmax)
                  continue;
               hits[n++] = (struct hit){ true, opaque, false, t,
                  (float)p + 0.5f, (float)oo[0], i, g, p, 7 + p, (float)oo[0] };
            }
         } else {
            for (uint32_t p = 0; p < TRI_PRIMS; p++) {
               double t, u, v; bool front;
               if (!ray_tri(oo, d, tri_verts[g][p], &t, &u, &v, &front))
                  continue;
               if (t <= tmin || t >= tmax)
                  continue;
               if ((flags & RF_CULL_BACK) && !front)
                  continue;
               hits[n++] = (struct hit){ false, opaque, front, (float)t,
                  (float)u, (float)v, i, g, p, front ? 0xFEu : 0xFFu, (float)oo[0] };
            }
         }
      }
   }
   for (unsigned a = 1; a < n; a++)
      for (unsigned b = a; b > 0; b--) {
         bool swap = hits[b - 1].t > hits[b].t ||
                     (hits[b - 1].t == hits[b].t &&
                      hits[b - 1].opaque == tie_non_opaque_first && hits[b].opaque != tie_non_opaque_first);
         if (!swap)
            break;
         struct hit x = hits[b]; hits[b] = hits[b - 1]; hits[b - 1] = x;
      }
   return n;
}

struct expect {
   uint32_t w[P_WORDS];
   bool check[P_WORDS], isf[P_WORDS];
};
static void eu(struct expect *e, unsigned i, uint32_t v) { e->w[i] = v; e->check[i] = true; e->isf[i] = false; }
static void ef(struct expect *e, unsigned i, float v) { memcpy(&e->w[i], &v, 4); e->check[i] = true; e->isf[i] = true; }

/* One traceRayEXT: any-hit and intersection decisions, then the closest-hit
 * or miss the ray ends in. Fills the payload words those shaders write. */
static void
trace_expect(struct expect *e, const float o[3], float tmin, float tmax,
             uint32_t flags, uint32_t sbt_off, uint32_t miss, uint32_t row,
             unsigned kind_w, unsigned t_w, unsigned cust_w, bool full)
{
   struct hit hits[8];
   unsigned n = trace_host(o, tmin, tmax, flags, hits);
   const struct hit *best = NULL;
   uint32_t ahs = 0;
   for (unsigned i = 0; i < n; i++) {
      const struct hit *h = &hits[i];
      if (best && h->t >= best->t)
         continue;
      bool accept = true, stop = false;
      if (!h->opaque) {
         ahs++;
         if (h->procedural) {
            accept = h->v <= 0.5f;
         } else if (row == ROW_TERMINATE) {
            stop = true;
         } else {
            accept = h->u >= 0.25f;
         }
      }
      if (accept)
         best = h;
      if (stop)
         break;
   }
   if (full)
      eu(e, P_AHS, ahs);

   if (!best) {
      eu(e, kind_w, miss ? K_MISS1 : K_MISS0);
      if (full) {
         eu(e, P_RECORD, miss_data[miss]);
         ef(e, P_TMIN, tmin);
         ef(e, P_TMAX, tmax);
         eu(e, P_FLAGS, flags);
         ef(e, P_WDIR_Z, -1.0f);
      }
      return;
   }
   if (flags & RF_SKIP_CHIT)
      return;
   uint32_t rec = sbt_off + insts[best->inst].sbt + best->geom;
   eu(e, kind_w, hit_records[rec].kind);
   ef(e, t_w, best->t);
   eu(e, cust_w, insts[best->inst].custom);
   if (!full) {
      /* the recursion row's second ray: its closest-hit traces a third, which
       * always misses */
      if (row == ROW_RECURSE && hit_records[rec].kind == K_CHIT2)
         eu(e, P_THIRD, K_MISS0);
      return;
   }
   eu(e, P_HITKIND, best->hitkind);
   eu(e, P_PRIM, best->prim);
   eu(e, P_INST, best->inst);
   eu(e, P_GEOM, best->geom);
   ef(e, P_ATTR_U, best->u);
   ef(e, P_ATTR_V, best->v);
   eu(e, P_RECORD, hit_records[rec].data);
   ef(e, P_OBJ_OX, best->obj_ox);
   ef(e, P_O2W_TX, insts[best->inst].tx);
   ef(e, P_W2O_XX, 1.0f);
   ef(e, P_TMIN, tmin);
   ef(e, P_TMAX, best->t);
   eu(e, P_FLAGS, flags);
   ef(e, P_WDIR_Z, -1.0f);

   if (hit_records[rec].kind == K_CHIT0 && row == ROW_RECURSE) {
      const float o2[3] = { o[0] + 20.0f, o[1], o[2] };
      trace_expect(e, o2, 0.0f, 100.0f, 0, 0, 0, row,
                   P_SEC_KIND, P_SEC_T, P_SEC_CUST, false);
   }
}

/* What the raygen shader does for launch cell (x, y). */
static void
build_expect(uint32_t x, uint32_t y, struct expect *e)
{
   memset(e, 0, sizeof(*e));
   float fx = (float)x * 0.03125f;
   float o[3] = { 0, 0, 2 };
   uint32_t flags = 0, sbt_off = 0, miss = 0;
   float tmin = 0.0f, tmax = 100.0f;
   switch (y) {
   case ROW_ALPHA:     o[0] = -0.95f + fx; o[1] = -0.9f; break;
   case ROW_TERMINATE: o[0] = 0.02f + fx * 0.5f; o[1] = 0.25f; break;
   case ROW_PROC_OPQ:  o[0] = 19.01f + fx; o[1] = -0.5f; break;
   case ROW_PROC:      o[0] = 20.01f + fx; o[1] = 0.5f; break;
   case ROW_MISS1:     o[0] = 5.0f + fx; miss = 1; break;
   case ROW_MISS0:
      o[0] = -5.0f - fx;
      flags = RF_CULL_BACK | RF_SKIP_CHIT;
      tmin = 0.5f; tmax = 50.0f + fx;
      break;
   case ROW_RECURSE:
      o[0] = -0.95f + fx; o[1] = -0.9f;
      flags = RF_CULL_NO_OPAQUE;
      break;
   case ROW_SBT_OFF:
      o[0] = -0.95f + fx; o[1] = -0.9f;
      flags = RF_OPAQUE;
      sbt_off = 4;
      break;
   default:
      o[0] = 2.3f + fx * 0.25f; o[1] = -0.9f;
      break;
   }
   eu(e, P_LAUNCH, x | (y << 16));
   eu(e, P_RGEN_REC, rgen_data);
   trace_expect(e, o, tmin, tmax, flags, sbt_off, miss, y,
                P_KIND, P_T, P_CUSTOM, true);
   uint32_t call = (x & 1) ? x * 5 + call_data[1] : x * 3 + call_data[0];
   if (!(x & 1) && (call & 3) == 0)
      call = call * 5 + call_data[1] + 7;
   eu(e, P_CALL, call);
   for (unsigned i = 0; i < P_WORDS; i++)
      if (!e->check[i])
         eu(e, i, 0);
}

static bool
word_matches(const struct expect *e, unsigned i, uint32_t got)
{
   if (!e->isf[i])
      return got == e->w[i];
   float a, b;
   memcpy(&a, &got, 4); memcpy(&b, &e->w[i], 4);
   return fabsf(a - b) <= 1e-5f * fmaxf(1.0f, fabsf(b));
}

static bool
record_matches(const struct expect *e, const uint32_t *got)
{
   for (unsigned i = 0; i < P_WORDS; i++)
      if (!word_matches(e, i, got[i]))
         return false;
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
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      0, 1, &mb, 0, NULL, 0, NULL);
   return as;
}

static VkShaderModule
load_module(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "FAILED: cannot open %s\n", path); return VK_NULL_HANDLE; }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *buf = malloc((size_t)sz);
   if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      fclose(f); free(buf); return VK_NULL_HANDLE;
   }
   fclose(f);
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = (size_t)sz, .pCode = buf,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   vkCreateShaderModule(dev, &smci, NULL, &sm);
   free(buf);
   return sm;
}

static VkDeviceSize
align_up(VkDeviceSize v, VkDeviceSize a)
{
   return (v + a - 1) / a * a;
}

int
main(int argc, char **argv)
{
   if (argc < 1 + NUM_STAGES) {
      fprintf(stderr, "FAILED: usage: rtpipe <%d SPIR-V stages>\n", NUM_STAGES);
      return 1;
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-rtpipe",
      .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rtprops,
   };
   vkGetPhysicalDeviceProperties2(pd, &props2);
   printf("device: %s\n", props2.properties.deviceName);
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
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
   };
   VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .rayTracingPipeline = VK_TRUE,
   };
   VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rtpf, .accelerationStructure = VK_TRUE,
   };
   VkPhysicalDeviceBufferDeviceAddressFeatures bdaf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &asf, .bufferDeviceAddress = VK_TRUE,
   };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &bdaf,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

#define LOAD(var, name) var = (void *)vkGetDeviceProcAddr(dev, name)
   LOAD(p_CreateAccelStruct, "vkCreateAccelerationStructureKHR");
   LOAD(p_DestroyAccelStruct, "vkDestroyAccelerationStructureKHR");
   LOAD(p_GetAccelBuildSizes, "vkGetAccelerationStructureBuildSizesKHR");
   LOAD(p_CmdBuildAccelStructs, "vkCmdBuildAccelerationStructuresKHR");
   LOAD(p_GetAccelAddress, "vkGetAccelerationStructureDeviceAddressKHR");
   LOAD(p_CreateRtPipelines, "vkCreateRayTracingPipelinesKHR");
   LOAD(p_GetGroupHandles, "vkGetRayTracingShaderGroupHandlesKHR");
   LOAD(p_CmdTraceRays, "vkCmdTraceRaysKHR");
#undef LOAD
   if (!p_CreateAccelStruct || !p_DestroyAccelStruct || !p_GetAccelBuildSizes ||
       !p_CmdBuildAccelStructs || !p_GetAccelAddress || !p_CreateRtPipelines ||
       !p_GetGroupHandles || !p_CmdTraceRays) {
      fprintf(stderr, "FAILED: ray-tracing entry points unavailable\n");
      return 1;
   }

   /* --- acceleration structures -------------------------------------- */
   const VkBufferUsageFlags in_usage =
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   VkBuffer tbuf[2], abuf[2];
   VkDeviceMemory tmem[2], amem[2];
   for (unsigned g = 0; g < 2; g++)
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

   VkAccelerationStructureGeometryKHR tg[2], ag[2];
   for (unsigned g = 0; g < 2; g++) {
      /* One any-hit call per candidate makes the any-hit count exact. */
      VkGeometryFlagsKHR gflags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      tg[g] = (VkAccelerationStructureGeometryKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
         .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
         .flags = gflags | (tri_opaque[g] ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0),
         .geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData.deviceAddress = buffer_addr(tbuf[g]),
            .vertexStride = 3 * sizeof(float),
            .maxVertex = 3 * TRI_PRIMS - 1,
            .indexType = VK_INDEX_TYPE_NONE_KHR,
         },
      };
      ag[g] = (VkAccelerationStructureGeometryKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
         .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
         .flags = gflags | (aabb_opaque[g] ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0),
         .geometry.aabbs = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
            .data.deviceAddress = buffer_addr(abuf[g]),
            .stride = sizeof(VkAabbPositionsKHR),
         },
      };
   }
   const uint32_t tri_counts[2] = { TRI_PRIMS, TRI_PRIMS }, aabb_counts[2] = { 1, 2 };
   VkAccelerationStructureKHR blas_t =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, tg, 2, tri_counts);
   VkAccelerationStructureKHR blas_p =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, ag, 2, aabb_counts);
   if (!blas_t || !blas_p) { fprintf(stderr, "FAILED: BLAS\n"); return 1; }

   VkAccelerationStructureInstanceKHR vinst[2];
   for (unsigned i = 0; i < 2; i++) {
      VkAccelerationStructureDeviceAddressInfoKHR adai = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
         .accelerationStructure = insts[i].procedural ? blas_p : blas_t,
      };
      memset(&vinst[i], 0, sizeof(vinst[i]));
      vinst[i].transform.matrix[0][0] = 1;
      vinst[i].transform.matrix[1][1] = 1;
      vinst[i].transform.matrix[2][2] = 1;
      vinst[i].transform.matrix[0][3] = insts[i].tx;
      vinst[i].instanceCustomIndex = insts[i].custom;
      vinst[i].mask = 0xff;
      vinst[i].instanceShaderBindingTableRecordOffset = insts[i].sbt;
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
   const uint32_t ninst = 2;
   VkAccelerationStructureKHR tlas =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &ig, 1, &ninst);
   if (!tlas) { fprintf(stderr, "FAILED: TLAS\n"); return 1; }

   /* --- pipeline ----------------------------------------------------- */
   static const VkShaderStageFlagBits stage_bits[NUM_STAGES] = {
      VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_SHADER_STAGE_MISS_BIT_KHR,
      VK_SHADER_STAGE_MISS_BIT_KHR, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR, VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
      VK_SHADER_STAGE_INTERSECTION_BIT_KHR, VK_SHADER_STAGE_CALLABLE_BIT_KHR,
      VK_SHADER_STAGE_CALLABLE_BIT_KHR,
   };
   VkPipelineShaderStageCreateInfo stages[NUM_STAGES];
   for (unsigned s = 0; s < NUM_STAGES; s++) {
      VkShaderModule sm = load_module(argv[1 + s]);
      if (!sm) { fprintf(stderr, "FAILED: module %s\n", argv[1 + s]); return 1; }
      stages[s] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = stage_bits[s], .module = sm, .pName = "main",
      };
   }
#define GENERAL(s) { .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR, \
      .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, .generalShader = (s), \
      .closestHitShader = VK_SHADER_UNUSED_KHR, .anyHitShader = VK_SHADER_UNUSED_KHR, \
      .intersectionShader = VK_SHADER_UNUSED_KHR }
   VkRayTracingShaderGroupCreateInfoKHR groups[NUM_GROUPS] = {
      [G_RGEN]  = GENERAL(S_RGEN),
      [G_MISS0] = GENERAL(S_MISS0),
      [G_MISS1] = GENERAL(S_MISS1),
      [G_TRI_OPAQUE] = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR, .closestHitShader = S_CHIT0,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR },
      [G_TRI_ALPHA] = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR, .closestHitShader = S_CHIT1,
         .anyHitShader = S_AHIT1, .intersectionShader = VK_SHADER_UNUSED_KHR },
      [G_PROC] = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR, .closestHitShader = S_CHIT2,
         .anyHitShader = S_AHIT2, .intersectionShader = S_RINT },
      [G_CALL0] = GENERAL(S_CALL0),
      [G_CALL1] = GENERAL(S_CALL1),
   };
#undef GENERAL

   VkDescriptorSetLayoutBinding dslb[2] = {
      { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkRayTracingPipelineCreateInfoKHR rtpci = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount = NUM_STAGES, .pStages = stages,
      .groupCount = NUM_GROUPS, .pGroups = groups,
      .maxPipelineRayRecursionDepth = 3,
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(p_CreateRtPipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtpci, NULL, &pipe));

   /* --- shader binding table ----------------------------------------- */
   const uint32_t hsize = rtprops.shaderGroupHandleSize;
   uint8_t *handles = malloc((size_t)hsize * NUM_GROUPS);
   CHECK(p_GetGroupHandles(dev, pipe, 0, NUM_GROUPS, (size_t)hsize * NUM_GROUPS, handles));
   const VkDeviceSize stride =
      align_up(hsize + sizeof(uint32_t), rtprops.shaderGroupHandleAlignment);
   const VkDeviceSize base_align = rtprops.shaderGroupBaseAlignment;
   const VkDeviceSize rgen_off = 0;
   const VkDeviceSize miss_off = align_up(rgen_off + stride, base_align);
   const VkDeviceSize hit_off  = align_up(miss_off + 2 * stride, base_align);
   const VkDeviceSize call_off = align_up(hit_off + NUM_HIT_RECORDS * stride, base_align);
   const VkDeviceSize sbt_size = call_off + 2 * stride;
   uint8_t *sbt = calloc(1, sbt_size);
#define RECORD(off, group, data) do {                                       \
      memcpy(sbt + (off), handles + (size_t)(group) * hsize, hsize);       \
      uint32_t _d = (data); memcpy(sbt + (off) + hsize, &_d, 4);           \
   } while (0)
   RECORD(rgen_off, G_RGEN, rgen_data);
   RECORD(miss_off, G_MISS0, miss_data[0]);
   RECORD(miss_off + stride, G_MISS1, miss_data[1]);
   for (unsigned r = 0; r < NUM_HIT_RECORDS; r++)
      RECORD(hit_off + r * stride, hit_records[r].group, hit_records[r].data);
   RECORD(call_off, G_CALL0, call_data[0]);
   RECORD(call_off + stride, G_CALL1, call_data[1]);
#undef RECORD
   VkBuffer sbtbuf; VkDeviceMemory sbtmem;
   if (!make_buffer(sbt_size, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, sbt,
                    &sbtbuf, &sbtmem)) {
      fprintf(stderr, "FAILED: SBT buffer\n"); return 1;
   }
   const VkDeviceAddress sbt_addr = buffer_addr(sbtbuf);
   VkStridedDeviceAddressRegionKHR rgen_region = { sbt_addr + rgen_off, stride, stride };
   VkStridedDeviceAddressRegionKHR miss_region = { sbt_addr + miss_off, stride, 2 * stride };
   VkStridedDeviceAddressRegionKHR hit_region  = { sbt_addr + hit_off, stride,
                                                   NUM_HIT_RECORDS * stride };
   VkStridedDeviceAddressRegionKHR call_region = { sbt_addr + call_off, stride, 2 * stride };

   /* --- output + descriptors ----------------------------------------- */
   const VkDeviceSize obytes = (VkDeviceSize)LAUNCH_W * LAUNCH_H * P_WORDS * 4;
   VkBuffer obuf; VkDeviceMemory omem;
   if (!make_buffer(obytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, &obuf, &omem)) {
      fprintf(stderr, "FAILED: output buffer\n"); return 1;
   }
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
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
   VkDescriptorBufferInfo odbi = { obuf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &was,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &odbi },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pl, 0, 1,
                           &ds, 0, NULL);
   p_CmdTraceRays(cmd, &rgen_region, &miss_region, &hit_region, &call_region,
                  LAUNCH_W, LAUNCH_H, 1);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify ------------------------------------------------------- */
   static const char *row_names[LAUNCH_H] = {
      "alpha", "terminate", "proc_opaque", "proc", "recurse", "miss0",
      "miss1", "sbt_offset", "tie",
   };
   uint32_t *rec;
   CHECK(vkMapMemory(dev, omem, 0, obytes, 0, (void **)&rec));
   unsigned bad_cells = 0, bad_rows = 0, checked = 0;
   for (uint32_t y = 0; y < LAUNCH_H; y++) {
      unsigned bad_here = 0;
      for (uint32_t x = 0; x < LAUNCH_W; x++) {
         const uint32_t *got = rec + (size_t)(y * LAUNCH_W + x) * P_WORDS;
         struct expect e;
         tie_non_opaque_first = false;
         build_expect(x, y, &e);
         if (y == ROW_TIE && !record_matches(&e, got)) {
            tie_non_opaque_first = true;
            build_expect(x, y, &e);
         }
         bool ok = true;
         for (unsigned i = 0; i < P_WORDS; i++) {
            checked++;
            if (!word_matches(&e, i, got[i])) {
               if (bad_here < 6) {
                  float a, b;
                  memcpy(&a, &got[i], 4); memcpy(&b, &e.w[i], 4);
                  printf("  %s x=%u word %u: got 0x%08x (%g) want 0x%08x (%g)\n",
                         row_names[y], x, i, got[i], a, e.w[i], b);
               }
               bad_here++;
               ok = false;
            }
         }
         if (!ok) bad_cells++;
      }
      if (bad_here) bad_rows++;
   }
   vkUnmapMemory(dev, omem);

   p_DestroyAccelStruct(dev, tlas, NULL);
   p_DestroyAccelStruct(dev, blas_t, NULL);
   p_DestroyAccelStruct(dev, blas_p, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad_cells) {
      printf("FAILED (%u of %u launch cells wrong in %u of %u rows)\n",
             bad_cells, LAUNCH_W * LAUNCH_H, bad_rows, LAUNCH_H);
      return 1;
   }
   printf("PASSED (ray-tracing pipeline: %u rows x %u cells, %u words checked)\n",
          LAUNCH_H, LAUNCH_W, checked);
   return 0;
}
