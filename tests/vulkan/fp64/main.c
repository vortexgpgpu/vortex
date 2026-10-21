/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * fp64 (shaderFloat64) on the Vortex device: through the toolchain's soft-float
 * library where the FPU has no D extension (rv32), natively where it has (rv64).
 *
 * Two parts, both checked bit-exactly against the host's IEEE double
 * arithmetic:
 *   - a compute dispatch (fp64.comp) running arithmetic, square roots,
 *     min/max, rounding, frexp/ldexp, conversions to and from every integer
 *     width, f32 and f16, comparisons, double packing, UBO and push-constant
 *     doubles and subgroup operations over per-lane inputs that put special
 *     values (NaN, infinities, signed zeros, denormals, extremes) in
 *     neighbouring lanes;
 *   - a draw (fp64.vert/.frag) that reads doubles from a storage buffer in the
 *     vertex shader, computes and stores in double there, and passes flat
 *     double, dvec3 and dvec4 varyings, which the fragment shader computes on
 *     and stores.
 *
 * Where Vulkan lets an implementation fuse a multiply and an add, both the
 * fused and the separately rounded result are accepted; where it leaves the
 * sign of a zero result open (min/max of +0 and -0, sign of a zero), either
 * zero is. Every other result must match to the bit, and a NaN must be a NaN.
 *
 * Every result slot starts as a sentinel, so a shader the device could not run
 * (a STRICT failure instead of a silent llvmpipe fallback) cannot pass.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOPS       41u
#define INVOCS     128u
#define LOCAL_MAX  64u
#define SENTINEL   0x7ff4dead5eed0001ull
#define FB_W       16u
#define FB_H       16u
#define NVERT      6u
#define NFS        9u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* ---- host reference helpers ------------------------------------------ */

static uint64_t
dbits(double d)
{
   uint64_t u;
   memcpy(&u, &d, 8);
   return u;
}

static double
bitsd(uint64_t u)
{
   double d;
   memcpy(&d, &u, 8);
   return d;
}

/* binary64 -> binary16, round to nearest even, as one rounding. */
static uint16_t
d2h(double d)
{
   const uint64_t u = dbits(d);
   const uint16_t s = (uint16_t)((u >> 48) & 0x8000u);
   const int e = (int)((u >> 52) & 0x7ff);
   const uint64_t m = u & ((1ull << 52) - 1);
   if (e == 0x7ff) {
      return s | 0x7c00u | (m ? 0x200u : 0u);
   }
   if (e == 0) {
      return s;      /* a binary64 denormal is far below half a binary16 ulp */
   }
   const int E = e - 1023;
   if (E > 15) {
      return s | 0x7c00u;
   }
   const uint64_t sig = m | (1ull << 52);
   const bool normal = E >= -14;
   const int shift = normal ? 42 : 28 - E;
   if (shift >= 54) {
      return s;
   }
   uint64_t q = sig >> shift;
   const uint64_t rem = sig & ((1ull << shift) - 1);
   const uint64_t half = 1ull << (shift - 1);
   if (rem > half || (rem == half && (q & 1u))) {
      q++;
   }
   if (!normal) {
      return s | (uint16_t)q;
   }
   const uint32_t bits = ((uint32_t)(E + 15) << 10) + (uint32_t)(q - 1024u);
   return s | (uint16_t)(bits >= 0x7c00u ? 0x7c00u : bits);
}

/* A deterministic 64-bit stream for the inputs past the special values. */
static uint64_t
mix(uint64_t x)
{
   x += 0x9e3779b97f4a7c15ull;
   x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
   x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}

/* a: anything, including every NaN and infinity. */
static double
in_a(uint32_t i)
{
   static const uint64_t special[] = {
      0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull,
      0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
      0x7ff8000000000000ull, 0xfff8000000000001ull, 0x0000000000000001ull,
      0x8000000000000001ull, 0x7fefffffffffffffull, 0xffefffffffffffffull,
      0x000123456789abcdull, 0x0010000000000000ull, 0x3fe0000000000000ull,
      0x3ff8000000000000ull, 0x4004000000000000ull, 0xc004000000000000ull,
      0x400c000000000000ull, 0x4341c37937e08000ull, 0x4330000000000001ull,
      0x3fb999999999999aull, 0xbfd3333333333333ull, 0x40fe240c9fbe76c9ull,
      0x40effc0000000000ull, 0x40effe0000000000ull, 0x3f10000000000000ull,
      0x3e70000000000000ull, 0x3e6ffffffffffffeull, 0x7e37e43c8800759cull,
      0xc01d000000000000ull, 0x400921fb54442d18ull, 0x40effbffffffffffull,
      0x3e78000000000000ull, 0x3f0ff80000000000ull, 0x4097400000000000ull,
   };
   const uint32_t n = sizeof(special) / sizeof(special[0]);
   return bitsd(i < n ? special[i] : mix(i));
}

/* b: no NaN; zeros, infinities and denormals included. */
static double
in_b(uint32_t i)
{
   static const uint64_t special[] = {
      0x3ff0000000000000ull, 0x8000000000000000ull, 0x0000000000000000ull,
      0x4008000000000000ull, 0xfff0000000000000ull, 0x7ff0000000000000ull,
      0x0000000000000003ull, 0x01a56e1fc2f8f359ull, 0x401e000000000000ull,
      0xc000000000000000ull, 0x7fe1ccf385ebc8a0ull, 0x3fd5555555555555ull,
   };
   const uint32_t n = sizeof(special) / sizeof(special[0]);
   if (i < n) {
      return bitsd(special[i]);
   }
   uint64_t u = mix(i * 3u + 1u);
   if (((u >> 52) & 0x7ff) == 0x7ff) {
      u &= ~(1ull << 62);
   }
   return bitsd(u);
}

/* c: zero or a normal value with |c| < 2^20, fractional parts included. */
static double
in_c(uint32_t i)
{
   static const double special[] = {
      0.0, -0.0, 0.5, -0.5, 1.5, -2.5, 3.75, 1e-7, -1e-7, 1048575.75,
      -1048575.25, 0.1, 123456.789, -98765.4321, 65504.0, 65520.0, -0.00006103515625,
   };
   const uint32_t n = sizeof(special) / sizeof(special[0]);
   if (i < n) {
      return special[i];
   }
   const uint64_t r = mix(i * 7u + 3u);
   return (double)((int64_t)(r % 2097151u) - 1048575) / 2.0 +
          (double)((r >> 32) % 1000000u) * 1e-7;
}

static int32_t
in_i(uint32_t i)
{
   static const int32_t special[] = {
      0, 1, -1, 2147483647, (int32_t)0x80000000, 42, -42, 16777217, -16777217,
      65535, 1000000007,
   };
   const uint32_t n = sizeof(special) / sizeof(special[0]);
   return i < n ? special[i] : (int32_t)(uint32_t)mix(i * 11u + 5u);
}

#define UBO_UD   0.3
#define UBO_UVX  -1.0e10
#define UBO_UVY  7.0 / 3.0
#define PC_PD    1.0 / 7.0

/* Acceptable results of one op: one or two bit patterns, a NaN class, or a
 * zero of either sign. */
struct want {
   uint64_t v[2];
   unsigned n;
   bool any_zero;
};

static struct want
w1(double d)
{
   struct want w = { { dbits(d), 0 }, 1, false };
   return w;
}

static struct want
w2(double d, double e)
{
   struct want w = { { dbits(d), dbits(e) }, 2, false };
   return w;
}

static struct want
wz(double d)
{
   struct want w = w1(d);
   w.any_zero = d == 0.0;
   return w;
}

static struct want
wbits(uint64_t u)
{
   struct want w = { { u, 0 }, 1, false };
   return w;
}

/* Whether `u` is a NaN of a `width`-bit float (0: an integer, never a NaN). */
static bool
is_nan_bits(uint64_t u, unsigned width)
{
   switch (width) {
   case 64: return ((u >> 52) & 0x7ff) == 0x7ff && (u & ((1ull << 52) - 1));
   case 32: return u <= 0xffffffffu && ((u >> 23) & 0xff) == 0xff && (u & 0x7fffffu);
   case 16: return u <= 0xffffu && ((u >> 10) & 0x1f) == 0x1f && (u & 0x3ffu);
   default: return false;
   }
}

static bool
matches(const struct want *w, uint64_t got, unsigned width)
{
   for (unsigned k = 0; k < w->n; k++) {
      if (got == w->v[k]) {
         return true;
      }
      if (is_nan_bits(got, width) && is_nan_bits(w->v[k], width)) {
         return true;
      }
   }
   return w->any_zero && (got & ~(1ull << 63)) == 0;
}

static double
host_min(double x, double y)
{
   return y < x ? y : x;
}

static double
host_max(double x, double y)
{
   return x < y ? y : x;
}

static const char *const op_name[NOPS] = {
   "add", "sub", "mul", "div", "fma", "sqrt", "rsq", "min", "max", "floor",
   "ceil", "trunc", "roundEven", "fract", "abs", "neg", "sign", "frexp.sig",
   "frexp.exp", "ldexp", "f2f32", "d2i", "d2u", "d2i64", "i2d", "u2d", "i64_2d",
   "u64_2d", "d2f16", "f32_2f16", "cmp", "pack", "mod", "clamp", "push+ubo",
   "ubo vec", "subgroupAdd", "subgroupMax", "broadcastFirst", "shuffleXor", "ids",
};

/* Everything but the subgroup ops, which need the lane layout. */
static struct want
expect(uint32_t i, uint32_t k)
{
   const double a = in_a(i), b = in_b(i), c = in_c(i);
   const int32_t iv = in_i(i);
   int e;
   switch (k) {
   case 0:  return w1(a + b);
   case 1:  return w1(a - b);
   case 2:  return w1(a * b);
   case 3:  return w1(a / b);
   case 4:  { volatile double p = a * c; return w2(p + b, fma(a, c, b)); }
   case 5:  return w1(sqrt(fabs(a)));
   case 6:  return w1(1.0 / sqrt(fabs(c)));
   case 7:  return wz(host_min(b, c));
   case 8:  return wz(host_max(b, c));
   case 9:  return w1(floor(a));
   case 10: return w1(ceil(a));
   case 11: return w1(trunc(a));
   case 12: return w1(rint(a));
   case 13: return w1(c - floor(c));
   case 14: return w1(fabs(a));
   case 15: return w1(-a);
   case 16: return wz(b > 0.0 ? 1.0 : b < 0.0 ? -1.0 : 0.0);
   case 17: return w1(frexp(c, &e));
   case 18: frexp(c, &e); return wbits((uint64_t)(int64_t)e);
   case 19: return w1(ldexp(c, (iv & 63) - 32));
   case 20: { float f = (float)a; uint32_t u; memcpy(&u, &f, 4); return wbits(u); }
   case 21: return wbits((uint64_t)(int64_t)(int32_t)c);
   case 22: return wbits((uint64_t)(uint32_t)fabs(c));
   case 23: return wbits((uint64_t)(int64_t)(c * 1048576.0));
   case 24: return w1((double)iv);
   case 25: return w1((double)(uint32_t)iv);
   case 26: return w1((double)(int64_t)((uint64_t)(int64_t)iv * 1099511627775ull));
   case 27: return w1((double)((uint64_t)(int64_t)iv * 0x8000000000000001ull));
   case 28: return wbits(d2h(a));
   case 29: return wbits(d2h((double)(float)c));
   case 30: {
      const uint32_t m = (a < b ? 1u : 0u) | (a <= b ? 2u : 0u) | (a > b ? 4u : 0u) |
                         (a >= b ? 8u : 0u) | (a == b ? 16u : 0u) | (a != b ? 32u : 0u) |
                         (isnan(a) ? 64u : 0u) | (isinf(a) ? 128u : 0u);
      return wbits(m);
   }
   case 31: {
      const uint64_t u = dbits(a);
      return wbits((u << 32) | (u >> 32));
   }
   case 32: {
      volatile double q = floor(c / b);
      volatile double p = b * q;
      return w2(c - p, fma(-b, q, c));
   }
   case 33: return w1(host_min(host_max(c, -1.0), 1.0));
   case 34: { volatile double p = (PC_PD) * c; return w2(p + UBO_UD, fma(PC_PD, c, UBO_UD)); }
   case 35: return w1(UBO_UVX / c - (UBO_UVY));
   default: return wbits(0);
   }
}

/* The float width an op's result is stored at; 0 for an integer result. */
static unsigned
op_width(uint32_t k)
{
   switch (k) {
   case 18: case 21: case 22: case 23: case 30: case 31: case 40:
      return 0;
   case 20:
      return 32;
   case 28: case 29:
      return 16;
   default:
      return 64;
   }
}

/* ---- Vulkan plumbing ---------------------------------------------------- */

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

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return VK_NULL_HANDLE;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
      sm = VK_NULL_HANDLE;
   }
   free(spv);
   return sm;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

struct hbuf {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *ptr;
   VkDeviceSize size;
};

static int
make_hbuf(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
          VkDeviceSize size, VkBufferUsageFlags usage, struct hbuf *b)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &b->buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, b->buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, &b->mem));
   CHECK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
   CHECK(vkMapMemory(dev, b->mem, 0, size, 0, &b->ptr));
   b->size = size;
   return 0;
}

static void
free_hbuf(VkDevice dev, struct hbuf *b)
{
   vkUnmapMemory(dev, b->mem);
   vkDestroyBuffer(dev, b->buf, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static int
submit_wait(VkDevice dev, VkQueue queue, VkCommandBuffer cb)
{
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   (void)dev;
   return 0;
}

static unsigned fails;

static void
report(const char *part, uint32_t i, const char *op, uint64_t got, const struct want *w)
{
   if (fails < 16)
      fprintf(stderr, "  %s [%u] %s: got 0x%016llx (%a), want 0x%016llx (%a)%s\n",
              part, i, op, (unsigned long long)got, bitsd(got),
              (unsigned long long)w->v[0], bitsd(w->v[0]),
              w->n > 1 ? " or its unfused/fused twin" : "");
   fails++;
}

/* ---- compute part -------------------------------------------------------- */

static int
run_compute(VkDevice dev, VkQueue queue, VkCommandPool cp,
            const VkPhysicalDeviceMemoryProperties *mp,
            const VkPhysicalDeviceProperties *props, uint32_t sg_size,
            const char *spv_path)
{
   struct hbuf ba, bb, bc, bi, bo, bu;
   const VkDeviceSize dsz = INVOCS * sizeof(double);
   const VkDeviceSize osz = (VkDeviceSize)INVOCS * NOPS * sizeof(uint64_t);
   const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
   if (make_hbuf(dev, mp, dsz, ssbo, &ba) || make_hbuf(dev, mp, dsz, ssbo, &bb) ||
       make_hbuf(dev, mp, dsz, ssbo, &bc) ||
       make_hbuf(dev, mp, INVOCS * sizeof(int32_t), ssbo, &bi) ||
       make_hbuf(dev, mp, osz, ssbo, &bo) ||
       make_hbuf(dev, mp, 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &bu))
      return 1;
   for (uint32_t i = 0; i < INVOCS; i++) {
      ((double *)ba.ptr)[i] = in_a(i);
      ((double *)bb.ptr)[i] = in_b(i);
      ((double *)bc.ptr)[i] = in_c(i);
      ((int32_t *)bi.ptr)[i] = in_i(i);
   }
   for (uint32_t w = 0; w < INVOCS * NOPS; w++)
      ((uint64_t *)bo.ptr)[w] = SENTINEL;
   const double ubo[4] = { UBO_UD, 0.0, UBO_UVX, UBO_UVY };
   memcpy(bu.ptr, ubo, sizeof ubo);
   const double pd = PC_PD;

   VkDescriptorSetLayoutBinding dslb[6];
   for (uint32_t k = 0; k < 6; k++) {
      dslb[k] = (VkDescriptorSetLayoutBinding){
         .binding = k,
         .descriptorType = k == 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                  : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 6, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(double) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   uint32_t local = LOCAL_MAX;
   while (local > props->limits.maxComputeWorkGroupSize[0] ||
          local > props->limits.maxComputeWorkGroupInvocations)
      local >>= 1;
   if (local % sg_size) {
      fprintf(stderr, "FAILED: workgroup %u is not a whole number of %u-wide subgroups\n",
              local, sg_size);
      return 1;
   }
   printf("compute: local_size_x=%u, subgroup size %u\n", local, sg_size);

   VkShaderModule sm = load_module(dev, spv_path);
   if (!sm) return 1;
   VkSpecializationMapEntry sme = { .constantID = 0, .offset = 0, .size = 4 };
   VkSpecializationInfo spec = {
      .mapEntryCount = 1, .pMapEntries = &sme, .dataSize = 4, .pData = &local,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
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
   const struct hbuf *bufs[6] = { &ba, &bb, &bc, &bi, &bo, &bu };
   VkDescriptorBufferInfo dbi[6];
   VkWriteDescriptorSet wds[6];
   for (uint32_t k = 0; k < 6; k++) {
      dbi[k] = (VkDescriptorBufferInfo){ bufs[k]->buf, 0, VK_WHOLE_SIZE };
      wds[k] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
         .dstBinding = k, .descriptorCount = 1,
         .descriptorType = dslb[k].descriptorType, .pBufferInfo = &dbi[k],
      };
   }
   vkUpdateDescriptorSets(dev, 6, wds, 0, NULL);

   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pd, &pd);
   vkCmdDispatch(cb, INVOCS / local, 1, 1);
   CHECK(vkEndCommandBuffer(cb));
   if (submit_wait(dev, queue, cb))
      return 1;

   const uint64_t *r = bo.ptr;
   unsigned sentinels = 0;
   for (uint32_t w = 0; w < INVOCS * NOPS; w++)
      sentinels += r[w] == SENTINEL;
   if (sentinels)
      fprintf(stderr, "  compute: %u result slots still hold the sentinel\n", sentinels);
   fails += sentinels;

   for (uint32_t i = 0; i < INVOCS; i++) {
      for (uint32_t k = 0; k < 36; k++) {
         const struct want w = expect(i, k);
         if (!matches(&w, r[i * NOPS + k], op_width(k)))
            report("compute", i, op_name[k], r[i * NOPS + k], &w);
      }
   }

   /* Subgroups: group the invocations by the (workgroup, subgroup) they report,
    * and check that report against the dispatch geometry first. */
   for (uint32_t i = 0; i < INVOCS; i++) {
      const uint64_t id = r[i * NOPS + 40];
      const uint32_t wg = (uint32_t)(id >> 16), sg = (uint32_t)(id >> 8) & 0xff,
                     lane = (uint32_t)id & 0xff;
      if (wg != i / local || lane >= sg_size || sg >= local / sg_size) {
         const struct want w = wbits(((uint64_t)(i / local) << 16));
         report("compute", i, op_name[40], id, &w);
         continue;
      }
      int32_t first = -1, partner = -1;
      double sum = 0.0, mx = -INFINITY;
      for (uint32_t j = 0; j < INVOCS; j++) {
         const uint64_t jd = r[j * NOPS + 40];
         if ((jd >> 8) != (id >> 8))
            continue;
         sum += (double)in_i(j);
         mx = host_max(mx, in_c(j));
         if ((jd & 0xff) == 0)
            first = (int32_t)j;
         if ((jd & 0xff) == (lane ^ 1u))
            partner = (int32_t)j;
      }
      struct want w = w1(sum);
      if (!matches(&w, r[i * NOPS + 36], 64))
         report("compute", i, op_name[36], r[i * NOPS + 36], &w);
      w = wz(mx);
      if (!matches(&w, r[i * NOPS + 37], 64))
         report("compute", i, op_name[37], r[i * NOPS + 37], &w);
      w = first >= 0 ? w1(in_a((uint32_t)first)) : wbits(SENTINEL);
      if (!matches(&w, r[i * NOPS + 38], 64))
         report("compute", i, op_name[38], r[i * NOPS + 38], &w);
      w = partner >= 0 ? w1(in_a((uint32_t)partner)) : wbits(SENTINEL);
      if (!matches(&w, r[i * NOPS + 39], 64))
         report("compute", i, op_name[39], r[i * NOPS + 39], &w);
   }

   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   for (uint32_t k = 0; k < 6; k++)
      free_hbuf(dev, (struct hbuf *)bufs[k]);
   return 0;
}

/* ---- draw part ------------------------------------------------------------ */

static const float vpos[NVERT][2] = {
   { -0.9f, -0.9f }, { -0.1f, -0.9f }, { -0.9f, 0.9f },
   {  0.1f, -0.9f }, {  0.9f, -0.9f }, {  0.1f, 0.9f },
};
static const double vval[NVERT] = {
   1.0 / 3.0, -2.5e-310, 1e300, -7.25, 0.1, 12345.6789,
};

static int
run_draw(VkDevice dev, VkQueue queue, VkCommandPool cp,
         const VkPhysicalDeviceMemoryProperties *mp,
         const char *vs_path, const char *fs_path)
{
   struct hbuf vb, ib, ob, rb;
   if (make_hbuf(dev, mp, sizeof vpos, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb) ||
       make_hbuf(dev, mp, sizeof vval, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ib) ||
       make_hbuf(dev, mp, (NVERT + 2 * NFS) * sizeof(uint64_t),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ob) ||
       make_hbuf(dev, mp, FB_W * FB_H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb))
      return 1;
   memcpy(vb.ptr, vpos, sizeof vpos);
   memcpy(ib.ptr, vval, sizeof vval);
   for (uint32_t w = 0; w < NVERT + 2 * NFS; w++)
      ((uint64_t *)ob.ptr)[w] = SENTINEL;

   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { FB_W, FB_H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = find_mem(mp, imr.memoryTypeBits, 0),
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference attref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &attref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = FB_W, .height = FB_H, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT },
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

   VkShaderModule vs = load_module(dev, vs_path), fs = load_module(dev, fs_path);
   if (!vs || !fs) return 1;
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkVertexInputBindingDescription vib = { 0, sizeof vpos[0], VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription via = {
      .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0,
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vib,
      .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &via,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport vp = { 0, 0, (float)FB_W, (float)FB_H, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { FB_W, FB_H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi[2] = { { ob.buf, 0, VK_WHOLE_SIZE }, { ib.buf, 0, VK_WHOLE_SIZE } };
   VkWriteDescriptorSet wds[2];
   for (uint32_t k = 0; k < 2; k++) {
      wds[k] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = k,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[k],
      };
   }
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

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
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb, .renderArea = { { 0, 0 }, { FB_W, FB_H } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
   const VkDeviceSize vbo = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buf, &vbo);
   vkCmdDraw(cmd, NVERT, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { FB_W, FB_H, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));
   if (submit_wait(dev, queue, cmd))
      return 1;

   /* Both triangles must have been rasterized, or the stored values mean nothing. */
   const uint8_t *px = rb.ptr;
   unsigned covered[2] = { 0, 0 };
   for (uint32_t p = 0; p < FB_W * FB_H; p++) {
      if (px[p * 4 + 1] == 64)
         covered[0]++;
      else if (px[p * 4 + 1] == 191)
         covered[1]++;
   }
   if (covered[0] < 8 || covered[1] < 8) {
      fprintf(stderr, "  draw: triangles cover %u and %u pixels\n", covered[0], covered[1]);
      fails++;
   }

   const uint64_t *o = ob.ptr;
   for (uint32_t v = 0; v < NVERT; v++) {
      const double x = vval[v];
      const struct want w = w1(x / 7.0 + sqrt(fabs(x)));
      if (!matches(&w, o[v], 64))
         report("vertex", v, "val/7+sqrt", o[v], &w);
   }
   static const char *const vec_name[7] = {
      "dvec3.x", "dvec3.y", "dvec3.z", "dvec4.x", "dvec4.y", "dvec4.z", "dvec4.w",
   };
   for (uint32_t tri = 0; tri < 2; tri++) {
      const double x = vval[tri * 3];
      volatile double x3 = x * 3.0;
      const double d = x3 + 1.0 / 3.0;
      volatile double sq = d * d;
      const uint64_t *f = o + NVERT + tri * NFS;
      struct want w = w2(sq - 2.0, fma(d, d, -2.0));
      if (!matches(&w, f[0], 64))
         report("fragment", tri, "d*d-2", f[0], &w);
      w = w1(d / 9.0);
      if (!matches(&w, f[1], 64))
         report("fragment", tri, "d/9", f[1], &w);
      const double vec[7] = { x * 2.0, x - 5.0, x / 3.0, x + 1.0, -x, x * 0.5, x * x };
      for (uint32_t c = 0; c < 7; c++) {
         w = w1(vec[c]);
         if (!matches(&w, f[2 + c], 64))
            report("fragment", tri, vec_name[c], f[2 + c], &w);
      }
   }

   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   free_hbuf(dev, &vb);
   free_hbuf(dev, &ib);
   free_hbuf(dev, &ob);
   free_hbuf(dev, &rb);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *cs_path = (argc > 1) ? argv[1] : "fp64.comp.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "fp64.frag.spv";
   const char *vs_path = (argc > 3) ? argv[3] : "fp64.vert.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-fp64",
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

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics+compute queue\n"); return 1; }

   /* Everything used here is advertised by the device, so a missing feature is
    * a failure: withdrawing one must not turn this test into a no-op. */
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   VkPhysicalDeviceSubgroupProperties sgp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 p2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp,
   };
   vkGetPhysicalDeviceProperties2(pd, &p2);
   const VkSubgroupFeatureFlags sg_need = VK_SUBGROUP_FEATURE_BASIC_BIT |
      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
   const struct { const char *name; bool ok; } need[] = {
      { "shaderFloat64", f2.features.shaderFloat64 },
      { "shaderInt64", f2.features.shaderInt64 },
      { "shaderFloat16", v12.shaderFloat16 },
      { "vertexPipelineStoresAndAtomics", f2.features.vertexPipelineStoresAndAtomics },
      { "fragmentStoresAndAtomics", f2.features.fragmentStoresAndAtomics },
      { "compute subgroup arithmetic/ballot/shuffle",
        (sgp.supportedOperations & sg_need) == sg_need &&
        (sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) },
   };
   bool missing = false;
   for (unsigned k = 0; k < sizeof(need) / sizeof(need[0]); k++) {
      if (!need[k].ok) {
         printf("  %s is not offered\n", need[k].name);
         missing = true;
      }
   }
   if (missing) {
      printf("FAILED: a feature this test needs is not offered\n");
      return 1;
   }

   VkPhysicalDeviceVulkan12Features v12_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderFloat16 = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 f2_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12_en,
      .features = {
         .shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE,
         .vertexPipelineStoresAndAtomics = VK_TRUE, .fragmentStoresAndAtomics = VK_TRUE,
      },
   };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2_en,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));

   if (run_compute(dev, queue, cp, &mp, &props, sgp.subgroupSize, cs_path) ||
       run_draw(dev, queue, cp, &mp, vs_path, fs_path))
      return 1;

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails) {
      printf("FAILED (%u mismatches)\n", fails);
      return 1;
   }
   printf("PASSED (%u invocations x %u fp64 ops; vertex and fragment fp64 with "
          "flat double, dvec3 and dvec4 varyings)\n", INVOCS, NOPS);
   return 0;
}
