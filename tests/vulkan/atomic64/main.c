/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * 64-bit integer atomics (VK_KHR_shader_atomic_int64) on the Vortex device:
 * add, and, or, xor, signed and unsigned min/max, exchange and compare-and-swap
 * on a storage buffer, workgroup shared memory and a buffer device address, all
 * contended within and across warps.
 *
 * The checks are what any serialisation of the atomics must satisfy, exactly:
 *   - add: every target gets n equal odd addends, so the returned values are
 *     init + j * addend (mod 2^64) for j = 0..n-1, each once, and the final
 *     value is init + n * addend;
 *   - and/or/xor: the final value is fixed; every value an and/or returned lies
 *     between the initial and the final value bitwise;
 *   - min/max: the final value is the extreme of the operands and the initial
 *     value, and every returned value is the initial value or an operand on the
 *     right side of the final value;
 *   - exchange: the returned values plus the final one are a permutation of the
 *     initial value and all operands;
 *   - compare-and-swap: a retry-loop counter ends exactly n steps on, and in a
 *     race for an empty slot exactly one lane sees it empty while every other
 *     lane sees the winner's claim.
 *
 * The operands make both halves of a word matter (carries across bit 31,
 * operands differing only in the high or only in the low word, sign bits in
 * both orders), so a 32-bit atomic or a torn word fails a check.
 *
 * Under MESA_VORTEX_STRICT=1 a shader the device cannot run fails the dispatch
 * instead of running on llvmpipe, and the sentinel in every result slot shows it.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOTAL     256u
#define NADD      2u
#define NSLOT     16u
#define NGRP_MAX  64u
#define LOCAL_MAX 64u
#define SENTINEL  0xbadbadbadbadbad1ull

/* Mirrors the shader's Acc block (std430: all 8-byte words, no padding). */
struct acc {
   uint64_t add[NADD];
   uint64_t band;
   uint64_t bor;
   uint64_t bxor;
   int64_t  smin;
   int64_t  smax;
   uint64_t umin;
   uint64_t umax;
   uint64_t xchg;
   uint64_t cas;
   uint64_t slot[NSLOT];
   uint64_t ret_add[TOTAL];
   uint64_t ret_or[TOTAL];
   uint64_t ret_and[TOTAL];
   int64_t  ret_smin[TOTAL];
   int64_t  ret_smax[TOTAL];
   uint64_t ret_umin[TOTAL];
   uint64_t ret_umax[TOTAL];
   uint64_t ret_xchg[TOTAL];
   uint64_t ret_slot[TOTAL];
   uint64_t ret_sadd[TOTAL];
   uint64_t ret_sxchg[TOTAL];
   uint64_t ret_badd[TOTAL];
   int64_t  ret_bsmax[TOTAL];
   uint64_t grp[NGRP_MAX * 8u];
   uint64_t op[TOTAL];
};

/* Mirrors the shader's Remote block, reached through a device address. */
struct remote {
   uint64_t add;
   int64_t  smax;
   uint64_t cas;
   uint64_t umin;
};

static const uint64_t addend[NADD]   = { 0x0000000100000001ull, 0x00000000fffffff1ull };
static const uint64_t add_init[NADD] = { 0x00000000fffffff0ull, 0x7ffffffffffffff0ull };
#define XOR_INIT       0x0123456789abcdefull
#define XCHG_INIT      0xdeadbeef00000000ull
#define CAS_INIT       0x00000000ffffff00ull
#define CAS_STEP       0x0000000100000001ull
/* the shader's shared-memory seeds and steps */
#define SH_ADD_INIT    0xfffffff0fffffff0ull
#define SH_ADD_STEP    0x0000000100000003ull
#define SH_XCHG_INIT   0x5a5a5a5aa5a5a5a5ull
#define SH_CAS_INIT    0x00000000ffffffe0ull
/* the device-address targets */
#define BDA_ADD_INIT   0x00000000fffffff0ull
#define BDA_ADD_STEP   0x00000000ffffffffull

/* Operand of invocation i: pairs of lanes share the high word, so their order
 * is decided by the low word, which names the lane; every other pair has the
 * sign bit set, so signed and unsigned orders disagree. */
static uint64_t
operand(uint32_t i)
{
   const uint32_t g  = i >> 1;
   const uint32_t hi = (g * 0x9e3779b1u) ^ ((g & 1u) ? 0x80000000u : 0u);
   return ((uint64_t)hi << 32) | (i + 1u);
}

static uint64_t
bit_operand(uint32_t i)
{
   return (1ull << (i % 32u)) | (1ull << (32u + (i * 7u + 3u) % 32u));
}

static uint64_t
claim(uint32_t i)
{
   return (uint64_t)(i + 1u) * 0x0000000100000001ull;
}

/* The lane whose operand is v, or UINT32_MAX. */
static uint32_t
operand_lane(uint64_t v)
{
   const uint32_t i = (uint32_t)v - 1u;
   return (i < TOTAL && operand(i) == v) ? i : UINT32_MAX;
}

/* The inverse of an odd a modulo 2^64. */
static uint64_t
inverse(uint64_t a)
{
   uint64_t x = a;
   for (int k = 0; k < 6; k++)
      x *= 2u - a * x;
   return x;
}

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static unsigned fails;

static void
fail(const char *what, uint32_t i, uint64_t got, uint64_t want)
{
   if (fails < 16)
      fprintf(stderr, "  %s [%u]: got 0x%016llx, want 0x%016llx\n", what, i,
              (unsigned long long)got, (unsigned long long)want);
   fails++;
}

/* Every add to one target returns init + j * a for a distinct j in [0, n). */
static void
check_add_chain(const char *what, const uint64_t *ret, uint32_t stride,
                uint32_t first, uint32_t n, uint64_t init, uint64_t a,
                uint64_t final)
{
   const uint64_t inv = inverse(a);
   bool *seen = calloc(n, 1);
   for (uint32_t j = 0; j < n; j++) {
      const uint32_t idx = first + j * stride;
      const uint64_t s = (ret[idx] - init) * inv;
      if (s >= n || seen[s]) {
         fail(what, idx, ret[idx], init);
         continue;
      }
      seen[s] = true;
   }
   free(seen);
   if (final != init + (uint64_t)n * a)
      fail(what, UINT32_MAX, final, init + (uint64_t)n * a);
}

/* A min or max over the operands of lanes [first, first + n): the final value,
 * and (when ret is given) the values the atomics returned. */
static void
check_minmax(const char *what, bool is_max, bool is_signed, uint64_t init,
             const uint64_t *ret, uint32_t first, uint32_t n, uint64_t final)
{
#define BEYOND(x, y) (is_signed ? (is_max ? (int64_t)(x) > (int64_t)(y)  \
                                          : (int64_t)(x) < (int64_t)(y)) \
                                : (is_max ? (x) > (y) : (x) < (y)))
   uint64_t want = init;
   for (uint32_t j = 0; j < n; j++) {
      const uint64_t v = operand(first + j);
      if (BEYOND(v, want))
         want = v;
   }
   if (final != want)
      fail(what, UINT32_MAX, final, want);
   if (!ret)
      return;
   for (uint32_t j = 0; j < n; j++) {
      const uint64_t r = ret[first + j];
      const uint32_t k = operand_lane(r);
      const bool known = (r == init) || (k >= first && k < first + n);
      if (!known || BEYOND(r, final))
         fail(what, first + j, r, init);
   }
#undef BEYOND
}

/* {returned} + {final} == {init} + {claims of lanes [first, first + n)}. */
static void
check_exchange(const char *what, const uint64_t *ret, uint32_t first,
               uint32_t n, uint64_t init, uint64_t final)
{
   bool *seen = calloc(n + 1, 1);
   for (uint32_t j = 0; j <= n; j++) {
      const uint64_t r = j < n ? ret[first + j] : final;
      uint32_t s = n;
      if (r != init) {
         const uint32_t k = (uint32_t)r - 1u;
         s = (k >= first && k < first + n && claim(k) == r) ? k - first : UINT32_MAX;
      }
      if (s == UINT32_MAX || seen[s]) {
         fail(what, first + j, r, init);
         continue;
      }
      seen[s] = true;
   }
   free(seen);
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

static int
make_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize size, VkBufferUsageFlags usage, bool bda,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = bda ? &mafi : NULL,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
   return 0;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "atomic64.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-atomic64",
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
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   /* The suite's device has every 64-bit atomic (A, Zacas and, on a 32-bit
    * device, AMO64), so a missing feature is a failure, not a skip. */
   if (!v12.shaderBufferInt64Atomics || !v12.shaderSharedInt64Atomics ||
       !v12.bufferDeviceAddress || !f2.features.shaderInt64) {
      printf("FAILED: 64-bit atomics or buffer device address not offered "
             "(buffer=%u shared=%u bda=%u int64=%u)\n",
             v12.shaderBufferInt64Atomics, v12.shaderSharedInt64Atomics,
             v12.bufferDeviceAddress, f2.features.shaderInt64);
      return 1;
   }
   VkPhysicalDeviceVulkan12Features v12_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderBufferInt64Atomics = VK_TRUE, .shaderSharedInt64Atomics = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 f2_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12_en,
      .features = { .shaderInt64 = VK_TRUE },
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

   VkBuffer abuf, rbuf;
   VkDeviceMemory amem, rmem;
   if (make_buffer(dev, &mp, sizeof(struct acc), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   false, &abuf, &amem) ||
       make_buffer(dev, &mp, sizeof(struct remote), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true, &rbuf, &rmem))
      return 1;

   struct acc *a;
   CHECK(vkMapMemory(dev, amem, 0, sizeof(struct acc), 0, (void **)&a));
   for (uint32_t w = 0; w < sizeof(struct acc) / 8u; w++)
      ((uint64_t *)a)[w] = SENTINEL;
   for (uint32_t k = 0; k < NADD; k++)
      a->add[k] = add_init[k];
   a->band = ~0ull;
   a->bor  = 0;
   a->bxor = XOR_INIT;
   a->smin = INT64_MAX;
   a->smax = INT64_MIN;
   a->umin = UINT64_MAX;
   a->umax = 0;
   a->xchg = XCHG_INIT;
   a->cas  = CAS_INIT;
   for (uint32_t s = 0; s < NSLOT; s++)
      a->slot[s] = 0;
   for (uint32_t i = 0; i < TOTAL; i++)
      a->op[i] = operand(i);
   vkUnmapMemory(dev, amem);

   struct remote *rp;
   CHECK(vkMapMemory(dev, rmem, 0, sizeof(struct remote), 0, (void **)&rp));
   rp->add  = BDA_ADD_INIT;
   rp->smax = INT64_MIN;
   rp->cas  = CAS_INIT;
   rp->umin = UINT64_MAX;
   vkUnmapMemory(dev, rmem);

   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = rbuf,
   };
   const VkDeviceAddress raddr = vkGetBufferDeviceAddress(dev, &bdai);

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0,
      .size = sizeof(VkDeviceAddress),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* Workgroups of several warps: the largest power of two the device allows. */
   uint32_t local = LOCAL_MAX;
   while (local > props.limits.maxComputeWorkGroupSize[0] ||
          local > props.limits.maxComputeWorkGroupInvocations)
      local >>= 1;
   const uint32_t groups = TOTAL / local;
   if (groups > NGRP_MAX) { fprintf(stderr, "FAILED: workgroups too small\n"); return 1; }
   printf("local_size_x=%u, %u workgroups\n", local, groups);

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

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
   };
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
   VkDescriptorBufferInfo dbi = { .buffer = abuf, .offset = 0, .range = VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
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
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(raddr), &raddr);
   vkCmdDispatch(cb, groups, 1, 1);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify -------------------------------------------------------- */
   CHECK(vkMapMemory(dev, amem, 0, sizeof(struct acc), 0, (void **)&a));
   CHECK(vkMapMemory(dev, rmem, 0, sizeof(struct remote), 0, (void **)&rp));

   unsigned sentinels = 0;
   for (uint32_t w = 0; w < offsetof(struct acc, grp) / 8u; w++)
      sentinels += ((uint64_t *)a)[w] == SENTINEL;

   uint64_t all_bits = 0, all_xor = 0;
   for (uint32_t i = 0; i < TOTAL; i++) {
      all_bits |= bit_operand(i);
      all_xor  ^= operand(i);
   }

   /* storage buffer */
   for (uint32_t k = 0; k < NADD; k++)
      check_add_chain("ssbo add", a->ret_add, NADD, k, TOTAL / NADD, add_init[k],
                      addend[k], a->add[k]);
   if (a->band != ~all_bits)
      fail("ssbo and", UINT32_MAX, a->band, ~all_bits);
   if (a->bor != all_bits)
      fail("ssbo or", UINT32_MAX, a->bor, all_bits);
   if (a->bxor != (XOR_INIT ^ all_xor))
      fail("ssbo xor", UINT32_MAX, a->bxor, XOR_INIT ^ all_xor);
   for (uint32_t i = 0; i < TOTAL; i++) {
      /* an or returns a word between 0 and the final one, holding none of the
       * bits only this lane sets; an and the mirror image */
      if ((a->ret_or[i] & ~a->bor) != 0)
         fail("ssbo or return", i, a->ret_or[i], a->bor);
      if ((a->band & ~a->ret_and[i]) != 0)
         fail("ssbo and return", i, a->ret_and[i], a->band);
   }
   check_minmax("ssbo smin", false, true, (uint64_t)INT64_MAX,
                (const uint64_t *)a->ret_smin, 0, TOTAL, (uint64_t)a->smin);
   check_minmax("ssbo smax", true, true, (uint64_t)INT64_MIN,
                (const uint64_t *)a->ret_smax, 0, TOTAL, (uint64_t)a->smax);
   check_minmax("ssbo umin", false, false, UINT64_MAX, a->ret_umin, 0, TOTAL, a->umin);
   check_minmax("ssbo umax", true, false, 0, a->ret_umax, 0, TOTAL, a->umax);
   check_exchange("ssbo exchange", a->ret_xchg, 0, TOTAL, XCHG_INIT, a->xchg);
   if (a->cas != CAS_INIT + (uint64_t)TOTAL * CAS_STEP)
      fail("ssbo compare-and-swap counter", UINT32_MAX, a->cas,
           CAS_INIT + (uint64_t)TOTAL * CAS_STEP);
   for (uint32_t s = 0; s < NSLOT; s++) {
      uint32_t winners = 0;
      for (uint32_t i = s; i < TOTAL; i += NSLOT) {
         if (a->ret_slot[i] == 0) {
            winners++;
            if (a->slot[s] != claim(i))
               fail("ssbo compare-and-swap slot", s, a->slot[s], claim(i));
         } else if (a->ret_slot[i] != a->slot[s]) {
            fail("ssbo compare-and-swap loser", i, a->ret_slot[i], a->slot[s]);
         }
      }
      if (winners != 1)
         fail("ssbo compare-and-swap winners", s, winners, 1);
   }

   /* shared memory, per workgroup */
   for (uint32_t g = 0; g < groups; g++) {
      const uint64_t *r = &a->grp[g * 8u];
      const uint32_t first = g * local;
      uint64_t bits = 0, x = 0;
      for (uint32_t j = 0; j < local; j++) {
         bits |= bit_operand(first + j);
         x    ^= operand(first + j);
      }
      check_add_chain("shared add", a->ret_sadd, 1, first, local, SH_ADD_INIT,
                      SH_ADD_STEP, r[0]);
      if (r[1] != ~bits)
         fail("shared and", g, r[1], ~bits);
      if (r[2] != bits)
         fail("shared or", g, r[2], bits);
      if (r[3] != (XOR_INIT ^ x))
         fail("shared xor", g, r[3], XOR_INIT ^ x);
      check_minmax("shared smin", false, true, (uint64_t)INT64_MAX, NULL, first, local, r[4]);
      check_minmax("shared umax", true, false, 0, NULL, first, local, r[5]);
      check_exchange("shared exchange", a->ret_sxchg, first, local, SH_XCHG_INIT, r[6]);
      if (r[7] != SH_CAS_INIT + local)
         fail("shared compare-and-swap counter", g, r[7], SH_CAS_INIT + local);
   }

   /* buffer device address */
   check_add_chain("bda add", a->ret_badd, 1, 0, TOTAL, BDA_ADD_INIT, BDA_ADD_STEP, rp->add);
   check_minmax("bda smax", true, true, (uint64_t)INT64_MIN,
                (const uint64_t *)a->ret_bsmax, 0, TOTAL, (uint64_t)rp->smax);
   check_minmax("bda umin", false, false, UINT64_MAX, NULL, 0, TOTAL, rp->umin);
   if (rp->cas != CAS_INIT + (uint64_t)TOTAL * CAS_STEP)
      fail("bda compare-and-swap counter", UINT32_MAX, rp->cas,
           CAS_INIT + (uint64_t)TOTAL * CAS_STEP);

   vkUnmapMemory(dev, rmem);
   vkUnmapMemory(dev, amem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyBuffer(dev, abuf, NULL);
   vkFreeMemory(dev, amem, NULL);
   vkDestroyBuffer(dev, rbuf, NULL);
   vkFreeMemory(dev, rmem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (sentinels)
      fprintf(stderr, "  %u result slots still hold the sentinel: the dispatch "
              "did not run on the device\n", sentinels);
   if (fails || sentinels) {
      printf("FAILED (%u mismatches)\n", fails);
      return 1;
   }
   printf("PASSED (%u invocations: 64-bit add/and/or/xor/min/max/exchange/"
          "compare-and-swap on buffer, shared and device address)\n", TOTAL);
   return 0;
}
