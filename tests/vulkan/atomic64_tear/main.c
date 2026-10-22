/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * 64-bit atomic loads and stores (OpAtomicLoad / OpAtomicStore) must be
 * indivisible on the Vortex device. On a 32-bit device a 64-bit load or store
 * is two word accesses unless the driver makes it one atomic access, and a
 * value read half before and half after a concurrent update is torn.
 *
 * Every invocation adds a step that moves both halves of a counter (whose low
 * word starts 16 below the carry into the high word) and atomically loads it
 * back, and atomically stores a word with equal halves and loads that back,
 * on a storage buffer, workgroup shared memory and a buffer device address.
 * The shader counts every load whose halves disagree; all counts must be zero
 * and the counters must end exactly TOTAL * ITERS steps on.
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
#define ITERS     16u
#define NGRP_MAX  64u
#define LOCAL_MAX 64u
#define SENTINEL  0xbadbadbadbadbad1ull
#define BASE      0x00000000fffffff0ull
#define STEP      0x0000000100000001ull

/* Mirrors the shader's Acc block (std430). */
struct acc {
   uint64_t ctr;
   uint64_t word;
   uint32_t bad[TOTAL * 6u];
   uint64_t grp[NGRP_MAX * 2u];
};

struct remote {
   uint64_t ctr;
   uint64_t word;
};

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
   const char *spv_path = (argc > 1) ? argv[1] : "atomic64_tear.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-atomic64-tear",
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
   for (uint32_t w = 0; w < TOTAL * 6u; w++)
      a->bad[w] = 0xbadbad01u;
   for (uint32_t w = 0; w < NGRP_MAX * 2u; w++)
      a->grp[w] = SENTINEL;
   a->ctr  = BASE;
   a->word = 0;
   vkUnmapMemory(dev, amem);

   struct remote *rp;
   CHECK(vkMapMemory(dev, rmem, 0, sizeof(struct remote), 0, (void **)&rp));
   rp->ctr  = BASE;
   rp->word = 0;
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

   static const char *const what[6] = {
      "buffer counter load", "buffer word load", "shared counter load",
      "shared word load", "bda counter load", "bda word load",
   };
   unsigned sentinels = 0;
   for (uint32_t i = 0; i < TOTAL; i++) {
      for (uint32_t k = 0; k < 6u; k++) {
         uint32_t b = a->bad[i * 6u + k];
         if (b == 0xbadbad01u) {
            sentinels++;
         } else if (b != 0) {
            fail(what[k], i, b, 0);
         }
      }
   }
   const uint64_t total = (uint64_t)TOTAL * ITERS;
   if (a->ctr != BASE + total * STEP)
      fail("buffer counter", UINT32_MAX, a->ctr, BASE + total * STEP);
   if (rp->ctr != BASE + total * STEP)
      fail("bda counter", UINT32_MAX, rp->ctr, BASE + total * STEP);
   if ((uint32_t)a->word != (uint32_t)(a->word >> 32))
      fail("buffer word", UINT32_MAX, a->word, 0);
   if ((uint32_t)rp->word != (uint32_t)(rp->word >> 32))
      fail("bda word", UINT32_MAX, rp->word, 0);
   for (uint32_t g = 0; g < groups; g++) {
      const uint64_t want = BASE + (uint64_t)local * ITERS * STEP;
      if (a->grp[g * 2u] != want)
         fail("shared counter", g, a->grp[g * 2u], want);
      uint64_t w = a->grp[g * 2u + 1u];
      if ((uint32_t)w != (uint32_t)(w >> 32))
         fail("shared word", g, w, 0);
   }

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
   printf("PASSED (%u invocations x %u: 64-bit atomic loads and stores never "
          "torn on buffer, shared and device address)\n", TOTAL, ITERS);
   return 0;
}
