/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A compute shader reaching its storage buffers through two descriptor sets.
 *
 * Each bound set becomes its own constant-buffer blob, and the byte offset the
 * driver records for a descriptor is an offset within the blob that descriptor
 * lives in. The draw path relocates blob by blob and matches each descriptor to
 * its own; the launch path stages one blob and applies every recorded offset to
 * it. Set-1 descriptors therefore have their rewrites land at meaningless
 * addresses inside set 0's blob, while the descriptors themselves keep the
 * host's pointers -- and the device dereferences those.
 *
 * Four bindings, well inside any per-shader limit, so the descriptor count is
 * not what is under test. Every buffer carries a value naming its (set,
 * binding), so the report says which ones arrived.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N           256u
#define LOCAL_SIZE_MAX  64u
/* Two sets of two bindings each -- must match csmultiset.comp. */
#define NSET        2u
#define NBIND       2u
#define NBUF        (NSET * NBIND)

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

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

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "csmultiset.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-csmultiset",
      .apiVersion = VK_API_VERSION_1_1,
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

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   /* --- storage buffers ------------------------------------------- */
   const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(uint32_t);
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer buf[NBUF];
   for (uint32_t b = 0; b < NBUF; b++) {
      CHECK(vkCreateBuffer(dev, &bci, NULL, &buf[b]));
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf[0], &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t mt = UINT32_MAX;
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   /* A separate allocation per buffer, so a binding that is never relocated
    * cannot be rescued by happening to alias one that was. */
   VkDeviceMemory mem[NBUF];
   for (uint32_t b = 0; b < NBUF; b++) {
      CHECK(vkAllocateMemory(dev, &mai, NULL, &mem[b]));
      CHECK(vkBindBufferMemory(dev, buf[b], mem[b], 0));
      uint32_t *p;
      CHECK(vkMapMemory(dev, mem[b], 0, bytes, 0, (void **)&p));
      for (uint32_t i = 0; i < N; i++) {
         p[i] = 0xdeadbeefu;
      }
      vkUnmapMemory(dev, mem[b]);
   }

   /* --- shader module + compute pipeline -------------------------- */
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

   /* Two identical layouts, bound as set 0 and set 1. Identical is what makes
    * the test sharp: the only thing distinguishing a set-1 descriptor from its
    * set-0 twin is which blob it lives in. */
   VkDescriptorSetLayoutBinding dslb[NBIND];
   for (uint32_t b = 0; b < NBIND; b++) {
      dslb[b] = (VkDescriptorSetLayoutBinding){
         .binding = b, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = NBIND, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl[NSET];
   for (uint32_t s = 0; s < NSET; s++) {
      CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl[s]));
   }
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = NSET, .pSetLayouts = dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- pick local_size_x from device limits ---------------------- *
    * Cap the requested size by what the device advertises, then round
    * down to the largest power of two that divides N. csmultiset.comp
    * declares layout(local_size_x_id = 0) so the value flows in as
    * specialization constant ID 0. */
   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX
                                                    : dev_max_x;
   while (local_size > 1 && (N % local_size) != 0)
      local_size >>= 1;
   if (local_size == 0)
      local_size = 1;
   printf("local_size_x=%u (device max=%u)\n", local_size, dev_max_x);

   VkSpecializationMapEntry sme = {
      .constantID = 0, .offset = 0, .size = sizeof(uint32_t),
   };
   VkSpecializationInfo spec_info = {
      .mapEntryCount = 1, .pMapEntries = &sme,
      .dataSize = sizeof(uint32_t), .pData = &local_size,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec_info,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = NBUF,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = NSET, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = NSET, .pSetLayouts = dsl,
   };
   VkDescriptorSet ds[NSET];
   CHECK(vkAllocateDescriptorSets(dev, &dsai, ds));
   VkDescriptorBufferInfo dbi[NBUF];
   VkWriteDescriptorSet wds[NBUF];
   for (uint32_t b = 0; b < NBUF; b++) {
      dbi[b] = (VkDescriptorBufferInfo){
         .buffer = buf[b], .offset = 0, .range = bytes,
      };
      wds[b] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds[b / NBIND], .dstBinding = b % NBIND, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[b],
      };
   }
   vkUpdateDescriptorSets(dev, NBUF, wds, 0, NULL);

   /* --- dispatch -------------------------------------------------- */
   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
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
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, NSET, ds, 0, NULL);
   vkCmdDispatch(cb, N / local_size, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back + verify ---------------------------------------- */
   /* Report per binding and name its set: the failure this test exists to catch
    * loses one whole set and keeps the other, so which set was lost is the
    * finding, not how many elements differed. */
   unsigned bad_bindings = 0;
   for (uint32_t b = 0; b < NBUF; b++) {
      uint32_t *p;
      CHECK(vkMapMemory(dev, mem[b], 0, bytes, 0, (void **)&p));
      unsigned fails = 0;
      for (uint32_t i = 0; i < N; i++) {
         if (p[i] != 1000u * b + i) {
            fails++;
         }
      }
      vkUnmapMemory(dev, mem[b]);
      if (fails) {
         fprintf(stderr, "  set %u binding %u: %u/%u elements wrong\n",
                 b / NBIND, b % NBIND, fails, N);
         bad_bindings++;
      }
   }

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad_bindings) {
      printf("FAILED (csmultiset: %u of %u bindings never arrived; a whole set "
             "lost while the other survives is the launch relocation ignoring "
             "which blob a descriptor belongs to)\n", bad_bindings, NBUF);
      return 1;
   }
   printf("PASSED (csmultiset: all %u bindings across %u descriptor sets carry "
          "their own value, %u elements each)\n", NBUF, NSET, N);
   return 0;
}
