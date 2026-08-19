/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_zero_initialize_workgroup_memory on the device path.
 *
 * A shared variable declared with a null initializer must read back as zero at
 * the start of every dispatch. Vortex local memory is not cleared between
 * dispatches, so the driver emits a cooperative zeroing pass; nothing in the
 * suite exercised it.
 *
 * The kernel records what its shared slot held on entry, then fills that slot
 * with 0xDEADBEEF. It is dispatched TWICE. The first dispatch proves nothing on
 * its own -- local memory may happen to be clean -- but it guarantees the second
 * runs over memory that is definitely dirty. Every slot must still read zero.
 *
 * That two-dispatch shape is the whole test. A driver emitting no zeroing at all
 * passes a single-dispatch version whenever the allocation starts clean, which
 * is how an unemitted pass stays hidden.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N           256u
/* Maximum local_size_x the test will request. The actual value is
 * specialized at pipeline creation to
 * min(LOCAL_SIZE_MAX, VkPhysicalDeviceProperties.limits.maxComputeWorkGroupSize[0])
 * — vortexpipe reports the Vortex hardware cap (NUM_THREADS × NUM_WARPS)
 * there, so the same shader runs on every Vortex config without baking
 * the cap into the .comp source. N must be a multiple of the chosen
 * value; the loop below rounds down if needed. */
#define LOCAL_SIZE_MAX  64u

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
   const char *spv_path = (argc > 1) ? argv[1] : "add1.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-compute-smoke",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   /* --- physical device ------------------------------------------- */
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* compute-capable queue family */
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   /* --- logical device + queue ------------------------------------ */
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

   /* --- storage buffer + host-visible memory ---------------------- */
   const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(uint32_t);
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t mt = UINT32_MAX;
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));

   /* Seed with a non-zero pattern, so "every slot is zero" cannot pass on a run
    * where the kernel never executed at all. */
   uint32_t *p;
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < N; i++) p[i] = 0xA5A5A5A5u;
   vkUnmapMemory(dev, mem);

   /* --- shader module --------------------------------------------- */
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

   /* --- descriptor + pipeline layout ------------------------------ */
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

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- workgroup size ------------------------------------------- *
    * Capped at the shader's SLOTS (16) because the shared array is sized for
    * that; then rounded down to a power of two dividing N, so every invocation
    * has a slot and every slot is written. */
   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = 16u < dev_max_x ? 16u : dev_max_x;
   while (local_size > 1 && (N % local_size) != 0) {
      local_size >>= 1;
   }
   if (local_size == 0) {
      local_size = 1;
   }
   printf("local_size_x=%u (device max=%u)\n", local_size, dev_max_x);

   /* --- compute pipeline (→ vortexpipe create_compute_state) ------ */
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

   /* --- descriptor set -------------------------------------------- */
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

   VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = bytes };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   /* --- command buffer: bind + dispatch --------------------------- */
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
                           0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, N / local_size, 1, 1);
   /* Dispatch the same pipeline a second time. The kernel is a read-modify-
    * write, so two dispatches must leave i+2 -- a second run that reuses the
    * kernel already resident on the device, and that must see the first run's
    * output as its input. One dispatch cannot distinguish either. */
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        1, &mb, 0, NULL, 0, NULL);
   vkCmdDispatch(cb, N / local_size, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back + verify ---------------------------------------- */
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   unsigned fails = 0;
   for (uint32_t i = 0; i < N; i++) {
      /* Zero on entry to the SECOND dispatch, which ran over local memory the
       * first one deliberately filled with 0xDEADBEEF. */
      if (p[i] != 0u) {
         if (fails < 5) {
            fprintf(stderr, "  seen[%u] = 0x%08x, want 0\n", i, p[i]);
         }
         fails++;
      }
   }
   vkUnmapMemory(dev, mem);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails) {
      printf("FAILED (zeroshared: %u/%u shared slots were non-zero at dispatch "
             "entry -- 0xdeadbeef is the previous dispatch's poison, so the null "
             "initializer was not honoured; 0xa5a5a5a5 means the kernel never "
             "ran)\n", fails, N);
      return 1;
   }
   printf("PASSED (zeroshared: %u shared slots read zero on entry to the second "
          "dispatch, over local memory the first one poisoned)\n", N);
   return 0;
}
