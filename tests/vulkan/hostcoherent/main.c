/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Host-coherent memory across submissions.
 *
 * One storage buffer in HOST_VISIBLE | HOST_COHERENT memory, mapped once and
 * never unmapped or flushed. Six submissions; before each the host writes a
 * fresh pattern straight into the mapping, after each it reads the result
 * through the same pointer once the fence has signalled:
 *
 *   even submission:  data = data * 3 + k
 *   odd submission:   data = data * 3 + k; data += 1 (llvmpipe); data = data * 3 + k'
 *
 * Every submission's pattern and constants differ, so a driver that serves the
 * device a stale copy of the buffer -- the previous submission's bytes, or the
 * bytes from before the llvmpipe dispatch -- produces a different number in
 * every element, never the expected one by accident.
 *
 * The odd submissions' middle dispatch has a workgroup larger than the device's
 * and shares memory, so it runs on llvmpipe, which writes host memory with no
 * gallium transfer to interpose on. MESA_VORTEX_STRICT is 0 for that reason.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N              1024u
#define LOCAL_SIZE_MAX 64u
#define BIG_SIZE       128u   /* inc_shared.comp's workgroup */
#define SUBMISSIONS    6u

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

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t size = 0;
   uint32_t *spv = read_spirv(path, &size);
   if (!spv)
      return VK_NULL_HANDLE;
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &ci, NULL, &sm) != VK_SUCCESS)
      sm = VK_NULL_HANDLE;
   free(spv);
   return sm;
}

static uint32_t
seed(uint32_t i, uint32_t s)
{
   return (i * 2654435761u) ^ (s * 40503u + 17u);
}

static uint32_t
expected(uint32_t i, uint32_t s)
{
   const uint32_t k0 = 11u * s + 1u, k1 = 7u * s + 5u;
   uint32_t v = seed(i, s) * 3u + k0;
   if (s & 1u) {
      v += 1u;
      v = v * 3u + k1;
   }
   return v;
}

int
main(int argc, char **argv)
{
   const char *spv_mad = (argc > 1) ? argv[1] : "mad.comp.spv";
   const char *spv_inc = (argc > 2) ? argv[2] : "inc_shared.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-hostcoherent",
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
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-coherent memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));

   /* Mapped once for the whole test: the host's writes reach the device, and
    * the device's reach the host, with no flush, invalidate or unmap. */
   volatile uint32_t *p;
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));

   VkShaderModule sm_mad = load_module(dev, spv_mad);
   VkShaderModule sm_inc = load_module(dev, spv_inc);
   if (!sm_mad || !sm_inc) { fprintf(stderr, "FAILED: shader modules\n"); return 1; }

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
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX : dev_max_x;
   while (local_size > 1 && (N % local_size) != 0)
      local_size >>= 1;
   printf("local_size_x=%u big=%u (device max=%u)\n", local_size, BIG_SIZE, dev_max_x);

   VkSpecializationMapEntry sme = { .constantID = 0, .offset = 0, .size = sizeof(uint32_t) };
   VkSpecializationInfo spec = {
      .mapEntryCount = 1, .pMapEntries = &sme,
      .dataSize = sizeof(uint32_t), .pData = &local_size,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm_mad, .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = pl,
   };
   VkPipeline pipe_mad, pipe_inc;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe_mad));
   cpci.stage.module = sm_inc;
   cpci.stage.pSpecializationInfo = NULL;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe_inc));

   VkDescriptorPoolSize dps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
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

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
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
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   const VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
   };
   unsigned fails = 0;
   for (uint32_t s = 0; s < SUBMISSIONS && !fails; s++) {
      for (uint32_t i = 0; i < N; i++)
         p[i] = seed(i, s);

      const uint32_t k0 = 11u * s + 1u, k1 = 7u * s + 5u;
      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkResetCommandBuffer(cb, 0));
      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_mad);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof k0, &k0);
      vkCmdDispatch(cb, N / local_size, 1, 1);
      if (s & 1u) {
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_inc);
         vkCmdDispatch(cb, N / BIG_SIZE, 1, 1);
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_mad);
         vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof k1, &k1);
         vkCmdDispatch(cb, N / local_size, 1, 1);
      }
      CHECK(vkEndCommandBuffer(cb));

      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cb,
      };
      CHECK(vkResetFences(dev, 1, &fence));
      CHECK(vkQueueSubmit(queue, 1, &si, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

      for (uint32_t i = 0; i < N; i++) {
         const uint32_t want = expected(i, s);
         if (p[i] != want) {
            if (fails < 5)
               fprintf(stderr, "  submission %u: data[%u] = 0x%08x, want 0x%08x\n",
                       s, i, p[i], want);
            fails++;
         }
      }
   }

   vkUnmapMemory(dev, mem);
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe_inc, NULL);
   vkDestroyPipeline(dev, pipe_mad, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm_inc, NULL);
   vkDestroyShaderModule(dev, sm_mad, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails) {
      printf("FAILED (%u mismatches: the device read a stale copy of host writes, "
             "or the host a stale copy of device writes)\n", fails);
      return 1;
   }
   printf("PASSED (hostcoherent: %u submissions over one persistently mapped buffer, "
          "host writes between them, llvmpipe dispatch inside every other one)\n",
          SUBMISSIONS);
   return 0;
}
