/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * SSBO write-readback + SSBO atomic test for the vortexpipe driver (W7).
 *
 * Binds two storage buffers to a compute dispatch:
 *   binding 0 "data"    — N uints, seeded to a poison value; the shader
 *                         overwrites each with i*3+1  (store_ssbo).
 *   binding 1 "counter" — one uint, seeded 0; every invocation does
 *                         atomicAdd(counter, i)       (ssbo_atomic iadd).
 *   binding 2 "src"     — N uints, seeded src[i] = i; the shader only reads
 *                         it                          (load_ssbo).
 *
 * src makes the dispatch bind a descriptor that is never written. Those are
 * uploaded and relocated like any other but need no copy back, so this also
 * covers the read-only descriptor path. Its seed is chosen so the expected
 * results below are unchanged.
 *
 * After the dispatch the host verifies BOTH:
 *   data[i]  == i*3 + 1        (a broken store_ssbo fails this)
 *   counter  == sum(0..N-1)    (a broken/racy atomic fails this)
 *
 * Runs under lavapipe with GALLIUM_DRIVER=vortexpipe, so it exercises
 * vp_nir_to_llvm's store_ssbo and ssbo_atomic lowering plus the W7
 * multi-binding SSBO descriptor relocation.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N              256u
#define LOCAL_SIZE_MAX  64u
#define POISON          0xdeadbeefu

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

/* Allocate a host-visible/coherent storage buffer and bind memory. */
static int
make_buffer(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize bytes,
            VkBuffer *out_buf, VkDeviceMemory *out_mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (vkCreateBuffer(dev, &bci, NULL, out_buf) != VK_SUCCESS) return 1;

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);

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
   if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) return 1;
   if (vkBindBufferMemory(dev, *out_buf, *out_mem, 0) != VK_SUCCESS) return 1;
   return 0;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "ssbo.comp.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-ssbo",
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

   /* --- buffers: data[N] + counter[1] + src[N] --------------------- */
   const VkDeviceSize data_bytes = (VkDeviceSize)N * sizeof(uint32_t);
   const VkDeviceSize ctr_bytes  = sizeof(uint32_t);
   VkBuffer data_buf, ctr_buf, src_buf;
   VkDeviceMemory data_mem, ctr_mem, src_mem;
   if (make_buffer(dev, pd, data_bytes, &data_buf, &data_mem)) return 1;
   if (make_buffer(dev, pd, ctr_bytes,  &ctr_buf,  &ctr_mem))  return 1;
   if (make_buffer(dev, pd, data_bytes, &src_buf,  &src_mem))  return 1;

   /* seed: data[i] = POISON (so a correct run must overwrite it), counter = 0 */
   uint32_t *p;
   CHECK(vkMapMemory(dev, data_mem, 0, data_bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < N; i++) p[i] = POISON;
   vkUnmapMemory(dev, data_mem);
   uint32_t *c;
   CHECK(vkMapMemory(dev, ctr_mem, 0, ctr_bytes, 0, (void **)&c));
   *c = 0u;
   vkUnmapMemory(dev, ctr_mem);
   /* src[i] = i, so data[i] still works out to i*3+1 */
   uint32_t *sp;
   CHECK(vkMapMemory(dev, src_mem, 0, data_bytes, 0, (void **)&sp));
   for (uint32_t i = 0; i < N; i++) sp[i] = i;
   vkUnmapMemory(dev, src_mem);

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

   /* --- descriptor + pipeline layout: three storage-buffer bindings - */
   VkDescriptorSetLayoutBinding dslb[3] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
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

   /* --- pick local_size_x from device limits ---------------------- */
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

   /* --- descriptor set (3 buffers) -------------------------------- */
   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3,
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

   VkDescriptorBufferInfo dbi[3] = {
      { .buffer = data_buf, .offset = 0, .range = data_bytes },
      { .buffer = ctr_buf,  .offset = 0, .range = ctr_bytes  },
      { .buffer = src_buf,  .offset = 0, .range = data_bytes },
   };
   VkWriteDescriptorSet wds[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 2, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi[2] },
   };
   vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

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
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back + verify ---------------------------------------- */
   unsigned fails = 0;
   CHECK(vkMapMemory(dev, data_mem, 0, data_bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < N; i++) {
      uint32_t want = i * 3u + 1u;
      if (p[i] != want) {
         if (fails < 5)
            fprintf(stderr, "  data[%u] = %u (0x%x), want %u\n",
                    i, p[i], p[i], want);
         fails++;
      }
   }
   vkUnmapMemory(dev, data_mem);

   uint32_t got_ctr;
   CHECK(vkMapMemory(dev, ctr_mem, 0, ctr_bytes, 0, (void **)&c));
   got_ctr = *c;
   vkUnmapMemory(dev, ctr_mem);
   uint32_t want_ctr = (N * (N - 1u)) / 2u;   /* sum_{i=0}^{N-1} i */
   int ctr_ok = (got_ctr == want_ctr);
   if (!ctr_ok)
      fprintf(stderr, "  atomic counter = %u, want %u\n", got_ctr, want_ctr);

   /* src is read-only to the shader, so it must come back bit-identical. The
    * driver skips copying it back at all; this catches a skip that got the
    * bookkeeping wrong and wrote somewhere it should not have. */
   unsigned src_fails = 0;
   CHECK(vkMapMemory(dev, src_mem, 0, data_bytes, 0, (void **)&sp));
   for (uint32_t i = 0; i < N; i++) {
      if (sp[i] != i) {
         if (src_fails < 5)
            fprintf(stderr, "  src[%u] = %u (0x%x), want %u\n",
                    i, sp[i], sp[i], i);
         src_fails++;
      }
   }
   vkUnmapMemory(dev, src_mem);

   /* cleanup (best-effort) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkFreeMemory(dev, data_mem, NULL);
   vkFreeMemory(dev, ctr_mem, NULL);
   vkFreeMemory(dev, src_mem, NULL);
   vkDestroyBuffer(dev, data_buf, NULL);
   vkDestroyBuffer(dev, ctr_buf, NULL);
   vkDestroyBuffer(dev, src_buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails || !ctr_ok || src_fails) {
      printf("FAILED (array %u/%u mismatches, counter %s, read-only src %u/%u"
             " mismatches)\n",
             fails, N, ctr_ok ? "ok" : "WRONG", src_fails, N);
      return 1;
   }
   printf("PASSED (%u elements data[i]=src[i]*3+1, atomic counter=%u,"
          " read-only src intact)\n", N, got_ctr);
   return 0;
}
