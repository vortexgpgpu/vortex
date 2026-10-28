/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Sparse buffers read and written by compute.
 *
 * Sparse buffer a has four pages: page 0 and page 2 are bound to pages 0 and 1
 * of one memory, pages 1 and 3 are not. Sparse buffer b binds its one page to
 * page 0 of the same memory, so it aliases a's page 0. A uniform buffer and the
 * indirect dispatch's arguments live in two more sparse buffers, bound to pages
 * 3 and 4. The host fills the memory through a mapping; nothing is copied.
 *
 * An indirect dispatch reads a's two bound pages, b, the uniform buffer and a's
 * page 2 again through its device address, and stores into a's page 2, which
 * the host then finds in page 1 of the memory. Then a's page 0 is rebound to
 * page 2 of the memory, filled with another pattern, and the dispatch is run
 * again with no other call: a's page 0 must read the new page, while b, still
 * on page 0, keeps the old one.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

struct ctx {
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice        dev;
   VkQueue         queue;
   VkCommandBuffer cb;
   VkDeviceSize    page;
   VkDeviceMemory  mem;
   uint8_t        *map;
};

struct params {
   VkDeviceAddress p;
   uint32_t page_words;
   uint32_t mode;
};

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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static bool
sparse_buffer(struct ctx *c, VkDeviceSize pages, VkBufferUsageFlags usage, VkBuffer *buf,
              VkMemoryRequirements *mr)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
      .size = pages * c->page, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   vkGetBufferMemoryRequirements(c->dev, *buf, mr);
   return true;
}

/* Bind page `page` of `buf` to page `mpage` of the memory, or unbind it. */
static bool
bind_page(struct ctx *c, VkBuffer buf, uint32_t page, int32_t mpage)
{
   VkSparseMemoryBind b = {
      .resourceOffset = page * c->page, .size = c->page,
      .memory = mpage < 0 ? VK_NULL_HANDLE : c->mem,
      .memoryOffset = mpage < 0 ? 0 : (VkDeviceSize)mpage * c->page,
   };
   VkSparseBufferMemoryBindInfo bbi = { .buffer = buf, .bindCount = 1, .pBinds = &b };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .bufferBindCount = 1, .pBufferBinds = &bbi,
   };
   CHECK(vkQueueBindSparse(c->queue, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

static uint32_t *
mpage(struct ctx *c, uint32_t p)
{
   return (uint32_t *)(c->map + p * c->page);
}

static uint32_t
pat(uint32_t salt, uint32_t i)
{
   return salt ^ (i * 2654435761u);
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "FAILED: usage: %s sb.comp.spv\n", argv[0]);
      return 1;
   }
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_buffer", .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateInstance\n");
      return 1;
   }
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &npd, &pd) < 0 || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.sparseBinding || !f2.features.sparseResidencyBuffer ||
       !f2.features.sparseResidencyAliased || !f12.bufferDeviceAddress) {
      printf("NotSupported: sparse residency and aliasing of buffers, or buffer device "
             "addresses, are not offered\n");
      return 0;
   }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs compute and sparse binding\n");
      return 0;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan12Features w12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .bufferDeviceAddress = VK_TRUE,
   };
   VkPhysicalDeviceFeatures want = {
      .sparseBinding = VK_TRUE, .sparseResidencyBuffer = VK_TRUE,
      .sparseResidencyAliased = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &w12,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &want,
   };
   struct ctx c;
   memset(&c, 0, sizeof c);
   if (vkCreateDevice(pd, &dci, NULL, &c.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c.mp);

   /* The page is the sparse buffers' binding granularity. */
   const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
   VkBuffer a, b, u, ind, out;
   VkMemoryRequirements mr;
   {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
         .size = 4096, .usage = ssbo,
      };
      VkBuffer probe;
      if (vkCreateBuffer(c.dev, &bci, NULL, &probe) != VK_SUCCESS) {
         fprintf(stderr, "FAILED: sparse buffer\n");
         return 1;
      }
      vkGetBufferMemoryRequirements(c.dev, probe, &mr);
      vkDestroyBuffer(c.dev, probe, NULL);
      c.page = mr.alignment > N * 8u ? mr.alignment : N * 8u;
   }
   if (!sparse_buffer(&c, 4, ssbo, &a, &mr) || !sparse_buffer(&c, 1, ssbo, &b, &mr) ||
       !sparse_buffer(&c, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &u, &mr) ||
       !sparse_buffer(&c, 1, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &ind, &mr)) {
      return 1;
   }
   uint32_t mt = find_mem(&c.mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = 5 * c.page, .memoryTypeIndex = mt,
   };
   if (mt == UINT32_MAX || vkAllocateMemory(c.dev, &mai, NULL, &c.mem) != VK_SUCCESS ||
       vkMapMemory(c.dev, c.mem, 0, VK_WHOLE_SIZE, 0, (void **)&c.map) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: memory\n");
      return 1;
   }
   const uint32_t pw = (uint32_t)(c.page / 4u);
   for (uint32_t i = 0; i < pw; i++) {
      mpage(&c, 0)[i] = pat(0xa0a0a0a0u, i);
      mpage(&c, 1)[i] = pat(0xa2a2a2a2u, i);
      mpage(&c, 2)[i] = pat(0xb0b0b0b0u, i);
   }
   for (uint32_t i = 0; i < 64u; i++) {
      for (uint32_t k = 0; k < 4u; k++) {
         mpage(&c, 3)[4u * i + k] = pat(0xc0c0c0c0u, i) + k;
      }
   }
   const VkDispatchIndirectCommand dic = { N / 16u, 1, 1 };
   memcpy(mpage(&c, 4), &dic, sizeof dic);
   if (!bind_page(&c, a, 0, 0) || !bind_page(&c, a, 2, 1) || !bind_page(&c, b, 0, 0) ||
       !bind_page(&c, u, 0, 3) || !bind_page(&c, ind, 0, 4)) {
      return 1;
   }

   VkBufferCreateInfo obci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = N * 5u * 4u,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkDeviceMemory out_mem;
   uint32_t *op;
   VkMemoryRequirements omr;
   if (vkCreateBuffer(c.dev, &obci, NULL, &out) != VK_SUCCESS) {
      return 1;
   }
   vkGetBufferMemoryRequirements(c.dev, out, &omr);
   mai.allocationSize = omr.size;
   mai.memoryTypeIndex = find_mem(&c.mp, omr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (vkAllocateMemory(c.dev, &mai, NULL, &out_mem) != VK_SUCCESS ||
       vkBindBufferMemory(c.dev, out, out_mem, 0) != VK_SUCCESS ||
       vkMapMemory(c.dev, out_mem, 0, VK_WHOLE_SIZE, 0, (void **)&op) != VK_SUCCESS) {
      return 1;
   }

   VkDescriptorSetLayoutBinding dslb[4] = {
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct params) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   if (vkCreateDescriptorSetLayout(c.dev, &dslci, NULL, &dsl) != VK_SUCCESS ||
       vkCreatePipelineLayout(c.dev, &plci, NULL, &pl) != VK_SUCCESS ||
       vkCreateDescriptorPool(c.dev, &dpci, NULL, &dp) != VK_SUCCESS ||
       vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: layouts or pools\n");
      return 1;
   }
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   if (vkAllocateCommandBuffers(c.dev, &cbai, &c.cb) != VK_SUCCESS ||
       vkAllocateDescriptorSets(c.dev, &dsai, &ds) != VK_SUCCESS) {
      return 1;
   }
   VkDescriptorBufferInfo dbi[4] = {
      { a, 0, 4 * c.page }, { b, 0, c.page }, { u, 0, 64u * 16u }, { out, 0, N * 20u },
   };
   VkWriteDescriptorSet wds[4];
   for (unsigned i = 0; i < 4; i++) {
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                  : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[i],
      };
   }
   vkUpdateDescriptorSets(c.dev, 4, wds, 0, NULL);

   size_t sz = 0;
   uint32_t *spv = read_spirv(argv[1], &sz);
   if (!spv) {
      return 1;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   if (vkCreateShaderModule(c.dev, &smci, NULL, &sm) != VK_SUCCESS) {
      return 1;
   }
   free(spv);
   VkComputePipelineCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
      .layout = pl,
   };
   VkPipeline pipe;
   if (vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe) != VK_SUCCESS) {
      return 1;
   }
   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = a,
   };
   const VkDeviceAddress a_addr = vkGetBufferDeviceAddress(c.dev, &bdai);

   for (uint32_t mode = 0; mode < 2; mode++) {
      if (mode == 1 && !bind_page(&c, a, 0, 2)) {
         return 1;
      }
      memset(op, 0, N * 20u);
      const struct params p = { a_addr, pw, mode };
      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
         .pCommandBuffers = &c.cb,
      };
      if (vkResetCommandBuffer(c.cb, 0) != VK_SUCCESS ||
          vkBeginCommandBuffer(c.cb, &cbbi) != VK_SUCCESS) {
         return 1;
      }
      vkCmdBindPipeline(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(c.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      vkCmdPushConstants(c.cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
      vkCmdDispatchIndirect(c.cb, ind, 0);
      if (vkEndCommandBuffer(c.cb) != VK_SUCCESS ||
          vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS ||
          vkQueueWaitIdle(c.queue) != VK_SUCCESS) {
         return 1;
      }
      unsigned bad = 0;
      char first[160] = "";
      for (uint32_t i = 0; i < N; i++) {
         const uint32_t want[6] = {
            pat(mode ? 0xb0b0b0b0u : 0xa0a0a0a0u, i), pat(0xa2a2a2a2u, i),
            pat(0xa0a0a0a0u, i), pat(0xc0c0c0c0u, i & 63u), pat(0xa2a2a2a2u, i),
            3u * i + mode,
         };
         const uint32_t got[6] = {
            op[5u * i], op[5u * i + 1u], op[5u * i + 2u], op[5u * i + 3u], op[5u * i + 4u],
            mpage(&c, 1)[pw / 2u + i],
         };
         static const char *const what[6] = {
            "a page 0", "a page 2", "b", "uniform", "a page 2 by address", "store to a page 2",
         };
         for (unsigned k = 0; k < 6; k++) {
            if (got[k] != want[k] && !bad++) {
               snprintf(first, sizeof first, "%s word %u: 0x%08x, want 0x%08x", what[k], i,
                        got[k], want[k]);
            }
         }
      }
      if (bad) {
         printf("FAILED (pass %u: %u words wrong; first %s)\n", mode, bad, first);
         return 1;
      }
      printf("pass %u: %u invocations read and wrote the bound pages right\n", mode, N);
   }
   printf("PASSED (sparse_buffer: storage, aliased, uniform, indirect and device-address "
          "access to partially resident sparse buffers, across a rebind)\n");
   return 0;
}
