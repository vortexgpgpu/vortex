/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Work the device must refuse: a fully bound sparse R32_UINT image 256 texels,
 * two 64 KiB tiles, wide. The reference layout keeps each tile's 128 x 128
 * texels together, which rows running linearly across both tiles do not, so
 * the device's linear addressing would reach the wrong memory. A compute
 * dispatch and a
 * rasterizer-discard draw, whose vertex shader runs alone for its stores, each
 * store into the image and read their texels back into a buffer.
 *
 * Both must be refused with the sparse-image reason and run on llvmpipe, which
 * the Makefile checks from the driver's placement lines, and every value read
 * back must be the one stored.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe, MESA_VORTEX_STRICT=0.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W     256u
#define H     4u
#define LOCAL 4u

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

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv)
      return VK_NULL_HANDLE;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS)
      sm = VK_NULL_HANDLE;
   free(spv);
   return sm;
}

int
main(int argc, char **argv)
{
   const char *cs_path = (argc > 1) ? argv[1] : "sf_store.comp.spv";
   const char *vs_path = (argc > 2) ? argv[2] : "sf_store.vert.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparsefb",
      .apiVersion = VK_API_VERSION_1_3,
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

   VkPhysicalDeviceVulkan13Features v13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.sparseBinding || !f2.features.vertexPipelineStoresAndAtomics ||
       !v13.dynamicRendering) {
      printf("NotSupported: sparseBinding, vertexPipelineStoresAndAtomics or "
             "dynamicRendering is not offered by this device\n");
      return 0;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & need) == need) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs graphics, compute and sparse binding\n");
      return 0;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan13Features v13_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 f2_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13_en,
      .features = { .sparseBinding = VK_TRUE, .vertexPipelineStoresAndAtomics = VK_TRUE },
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

   /* --- the sparse image, fully bound ------------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .extent = { W, 2u * H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   uint32_t imt = find_mem(&mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no sparse image memory\n"); return 1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   VkSparseMemoryBind bind = { .resourceOffset = 0, .size = imr.size, .memory = imem };
   VkSparseImageOpaqueMemoryBindInfo obi = { .image = img, .bindCount = 1, .pBinds = &bind };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageOpaqueBindCount = 1, .pImageOpaqueBinds = &obi,
   };
   CHECK(vkQueueBindSparse(queue, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   /* --- the read-back buffer ---------------------------------------- */
   const uint32_t texels = 2u * W * H;
   const VkDeviceSize bytes = (VkDeviceSize)texels * sizeof(uint32_t);
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer out;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &out));
   VkMemoryRequirements omr;
   vkGetBufferMemoryRequirements(dev, out, &omr);
   uint32_t omt = find_mem(&mp, omr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (omt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   VkMemoryAllocateInfo omai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = omr.size, .memoryTypeIndex = omt,
   };
   VkDeviceMemory omem;
   CHECK(vkAllocateMemory(dev, &omai, NULL, &omem));
   CHECK(vkBindBufferMemory(dev, out, omem, 0));
   uint32_t *p;
   CHECK(vkMapMemory(dev, omem, 0, bytes, 0, (void **)&p));
   memset(p, 0, (size_t)bytes);
   vkUnmapMemory(dev, omem);

   /* --- descriptors and pipelines ----------------------------------- */
   const VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = stages },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = stages },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = { .stageFlags = stages, .offset = 0, .size = 2 * sizeof(uint32_t) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkShaderModule csm = load_module(dev, cs_path);
   VkShaderModule vsm = load_module(dev, vs_path);
   if (!csm || !vsm) { fprintf(stderr, "FAILED: shader modules\n"); return 1; }
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = csm, .pName = "main",
      },
      .layout = pl,
   };
   VkPipeline cpipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe));

   VkPipelineShaderStageCreateInfo vss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vsm, .pName = "main",
   };
   VkPipelineVertexInputStateCreateInfo vis = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ias = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .rasterizerDiscardEnable = VK_TRUE,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineRenderingCreateInfo prci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &prci,
      .stageCount = 1, .pStages = &vss,
      .pVertexInputState = &vis, .pInputAssemblyState = &ias,
      .pRasterizationState = &rs, .layout = pl,
   };
   VkPipeline gpipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &gpipe));
   vkDestroyShaderModule(dev, csm, NULL);
   vkDestroyShaderModule(dev, vsm, NULL);

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
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
   VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorBufferInfo dbi = { .buffer = out, .offset = 0, .range = bytes };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

   /* --- record: the dispatch, then the discard draw ------------------ */
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   VkImageMemoryBarrier togen = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                        0, 0, NULL, 0, NULL, 1, &togen);
   const uint32_t extent[2] = { W, H };
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
   vkCmdPushConstants(cb, pl, stages, 0, sizeof extent, extent);
   vkCmdDispatch(cb, W / LOCAL, H / LOCAL, 1);

   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { 1, 1 } }, .layerCount = 1,
   };
   vkCmdBeginRendering(cb, &ri);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
   vkCmdPushConstants(cb, pl, stages, 0, sizeof extent, extent);
   vkCmdDraw(cb, W * H, 1, 0, 0);
   vkCmdEndRendering(cb);
   VkMemoryBarrier tohost = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &tohost, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned bad = 0;
   uint32_t first = UINT32_MAX, got = 0;
   CHECK(vkMapMemory(dev, omem, 0, bytes, 0, (void **)&p));
   for (uint32_t k = 0; k < texels; k++) {
      if (p[k] != (0xc0000000u ^ (k * 2654435761u))) {
         if (first == UINT32_MAX) { first = k; got = p[k]; }
         bad++;
      }
   }
   vkUnmapMemory(dev, omem);
   if (bad) {
      printf("FAILED (%u/%u values wrong; first at %u (%s): got 0x%08x, want 0x%08x)\n",
             bad, texels, first, first < W * H ? "dispatch" : "draw", got,
             0xc0000000u ^ (first * 2654435761u));
      return 1;
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, gpipe, NULL);
   vkDestroyPipeline(dev, cpipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyBuffer(dev, out, NULL);
   vkFreeMemory(dev, omem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("PASSED (sparsefb: %u texels stored and read back through a two-tile-wide "
          "sparse image by a dispatch and a discard draw)\n", texels);
   return 0;
}
