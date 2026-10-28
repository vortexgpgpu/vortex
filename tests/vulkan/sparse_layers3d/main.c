/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Views of a sparse 3D image that start at a slice inside its tiles.
 *
 * A fully bound sparse R32_UINT 3D image, 64 x 32 x 32, is two 32 x 32 x 16
 * tiles wide and two deep, so a slice's tiles are not the next slice band's. Slices 14 to 17, which straddle the two tiles, are viewed two
 * ways, and a dispatch reads every texel of each view into a buffer:
 *  - a sliced 3D storage view (VK_EXT_image_sliced_view_of_3d), which runs on
 *    the device, where every texel must be its slice's;
 *  - a four-layer 2D-array storage view. It needs the image 2D-array
 *    compatible, which Vulkan does not allow on a sparse image
 *    (VUID-VkImageCreateInfo-flags-09403) and lavapipe does not reject. The
 *    device does not address it, so its dispatch must run on llvmpipe, whose
 *    values are reported, not checked.
 * The Makefile checks the placement. llvmpipe steps a view that starts inside
 * a tile's depth wrongly past the tile edge, so no value here is taken from it.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W      64u
#define N      32u
#define FIRST  14u
#define LAYERS 4u
#define LOCAL  4u

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

static uint32_t
texel(uint32_t x, uint32_t y, uint32_t z)
{
   return 0xa0000000u | (z << 16) | (y << 8) | x;
}

int
main(int argc, char **argv)
{
   const char *paths[2] = { (argc > 1) ? argv[1] : "sl_slice.comp.spv",
                            (argc > 2) ? argv[2] : "sl_load.comp.spv" };

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_layers3d",
      .apiVersion = VK_API_VERSION_1_1,
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
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceImageSlicedViewOf3DFeaturesEXT sliced = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &sliced,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.sparseBinding || !f2.features.sparseResidencyImage3D ||
       !sliced.imageSlicedViewOf3D) {
      printf("NotSupported: sparseBinding, sparseResidencyImage3D or "
             "imageSlicedViewOf3D is not offered\n");
      return 0;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   const VkQueueFlags need = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & need) == need) { qf = i; break; }
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
   VkPhysicalDeviceImageSlicedViewOf3DFeaturesEXT en_sliced = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT,
      .imageSlicedViewOf3D = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &en_sliced,
      .features = { .sparseBinding = VK_TRUE, .sparseResidencyImage3D = VK_TRUE },
   };
   const char *ext = VK_EXT_IMAGE_SLICED_VIEW_OF_3D_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &en,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &ext,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, qf, 0, &q);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   const VkMemoryPropertyFlags hv =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   /* --- the sparse 3D image, fully bound ------------------------------ */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
               VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
      .imageType = VK_IMAGE_TYPE_3D, .format = VK_FORMAT_R32_UINT,
      .extent = { W, N, N }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
   CHECK(vkQueueBindSparse(q, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(q));

   /* --- staging and result buffers ------------------------------------ */
   const uint32_t texels = W * N * N, out_texels = W * N * LAYERS;
   VkBuffer buf[2];
   VkDeviceMemory bmem[2];
   uint32_t *bp[2];
   const VkDeviceSize bsz[2] = { texels * 4u, out_texels * 4u };
   const VkBufferUsageFlags bus[2] = { VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
   for (unsigned i = 0; i < 2; i++) {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bsz[i], .usage = bus[i],
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &buf[i]));
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, buf[i], &mr);
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size, .memoryTypeIndex = find_mem(&mp, mr.memoryTypeBits, hv),
      };
      CHECK(vkAllocateMemory(dev, &mai, NULL, &bmem[i]));
      CHECK(vkBindBufferMemory(dev, buf[i], bmem[i], 0));
      CHECK(vkMapMemory(dev, bmem[i], 0, VK_WHOLE_SIZE, 0, (void **)&bp[i]));
   }
   for (uint32_t z = 0; z < N; z++)
      for (uint32_t y = 0; y < N; y++)
         for (uint32_t x = 0; x < W; x++)
            bp[0][(z * N + y) * W + x] = texel(x, y, z);

   /* --- the views, the sampler, the pipelines -------------------------- */
   VkImageViewSlicedCreateInfoEXT slice = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_SLICED_CREATE_INFO_EXT,
      .sliceOffset = FIRST, .sliceCount = LAYERS,
   };
   VkImageViewCreateInfo ivci[2] = {
      { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &slice, .image = img,
        .viewType = VK_IMAGE_VIEW_TYPE_3D, .format = VK_FORMAT_R32_UINT,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
      { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format = VK_FORMAT_R32_UINT,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, FIRST, LAYERS } },
   };
   VkImageView view[2];
   for (unsigned i = 0; i < 2; i++)
      CHECK(vkCreateImageView(dev, &ivci[i], NULL, &view[i]));

   VkDescriptorSetLayout dsl[2];
   VkPipelineLayout pl[2];
   VkPipeline pipe[2];
   VkDescriptorPoolSize ps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 2, .pPoolSizes = ps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSet ds[2];
   for (unsigned i = 0; i < 2; i++) {
      VkDescriptorSetLayoutBinding b[2] = {
         { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
         { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      };
      VkDescriptorSetLayoutCreateInfo dslci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 2, .pBindings = b,
      };
      CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl[i]));
      VkPipelineLayoutCreateInfo plci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1, .pSetLayouts = &dsl[i],
      };
      CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl[i]));
      VkShaderModule sm = load_module(dev, paths[i]);
      if (!sm) { fprintf(stderr, "FAILED: shader %s\n", paths[i]); return 1; }
      VkComputePipelineCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
         .layout = pl[i],
      };
      CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe[i]));
      vkDestroyShaderModule(dev, sm, NULL);
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl[i],
      };
      CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds[i]));
      VkDescriptorImageInfo dii = { VK_NULL_HANDLE, view[i], VK_IMAGE_LAYOUT_GENERAL };
      VkDescriptorBufferInfo dbi = { buf[1], 0, VK_WHOLE_SIZE };
      VkWriteDescriptorSet w[2] = {
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[i], .dstBinding = 0,
           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           .pImageInfo = &dii },
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[i], .dstBinding = 1,
           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           .pBufferInfo = &dbi },
      };
      vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
   }

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   const VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };

   /* Upload every slice. */
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   VkImageMemoryBarrier ib = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &ib);
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, N, N },
   };
   vkCmdCopyBufferToImage(cb, buf[0], img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
   CHECK(vkEndCommandBuffer(cb));
   CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(q));

   static const char *const name[2] = { "sliced 3D storage", "2D-array storage" };
   unsigned bad = 0;
   for (unsigned i = 0; i < 2; i++) {
      memset(bp[1], 0, out_texels * 4u);
      CHECK(vkResetCommandBuffer(cb, 0));
      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe[i]);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[i], 0, 1, &ds[i], 0, NULL);
      vkCmdDispatch(cb, W / LOCAL, N / LOCAL, LAYERS);
      CHECK(vkEndCommandBuffer(cb));
      CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(q));
      unsigned differ = 0;
      char first[96] = "";
      for (uint32_t l = 0; l < LAYERS; l++)
         for (uint32_t y = 0; y < N; y++)
            for (uint32_t x = 0; x < W; x++) {
               const uint32_t got = bp[1][(l * N + y) * W + x];
               if (got != texel(x, y, FIRST + l) && !differ++)
                  snprintf(first, sizeof first, "; first (%u,%u,%u): got 0x%08x, want 0x%08x",
                           x, y, l, got, texel(x, y, FIRST + l));
            }
      printf("%s view: %u/%u texels differ from slices %u..%u%s%s\n", name[i], differ,
             out_texels, FIRST, FIRST + LAYERS - 1u, first, i == 1 ? " (not checked)" : "");
      if (i == 0)
         bad += differ;
   }
   if (bad) {
      printf("FAILED (%u texels of the sliced view are not their slices')\n", bad);
      return 1;
   }

   printf("PASSED (sparse_layers3d: views of slices %u..%u of a sparse 3D image, across "
          "its tile edge, read by two dispatches)\n", FIRST, FIRST + LAYERS - 1u);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return 0;
}
