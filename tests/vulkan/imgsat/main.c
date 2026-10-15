/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Integer storage-image stores past each format's range.
 *
 * The spec leaves the stored value undefined when a typed store's integer does
 * not fit the format; lavapipe saturates it to the field, and this pins that:
 * an 8-bit unsigned image given 300 stores 255, a signed one given -300 stores
 * -128, where keeping the low bits would store 44 and 0xd4. Two texels per
 * image: a value past the range by a little, and the first power of two past
 * it (low bits all zero, so a wrap stores 0 and only a clamp stores the
 * maximum); the signed images take the positive side instead. The A2B10G10R10
 * image saturates its red and alpha fields on their own.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct img {
   const char *name;
   VkFormat    fmt;
   unsigned    bpp;
   uint8_t     expect[2][4];   /* the two texels' bytes, low first */
};

static const struct img IMGS[5] = {
   { "r8ui",       VK_FORMAT_R8_UINT,  1, { { 0xff }, { 0xff } } },
   { "r8i",        VK_FORMAT_R8_SINT,  1, { { 0x80 }, { 0x7f } } },
   { "r16ui",      VK_FORMAT_R16_UINT, 2, { { 0xff, 0xff }, { 0xff, 0xff } } },
   { "r16i",       VK_FORMAT_R16_SINT, 2, { { 0x00, 0x80 }, { 0xff, 0x7f } } },
   /* 1023 | 5 << 10 | 1023 << 20 | 3 << 30, then 1023 | 1 << 30 */
   { "rgb10a2ui",  VK_FORMAT_A2B10G10R10_UINT_PACK32, 4,
     { { 0xff, 0x17, 0xf0, 0xff }, { 0xff, 0x03, 0x00, 0xc0 } } },
};

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

int
main(int argc, char **argv)
{
   const char *cs_path = (argc > 1) ? argv[1] : "imgsat.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-imgsat",
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
   /* r8, r16 and rgb10_a2ui are extended storage formats. */
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!have.shaderStorageImageExtendedFormats) {
      fprintf(stderr, "FAILED: shaderStorageImageExtendedFormats not supported\n");
      return 1;
   }
   VkPhysicalDeviceFeatures want = { .shaderStorageImageExtendedFormats = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   VkImage img[5];
   VkDeviceMemory imem[5];
   VkImageView view[5];
   VkBuffer rb[5];
   VkDeviceMemory rmem[5];
   for (unsigned i = 0; i < 5; i++) {
      VkImageCreateInfo imci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = IMGS[i].fmt,
         .extent = { 2, 1, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      CHECK(vkCreateImage(dev, &imci, NULL, &img[i]));
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, img[i], &mr);
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size, .memoryTypeIndex = find_mem(&mp, mr.memoryTypeBits, 0),
      };
      CHECK(vkAllocateMemory(dev, &mai, NULL, &imem[i]));
      CHECK(vkBindImageMemory(dev, img[i], imem[i], 0));
      VkImageViewCreateInfo ivci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = IMGS[i].fmt,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      };
      CHECK(vkCreateImageView(dev, &ivci, NULL, &view[i]));
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 2u * IMGS[i].bpp, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &rb[i]));
      VkMemoryRequirements bmr;
      vkGetBufferMemoryRequirements(dev, rb[i], &bmr);
      VkMemoryAllocateInfo bmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = bmr.size, .memoryTypeIndex = find_mem(&mp, bmr.memoryTypeBits, host),
      };
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &rmem[i]));
      CHECK(vkBindBufferMemory(dev, rb[i], rmem[i], 0));
   }

   size_t sz = 0;
   uint32_t *spv = read_spirv(cs_path, &sz);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule cs;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &cs));
   free(spv);
   VkDescriptorSetLayoutBinding b[5];
   for (unsigned i = 0; i < 5; i++) {
      b[i] = (VkDescriptorSetLayoutBinding){
         .binding = i, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      };
   }
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 5, .pBindings = b,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main" },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorImageInfo dii[5];
   VkWriteDescriptorSet wds[5];
   for (unsigned i = 0; i < 5; i++) {
      dii[i] = (VkDescriptorImageInfo){ .imageView = view[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = i,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &dii[i],
      };
   }
   vkUpdateDescriptorSets(dev, 5, wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   for (unsigned i = 0; i < 5; i++) {
      VkImageMemoryBarrier toc = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img[i],
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &toc);
   }
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
   vkCmdDispatch(cmd, 1, 1, 1);
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
   for (unsigned i = 0; i < 5; i++) {
      VkBufferImageCopy region = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { 2, 1, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img[i], VK_IMAGE_LAYOUT_GENERAL, rb[i], 1, &region);
   }
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned failed = 0;
   for (unsigned i = 0; i < 5; i++) {
      uint8_t *v;
      CHECK(vkMapMemory(dev, rmem[i], 0, VK_WHOLE_SIZE, 0, (void **)&v));
      bool ok = true;
      for (unsigned k = 0; k < 2; k++) {
         if (memcmp(v + k * IMGS[i].bpp, IMGS[i].expect[k], IMGS[i].bpp)) {
            ok = false;
            printf("  %s texel %u:", IMGS[i].name, k);
            for (unsigned j = 0; j < IMGS[i].bpp; j++)
               printf(" %02x", v[k * IMGS[i].bpp + j]);
            printf(", expected");
            for (unsigned j = 0; j < IMGS[i].bpp; j++)
               printf(" %02x", IMGS[i].expect[k][j]);
            printf("\n");
         }
      }
      vkUnmapMemory(dev, rmem[i]);
      printf("%s: %s\n", IMGS[i].name, ok ? "pass" : "FAIL");
      if (!ok) failed++;
   }
   if (failed) {
      printf("FAILED (%u image(s))\n", failed);
      return 1;
   }
   printf("PASSED (integer storage-image stores saturate)\n");
   return 0;
}
