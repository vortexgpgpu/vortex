/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * 4:2:2 packed textures sampled through a sampler Y'CbCr conversion: a
 * G8B8G8R8_422 and a B8G8R8G8_422 image, uploaded through a buffer copy and
 * read by a compute shader at every texel centre with nearest filtering and an
 * identity model. Two texels share one 32-bit word: each has its own luma and
 * both the pair's chroma, so every texel checks the byte its luma comes from
 * (even and odd differ) and the two chroma bytes.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W      32u
#define H      8u
#define TEXELS (W * H)
#define LOCAL  8u

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

static int
alloc_bind_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
                  VkDeviceSize bytes, VkBufferUsageFlags usage,
                  VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   const uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
   return 0;
}

/* The packed word of texel pair p in row y, as bytes in memory order. */
static void
pair_bytes(uint32_t y, uint32_t p, uint8_t b[4])
{
   for (uint32_t i = 0; i < 4; i++)
      b[i] = (uint8_t)((y * 53u + p * 29u + i * 71u + 17u) & 0xffu);
}

/* What the identity model returns for texel (x, y): (Cr, Y, Cb, 1). */
static void
expected(VkFormat fmt, uint32_t x, uint32_t y, float out[4])
{
   uint8_t b[4];
   pair_bytes(y, x / 2u, b);
   const bool odd = (x & 1u) != 0;
   uint8_t luma, cb, cr;
   if (fmt == VK_FORMAT_G8B8G8R8_422_UNORM) {   /* G0 B G1 R */
      luma = odd ? b[2] : b[0];
      cb = b[1];
      cr = b[3];
   } else {                                      /* B G0 R G1 */
      luma = odd ? b[3] : b[1];
      cb = b[0];
      cr = b[2];
   }
   out[0] = cr / 255.0f;
   out[1] = luma / 255.0f;
   out[2] = cb / 255.0f;
   out[3] = 1.0f;
}

static int
run_format(VkPhysicalDevice pd, VkDevice dev, VkQueue queue, uint32_t qf,
           const VkPhysicalDeviceMemoryProperties *mp, VkFormat fmt,
           const char *name, const uint32_t *spv, size_t spv_size)
{
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pd, fmt, &fp);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                     VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT;
   if ((fp.optimalTilingFeatures & need) != need) {
      fprintf(stderr, "FAILED: %s lacks sampled/transfer/midpoint support (optimal 0x%x)\n",
              name, (unsigned)fp.optimalTilingFeatures);
      return 1;
   }

   VkSamplerYcbcrConversionCreateInfo ycci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
      .format = fmt,
      .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
      .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
      .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
      .xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .chromaFilter = VK_FILTER_NEAREST,
   };
   VkSamplerYcbcrConversion conv;
   CHECK(vkCreateSamplerYcbcrConversion(dev, &ycci, NULL, &conv));
   VkSamplerYcbcrConversionInfo yci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, .conversion = conv,
   };

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   uint32_t imt = find_mem(mp, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (imt == UINT32_MAX)
      imt = find_mem(mp, imr.memoryTypeBits, 0);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &yci,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = &yci,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler smp;
   CHECK(vkCreateSampler(dev, &sci, NULL, &smp));

   /* Seed: W/2 packed words per row. */
   const VkDeviceSize seed_bytes = (VkDeviceSize)(W / 2u) * 4u * H;
   VkBuffer seed;
   VkDeviceMemory seed_mem;
   if (alloc_bind_buffer(dev, mp, seed_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &seed, &seed_mem))
      return 1;
   uint8_t *sp;
   CHECK(vkMapMemory(dev, seed_mem, 0, seed_bytes, 0, (void **)&sp));
   for (uint32_t y = 0; y < H; y++)
      for (uint32_t p = 0; p < W / 2u; p++)
         pair_bytes(y, p, sp + (y * (W / 2u) + p) * 4u);
   vkUnmapMemory(dev, seed_mem);

   const VkDeviceSize out_bytes = (VkDeviceSize)TEXELS * 4u * sizeof(float);
   VkBuffer out;
   VkDeviceMemory out_mem;
   if (alloc_bind_buffer(dev, mp, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &out_mem))
      return 1;

   VkDescriptorSetLayoutBinding b[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = &smp },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = b,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
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
   VkDescriptorImageInfo dii = {
      .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkDescriptorBufferInfo dbi = { out, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

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
   VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   VkImageMemoryBarrier tod = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &tod);
   VkBufferImageCopy bic = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 },
   };
   vkCmdCopyBufferToImage(cb, seed, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
   VkImageMemoryBarrier tos = tod;
   tos.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   tos.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   tos.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   tos.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, NULL, 0, NULL, 1, &tos);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   const uint32_t pc[2] = { W, H };
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
   vkCmdDispatch(cb, W / LOCAL, H, 1);
   VkMemoryBarrier toh = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &toh, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   float *o;
   CHECK(vkMapMemory(dev, out_mem, 0, out_bytes, 0, (void **)&o));
   unsigned bad = 0, odd_bad = 0;
   uint32_t first = UINT32_MAX;
   for (uint32_t y = 0; y < H; y++) {
      for (uint32_t x = 0; x < W; x++) {
         float want[4];
         expected(fmt, x, y, want);
         const float *got = o + (y * W + x) * 4u;
         bool ok = true;
         for (int c = 0; c < 4; c++)
            ok = ok && fabsf(got[c] - want[c]) <= 0.5f / 255.0f;
         if (!ok) {
            if (first == UINT32_MAX)
               first = y * W + x;
            bad++;
            odd_bad += x & 1u;
         }
      }
   }
   if (bad) {
      const uint32_t fx = first % W, fy = first / W;
      float want[4];
      expected(fmt, fx, fy, want);
      const float *got = o + first * 4u;
      printf("FAILED (%s: %u/%u texels wrong, %u of them odd; first (%u,%u): "
             "got (%.4f %.4f %.4f %.4f), want (%.4f %.4f %.4f %.4f))\n",
             name, bad, TEXELS, odd_bad, fx, fy, got[0], got[1], got[2], got[3],
             want[0], want[1], want[2], want[3]);
      return 1;
   }
   vkUnmapMemory(dev, out_mem);
   printf("%s: %u texels right\n", name, TEXELS);

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyBuffer(dev, out, NULL);
   vkFreeMemory(dev, out_mem, NULL);
   vkDestroyBuffer(dev, seed, NULL);
   vkFreeMemory(dev, seed_mem, NULL);
   vkDestroySampler(dev, smp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroySamplerYcbcrConversion(dev, conv, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *path = (argc > 1) ? argv[1] : "t422.comp.spv";
   size_t spv_size = 0;
   uint32_t *spv = read_spirv(path, &spv_size);
   if (!spv)
      return 1;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-tex-422",
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

   VkPhysicalDeviceSamplerYcbcrConversionFeatures yf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &yf,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!yf.samplerYcbcrConversion) {
      fprintf(stderr, "FAILED: no samplerYcbcrConversion\n");
      return 1;
   }

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
   VkPhysicalDeviceSamplerYcbcrConversionFeatures yon = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
      .samplerYcbcrConversion = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &yon,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   if (run_format(pd, dev, queue, qf, &mp, VK_FORMAT_G8B8G8R8_422_UNORM, "g8b8g8r8_422",
                  spv, spv_size) ||
       run_format(pd, dev, queue, qf, &mp, VK_FORMAT_B8G8R8G8_422_UNORM, "b8g8r8g8_422",
                  spv, spv_size))
      return 1;

   free(spv);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (tex_422: G8B8G8R8_422 and B8G8R8G8_422, %u texels each, sampled "
          "through a Y'CbCr conversion from compute)\n", TEXELS);
   return 0;
}
