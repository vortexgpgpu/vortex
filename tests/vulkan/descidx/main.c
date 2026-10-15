/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Descriptor arrays indexed per lane, in a compute shader and then in a fragment
 * shader (a 16x16 attachment-less pass, pixel (x, y) being lane y * 16 + x).
 *
 *   binding 0 "src[4]" -- four read-only storage buffers, src[k][j] = k*1000 + j
 *   binding 1 "dst[4]" -- four storage buffers, seeded POISON
 *   binding 2 "out"    -- N uints, seeded POISON
 *   binding 3 "img[4]" -- four 8x8 R32_UINT storage images, texel j of image k
 *                         seeded k*100000 + j*7
 *   binding 4 "tbo[4]" -- R32_UINT uniform texel buffers viewing src
 *   binding 5 "vsout"  -- 3 uints the draw's vertex shader stores to
 *   binding 6 "rimg[4]" -- img's images again, declared without a format
 *   binding 7 "simg[4]" -- 8x8 R32_UINT sampled images, texel j of k = k*10 + j*3
 *   binding 8 "smp[2]"  -- nearest samplers: smp[0] clamps u, smp[1] repeats it
 *
 * Lane i reads src[i & 3][i >> 2], texel i >> 2 of img[i & 3] (and of rimg[i & 3])
 * and element i >> 2 of tbo[i & 3], and samples simg[i & 3] through smp[1] and
 * simg[2] through smp[i & 1] one period right of that texel, sums them into
 * out[i], then writes i to dst[i & 3][i >> 2] and to that texel of img, so every
 * warp reaches all four descriptors of each array and the index is known only
 * on the device. A descriptor the driver did not relocate reads host memory on
 * the device; a store it does not write back never reaches dst or the image. Both fail the checks below, and src must come back
 * untouched. Both stages run the same checks, with dst, out and the images
 * re-seeded in between. Every buffer is a range of one allocation, so the
 * draw's vertex stores through a fixed descriptor share memory with its
 * fragment stores through indexed ones, and both must come home.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIDE   16u
#define N      (SIDE * SIDE)
#define NBUF   4u
#define PER    (N / NBUF)
#define POISON 0xdeadbeefu
#define IMG_W  8u
#define IMG_H  (PER / IMG_W)

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

static int
make_module(VkDevice dev, const char *path, VkShaderModule *out)
{
   size_t size = 0;
   uint32_t *spv = read_spirv(path, &size);
   if (!spv)
      return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = spv,
   };
   VkResult r = vkCreateShaderModule(dev, &smci, NULL, out);
   free(spv);
   return r != VK_SUCCESS;
}

static uint32_t
host_memory_type(VkPhysicalDevice pd, uint32_t type_bits)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

/* A linear, host-visible storage image the host seeds and checks in place;
 * `*out_pitch` is its row pitch in texels. */
static int
make_image(VkDevice dev, VkPhysicalDevice pd, VkImageUsageFlags usage, VkImage *out_img,
           VkImageView *out_view, VkDeviceMemory *out_mem, uint32_t **out_map,
           uint32_t *out_pitch)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .extent = { IMG_W, IMG_H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
   };
   if (vkCreateImage(dev, &ici, NULL, out_img) != VK_SUCCESS) return 1;
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, *out_img, &mr);
   const uint32_t mt = host_memory_type(pd, mr.memoryTypeBits);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible image memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) return 1;
   if (vkBindImageMemory(dev, *out_img, *out_mem, 0) != VK_SUCCESS) return 1;
   VkImageSubresource sub = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT };
   VkSubresourceLayout lay;
   vkGetImageSubresourceLayout(dev, *out_img, &sub, &lay);
   if (vkMapMemory(dev, *out_mem, 0, VK_WHOLE_SIZE, 0, (void **)out_map) != VK_SUCCESS)
      return 1;
   *out_map += lay.offset / sizeof(uint32_t);
   *out_pitch = (uint32_t)(lay.rowPitch / sizeof(uint32_t));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *out_img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   return vkCreateImageView(dev, &vci, NULL, out_view) != VK_SUCCESS;
}

/* Buffers placed in one host-visible allocation, the way an application's
 * suballocator packs them: buffer i of bytes[i] maps at (*out_map)[i]. */
static int
make_buffers(VkDevice dev, VkPhysicalDevice pd, uint32_t n, const VkDeviceSize *bytes,
             VkBuffer *out_buf, VkDeviceMemory *out_mem, uint32_t **out_map)
{
   VkDeviceSize off[16], total = 0;
   uint32_t type_bits = ~0u;
   if (n > 16) return 1;
   for (uint32_t i = 0; i < n; i++) {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = bytes[i],
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      if (vkCreateBuffer(dev, &bci, NULL, &out_buf[i]) != VK_SUCCESS) return 1;
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, out_buf[i], &mr);
      total = (total + mr.alignment - 1) / mr.alignment * mr.alignment;
      off[i] = total;
      total += mr.size;
      type_bits &= mr.memoryTypeBits;
   }
   const uint32_t mt = host_memory_type(pd, type_bits);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = total, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) return 1;
   uint8_t *map = NULL;
   if (vkMapMemory(dev, *out_mem, 0, VK_WHOLE_SIZE, 0, (void **)&map) != VK_SUCCESS) return 1;
   for (uint32_t i = 0; i < n; i++) {
      if (vkBindBufferMemory(dev, out_buf[i], *out_mem, off[i]) != VK_SUCCESS) return 1;
      out_map[i] = (uint32_t *)(map + off[i]);
   }
   return 0;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-descidx",
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

   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f12.shaderStorageBufferArrayNonUniformIndexing ||
       !f12.shaderStorageImageArrayNonUniformIndexing ||
       !f12.shaderUniformTexelBufferArrayNonUniformIndexing ||
       !f12.shaderUniformTexelBufferArrayDynamicIndexing ||
       !f12.shaderSampledImageArrayNonUniformIndexing ||
       !f2.features.fragmentStoresAndAtomics ||
       !f2.features.vertexPipelineStoresAndAtomics ||
       !f2.features.shaderStorageImageReadWithoutFormat) {
      fprintf(stderr, "FAILED: non-uniform storage-buffer/image/texel-buffer indexing,"
              " vertex/fragment stores or format-less image reads not supported\n");
      return 1;
   }
   VkPhysicalDeviceVulkan12Features en12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
      .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
      .shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
      .shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 en2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &en12,
      .features = { .shaderStorageBufferArrayDynamicIndexing = VK_TRUE,
                    .shaderStorageImageArrayDynamicIndexing = VK_TRUE,
                    .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
                    .vertexPipelineStoresAndAtomics = VK_TRUE,
                    .shaderStorageImageReadWithoutFormat = VK_TRUE,
                    .fragmentStoresAndAtomics = VK_TRUE },
   };

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) ==
          (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics+compute queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &en2,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   /* All buffers share one allocation, so a store through a fixed descriptor
    * and one through an indexed descriptor land in the same memory. */
   const VkDeviceSize per_bytes = PER * sizeof(uint32_t);
   const VkDeviceSize out_bytes = N * sizeof(uint32_t);
   const VkDeviceSize vs_bytes = 3 * sizeof(uint32_t);
   enum { B_SRC = 0, B_DST = NBUF, B_OUT = 2 * NBUF, B_VS, B_COUNT };
   VkDeviceSize bytes[B_COUNT];
   for (uint32_t k = 0; k < NBUF; k++)
      bytes[B_SRC + k] = bytes[B_DST + k] = per_bytes;
   bytes[B_OUT] = out_bytes;
   bytes[B_VS] = vs_bytes;
   VkBuffer bufs[B_COUNT];
   uint32_t *maps[B_COUNT];
   VkDeviceMemory buf_mem;
   if (make_buffers(dev, pd, B_COUNT, bytes, bufs, &buf_mem, maps)) return 1;
   VkBuffer *src_buf = &bufs[B_SRC], *dst_buf = &bufs[B_DST];
   VkBuffer out_buf = bufs[B_OUT], vs_buf = bufs[B_VS];
   uint32_t **src = &maps[B_SRC], **dst = &maps[B_DST];
   uint32_t *out = maps[B_OUT], *vsout = maps[B_VS];
   for (uint32_t k = 0; k < NBUF; k++)
      for (uint32_t j = 0; j < PER; j++) {
         src[k][j] = k * 1000u + j;
         dst[k][j] = POISON;
      }
   VkImage img[NBUF];
   VkImageView img_view[NBUF];
   VkDeviceMemory img_mem[NBUF];
   uint32_t *texel[NBUF], pitch[NBUF];
   for (uint32_t k = 0; k < NBUF; k++)
      if (make_image(dev, pd, VK_IMAGE_USAGE_STORAGE_BIT, &img[k], &img_view[k],
                     &img_mem[k], &texel[k], &pitch[k]))
         return 1;
   /* Sampled only, never written: texel j of simg[k] is k*10 + j*3. */
   VkImage simg[NBUF];
   VkImageView simg_view[NBUF];
   VkDeviceMemory simg_mem[NBUF];
   uint32_t *stexel[NBUF], spitch[NBUF];
   for (uint32_t k = 0; k < NBUF; k++) {
      if (make_image(dev, pd, VK_IMAGE_USAGE_SAMPLED_BIT, &simg[k], &simg_view[k],
                     &simg_mem[k], &stexel[k], &spitch[k]))
         return 1;
      for (uint32_t j = 0; j < PER; j++)
         stexel[k][(j / IMG_W) * spitch[k] + j % IMG_W] = k * 10u + j * 3u;
   }
   VkSampler smp[2];
   for (uint32_t m = 0; m < 2; m++) {
      const VkSamplerAddressMode u = m ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                       : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      VkSamplerCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
         .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
         .addressModeU = u, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      };
      CHECK(vkCreateSampler(dev, &sci, NULL, &smp[m]));
   }
#define TEXEL(k, j) texel[k][((j) / IMG_W) * pitch[k] + (j) % IMG_W]
   VkBufferView tbo_view[NBUF];
   for (uint32_t k = 0; k < NBUF; k++) {
      VkBufferViewCreateInfo bvci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = src_buf[k], .format = VK_FORMAT_R32_UINT, .range = per_bytes,
      };
      CHECK(vkCreateBufferView(dev, &bvci, NULL, &tbo_view[k]));
   }
   for (uint32_t i = 0; i < N; i++)
      out[i] = POISON;

   VkShaderModule cs, vs, fs;
   if (make_module(dev, "descidx.comp.spv", &cs) || make_module(dev, "descidx.vert.spv", &vs) ||
       make_module(dev, "descidx.frag.spv", &fs))
      return 1;

   VkDescriptorSetLayoutBinding dslb[9] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT },
      { .binding = 6, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 7, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = NBUF, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 8, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = 2, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 9, .pBindings = dslb,
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
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main",
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .width = SIDE, .height = SIDE, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport vp = { 0, 0, (float)SIDE, (float)SIDE, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { SIDE, SIDE } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline gpipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &gpipe));

   VkDescriptorPoolSize dps[5] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * NBUF + 2 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 2 * NBUF },
      { .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .descriptorCount = NBUF },
      { .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = NBUF },
      { .type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 2 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 5, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

   VkDescriptorBufferInfo sbi[NBUF], dbi[NBUF];
   for (uint32_t k = 0; k < NBUF; k++) {
      sbi[k] = (VkDescriptorBufferInfo){ .buffer = src_buf[k], .range = per_bytes };
      dbi[k] = (VkDescriptorBufferInfo){ .buffer = dst_buf[k], .range = per_bytes };
   }
   VkDescriptorBufferInfo obi = { .buffer = out_buf, .range = out_bytes };
   VkDescriptorBufferInfo vbi = { .buffer = vs_buf, .range = vs_bytes };
   VkDescriptorImageInfo iii[NBUF];
   for (uint32_t k = 0; k < NBUF; k++)
      iii[k] = (VkDescriptorImageInfo){ .imageView = img_view[k],
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorImageInfo sii[NBUF], smi[2];
   for (uint32_t k = 0; k < NBUF; k++)
      sii[k] = (VkDescriptorImageInfo){ .imageView = simg_view[k],
                                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   for (uint32_t m = 0; m < 2; m++)
      smi[m] = (VkDescriptorImageInfo){ .sampler = smp[m] };
   VkWriteDescriptorSet wds[9] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = sbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 2,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &obi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 3,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = iii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 4,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
        .pTexelBufferView = tbo_view },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 5,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &vbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 6,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = iii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 7,
        .descriptorCount = NBUF, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = sii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 8,
        .descriptorCount = 2, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = smi },
   };
   vkUpdateDescriptorSets(dev, 9, wds, 0, NULL);

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
   static const char *const stage_name[2] = { "compute", "fragment" };
   unsigned failed = 0;
   for (uint32_t st = 0; st < 2; st++) {
      for (uint32_t k = 0; k < NBUF; k++)
         for (uint32_t j = 0; j < PER; j++) {
            dst[k][j] = POISON;
            TEXEL(k, j) = k * 100000u + j * 7u;
         }
      for (uint32_t i = 0; i < N; i++)
         out[i] = POISON;
      for (uint32_t v = 0; v < 3; v++)
         vsout[v] = POISON;

      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      /* The host's seed, made visible to the shader, in the layout the
       * descriptors name. */
      VkImageMemoryBarrier imb[2 * NBUF];
      for (uint32_t k = 0; k < 2 * NBUF; k++)
         imb[k] = (VkImageMemoryBarrier){
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = st == 0 ? VK_IMAGE_LAYOUT_PREINITIALIZED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = k < NBUF ? img[k] : simg[k - NBUF],
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           0, 0, NULL, 0, NULL, 2 * NBUF, imb);
      if (st == 0) {
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
         vkCmdDispatch(cb, N / 16u, 1, 1);
      } else {
         VkRenderPassBeginInfo rpbi = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = rp, .framebuffer = fb,
            .renderArea = { { 0, 0 }, { SIDE, SIDE } },
         };
         vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gpipe);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
         vkCmdDraw(cb, 3, 1, 0, 0);
         vkCmdEndRenderPass(cb);
      }
      VkMemoryBarrier hmb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hmb, 0, NULL, 0, NULL);
      CHECK(vkEndCommandBuffer(cb));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cb,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      unsigned out_fails = 0, dst_fails = 0, img_fails = 0, src_fails = 0;
      for (uint32_t i = 0; i < N; i++) {
         const uint32_t k = i & 3u, j = i >> 2;
         /* simg[k] through the repeating sampler: column j & 7; simg[2] through
          * smp[k & 1]: the same, or the last column when it clamps. */
         const uint32_t jc = (k & 1u) ? j : (j & ~7u) + 7u;
         const uint32_t want = 2u * (k * 1000u + j) + 2u * (k * 100000u + j * 7u) +
                               (k * 10u + j * 3u) + (20u + jc * 3u);
         if (out[i] != want && out_fails++ < 5)
            fprintf(stderr, "  %s: out[%u] = 0x%x, want %u\n", stage_name[st], i, out[i],
                    want);
         if (dst[k][j] != i && dst_fails++ < 5)
            fprintf(stderr, "  %s: dst[%u][%u] = 0x%x, want %u\n", stage_name[st], k, j,
                    dst[k][j], i);
         if (TEXEL(k, j) != i && img_fails++ < 5)
            fprintf(stderr, "  %s: img[%u] texel %u = 0x%x, want %u\n", stage_name[st], k,
                    j, TEXEL(k, j), i);
         if (src[k][j] != k * 1000u + j)
            src_fails++;
      }
      /* The vertex stage stores through a fixed descriptor in the draw whose
       * fragment stage stores through indexed ones, into the same allocation. */
      unsigned vs_fails = 0;
      for (uint32_t v = 0; st == 1 && v < 3; v++)
         if (vsout[v] != 0x1000u + v && vs_fails++ < 3)
            fprintf(stderr, "  vertex: vsout[%u] = 0x%x, want 0x%x\n", v, vsout[v], 0x1000u + v);
      if (vs_fails)
         printf("vertex: FAIL  %u/3 fixed-descriptor stores lost\n", vs_fails);
      const bool ok = !out_fails && !dst_fails && !img_fails && !src_fails && !vs_fails;
      printf("%s: %s  %u/%u indexed loads, %u/%u buffer and %u/%u image indexed stores"
             " wrong, %u read-only words changed\n", stage_name[st], ok ? "pass" : "FAIL",
             out_fails, N, dst_fails, N, img_fails, N, src_fails);
      failed += !ok;
   }

   if (failed) {
      printf("FAILED (descidx: %u stage(s))\n", failed);
      return 1;
   }
   printf("PASSED (descidx: buffer, texel-buffer, image, format-less image and"
          " texture/sampler accesses through per-lane descriptor indices, compute and"
          " fragment; vertex stores beside them)\n");
   return 0;
}
