/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Float atomics on sparse images several 64 KiB tiles wide: a 300 x 140 2D and
 * a 48 x 48 x 20 3D R32_SFLOAT image, bound whole. 1024 invocations add into
 * four accumulators of each image, every one in a different tile, so 256
 * contended adds meet on each; every invocation also exchanges a texel of its
 * own. The host seeds and checks the images through a mapping of their
 * memory, each texel at its place in its tile; lavapipe stores a tile's texels
 * x, then y, then z.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N    1024u
#define TILE 65536u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static const int32_t acc2[4][2] = { { 5, 1 }, { 140, 2 }, { 290, 3 }, { 200, 130 } };
static const int32_t acc3[4][3] = { { 3, 3, 3 }, { 40, 5, 2 }, { 7, 40, 17 }, { 45, 45, 19 } };

struct img {
   VkImage        image;
   VkImageView    view;
   VkDeviceMemory mem;
   uint8_t       *map;
   uint32_t       w, h, d, gw, gh, gd;
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
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits)
{
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

/* Texel (x, y, z) of `m`, in its tile of the whole-bound memory. */
static float *
at(const struct img *m, uint32_t x, uint32_t y, uint32_t z)
{
   const uint32_t tx = (m->w + m->gw - 1) / m->gw, ty = (m->h + m->gh - 1) / m->gh;
   const uint32_t t = (z / m->gd * ty + y / m->gh) * tx + x / m->gw;
   const uint32_t in = (x % m->gw) + (y % m->gh) * m->gw + (z % m->gd) * m->gw * m->gh;
   return (float *)(m->map + (size_t)t * TILE + in * 4u);
}

static int
make_img(VkPhysicalDevice pd, VkDevice dev, VkQueue q,
         const VkPhysicalDeviceMemoryProperties *mp, VkImageType type, struct img *m)
{
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT;
   uint32_t np = 1;
   VkSparseImageFormatProperties sfp;
   vkGetPhysicalDeviceSparseImageFormatProperties(pd, VK_FORMAT_R32_SFLOAT, type,
      VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &np, &sfp);
   if (np < 1) {
      fprintf(stderr, "FAILED: no sparse residency for R32_SFLOAT\n");
      return 1;
   }
   m->gw = sfp.imageGranularity.width;
   m->gh = sfp.imageGranularity.height;
   m->gd = sfp.imageGranularity.depth;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      .imageType = type, .format = VK_FORMAT_R32_SFLOAT, .extent = { m->w, m->h, m->d },
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(dev, &imci, NULL, &m->image));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, m->image, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = find_mem(mp, mr.memoryTypeBits),
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, &m->mem));
   CHECK(vkMapMemory(dev, m->mem, 0, VK_WHOLE_SIZE, 0, (void **)&m->map));
   memset(m->map, 0, mr.size);
   VkSparseMemoryBind b = { .resourceOffset = 0, .size = mr.size, .memory = m->mem };
   VkSparseImageOpaqueMemoryBindInfo obi = { .image = m->image, .bindCount = 1, .pBinds = &b };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageOpaqueBindCount = 1, .pImageOpaqueBinds = &obi,
   };
   CHECK(vkQueueBindSparse(q, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(q));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = m->image,
      .viewType = type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R32_SFLOAT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   CHECK(vkCreateImageView(dev, &ivci, NULL, &m->view));
   return 0;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "FAILED: usage: %s sa.comp.spv\n", argv[0]);
      return 1;
   }
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_atomic", .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &npd, &pd) < 0 || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &af,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.sparseBinding || !f2.features.sparseResidencyImage2D ||
       !f2.features.sparseResidencyImage3D || !af.shaderImageFloat32Atomics ||
       !af.shaderImageFloat32AtomicAdd || !af.sparseImageFloat32Atomics ||
       !af.sparseImageFloat32AtomicAdd) {
      printf("NotSupported: float atomics on sparse images are not offered\n");
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
   VkPhysicalDeviceShaderAtomicFloatFeaturesEXT waf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .shaderImageFloat32Atomics = VK_TRUE, .shaderImageFloat32AtomicAdd = VK_TRUE,
      .sparseImageFloat32Atomics = VK_TRUE, .sparseImageFloat32AtomicAdd = VK_TRUE,
   };
   VkPhysicalDeviceFeatures want = {
      .sparseBinding = VK_TRUE, .sparseResidencyImage2D = VK_TRUE,
      .sparseResidencyImage3D = VK_TRUE,
   };
   const char *ext = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &waf,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &ext, .pEnabledFeatures = &want,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, qf, 0, &q);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   struct img i2 = { .w = 300, .h = 140, .d = 1 }, i3 = { .w = 48, .h = 48, .d = 20 };
   if (make_img(pd, dev, q, &mp, VK_IMAGE_TYPE_2D, &i2) ||
       make_img(pd, dev, q, &mp, VK_IMAGE_TYPE_3D, &i3)) {
      return 1;
   }
   for (uint32_t i = 0; i < N; i++) {
      *at(&i2, i % 300u, 4u + i / 300u, 0) = -(float)i - 1.0f;
   }

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = N * 4u,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer out;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &out));
   VkMemoryRequirements omr;
   vkGetBufferMemoryRequirements(dev, out, &omr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = omr.size, .memoryTypeIndex = find_mem(&mp, omr.memoryTypeBits),
   };
   VkDeviceMemory omem;
   float *op;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &omem));
   CHECK(vkBindBufferMemory(dev, out, omem, 0));
   CHECK(vkMapMemory(dev, omem, 0, VK_WHOLE_SIZE, 0, (void **)&op));

   VkDescriptorSetLayoutBinding dslb[3] = {
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
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
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
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
   VkDescriptorImageInfo dii[2] = {
      { .imageView = i2.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
      { .imageView = i3.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
   };
   VkDescriptorBufferInfo dbi = { out, 0, N * 4u };
   VkWriteDescriptorSet wds[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 2,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

   size_t sz = 0;
   uint32_t *spv = read_spirv(argv[1], &sz);
   if (!spv) {
      return 1;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(spv);
   VkComputePipelineCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   VkImageMemoryBarrier ib[2];
   for (unsigned k = 0; k < 2; k++) {
      ib[k] = (VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = k ? i3.image : i2.image,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      };
   }
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, ib);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, N / 16u, 1, 1);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(q));

   unsigned bad = 0;
   char first[160] = "";
   for (unsigned k = 0; k < 4; k++) {
      const float s2 = *at(&i2, (uint32_t)acc2[k][0], (uint32_t)acc2[k][1], 0);
      const float s3 = *at(&i3, (uint32_t)acc3[k][0], (uint32_t)acc3[k][1],
                           (uint32_t)acc3[k][2]);
      if ((s2 != (float)(N / 4u) || s3 != 2.0f * (float)(N / 4u)) && !bad++) {
         snprintf(first, sizeof first, "accumulator %u: 2D %g, 3D %g, want %g and %g", k,
                  s2, s3, (float)(N / 4u), 2.0f * (float)(N / 4u));
      }
   }
   for (uint32_t i = 0; i < N; i++) {
      const float now = *at(&i2, i % 300u, 4u + i / 300u, 0);
      if ((op[i] != -(float)i - 1.0f || now != (float)i) && !bad++) {
         snprintf(first, sizeof first, "exchange %u: returned %g, texel %g, want %g and %u",
                  i, op[i], now, -(float)i - 1.0f, i);
      }
   }
   if (bad) {
      printf("FAILED (sparse_atomic: %u results wrong; first %s)\n", bad, first);
      return 1;
   }
   printf("PASSED (sparse_atomic: %u contended float adds on 2D and 3D sparse images, four "
          "accumulators in separate tiles, and %u exchanges)\n", 2u * N, N);
   return 0;
}
