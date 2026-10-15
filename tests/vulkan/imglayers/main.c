/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Storage images with more than one layer: a 2D array, a 3D image, a cube, a
 * 1D array (whose layer is its second coordinate) and a cube declared without a
 * format, which takes the typeless store path. A storage texel buffer, viewed
 * from a non-zero offset, rides along: it has no rows at all, and the view's
 * offset is part of its base.
 *
 * A storage access names a texel by x, y and a third coordinate that is an
 * array layer, a 3D slice or a cube face, and each of those is a whole image
 * stride away. Every texel of the three images is stored and then atomically
 * incremented by the invocation of its own coordinate, and every one is read
 * back: an address that dropped the third coordinate lands every layer on
 * layer 0, which leaves the others at their zero fill and layer 0 holding the
 * last layer's code incremented several times. imageSize must report the
 * layer and slice counts too. The format-less cube is only stored, so its texels
 * hold their code without the increment.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N       4u   /* width and height */
#define LAYERS  3u   /* array layers and 3D slices */
#define FACES   6u

struct img {
   const char     *name;
   VkImageType     type;
   VkImageViewType view_type;
   uint32_t        h, depth, layers;
   VkImageCreateFlags flags;
   uint32_t        which;      /* the shader's code tag */
   uint32_t        nz;         /* layers, slices or faces */
   uint32_t        binding;
   uint32_t        add;        /* the atomic increment the texels carry */
};

#define NIMG 5u
static const struct img IMGS[NIMG] = {
   { "array",   VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, N, 1, LAYERS, 0, 1, LAYERS, 0, 1 },
   { "3d",      VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,       N, LAYERS, 1, 0, 2, LAYERS, 1, 1 },
   { "cube",    VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE,     N, 1, FACES,
     VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 3, FACES, 2, 1 },
   { "array1d", VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, 1, LAYERS, 0, 4, LAYERS, 4, 1 },
   { "cube_nf", VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE,     N, 1, FACES,
     VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 5, FACES, 5, 0 },
};
#define SIZES_BINDING 3u
#define TBUF_BINDING  6u

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
   const char *cs_path = (argc > 1) ? argv[1] : "imglayers.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-imglayers",
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
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!have.shaderStorageImageWriteWithoutFormat) {
      fprintf(stderr, "FAILED: shaderStorageImageWriteWithoutFormat not supported\n");
      return 1;
   }
   VkPhysicalDeviceFeatures want = { .shaderStorageImageWriteWithoutFormat = VK_TRUE };
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

   /* --- the images, a readback buffer each, and the size buffer -------- */
   VkImage img[NIMG];
   VkDeviceMemory imem[NIMG];
   VkImageView view[NIMG];
   VkBuffer rb[NIMG];
   VkDeviceMemory rmem[NIMG];
   for (unsigned i = 0; i < NIMG; i++) {
      const struct img *d = &IMGS[i];
      VkImageCreateInfo imci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = d->flags,
         .imageType = d->type, .format = VK_FORMAT_R32_UINT,
         .extent = { N, d->h, d->depth }, .mipLevels = 1, .arrayLayers = d->layers,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
         .image = img[i], .viewType = d->view_type, .format = VK_FORMAT_R32_UINT,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, d->layers },
      };
      CHECK(vkCreateImageView(dev, &ivci, NULL, &view[i]));
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = (VkDeviceSize)N * d->h * d->nz * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
   /* The texel buffer: N * N elements, viewed from the smallest offset the
    * device allows past zero, zero-filled through the mapping. */
   const VkDeviceSize tb_off = props.limits.minTexelBufferOffsetAlignment > 4
                             ? props.limits.minTexelBufferOffsetAlignment : 4;
   const VkDeviceSize tb_range = (VkDeviceSize)N * N * 4;
   VkBuffer tb;
   VkDeviceMemory tbmem;
   VkBufferView tbview;
   uint32_t *tbmap;
   {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = tb_off + tb_range, .usage = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &tb));
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, tb, &mr);
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size, .memoryTypeIndex = find_mem(&mp, mr.memoryTypeBits, host),
      };
      CHECK(vkAllocateMemory(dev, &mai, NULL, &tbmem));
      CHECK(vkBindBufferMemory(dev, tb, tbmem, 0));
      CHECK(vkMapMemory(dev, tbmem, 0, VK_WHOLE_SIZE, 0, (void **)&tbmap));
      memset(tbmap, 0, (size_t)(tb_off + tb_range));
      VkBufferViewCreateInfo bvci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = tb, .format = VK_FORMAT_R32_UINT, .offset = tb_off, .range = tb_range,
      };
      CHECK(vkCreateBufferView(dev, &bvci, NULL, &tbview));
   }
   VkBuffer sb;
   VkDeviceMemory smem;
   {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 64, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &sb));
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, sb, &mr);
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size, .memoryTypeIndex = find_mem(&mp, mr.memoryTypeBits, host),
      };
      CHECK(vkAllocateMemory(dev, &mai, NULL, &smem));
      CHECK(vkBindBufferMemory(dev, sb, smem, 0));
   }

   /* --- pipeline ------------------------------------------------------- */
   size_t sz = 0;
   uint32_t *spv = read_spirv(cs_path, &sz);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule cs;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &cs));
   free(spv);
   VkDescriptorSetLayoutBinding b[NIMG + 2];
   for (unsigned i = 0; i <= NIMG; i++) {
      b[i] = (VkDescriptorSetLayoutBinding){
         .binding = i < NIMG ? IMGS[i].binding : SIZES_BINDING, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .descriptorType = i < NIMG ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      };
   }
   b[NIMG + 1] = (VkDescriptorSetLayoutBinding){
      .binding = TBUF_BINDING, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = NIMG + 2, .pBindings = b,
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
   VkDescriptorPoolSize dps[3] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, NIMG }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorImageInfo dii[NIMG];
   VkWriteDescriptorSet wds[NIMG + 2];
   for (unsigned i = 0; i < NIMG; i++) {
      dii[i] = (VkDescriptorImageInfo){ .imageView = view[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
         .dstBinding = IMGS[i].binding,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &dii[i],
      };
   }
   VkDescriptorBufferInfo dbi = { sb, 0, 64 };
   wds[NIMG] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = SIZES_BINDING,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi,
   };
   wds[NIMG + 1] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = TBUF_BINDING, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, .pTexelBufferView = &tbview,
   };
   vkUpdateDescriptorSets(dev, NIMG + 2, wds, 0, NULL);

   /* --- record: zero-fill, dispatch, read back -------------------------- */
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
   for (unsigned i = 0; i < NIMG; i++) {
      VkImageMemoryBarrier toc = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img[i],
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, IMGS[i].layers },
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 0, NULL, 0, NULL, 1, &toc);
      VkClearColorValue zero = { .uint32 = { 0, 0, 0, 0 } };
      VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, IMGS[i].layers };
      vkCmdClearColorImage(cmd, img[i], VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rng);
   }
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
   vkCmdDispatch(cmd, 1, 1, FACES);
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
   for (unsigned i = 0; i < NIMG; i++) {
      VkBufferImageCopy region = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, IMGS[i].layers },
         .imageExtent = { N, IMGS[i].h, IMGS[i].depth },
      };
      vkCmdCopyImageToBuffer(cmd, img[i], VK_IMAGE_LAYOUT_GENERAL, rb[i], 1, &region);
   }
   VkMemoryBarrier hb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- check ------------------------------------------------------------ */
   unsigned failed = 0;
   for (unsigned i = 0; i < NIMG; i++) {
      const struct img *d = &IMGS[i];
      uint32_t *v;
      CHECK(vkMapMemory(dev, rmem[i], 0, VK_WHOLE_SIZE, 0, (void **)&v));
      unsigned bad = 0;
      for (uint32_t z = 0; z < d->nz; z++) {
         for (uint32_t y = 0; y < d->h; y++) {
            for (uint32_t x = 0; x < N; x++) {
               const uint32_t want = ((d->which << 24) | (z << 16) | (y << 8) | x) + d->add;
               const uint32_t got = v[(z * d->h + y) * N + x];
               if (got != want && bad++ < 3)
                  printf("  %s (%u,%u,%u): 0x%08x, expected 0x%08x\n", d->name, x, y, z, got, want);
            }
         }
      }
      vkUnmapMemory(dev, rmem[i]);
      printf("%s: %s (%u of %u texels wrong)\n", d->name, bad ? "FAIL" : "pass", bad,
             N * d->h * d->nz);
      if (bad) failed++;
   }
   {
      const uint32_t *tv = (const uint32_t *)((const uint8_t *)tbmap + tb_off);
      unsigned bad = 0;
      for (uint32_t y = 0; y < N; y++) {
         for (uint32_t x = 0; x < N; x++) {
            const uint32_t want = ((6u << 24) | (y << 8) | x) + 1u;
            if (tv[y * N + x] != want && bad++ < 3)
               printf("  texel buffer [%u]: 0x%08x, expected 0x%08x\n", y * N + x,
                      tv[y * N + x], want);
         }
      }
      for (uint32_t k = 0; k < tb_off / 4; k++)
         bad += tbmap[k] != 0;   /* below the view: never written */
      printf("texel buffer: %s (%u of %u elements wrong, view offset %u)\n", bad ? "FAIL" : "pass",
             bad, N * N, (unsigned)tb_off);
      if (bad) failed++;
   }
   int32_t *sz3;
   CHECK(vkMapMemory(dev, smem, 0, 64, 0, (void **)&sz3));
   static const int32_t want_sz[4][3] = {
      { N, N, LAYERS }, { N, N, LAYERS }, { N, N, 0 }, { N, LAYERS, N * N },
   };
   for (unsigned i = 0; i < 4; i++) {
      if (memcmp(&sz3[i * 4], want_sz[i], sizeof want_sz[i])) {
         printf("%s: FAIL (imageSize %d,%d,%d, expected %d,%d,%d)\n", IMGS[i].name,
                sz3[i * 4], sz3[i * 4 + 1], sz3[i * 4 + 2], want_sz[i][0], want_sz[i][1],
                want_sz[i][2]);
         failed++;
      }
   }
   vkUnmapMemory(dev, smem);

   if (failed) {
      printf("FAILED (%u check(s))\n", failed);
      return 1;
   }
   printf("PASSED (layered storage images: every layer, slice and face; a texel buffer)\n");
   return 0;
}
