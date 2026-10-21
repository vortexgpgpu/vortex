/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * sparseImageLoadARB on ordinary (non-sparse) storage images.
 *
 * A sparse image load returns the texel plus a residency code, and an ordinary
 * image is resident everywhere, so every texel must come back with its seeded
 * value and a code sparseTexelsResidentARB reports resident. Each case covers
 * one way the driver reaches the texel: a 2D, 2D array, cube and 3D R32_UINT
 * view through the typed load, an RGBA8 view through its packed decode, and an
 * R32_UINT view declared without a format through the format-less load.
 *
 * Each case seeds its image through a buffer copy with its own pattern, so a
 * case that reads a stale texel or a texel of the wrong layer is caught.
 *
 * A last case reaches a sparse image: a fully bound R32_UINT 2D array one 64 KiB
 * tile wide and three tall, read and written through a view of its second
 * layer. Its tiled layout matches linear addressing, so it runs on the device,
 * and a view that does not start at the image's origin catches an address that
 * adds the view's offset twice.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL 4u
#define WANT_RESIDENT 1u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

struct sl_case {
   const char         *name;
   VkImageType         type;
   VkImageViewType     view;
   VkFormat            format;
   VkImageCreateFlags  flags;
   uint32_t            w, h, depth, layers;
   uint32_t            salt;
};

/* In the order of the SHADERS list, which is the order of argv. */
static const struct sl_case cases[] = {
   { "2d",       VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,       VK_FORMAT_R32_UINT,
     0, 20, 12, 1, 1, 0x2d000000u },
   { "2darray",  VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R32_UINT,
     0, 12, 8, 1, 3, 0x2a000000u },
   { "cube",     VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE,     VK_FORMAT_R32_UINT,
     VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 8, 8, 1, 6, 0xcb000000u },
   { "3d",       VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,       VK_FORMAT_R32_UINT,
     0, 8, 12, 5, 1, 0x3d000000u },
   { "rgba8",    VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,       VK_FORMAT_R8G8B8A8_UNORM,
     0, 16, 12, 1, 1, 0x8a000000u },
   { "noformat", VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,       VK_FORMAT_R32_UINT,
     0, 20, 12, 1, 1, 0x0f000000u },
};
#define NUM_CASES (sizeof(cases) / sizeof(cases[0]))

/* The sparse case: R32_UINT, whose 64 KiB tile is 128 x 128 texels. */
#define SP_W      128u
#define SP_H      384u
#define SP_LAYERS 2u
#define SP_LAYER  1u

struct ctx {
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice        dev;
   VkQueue         queue;
   VkCommandBuffer cb;
   VkDescriptorSetLayout dsl;
   VkPipelineLayout      pl;
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

static bool
make_buffer(struct ctx *c, VkDeviceSize bytes, VkBufferUsageFlags usage,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   uint32_t mt = find_mem(&c->mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return false; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(c->dev, *buf, *mem, 0));
   return true;
}

/* The seeded value of texel i: distinct per texel and per case. */
static uint32_t
texel_value(const struct sl_case *k, uint32_t i)
{
   return k->salt ^ (i * 2654435761u);
}

static bool
run_case(struct ctx *c, const struct sl_case *k, const char *spv_path)
{
   const uint32_t slices = k->depth > 1 ? k->depth : k->layers;
   const uint32_t texels = k->w * k->h * slices;

   size_t sz = 0;
   uint32_t *spv = read_spirv(spv_path, &sz);
   if (!spv) return false;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   free(spv);
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = c->pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   vkDestroyShaderModule(c->dev, sm, NULL);

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = k->flags, .imageType = k->type, .format = k->format,
      .extent = { k->w, k->h, k->depth }, .mipLevels = 1, .arrayLayers = k->layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(c->dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, img, &imr);
   uint32_t imt = find_mem(&c->mp, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (imt == UINT32_MAX)
      imt = find_mem(&c->mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return false; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(c->dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(c->dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = k->view, .format = k->format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, k->layers },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &view));

   const VkDeviceSize seed_bytes = (VkDeviceSize)texels * sizeof(uint32_t);
   const VkDeviceSize out_bytes = 2u * seed_bytes;
   VkBuffer seed, out;
   VkDeviceMemory seed_mem, out_mem;
   if (!make_buffer(c, seed_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &seed, &seed_mem) ||
       !make_buffer(c, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &out_mem))
      return false;
   uint32_t *p;
   CHECK(vkMapMemory(c->dev, seed_mem, 0, seed_bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < texels; i++)
      p[i] = texel_value(k, i);
   vkUnmapMemory(c->dev, seed_mem);
   CHECK(vkMapMemory(c->dev, out_mem, 0, out_bytes, 0, (void **)&p));
   memset(p, 0xa5, (size_t)out_bytes);
   vkUnmapMemory(c->dev, out_mem);

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &c->dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, &ds));
   VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorBufferInfo dbi = { .buffer = out, .offset = 0, .range = out_bytes };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 2, wds, 0, NULL);

   CHECK(vkResetCommandBuffer(c->cb, 0));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(c->cb, &cbbi));
   VkImageMemoryBarrier tob = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, k->layers },
   };
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tob);
   VkBufferImageCopy region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, k->layers },
      .imageExtent = { k->w, k->h, k->depth },
   };
   vkCmdCopyBufferToImage(c->cb, seed, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &region);
   VkImageMemoryBarrier togen = tob;
   togen.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   togen.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   togen.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   togen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &togen);
   vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &ds, 0, NULL);
   const uint32_t extent[3] = { k->w, k->h, slices };
   vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof extent, extent);
   vkCmdDispatch(c->cb, (k->w + LOCAL - 1) / LOCAL, (k->h + LOCAL - 1) / LOCAL, slices);
   VkMemoryBarrier tohost = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &tohost, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(c->cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &c->cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   unsigned bad_value = 0, bad_code = 0;
   uint32_t first = UINT32_MAX, got_v = 0, got_r = 0;
   CHECK(vkMapMemory(c->dev, out_mem, 0, out_bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < texels; i++) {
      const bool v_ok = p[2u * i] == texel_value(k, i);
      const bool r_ok = p[2u * i + 1u] == WANT_RESIDENT;
      bad_value += !v_ok;
      bad_code += !r_ok;
      if ((!v_ok || !r_ok) && first == UINT32_MAX) {
         first = i; got_v = p[2u * i]; got_r = p[2u * i + 1u];
      }
   }
   vkUnmapMemory(c->dev, out_mem);

   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyImage(c->dev, img, NULL);
   vkFreeMemory(c->dev, imem, NULL);
   vkDestroyBuffer(c->dev, seed, NULL);
   vkFreeMemory(c->dev, seed_mem, NULL);
   vkDestroyBuffer(c->dev, out, NULL);
   vkFreeMemory(c->dev, out_mem, NULL);

   if (bad_value || bad_code) {
      printf("FAILED (%s: %u/%u texels wrong, %u/%u residency codes wrong; first at %u: "
             "got 0x%08x resident=%u, want 0x%08x resident=%u)\n",
             k->name, bad_value, texels, bad_code, texels, first, got_v, got_r,
             texel_value(k, first), WANT_RESIDENT);
      return false;
   }
   printf("%s: %u texels, values and residency codes correct\n", k->name, texels);
   return true;
}

static uint32_t
sparse_value(uint32_t layer, uint32_t i)
{
   return 0x5a000000u ^ (layer << 20) ^ (i * 40503u);
}

/* imageLoad + imageStore of every texel of layer SP_LAYER of a fully bound
 * sparse 2D array through a 2D view of that layer; layer 0 must keep its seed. */
static bool
run_sparse_layer(struct ctx *c, const char *spv_path)
{
   const uint32_t texels = SP_W * SP_H;
   size_t sz = 0;
   uint32_t *spv = read_spirv(spv_path, &sz);
   if (!spv) return false;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   free(spv);
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = c->pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   vkDestroyShaderModule(c->dev, sm, NULL);

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .extent = { SP_W, SP_H, 1 }, .mipLevels = 1, .arrayLayers = SP_LAYERS,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(c->dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, img, &imr);
   uint32_t imt = find_mem(&c->mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no sparse image memory\n"); return false; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(c->dev, &imai, NULL, &imem));
   VkSparseMemoryBind bind = { .resourceOffset = 0, .size = imr.size, .memory = imem };
   VkSparseImageOpaqueMemoryBindInfo obi = { .image = img, .bindCount = 1, .pBinds = &bind };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageOpaqueBindCount = 1, .pImageOpaqueBinds = &obi,
   };
   CHECK(vkQueueBindSparse(c->queue, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, SP_LAYER, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &view));

   const VkDeviceSize bytes = (VkDeviceSize)texels * SP_LAYERS * sizeof(uint32_t);
   VkBuffer stage, dummy;
   VkDeviceMemory stage_mem, dummy_mem;
   if (!make_buffer(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, &stage, &stage_mem) ||
       !make_buffer(c, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dummy, &dummy_mem))
      return false;
   uint32_t *p;
   CHECK(vkMapMemory(c->dev, stage_mem, 0, bytes, 0, (void **)&p));
   for (uint32_t l = 0; l < SP_LAYERS; l++)
      for (uint32_t i = 0; i < texels; i++)
         p[l * texels + i] = sparse_value(l, i);
   vkUnmapMemory(c->dev, stage_mem);

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &c->dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, &ds));
   VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorBufferInfo dbi = { .buffer = dummy, .offset = 0, .range = 16 };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 2, wds, 0, NULL);

   CHECK(vkResetCommandBuffer(c->cb, 0));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(c->cb, &cbbi));
   VkImageMemoryBarrier tob = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, SP_LAYERS },
   };
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tob);
   VkBufferImageCopy region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, SP_LAYERS },
      .imageExtent = { SP_W, SP_H, 1 },
   };
   vkCmdCopyBufferToImage(c->cb, stage, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &region);
   VkImageMemoryBarrier togen = tob;
   togen.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   togen.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
   togen.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   togen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &togen);
   vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &ds, 0, NULL);
   const uint32_t extent[3] = { SP_W, SP_H, 1 };
   vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof extent, extent);
   vkCmdDispatch(c->cb, SP_W / LOCAL, SP_H / LOCAL, 1);
   VkImageMemoryBarrier tosrc = tob;
   tosrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   tosrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   tosrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
   tosrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tosrc);
   vkCmdCopyImageToBuffer(c->cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stage, 1, &region);
   CHECK(vkEndCommandBuffer(c->cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &c->cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   unsigned bad = 0;
   uint32_t first = UINT32_MAX, got = 0, want_first = 0;
   CHECK(vkMapMemory(c->dev, stage_mem, 0, bytes, 0, (void **)&p));
   for (uint32_t l = 0; l < SP_LAYERS; l++)
      for (uint32_t i = 0; i < texels; i++) {
         const uint32_t want = l == SP_LAYER ? sparse_value(l, i) * 2u + 1u : sparse_value(l, i);
         if (p[l * texels + i] != want) {
            if (first == UINT32_MAX) {
               first = l * texels + i; got = p[l * texels + i]; want_first = want;
            }
            bad++;
         }
      }
   vkUnmapMemory(c->dev, stage_mem);

   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyImage(c->dev, img, NULL);
   vkFreeMemory(c->dev, imem, NULL);
   vkDestroyBuffer(c->dev, stage, NULL);
   vkFreeMemory(c->dev, stage_mem, NULL);
   vkDestroyBuffer(c->dev, dummy, NULL);
   vkFreeMemory(c->dev, dummy_mem, NULL);

   if (bad) {
      printf("FAILED (sparse layer %u view: %u/%u texels wrong; first at layer %u texel %u: "
             "got 0x%08x, want 0x%08x)\n", SP_LAYER, bad, texels * SP_LAYERS,
             first / texels, first % texels, got, want_first);
      return false;
   }
   printf("sparse: %u texels of a %ux%ux%u sparse array through a layer-%u view, "
          "layer 0 untouched\n", texels, SP_W, SP_H, SP_LAYERS, SP_LAYER);
   return true;
}

int
main(int argc, char **argv)
{
   if (argc < 2 + (int)NUM_CASES) {
      fprintf(stderr, "FAILED: usage: %s <%u shader .spv files>\n", argv[0],
              (unsigned)NUM_CASES + 1u);
      return 1;
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-imgsparse",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
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

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!have.shaderResourceResidency || !have.shaderStorageImageReadWithoutFormat ||
       !have.sparseBinding) {
      printf("NotSupported: shaderResourceResidency, shaderStorageImageReadWithoutFormat "
             "or sparseBinding is not offered by this device\n");
      return 0;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs both compute and sparse binding\n");
      return 0;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures want = {
      .shaderResourceResidency = VK_TRUE,
      .shaderStorageImageReadWithoutFormat = VK_TRUE,
      .sparseBinding = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
   };
   struct ctx c;
   memset(&c, 0, sizeof c);
   if (vkCreateDevice(pd, &dci, NULL, &c.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c.mp);

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 3 * sizeof(uint32_t),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &c.dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   if (vkCreateDescriptorSetLayout(c.dev, &dslci, NULL, &c.dsl) != VK_SUCCESS ||
       vkCreatePipelineLayout(c.dev, &plci, NULL, &c.pl) != VK_SUCCESS ||
       vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: layout or command pool creation\n");
      return 1;
   }
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (vkAllocateCommandBuffers(c.dev, &cbai, &c.cb) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkAllocateCommandBuffers\n");
      return 1;
   }

   for (unsigned i = 0; i < NUM_CASES; i++)
      if (!run_case(&c, &cases[i], argv[1 + i]))
         return 1;
   if (!run_sparse_layer(&c, argv[1 + NUM_CASES]))
      return 1;

   vkDestroyCommandPool(c.dev, cp, NULL);
   vkDestroyPipelineLayout(c.dev, c.pl, NULL);
   vkDestroyDescriptorSetLayout(c.dev, c.dsl, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("PASSED (imgsparse: sparseImageLoadARB on %u image shapes, every texel "
          "resident with its seeded value; load/store through a sparse layer view)\n",
          (unsigned)NUM_CASES);
   return 0;
}
