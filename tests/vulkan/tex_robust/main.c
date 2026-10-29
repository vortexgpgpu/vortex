/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Robust texel fetches on a device with robustImageAccess2 and nullDescriptor,
 * in three compute dispatches.
 *
 * Probes (trobust.comp): texelFetch inside and outside an R32_UINT 2D array
 * (4x4, 2 layers, 3 levels), an R32_UINT 3D image (4x4x4, 3 levels), an R32_UINT
 * texel buffer of 8 elements, an R32_UINT 1D array (4 wide, 2 layers), an R32_UINT
 * 4-sample image cleared to 77, and a BC1 RGBA image; a fetch through a null
 * texel buffer; textureSize of a null sampled image and a null texel buffer.
 * Every probe inside reads its texel; outside -- past a row or column, the last
 * layer, a level's last 3D slice, the last level, the last sample, the buffer's
 * end, or at a negative coordinate -- it reads (0, 0, 0, 1), or (0, 0, 0, 0) for
 * the BC1 RGBA format, which has alpha. A null descriptor reads (0, 0, 0, 0) and
 * has size 0.
 *
 * A texel buffer slot nulled through a template (tbnull.comp): the shader's only
 * texture is element 1 of a 2-element uniform texel buffer binding, written live
 * and then rewritten to null through a descriptor update template at
 * dstArrayElement 1. It must read (0, 0, 0, 0) and have size 0.
 *
 * More than 16 (texture, sampler) pairs (pairs.comp): five 1x1 textures each
 * sampled with each of four samplers, 20 pairs at fixed slots, all on the device.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AW 4u
#define AH 4u
#define AL 2u
#define VW 4u
#define VH 4u
#define VD 4u
#define LEVELS 3u
#define TB_ELEMS 8u
#define W1 4u
#define L1 2u
#define MS_W 2u
#define MS_SAMPLES 4u
#define MS_VAL 77u
#define NTEX 5u
#define NSMP 4u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

enum { K_ARR, K_VOL, K_TB, K_NULL, K_ARR1D, K_MS, K_BC, K_NULLIMG_SIZE, K_NULLTB_SIZE };

struct probe { int32_t kind, x, y, z, lod; };

static const struct probe probes[] = {
   /* the 2D array, inside */
   { K_ARR, 0, 0, 0, 0 }, { K_ARR, 3, 3, 1, 0 }, { K_ARR, 1, 1, 1, 1 }, { K_ARR, 0, 0, 1, 2 },
   /* outside: column, row, layer, level, negative, the smaller level's column */
   { K_ARR, 4, 0, 0, 0 }, { K_ARR, 0, 4, 0, 0 }, { K_ARR, 0, 0, 2, 0 },
   { K_ARR, 0, 0, 0, 3 }, { K_ARR, -1, 0, 0, 0 }, { K_ARR, 2, 0, 0, 1 },
   /* the 3D image, inside */
   { K_VOL, 3, 3, 3, 0 }, { K_VOL, 1, 1, 1, 1 }, { K_VOL, 0, 0, 0, 2 },
   /* outside: a slice at level 0; slice 2 at level 1, where the depth is 2 but
    * the base level's is 4; a column */
   { K_VOL, 0, 0, 4, 0 }, { K_VOL, 0, 0, 2, 1 }, { K_VOL, 4, 0, 0, 0 },
   /* the texel buffer: inside, past the end, negative */
   { K_TB, 0, 0, 0, 0 }, { K_TB, 7, 0, 0, 0 }, { K_TB, 8, 0, 0, 0 },
   { K_TB, 100, 0, 0, 0 }, { K_TB, -1, 0, 0, 0 },
   /* the null texel buffer */
   { K_NULL, 0, 0, 0, 0 }, { K_NULL, 5, 0, 0, 0 },
   /* the 1D array: inside, past the last layer, past the row */
   { K_ARR1D, 3, 1, 0, 0 }, { K_ARR1D, 0, 2, 0, 0 }, { K_ARR1D, 4, 0, 0, 0 },
   /* the 4-sample image: the last sample, past it, past the row */
   { K_MS, 1, 1, 3, 0 }, { K_MS, 0, 0, 4, 0 }, { K_MS, 2, 0, 0, 0 },
   /* BC1 RGBA: inside, and outside, where alpha is 0 */
   { K_BC, 3, 3, 0, 0 }, { K_BC, 4, 0, 0, 0 },
   /* sizes of the null sampled image and the null texel buffer */
   { K_NULLIMG_SIZE, 0, 0, 0, 0 }, { K_NULLTB_SIZE, 0, 0, 0, 0 },
};
#define NPROBES (sizeof(probes) / sizeof(probes[0]))

static uint32_t arr_val(uint32_t x, uint32_t y, uint32_t L, uint32_t l)
{
   return 1u + x + 10u * y + 100u * L + 1000u * l;
}

static uint32_t vol_val(uint32_t x, uint32_t y, uint32_t z, uint32_t l)
{
   return 5001u + x + 10u * y + 100u * z + 1000u * l;
}

static uint32_t tb_val(uint32_t i) { return 7000u + i; }

static uint32_t arr1d_val(uint32_t x, uint32_t L) { return 9001u + x + 10u * L; }

static uint32_t lvl(uint32_t n, int32_t l) { return (n >> l) ? (n >> l) : 1u; }

static bool in_range(int32_t v, uint32_t n) { return v >= 0 && v < (int32_t)n; }

static const uint32_t F1 = 0x3f800000u;   /* 1.0f */

static void expected(const struct probe *p, uint32_t out[4])
{
   out[0] = out[1] = out[2] = 0;
   out[3] = 1;
   const bool lod_in = in_range(p->lod, LEVELS);
   switch (p->kind) {
   case K_ARR:
      if (lod_in && in_range(p->x, lvl(AW, p->lod)) && in_range(p->y, lvl(AH, p->lod)) &&
          in_range(p->z, AL))
         out[0] = arr_val(p->x, p->y, p->z, p->lod);
      break;
   case K_VOL:
      if (lod_in && in_range(p->x, lvl(VW, p->lod)) && in_range(p->y, lvl(VH, p->lod)) &&
          in_range(p->z, lvl(VD, p->lod)))
         out[0] = vol_val(p->x, p->y, p->z, p->lod);
      break;
   case K_TB:
      if (in_range(p->x, TB_ELEMS))
         out[0] = tb_val(p->x);
      break;
   case K_ARR1D:
      if (in_range(p->x, W1) && in_range(p->y, L1))
         out[0] = arr1d_val(p->x, p->y);
      break;
   case K_MS:
      if (in_range(p->x, MS_W) && in_range(p->y, MS_W) && in_range(p->z, MS_SAMPLES))
         out[0] = MS_VAL;
      break;
   case K_BC:
      /* The block is white and opaque; outside, transparent black. */
      if (in_range(p->x, 4) && in_range(p->y, 4)) {
         out[0] = out[1] = out[2] = out[3] = F1;
      } else {
         out[3] = 0;
      }
      break;
   default:
      out[3] = 0;   /* a null descriptor, and a size */
      break;
   }
}

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

struct ctx {
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
   VkCommandPool cp;
};

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
alloc_bind_buffer(const struct ctx *c, VkDeviceSize bytes, VkBufferUsageFlags usage,
                  VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   const uint32_t mt = find_mem(&c->mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(c->dev, *buf, *mem, 0));
   return 0;
}

static int
make_image(const struct ctx *c, VkImageType type, VkFormat fmt, VkExtent3D ext,
           uint32_t levels, uint32_t layers, VkSampleCountFlagBits samples,
           VkImage *img, VkDeviceMemory *mem)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = type, .format = fmt, .extent = ext,
      .mipLevels = levels, .arrayLayers = layers,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &ici, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, *img, &mr);
   uint32_t mt = find_mem(&c->mp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (mt == UINT32_MAX)
      mt = find_mem(&c->mp, mr.memoryTypeBits, 0);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(c->dev, *img, *mem, 0));
   return 0;
}

static int
make_view(const struct ctx *c, VkImage img, VkImageViewType type, VkFormat fmt,
          uint32_t levels, uint32_t layers, VkImageView *view)
{
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = type, .format = fmt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, view));
   return 0;
}

static int
make_pipeline(const struct ctx *c, const VkDescriptorSetLayoutBinding *b, uint32_t nb,
              const uint32_t *spv, size_t spv_size, VkDescriptorSetLayout *dsl,
              VkPipelineLayout *pl, VkPipeline *pipe)
{
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nb, .pBindings = b,
   };
   CHECK(vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = dsl,
   };
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, pl));
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = *pl,
   };
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, pipe));
   vkDestroyShaderModule(c->dev, sm, NULL);
   return 0;
}

static int
alloc_set(const struct ctx *c, VkDescriptorSetLayout dsl, const VkDescriptorPoolSize *dps,
          uint32_t nps, VkDescriptorPool *dp, VkDescriptorSet *ds)
{
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = nps, .pPoolSizes = dps,
   };
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = *dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, ds));
   return 0;
}

static int
begin(const struct ctx *c, VkCommandBuffer *cb)
{
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = c->cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, cb));
   VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(*cb, &cbbi));
   return 0;
}

static int
submit(const struct ctx *c, VkCommandBuffer cb)
{
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
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   vkFreeCommandBuffers(c->dev, c->cp, 1, &cb);
   return 0;
}

static void
to_layout(VkCommandBuffer cb, VkImage img, uint32_t levels, uint32_t layers,
          VkImageLayout from, VkImageLayout to)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = to == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ? VK_ACCESS_TRANSFER_WRITE_BIT
                                                                  : VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, NULL, 0, NULL, 1, &b);
}

/* ---- the probes ---------------------------------------------------------- */

static int
run_probes(const struct ctx *c, VkSampler smp, const uint32_t *spv, size_t spv_size)
{
   VkImage arr, vol, a1d, ms, bc;
   VkDeviceMemory arr_mem, vol_mem, a1d_mem, ms_mem, bc_mem;
   if (make_image(c, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, (VkExtent3D){ AW, AH, 1 }, LEVELS,
                  AL, VK_SAMPLE_COUNT_1_BIT, &arr, &arr_mem) ||
       make_image(c, VK_IMAGE_TYPE_3D, VK_FORMAT_R32_UINT, (VkExtent3D){ VW, VH, VD }, LEVELS,
                  1, VK_SAMPLE_COUNT_1_BIT, &vol, &vol_mem) ||
       make_image(c, VK_IMAGE_TYPE_1D, VK_FORMAT_R32_UINT, (VkExtent3D){ W1, 1, 1 }, 1, L1,
                  VK_SAMPLE_COUNT_1_BIT, &a1d, &a1d_mem) ||
       make_image(c, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, (VkExtent3D){ MS_W, MS_W, 1 }, 1, 1,
                  VK_SAMPLE_COUNT_4_BIT, &ms, &ms_mem) ||
       make_image(c, VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, (VkExtent3D){ 4, 4, 1 },
                  1, 1, VK_SAMPLE_COUNT_1_BIT, &bc, &bc_mem))
      return 1;

   /* The seed: every level of every layer and slice, packed, then the 1D array
    * and the BC1 block (white color0, black color1, all indices 0). */
   VkBufferImageCopy arr_bic[LEVELS], vol_bic[LEVELS];
   uint32_t words = 0;
   for (uint32_t l = 0; l < LEVELS; l++) {
      arr_bic[l] = (VkBufferImageCopy){
         .bufferOffset = words * 4u,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, AL },
         .imageExtent = { lvl(AW, l), lvl(AH, l), 1 },
      };
      words += lvl(AW, l) * lvl(AH, l) * AL;
   }
   for (uint32_t l = 0; l < LEVELS; l++) {
      vol_bic[l] = (VkBufferImageCopy){
         .bufferOffset = words * 4u,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1 },
         .imageExtent = { lvl(VW, l), lvl(VH, l), lvl(VD, l) },
      };
      words += lvl(VW, l) * lvl(VH, l) * lvl(VD, l);
   }
   const VkBufferImageCopy a1d_bic = {
      .bufferOffset = words * 4u,
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, L1 },
      .imageExtent = { W1, 1, 1 },
   };
   words += W1 * L1;
   const VkBufferImageCopy bc_bic = {
      .bufferOffset = words * 4u,
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { 4, 4, 1 },
   };
   words += 2;
   VkBuffer seed;
   VkDeviceMemory seed_mem;
   if (alloc_bind_buffer(c, words * 4u, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &seed, &seed_mem))
      return 1;
   uint32_t *sp;
   CHECK(vkMapMemory(c->dev, seed_mem, 0, words * 4u, 0, (void **)&sp));
   for (uint32_t l = 0; l < LEVELS; l++) {
      uint32_t *w = sp + arr_bic[l].bufferOffset / 4u;
      for (uint32_t L = 0; L < AL; L++)
         for (uint32_t y = 0; y < lvl(AH, l); y++)
            for (uint32_t x = 0; x < lvl(AW, l); x++)
               *w++ = arr_val(x, y, L, l);
      w = sp + vol_bic[l].bufferOffset / 4u;
      for (uint32_t z = 0; z < lvl(VD, l); z++)
         for (uint32_t y = 0; y < lvl(VH, l); y++)
            for (uint32_t x = 0; x < lvl(VW, l); x++)
               *w++ = vol_val(x, y, z, l);
   }
   uint32_t *w = sp + a1d_bic.bufferOffset / 4u;
   for (uint32_t L = 0; L < L1; L++)
      for (uint32_t x = 0; x < W1; x++)
         *w++ = arr1d_val(x, L);
   w = sp + bc_bic.bufferOffset / 4u;
   w[0] = 0x0000ffffu;   /* color0 white, color1 black: four-color mode */
   w[1] = 0u;            /* every texel color0 */
   vkUnmapMemory(c->dev, seed_mem);

   VkBuffer tb, pb, ob;
   VkDeviceMemory tb_mem, pb_mem, ob_mem;
   const VkDeviceSize pbytes = NPROBES * 2u * 4u * sizeof(int32_t);
   const VkDeviceSize obytes = NPROBES * 4u * sizeof(uint32_t);
   if (alloc_bind_buffer(c, TB_ELEMS * 4u, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                         &tb, &tb_mem) ||
       alloc_bind_buffer(c, pbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &pb, &pb_mem) ||
       alloc_bind_buffer(c, obytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ob, &ob_mem))
      return 1;
   uint32_t *tp;
   CHECK(vkMapMemory(c->dev, tb_mem, 0, TB_ELEMS * 4u, 0, (void **)&tp));
   for (uint32_t i = 0; i < TB_ELEMS; i++)
      tp[i] = tb_val(i);
   vkUnmapMemory(c->dev, tb_mem);
   int32_t *pp;
   CHECK(vkMapMemory(c->dev, pb_mem, 0, pbytes, 0, (void **)&pp));
   for (uint32_t i = 0; i < NPROBES; i++) {
      const int32_t q[8] = { probes[i].kind, probes[i].x, probes[i].y, probes[i].z,
                             probes[i].lod, 0, 0, 0 };
      memcpy(pp + 8u * i, q, sizeof q);
   }
   vkUnmapMemory(c->dev, pb_mem);
   uint32_t *op;
   CHECK(vkMapMemory(c->dev, ob_mem, 0, obytes, 0, (void **)&op));
   memset(op, 0xab, obytes);
   vkUnmapMemory(c->dev, ob_mem);

   VkImageView arr_view, vol_view, a1d_view, ms_view, bc_view;
   if (make_view(c, arr, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R32_UINT, LEVELS, AL,
                 &arr_view) ||
       make_view(c, vol, VK_IMAGE_VIEW_TYPE_3D, VK_FORMAT_R32_UINT, LEVELS, 1, &vol_view) ||
       make_view(c, a1d, VK_IMAGE_VIEW_TYPE_1D_ARRAY, VK_FORMAT_R32_UINT, 1, L1, &a1d_view) ||
       make_view(c, ms, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_UINT, 1, 1, &ms_view) ||
       make_view(c, bc, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1, 1, &bc_view))
      return 1;
   VkBufferViewCreateInfo bvci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = tb,
      .format = VK_FORMAT_R32_UINT, .range = VK_WHOLE_SIZE,
   };
   VkBufferView tb_view;
   CHECK(vkCreateBufferView(c->dev, &bvci, NULL, &tb_view));

   const VkShaderStageFlags CS = VK_SHADER_STAGE_COMPUTE_BIT;
   VkDescriptorSetLayoutBinding b[10] = {
      { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 2, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, CS, NULL },
      { 3, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, CS, NULL },
      { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL },
      { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, CS, NULL },
      { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, CS, NULL },
   };
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline pipe;
   if (make_pipeline(c, b, 10, spv, spv_size, &dsl, &pl, &pipe))
      return 1;
   VkDescriptorPoolSize dps[3] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
   };
   VkDescriptorPool dp;
   VkDescriptorSet ds;
   if (alloc_set(c, dsl, dps, 3, &dp, &ds))
      return 1;
   const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   VkDescriptorImageInfo dii[6] = {
      { smp, arr_view, RO }, { smp, vol_view, RO }, { smp, a1d_view, RO },
      { smp, ms_view, RO }, { smp, bc_view, RO }, { smp, VK_NULL_HANDLE, RO },
   };
   const VkBufferView null_view = VK_NULL_HANDLE;
   VkDescriptorBufferInfo dbi[2] = { { pb, 0, VK_WHOLE_SIZE }, { ob, 0, VK_WHOLE_SIZE } };
   const uint32_t img_binding[6] = { 0, 1, 4, 5, 6, 7 };
   VkWriteDescriptorSet wds[10];
   for (uint32_t i = 0; i < 6; i++) {
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
         .dstBinding = img_binding[i], .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii[i],
      };
   }
   for (uint32_t i = 0; i < 2; i++) {
      wds[6 + i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 2 + i,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .pTexelBufferView = i ? &null_view : &tb_view,
      };
      wds[8 + i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 8 + i,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[i],
      };
   }
   vkUpdateDescriptorSets(c->dev, 10, wds, 0, NULL);

   VkCommandBuffer cb;
   if (begin(c, &cb))
      return 1;
   const VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, UND = VK_IMAGE_LAYOUT_UNDEFINED;
   to_layout(cb, arr, LEVELS, AL, UND, DST);
   to_layout(cb, vol, LEVELS, 1, UND, DST);
   to_layout(cb, a1d, 1, L1, UND, DST);
   to_layout(cb, ms, 1, 1, UND, DST);
   to_layout(cb, bc, 1, 1, UND, DST);
   vkCmdCopyBufferToImage(cb, seed, arr, DST, LEVELS, arr_bic);
   vkCmdCopyBufferToImage(cb, seed, vol, DST, LEVELS, vol_bic);
   vkCmdCopyBufferToImage(cb, seed, a1d, DST, 1, &a1d_bic);
   vkCmdCopyBufferToImage(cb, seed, bc, DST, 1, &bc_bic);
   const VkClearColorValue msv = { .uint32 = { MS_VAL, MS_VAL, MS_VAL, MS_VAL } };
   const VkImageSubresourceRange msr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   vkCmdClearColorImage(cb, ms, DST, &msv, 1, &msr);
   to_layout(cb, arr, LEVELS, AL, DST, RO);
   to_layout(cb, vol, LEVELS, 1, DST, RO);
   to_layout(cb, a1d, 1, L1, DST, RO);
   to_layout(cb, ms, 1, 1, DST, RO);
   to_layout(cb, bc, 1, 1, DST, RO);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, NPROBES, 1, 1);
   if (submit(c, cb))
      return 1;

   static const char *kname[] = { "array", "3d", "texel buffer", "null texel buffer",
                                  "1d array", "multisample", "bc1 rgba",
                                  "null image size", "null texel buffer size" };
   CHECK(vkMapMemory(c->dev, ob_mem, 0, obytes, 0, (void **)&op));
   unsigned bad = 0;
   for (uint32_t i = 0; i < NPROBES; i++) {
      uint32_t want[4];
      expected(&probes[i], want);
      const uint32_t *got = op + 4u * i;
      if (memcmp(got, want, sizeof want)) {
         if (bad++ < 10)
            printf("  %s (%d,%d,%d) level %d: got (%#x %#x %#x %#x), want (%#x %#x %#x %#x)\n",
                   kname[probes[i].kind], probes[i].x, probes[i].y, probes[i].z, probes[i].lod,
                   got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
      }
   }
   vkUnmapMemory(c->dev, ob_mem);
   if (bad) {
      printf("FAILED (tex_robust: %u/%u probes wrong)\n", bad, (unsigned)NPROBES);
      return 1;
   }
   printf("probes: %u texel fetches and size queries right\n", (unsigned)NPROBES);

   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   vkDestroyBufferView(c->dev, tb_view, NULL);
   VkImageView views[5] = { arr_view, vol_view, a1d_view, ms_view, bc_view };
   VkImage imgs[5] = { arr, vol, a1d, ms, bc };
   VkDeviceMemory imems[5] = { arr_mem, vol_mem, a1d_mem, ms_mem, bc_mem };
   for (int i = 0; i < 5; i++) {
      vkDestroyImageView(c->dev, views[i], NULL);
      vkDestroyImage(c->dev, imgs[i], NULL);
      vkFreeMemory(c->dev, imems[i], NULL);
   }
   VkBuffer bufs[4] = { seed, tb, pb, ob };
   VkDeviceMemory bmems[4] = { seed_mem, tb_mem, pb_mem, ob_mem };
   for (int i = 0; i < 4; i++) {
      vkDestroyBuffer(c->dev, bufs[i], NULL);
      vkFreeMemory(c->dev, bmems[i], NULL);
   }
   return 0;
}

/* ---- a texel buffer element nulled through a template ------------------- */

static int
run_tbnull(const struct ctx *c, const uint32_t *spv, size_t spv_size)
{
   VkBuffer tb, ob;
   VkDeviceMemory tb_mem, ob_mem;
   if (alloc_bind_buffer(c, TB_ELEMS * 4u, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                         &tb, &tb_mem) ||
       alloc_bind_buffer(c, 32u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ob, &ob_mem))
      return 1;
   uint32_t *tp;
   CHECK(vkMapMemory(c->dev, tb_mem, 0, TB_ELEMS * 4u, 0, (void **)&tp));
   for (uint32_t i = 0; i < TB_ELEMS; i++)
      tp[i] = tb_val(i);
   vkUnmapMemory(c->dev, tb_mem);
   VkBufferViewCreateInfo bvci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = tb,
      .format = VK_FORMAT_R32_UINT, .range = VK_WHOLE_SIZE,
   };
   VkBufferView tb_view;
   CHECK(vkCreateBufferView(c->dev, &bvci, NULL, &tb_view));

   const VkShaderStageFlags CS = VK_SHADER_STAGE_COMPUTE_BIT;
   VkDescriptorSetLayoutBinding b[2] = {
      { 0, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2, CS, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, CS, NULL },
   };
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline pipe;
   if (make_pipeline(c, b, 2, spv, spv_size, &dsl, &pl, &pipe))
      return 1;
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPool dp;
   VkDescriptorSet ds;
   if (alloc_set(c, dsl, dps, 2, &dp, &ds))
      return 1;
   VkDescriptorBufferInfo dbi = { ob, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .dstArrayElement = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .pTexelBufferView = &tb_view },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 2, w, 0, NULL);
   VkDescriptorUpdateTemplateEntry tent = {
      .dstBinding = 0, .dstArrayElement = 1, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      .offset = 0, .stride = sizeof(VkBufferView),
   };
   VkDescriptorUpdateTemplateCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
      .descriptorUpdateEntryCount = 1, .pDescriptorUpdateEntries = &tent,
      .templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
      .descriptorSetLayout = dsl,
   };
   VkDescriptorUpdateTemplate tmpl;
   CHECK(vkCreateDescriptorUpdateTemplate(c->dev, &tci, NULL, &tmpl));
   const VkBufferView null_view = VK_NULL_HANDLE;
   vkUpdateDescriptorSetWithTemplate(c->dev, ds, tmpl, &null_view);

   uint32_t *op;
   CHECK(vkMapMemory(c->dev, ob_mem, 0, 32u, 0, (void **)&op));
   memset(op, 0xab, 32u);
   vkUnmapMemory(c->dev, ob_mem);
   VkCommandBuffer cb;
   if (begin(c, &cb))
      return 1;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, 1, 1, 1);
   if (submit(c, cb))
      return 1;
   CHECK(vkMapMemory(c->dev, ob_mem, 0, 32u, 0, (void **)&op));
   const bool ok = !op[0] && !op[1] && !op[2] && !op[3] && !op[4];
   printf("texel buffer nulled through a template: fetch (%#x %#x %#x %#x), size %u\n",
          op[0], op[1], op[2], op[3], op[4]);
   vkUnmapMemory(c->dev, ob_mem);
   if (!ok) {
      printf("FAILED (tex_robust: a texel buffer element nulled through a template still "
             "reads or sizes as its old view)\n");
      return 1;
   }

   vkDestroyDescriptorUpdateTemplate(c->dev, tmpl, NULL);
   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   vkDestroyBufferView(c->dev, tb_view, NULL);
   vkDestroyBuffer(c->dev, tb, NULL);
   vkFreeMemory(c->dev, tb_mem, NULL);
   vkDestroyBuffer(c->dev, ob, NULL);
   vkFreeMemory(c->dev, ob_mem, NULL);
   return 0;
}

/* ---- more than 16 (texture, sampler) pairs ------------------------------- */

static int
run_pairs(const struct ctx *c, const uint32_t *spv, size_t spv_size)
{
   VkImage img[NTEX];
   VkDeviceMemory imem[NTEX];
   VkImageView view[NTEX];
   VkSampler smp[NSMP];
   for (uint32_t i = 0; i < NTEX; i++) {
      if (make_image(c, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, (VkExtent3D){ 1, 1, 1 }, 1, 1,
                     VK_SAMPLE_COUNT_1_BIT, &img[i], &imem[i]) ||
          make_view(c, img[i], VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_UINT, 1, 1, &view[i]))
         return 1;
   }
   for (uint32_t j = 0; j < NSMP; j++) {
      VkSamplerCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
         .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
         .addressModeU = j & 1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                               : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeV = j & 2 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                               : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .maxLod = (float)j,
      };
      CHECK(vkCreateSampler(c->dev, &sci, NULL, &smp[j]));
   }
   VkBuffer ob;
   VkDeviceMemory ob_mem;
   const VkDeviceSize obytes = NTEX * NSMP * sizeof(uint32_t);
   if (alloc_bind_buffer(c, obytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ob, &ob_mem))
      return 1;

   const VkShaderStageFlags CS = VK_SHADER_STAGE_COMPUTE_BIT;
   VkDescriptorSetLayoutBinding b[3] = {
      { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NTEX, CS, NULL },
      { 1, VK_DESCRIPTOR_TYPE_SAMPLER, NSMP, CS, NULL },
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, CS, NULL },
   };
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline pipe;
   if (make_pipeline(c, b, 3, spv, spv_size, &dsl, &pl, &pipe))
      return 1;
   VkDescriptorPoolSize dps[3] = {
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NTEX }, { VK_DESCRIPTOR_TYPE_SAMPLER, NSMP },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPool dp;
   VkDescriptorSet ds;
   if (alloc_set(c, dsl, dps, 3, &dp, &ds))
      return 1;
   VkDescriptorImageInfo ti[NTEX], si[NSMP];
   for (uint32_t i = 0; i < NTEX; i++)
      ti[i] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, view[i],
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   for (uint32_t j = 0; j < NSMP; j++)
      si[j] = (VkDescriptorImageInfo){ smp[j], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
   VkDescriptorBufferInfo dbi = { ob, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = NTEX, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = ti },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = NSMP, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = si },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 2,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 3, w, 0, NULL);

   VkCommandBuffer cb;
   if (begin(c, &cb))
      return 1;
   const VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   for (uint32_t i = 0; i < NTEX; i++) {
      to_layout(cb, img[i], 1, 1, VK_IMAGE_LAYOUT_UNDEFINED, DST);
      const VkClearColorValue v = { .uint32 = { 100u + i, 0, 0, 0 } };
      const VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdClearColorImage(cb, img[i], DST, &v, 1, &r);
      to_layout(cb, img[i], 1, 1, DST, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
   }
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, 1, 1, 1);
   if (submit(c, cb))
      return 1;
   uint32_t *op;
   CHECK(vkMapMemory(c->dev, ob_mem, 0, obytes, 0, (void **)&op));
   unsigned bad = 0;
   for (uint32_t i = 0; i < NTEX; i++)
      for (uint32_t j = 0; j < NSMP; j++)
         bad += op[i * NSMP + j] != 100u + i;
   vkUnmapMemory(c->dev, ob_mem);
   if (bad) {
      printf("FAILED (tex_robust: %u/%u of %u (texture, sampler) pairs sampled wrong)\n",
             bad, NTEX * NSMP, NTEX * NSMP);
      return 1;
   }
   printf("pairs: %u (texture, sampler) pairs sampled right\n", NTEX * NSMP);

   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   for (uint32_t i = 0; i < NTEX; i++) {
      vkDestroyImageView(c->dev, view[i], NULL);
      vkDestroyImage(c->dev, img[i], NULL);
      vkFreeMemory(c->dev, imem[i], NULL);
   }
   for (uint32_t j = 0; j < NSMP; j++)
      vkDestroySampler(c->dev, smp[j], NULL);
   vkDestroyBuffer(c->dev, ob, NULL);
   vkFreeMemory(c->dev, ob_mem, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *paths[3] = { "trobust.comp.spv", "tbnull.comp.spv", "pairs.comp.spv" };
   uint32_t *spv[3];
   size_t spv_size[3];
   for (int i = 0; i < 3; i++) {
      if (argc > i + 1)
         paths[i] = argv[i + 1];
      spv[i] = read_spirv(paths[i], &spv_size[i]);
      if (!spv[i])
         return 1;
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-tex-robust",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo inci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&inci, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceRobustness2FeaturesEXT r2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceImageRobustnessFeaturesEXT ir = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT, .pNext = &r2,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &ir,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!r2.robustImageAccess2 || !r2.nullDescriptor || !ir.robustImageAccess ||
       !f2.features.textureCompressionBC) {
      fprintf(stderr, "FAILED: no robustImageAccess2/nullDescriptor/robustImageAccess/BC\n");
      return 1;
   }

   struct ctx c;
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   c.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
   free(qfp);
   if (c.qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceRobustness2FeaturesEXT r2on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .robustImageAccess2 = VK_TRUE, .nullDescriptor = VK_TRUE,
   };
   VkPhysicalDeviceImageRobustnessFeaturesEXT iron = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT, .pNext = &r2on,
      .robustImageAccess = VK_TRUE,
   };
   VkPhysicalDeviceFeatures core = { .textureCompressionBC = VK_TRUE };
   const char *exts[] = { "VK_EXT_robustness2", "VK_EXT_image_robustness" };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &iron,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts,
      .pEnabledFeatures = &core,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &c.dev));
   vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c.mp);
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c.qf,
   };
   CHECK(vkCreateCommandPool(c.dev, &cpi, NULL, &c.cp));
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = (float)LEVELS,
   };
   VkSampler smp;
   CHECK(vkCreateSampler(c.dev, &sci, NULL, &smp));

   /* Every part runs, so a failure in one does not hide another. */
   const int bad = run_probes(&c, smp, spv[0], spv_size[0]) |
                   run_tbnull(&c, spv[1], spv_size[1]) |
                   run_pairs(&c, spv[2], spv_size[2]);
   if (bad)
      return 1;

   vkDestroySampler(c.dev, smp, NULL);
   vkDestroyCommandPool(c.dev, c.cp, NULL);
   for (int i = 0; i < 3; i++)
      free(spv[i]);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (tex_robust: %u fetches and size queries inside and outside the view, a "
          "texel buffer nulled through a template, %u (texture, sampler) pairs)\n",
          (unsigned)NPROBES, NTEX * NSMP);
   return 0;
}
