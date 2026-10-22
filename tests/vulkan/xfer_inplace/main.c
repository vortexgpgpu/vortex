/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A chain of transfers through images whose rows carry padding.
 *
 *   1. buffer -> image A (37x29, layer 1 of 2)
 *   2. image A -> image B, a 20x15 box into level 1 of B (32x24) at (3, 2)
 *   3. a 2x nearest blit of that box from B into image C (50x40) at (5, 4)
 *   4. a 7x5 clear inside C at (10, 8), from a render pass
 *   5. C sampled by a compute shader with texelFetch, and copied to a buffer
 *
 * Every texel of C is checked twice -- as the copy-out holds it and as the
 * shader sampled it -- against a host model of the chain. Each step reads what
 * the one before it wrote, so a step that wrote a stale copy, missed its row
 * padding or left its result where the next one does not look shows. The case
 * repeats four times with new images and a pattern and clear colours of its
 * own, so an image that inherits the previous case's texels, or a copy served
 * from a cache keyed on a recycled address, fails the case.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FMT      VK_FORMAT_R8G8B8A8_UNORM
#define AW       37u
#define AH       29u
#define BW       64u
#define BH       48u
#define CW       50u
#define CH       40u
#define BOX_W    20u
#define BOX_H    15u
#define BOX_X    3u                    /* the box's corner in B level 1 */
#define BOX_Y    2u
#define DST_X    5u                    /* the blit's corner in C */
#define DST_Y    4u
#define CLR_X    10u                   /* the clear's box in C */
#define CLR_Y    8u
#define CLR_W    7u
#define CLR_H    5u
#define CASES    4u

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

static uint32_t
pattern(uint32_t x, uint32_t y, uint32_t c)
{
   uint32_t v = (x * 2654435761u) ^ (y * 40503u) ^ (c * 0x9e3779b9u);
   v ^= v >> 13;
   return v * 0x85ebca6bu;
}

/* Clear colours are given as bytes, so the unorm value each channel stores is
 * exact and no rounding mode is under test. */
static void
unorm(uint32_t v, float f[4])
{
   for (unsigned i = 0; i < 4; i++)
      f[i] = (float)((v >> (8 * i)) & 0xffu) / 255.0f;
}

struct ctx {
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
};

static VkResult
make_image(struct ctx *c, uint32_t w, uint32_t h, uint32_t levels, uint32_t layers,
           VkImageUsageFlags usage, VkImage *img, VkDeviceMemory *mem)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = FMT, .extent = { w, h, 1 }, .mipLevels = levels, .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
   };
   VkResult r = vkCreateImage(c->dev, &ici, NULL, img);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, *img, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&c->mp, mr.memoryTypeBits, 0),
   };
   r = vkAllocateMemory(c->dev, &mai, NULL, mem);
   return r == VK_SUCCESS ? vkBindImageMemory(c->dev, *img, *mem, 0) : r;
}

static VkResult
make_buffer(struct ctx *c, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *buf,
            VkDeviceMemory *mem, void **map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
   };
   VkResult r = vkCreateBuffer(c->dev, &bci, NULL, buf);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&c->mp, mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   r = vkAllocateMemory(c->dev, &mai, NULL, mem);
   if (r == VK_SUCCESS)
      r = vkBindBufferMemory(c->dev, *buf, *mem, 0);
   return r == VK_SUCCESS ? vkMapMemory(c->dev, *mem, 0, size, 0, map) : r;
}

static void
barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to,
        uint32_t levels, uint32_t layers)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static int
run_case(struct ctx *c, uint32_t k, VkShaderModule cs)
{
   const uint32_t rgba_b = 0xff4c3300u | (25u * k);
   const uint32_t rgba_c = 0xcc990066u | ((25u * k) << 8);
   const uint32_t rgba_box = 0x80001ae6u | ((25u * (k + 1)) << 16) | (0xb3u << 8);
   float clear_b[4], clear_c[4], clear_box[4];
   unorm(rgba_b, clear_b);
   unorm(rgba_c, clear_c);
   unorm(rgba_box, clear_box);

   VkImage ia, ib, ic;
   VkDeviceMemory ma, mb, mc;
   const VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(make_image(c, AW, AH, 1, 2, xfer, &ia, &ma));
   CHECK(make_image(c, BW, BH, 2, 1, xfer, &ib, &mb));
   CHECK(make_image(c, CW, CH, 1, 1,
                    xfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    &ic, &mc));

   VkBuffer src, dst, out;
   VkDeviceMemory msrc, mdst, mout;
   uint32_t *psrc, *pdst, *pout;
   CHECK(make_buffer(c, AW * AH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src, &msrc, (void **)&psrc));
   CHECK(make_buffer(c, CW * CH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst, &mdst, (void **)&pdst));
   CHECK(make_buffer(c, CW * CH * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &mout,
                     (void **)&pout));
   for (uint32_t y = 0; y < AH; y++)
      for (uint32_t x = 0; x < AW; x++)
         psrc[y * AW + x] = pattern(x, y, k);
   memset(pdst, 0, CW * CH * 4);
   memset(pout, 0, CW * CH * 4);

   /* C as a colour attachment (the clear) and as a sampled image. */
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = ic,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FMT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c->dev, &vci, NULL, &view));
   VkAttachmentDescription att = {
      .format = FMT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 1, .pAttachments = &view, .width = CW, .height = CH, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &fb));

   /* The sampling dispatch. */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler smp;
   CHECK(vkCreateSampler(c->dev, &sci, NULL, &smp));
   VkDescriptorSetLayoutBinding binds[2] = {
      { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = binds,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, &pl));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main" },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, &ds));
   VkDescriptorImageInfo dii = { smp, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   VkDescriptorBufferInfo dbi = { out, 0, CW * CH * 4 };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 2, wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(c->dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   const VkImageLayout DST = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   const VkImageLayout SRC = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   barrier(cmd, ia, VK_IMAGE_LAYOUT_UNDEFINED, DST, 1, 2);
   barrier(cmd, ib, VK_IMAGE_LAYOUT_UNDEFINED, DST, 2, 1);
   barrier(cmd, ic, VK_IMAGE_LAYOUT_UNDEFINED, DST, 1, 1);
   const VkImageSubresourceRange all_b = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 1 };
   const VkImageSubresourceRange all_c = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   VkClearColorValue cb, cc;
   memcpy(cb.float32, clear_b, sizeof clear_b);
   memcpy(cc.float32, clear_c, sizeof clear_c);
   vkCmdClearColorImage(cmd, ib, DST, &cb, 1, &all_b);
   vkCmdClearColorImage(cmd, ic, DST, &cc, 1, &all_c);

   /* 1: buffer -> A, layer 1 */
   VkBufferImageCopy up = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1 },
      .imageExtent = { AW, AH, 1 },
   };
   vkCmdCopyBufferToImage(cmd, src, ia, DST, 1, &up);
   barrier(cmd, ia, DST, SRC, 1, 2);
   barrier(cmd, ib, DST, DST, 2, 1);

   /* 2: A -> B level 1 */
   VkImageCopy cpy = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1 },
      .dstOffset = { BOX_X, BOX_Y, 0 },
      .extent = { BOX_W, BOX_H, 1 },
   };
   vkCmdCopyImage(cmd, ia, SRC, ib, DST, 1, &cpy);
   barrier(cmd, ib, DST, SRC, 2, 1);
   barrier(cmd, ic, DST, DST, 1, 1);

   /* 3: the box blitted 2x into C */
   VkImageBlit blit = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1 },
      .srcOffsets = { { BOX_X, BOX_Y, 0 }, { BOX_X + BOX_W, BOX_Y + BOX_H, 1 } },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstOffsets = { { DST_X, DST_Y, 0 }, { DST_X + 2 * BOX_W, DST_Y + 2 * BOX_H, 1 } },
   };
   vkCmdBlitImage(cmd, ib, SRC, ic, DST, 1, &blit, VK_FILTER_NEAREST);
   barrier(cmd, ic, DST, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 1);

   /* 4: a box of C cleared from a render pass */
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { CW, CH } },
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   VkClearAttachment ca = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0 };
   memcpy(ca.clearValue.color.float32, clear_box, sizeof clear_box);
   VkClearRect cr = { { { CLR_X, CLR_Y }, { CLR_W, CLR_H } }, 0, 1 };
   vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
   vkCmdEndRenderPass(cmd);

   /* 5: C sampled, and copied out */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   const uint32_t dims[2] = { CW, CH };
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof dims, dims);
   vkCmdDispatch(cmd, (CW * CH + 15) / 16, 1, 1);
   barrier(cmd, ic, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, SRC, 1, 1);
   VkBufferImageCopy down = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { CW, CH, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, ic, SRC, dst, 1, &down);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   unsigned fails = 0;
   for (uint32_t y = 0; y < CH; y++) {
      for (uint32_t x = 0; x < CW; x++) {
         uint32_t want = rgba_c;
         if (x >= CLR_X && x < CLR_X + CLR_W && y >= CLR_Y && y < CLR_Y + CLR_H)
            want = rgba_box;
         else if (x >= DST_X && x < DST_X + 2 * BOX_W && y >= DST_Y && y < DST_Y + 2 * BOX_H)
            want = pattern((x - DST_X) / 2, (y - DST_Y) / 2, k);
         const uint32_t got_copy = pdst[y * CW + x], got_tex = pout[y * CW + x];
         if (got_copy != want || got_tex != want) {
            if (fails < 6)
               fprintf(stderr, "  case %u (%u,%u): copied 0x%08x sampled 0x%08x, want 0x%08x\n",
                       k, x, y, got_copy, got_tex, want);
            fails++;
         }
      }
   }

   vkDestroyCommandPool(c->dev, cp, NULL);
   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   vkDestroySampler(c->dev, smp, NULL);
   vkDestroyFramebuffer(c->dev, fb, NULL);
   vkDestroyRenderPass(c->dev, rp, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyBuffer(c->dev, src, NULL);
   vkDestroyBuffer(c->dev, dst, NULL);
   vkDestroyBuffer(c->dev, out, NULL);
   vkFreeMemory(c->dev, msrc, NULL);
   vkFreeMemory(c->dev, mdst, NULL);
   vkFreeMemory(c->dev, mout, NULL);
   vkDestroyImage(c->dev, ia, NULL);
   vkDestroyImage(c->dev, ib, NULL);
   vkDestroyImage(c->dev, ic, NULL);
   vkFreeMemory(c->dev, ma, NULL);
   vkFreeMemory(c->dev, mb, NULL);
   vkFreeMemory(c->dev, mc, NULL);
   if (fails) {
      printf("FAILED (case %u: %u texels of the chain's result differ)\n", k, fails);
      return 1;
   }
   printf("case %u: %u texels match, copied and sampled\n", k, CW * CH);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *cs_path = (argc > 1) ? argv[1] : "sample.comp.spv";
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-xfer-inplace", .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   struct ctx c;
   uint32_t npd = 1;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &c.pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(c.pd, &props);
   printf("device: %s\n", props.deviceName);
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   c.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { c.qf = i; break; }
   free(qfp);
   if (c.qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   CHECK(vkCreateDevice(c.pd, &dci, NULL, &c.dev));
   vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);

   size_t sz = 0;
   uint32_t *spv = read_spirv(cs_path, &sz);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule cs;
   CHECK(vkCreateShaderModule(c.dev, &smci, NULL, &cs));
   free(spv);

   for (uint32_t k = 0; k < CASES; k++) {
      if (run_case(&c, k, cs))
         return 1;
   }
   vkDestroyShaderModule(c.dev, cs, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (xfer_inplace: %u cases of buffer->image->image, scaled blit, box clear, "
          "sampled and copied)\n", CASES);
   return 0;
}
