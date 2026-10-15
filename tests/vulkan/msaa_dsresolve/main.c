/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Render-pass depth/stencil resolves of a 4x attachment
 * (VkSubpassDescriptionDepthStencilResolve).
 *
 * Every case draws a sloped plane into a 16x16 4x depth/stencil attachment, so
 * the four samples of a pixel hold four different depths. Where the format has
 * stencil, a first draw writes 0x5A into every sample and a second, limited to
 * sample 0 by the pipeline sample mask and to the pixels with x + y even by a
 * discard, writes 0xC3: sample 0 then differs from the others in a per-pixel
 * pattern. The pass resolves into a single-sample target cleared beforehand,
 * which is copied to a buffer and checked:
 *
 *   SAMPLE_ZERO  depth = the plane at sample 0's position (0.375, 0.125),
 *                stencil = sample 0's.
 *   AVERAGE      depth = the mean of the four sample depths.
 *
 * within the format's precision. A partial-area case resolves only inside its
 * render area; every pixel outside it must keep the clear.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH    16u
#define HEIGHT   16u
#define SAMPLES  4u

/* Neighbouring samples of a pixel differ by at least 1/256 in depth; this
 * bound only covers the interpolation's rounding, a few ulp of a float. */
#define FLOAT_TOL 1e-6

#define STENCIL_ALL   0x5Au
#define STENCIL_EVEN0 0xC3u

/* What the resolve target holds before the pass. */
#define CLEAR_DEPTH   0.0f
#define CLEAR_STENCIL 0x11u

/* The partial render area. */
#define PART_X  3
#define PART_Y  5
#define PART_W  9u
#define PART_H  7u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static const double k_pos[SAMPLES][2] = {
   { 0.375, 0.125 }, { 0.875, 0.375 }, { 0.125, 0.625 }, { 0.625, 0.875 },
};

struct ds_case {
   const char           *name;
   VkFormat              format;
   VkResolveModeFlagBits depth_mode;
   bool                  partial;
};

struct env {
   VkPhysicalDevice                 pd;
   VkDevice                         dev;
   VkQueue                          queue;
   VkCommandBuffer                  cmd;
   VkPhysicalDeviceMemoryProperties mp;
   VkPipelineLayout                 layout;
   VkShaderModule                   vs, fs_all, fs_even;
   VkBuffer                         buf;
   const uint8_t                   *data;   /* mapped read-back buffer */
};

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "FAILED: cannot open %s\n", path);
      return NULL;
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *buf = malloc((size_t)sz);
   if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      free(buf);
      buf = NULL;
   }
   fclose(f);
   *out_size = (size_t)sz;
   return buf;
}

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return VK_NULL_HANDLE;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
      sm = VK_NULL_HANDLE;
   }
   free(spv);
   return sm;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
bind_memory(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            const VkMemoryRequirements *mr, VkMemoryPropertyFlags want,
            VkDeviceMemory *out)
{
   uint32_t mt = find_mem(mp, mr->memoryTypeBits, want);
   if (mt == UINT32_MAX) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr->size, .memoryTypeIndex = mt,
   };
   return vkAllocateMemory(dev, &mai, NULL, out);
}

static bool
has_stencil(VkFormat fmt)
{
   return fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT ||
          fmt == VK_FORMAT_D16_UNORM_S8_UINT;
}

/* Bits of a UNORM depth format, 0 for a float one. */
static unsigned
depth_bits(VkFormat fmt)
{
   switch (fmt) {
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_D16_UNORM_S8_UINT:
      return 16;
   case VK_FORMAT_X8_D24_UNORM_PACK32:
   case VK_FORMAT_D24_UNORM_S8_UINT:
      return 24;
   default:
      return 0;
   }
}

static const char *
format_name(VkFormat fmt)
{
   switch (fmt) {
   case VK_FORMAT_D16_UNORM:           return "D16";
   case VK_FORMAT_X8_D24_UNORM_PACK32: return "X8_D24";
   case VK_FORMAT_D32_SFLOAT:          return "D32F";
   case VK_FORMAT_D24_UNORM_S8_UINT:   return "D24S8";
   case VK_FORMAT_D32_SFLOAT_S8_UINT:  return "D32FS8";
   default:                            return "?";
   }
}

static VkImageAspectFlags
aspects(VkFormat fmt)
{
   return VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

/* Whether the format can be a 4x depth/stencil attachment that is also
 * sampled, and a single-sample attachment that is copied to and from. */
static bool
format_ok(VkPhysicalDevice pd, VkFormat fmt)
{
   VkImageFormatProperties ifp;
   if (vkGetPhysicalDeviceImageFormatProperties(pd, fmt, VK_IMAGE_TYPE_2D,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0,
          &ifp) != VK_SUCCESS || !(ifp.sampleCounts & SAMPLES)) {
      return false;
   }
   return vkGetPhysicalDeviceImageFormatProperties(pd, fmt, VK_IMAGE_TYPE_2D,
             VK_IMAGE_TILING_OPTIMAL,
             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0,
             &ifp) == VK_SUCCESS;
}

/* What plane.vert puts at window position (x, y). */
static double
plane_depth(double x, double y)
{
   return 0.125 + x / 32.0 + y / 64.0;
}

/* What the two draws leave in the stencil of sample k of pixel (x, y). */
static uint32_t
expect_stencil(uint32_t x, uint32_t y, uint32_t k)
{
   return (k == 0 && ((x + y) & 1) == 0) ? STENCIL_EVEN0 : STENCIL_ALL;
}

static bool
in_area(const struct ds_case *c, uint32_t x, uint32_t y)
{
   if (!c->partial) {
      return true;
   }
   return x >= PART_X && x < PART_X + PART_W && y >= PART_Y && y < PART_Y + PART_H;
}

struct image {
   VkImage        img;
   VkDeviceMemory mem;
   VkImageView    view;
};

static VkResult
make_image(struct env *e, VkFormat fmt, VkSampleCountFlagBits samples,
           VkImageUsageFlags usage, struct image *out)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = vkCreateImage(e->dev, &ici, NULL, &out->img);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(e->dev, out->img, &mr);
   r = bind_memory(e->dev, &e->mp, &mr, 0, &out->mem);
   if (r == VK_SUCCESS) {
      r = vkBindImageMemory(e->dev, out->img, out->mem, 0);
   }
   if (r != VK_SUCCESS) {
      return r;
   }
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = out->img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { .aspectMask = aspects(fmt), .levelCount = 1, .layerCount = 1 },
   };
   return vkCreateImageView(e->dev, &vci, NULL, &out->view);
}

static void
destroy_image(VkDevice dev, struct image *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkDestroyImage(dev, im->img, NULL);
   vkFreeMemory(dev, im->mem, NULL);
}

/* A one-subpass pass drawing into the 4x attachment and resolving it into the
 * single-sample one, which keeps what it held outside the render area. */
static VkResult
make_render_pass(VkDevice dev, const struct ds_case *c, VkRenderPass *out)
{
   const bool stencil = has_stencil(c->format);
   VkAttachmentDescription2 atts[2] = {
      { .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
        .format = c->format, .samples = (VkSampleCountFlagBits)SAMPLES,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                 : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
      { .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
        .format = c->format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = stencil ? VK_ATTACHMENT_LOAD_OP_LOAD
                                 : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = stencil ? VK_ATTACHMENT_STORE_OP_STORE
                                  : VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference2 ds_ref = {
      .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .aspectMask = aspects(c->format),
   };
   VkAttachmentReference2 rs_ref = ds_ref;
   rs_ref.attachment = 1;
   /* A combined format resolves both aspects in one mode: the device reports
    * neither independentResolve nor independentResolveNone. */
   VkSubpassDescriptionDepthStencilResolve dsr = {
      .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
      .depthResolveMode = c->depth_mode,
      .stencilResolveMode = stencil ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                    : VK_RESOLVE_MODE_NONE,
      .pDepthStencilResolveAttachment = &rs_ref,
   };
   VkSubpassDescription2 sub = {
      .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
      .pNext = &dsr,
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .pDepthStencilAttachment = &ds_ref,
   };
   VkSubpassDependency2 deps[2] = {
      { .sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
        .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT },
      { .sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
        .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT },
   };
   VkRenderPassCreateInfo2 rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
      .attachmentCount = 2, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
      .dependencyCount = 2, .pDependencies = deps,
   };
   return vkCreateRenderPass2(dev, &rpci, NULL, out);
}

static VkResult
make_pipeline(struct env *e, VkShaderModule fs, VkRenderPass rp, bool stencil,
              VkSampleMask mask, uint32_t ref, VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo st[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = e->vs, .pName = "main" },
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
   VkViewport vp = { 0, 0, (float)WIDTH, (float)HEIGHT, 0.0f, 1.0f };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)SAMPLES,
      .pSampleMask = &mask,
   };
   VkStencilOpState sop = {
      .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xFF, .writeMask = 0xFF, .reference = ref,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = stencil, .front = sop, .back = sop,
      .maxDepthBounds = 1.0f,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
   };
   VkDynamicState dyn = VK_DYNAMIC_STATE_SCISSOR;
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1, .pDynamicStates = &dyn,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &dss,
      .pColorBlendState = &cb, .pDynamicState = &ds,
      .layout = e->layout, .renderPass = rp, .subpass = 0,
   };
   return vkCreateGraphicsPipelines(e->dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static void
layout_barrier(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect,
               VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
               VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
               VkPipelineStageFlags dst_stage)
{
   VkImageMemoryBarrier imb = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access, .dstAccessMask = dst_access,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1, .layerCount = 1 },
   };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &imb);
}

/* Byte offset of the stencil copy in the read-back buffer, after the depth. */
#define STENCIL_OFFSET ((VkDeviceSize)WIDTH * HEIGHT * 4)

/* The resolved depth of pixel (x, y): a float, or a UNORM code. */
static double
read_depth(const struct env *e, VkFormat fmt, uint32_t x, uint32_t y)
{
   const size_t i = (size_t)y * WIDTH + x;
   switch (depth_bits(fmt)) {
   case 16: {
      uint16_t v;
      memcpy(&v, e->data + i * 2, sizeof(v));
      return v;
   }
   case 24: {
      uint32_t v;
      memcpy(&v, e->data + i * 4, sizeof(v));
      return v & 0xFFFFFFu;
   }
   default: {
      float v;
      memcpy(&v, e->data + i * 4, sizeof(v));
      return v;
   }
   }
}

/* What a UNORM depth attachment stores for depth z. */
static double
unorm_code(double z, unsigned bits)
{
   return round(z * (double)((1u << bits) - 1u));
}

/* The resolved depth expected at pixel (x, y), in the units read_depth()
 * returns, and how far from it the result may be. */
static double
expect_depth(const struct ds_case *c, uint32_t x, uint32_t y, double *tol)
{
   const unsigned bits = depth_bits(c->format);
   if (!in_area(c, x, y)) {
      *tol = 0.0;
      return bits ? unorm_code(CLEAR_DEPTH, bits) : CLEAR_DEPTH;
   }
   double z[SAMPLES];
   for (uint32_t k = 0; k < SAMPLES; k++) {
      z[k] = plane_depth(x + k_pos[k][0], y + k_pos[k][1]);
      if (bits) {
         z[k] = unorm_code(z[k], bits);
      }
   }
   /* One code either way: the sample's own quantisation, or the rounding of
    * the averaged codes. */
   *tol = bits ? 1.0 : FLOAT_TOL;
   if (c->depth_mode == VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) {
      return z[0];
   }
   return (z[0] + z[1] + z[2] + z[3]) / SAMPLES;
}

/* Draws the case, resolves it, copies the resolve target out and checks it.
 * Returns the number of failing pixels, or -1 on an API error. */
static int
run_case(struct env *e, const struct ds_case *c)
{
   const bool stencil = has_stencil(c->format);
   const VkImageAspectFlags asp = aspects(c->format);
   struct image ms_img = { 0 }, rs_img = { 0 };
   VkRenderPass rp = VK_NULL_HANDLE;
   VkFramebuffer fb = VK_NULL_HANDLE;
   VkPipeline pipe_all = VK_NULL_HANDLE, pipe_even = VK_NULL_HANDLE;
   int bad = -1;

   if (make_image(e, c->format, (VkSampleCountFlagBits)SAMPLES,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  &ms_img) != VK_SUCCESS ||
       make_image(e, c->format, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  &rs_img) != VK_SUCCESS ||
       make_render_pass(e->dev, c, &rp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: %s images or render pass\n", c->name);
      goto out;
   }
   VkImageView views[2] = { ms_img.view, rs_img.view };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = views,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   if (vkCreateFramebuffer(e->dev, &fbci, NULL, &fb) != VK_SUCCESS ||
       make_pipeline(e, e->fs_all, rp, stencil, 0xF, STENCIL_ALL, &pipe_all) != VK_SUCCESS ||
       (stencil && make_pipeline(e, e->fs_even, rp, stencil, 0x1, STENCIL_EVEN0,
                                 &pipe_even) != VK_SUCCESS)) {
      fprintf(stderr, "FAILED: %s framebuffer or pipelines\n", c->name);
      goto out;
   }

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   vkResetCommandBuffer(e->cmd, 0);
   vkBeginCommandBuffer(e->cmd, &cbbi);

   layout_barrier(e->cmd, rs_img.img, asp, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   VkClearDepthStencilValue rs_clear = { CLEAR_DEPTH, CLEAR_STENCIL };
   VkImageSubresourceRange range = { .aspectMask = asp, .levelCount = 1, .layerCount = 1 };
   vkCmdClearDepthStencilImage(e->cmd, rs_img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               &rs_clear, 1, &range);
   layout_barrier(e->cmd, rs_img.img, asp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                  VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

   VkRect2D area = { { 0, 0 }, { WIDTH, HEIGHT } };
   if (c->partial) {
      area = (VkRect2D){ { PART_X, PART_Y }, { PART_W, PART_H } };
   }
   VkClearValue clears[2] = {
      { .depthStencil = { 1.0f, 0 } },
      { .depthStencil = { 0.0f, 0 } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb, .renderArea = area,
      .clearValueCount = 2, .pClearValues = clears,
   };
   vkCmdBeginRenderPass(e->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdSetScissor(e->cmd, 0, 1, &area);
   vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_all);
   vkCmdDraw(e->cmd, 6, 1, 0, 0);
   if (stencil) {
      vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_even);
      vkCmdDraw(e->cmd, 6, 1, 0, 0);
   }
   vkCmdEndRenderPass(e->cmd);

   VkBufferImageCopy regions[2] = {
      { .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
        .imageExtent = { WIDTH, HEIGHT, 1 } },
      { .bufferOffset = STENCIL_OFFSET,
        .imageSubresource = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1 },
        .imageExtent = { WIDTH, HEIGHT, 1 } },
   };
   vkCmdCopyImageToBuffer(e->cmd, rs_img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          e->buf, stencil ? 2 : 1, regions);
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(e->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
   vkEndCommandBuffer(e->cmd);

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &e->cmd,
   };
   VkResult r = vkQueueSubmit(e->queue, 1, &si, VK_NULL_HANDLE);
   if (r == VK_SUCCESS) {
      r = vkQueueWaitIdle(e->queue);
   }
   if (r != VK_SUCCESS) {
      fprintf(stderr, "FAILED: %s submit -> VkResult %d\n", c->name, (int)r);
      goto out;
   }

   bad = 0;
   const unsigned bits = depth_bits(c->format);
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         double tol;
         const double want = expect_depth(c, x, y, &tol);
         const double got = read_depth(e, c->format, x, y);
         bool ok = fabs(got - want) <= tol;
         uint32_t s_got = 0, s_want = 0;
         if (stencil) {
            s_got = e->data[STENCIL_OFFSET + (size_t)y * WIDTH + x];
            s_want = in_area(c, x, y) ? expect_stencil(x, y, 0) : CLEAR_STENCIL;
            ok = ok && s_got == s_want;
         }
         if (ok) {
            continue;
         }
         if (bad < 4) {
            const double scale = bits ? (double)((1u << bits) - 1u) : 1.0;
            printf("  %s (%u,%u): depth %.9g want %.9g (sample0 %.9g, centre %.9g)",
                   c->name, x, y, got, want,
                   plane_depth(x + k_pos[0][0], y + k_pos[0][1]) * scale,
                   plane_depth(x + 0.5, y + 0.5) * scale);
            if (stencil) {
               printf(" stencil 0x%02X want 0x%02X", s_got, s_want);
            }
            printf("\n");
         }
         bad++;
      }
   }
   printf("%-8s: %s  %s depth %s%s%s  bad pixels=%d of %u\n", c->name,
          bad ? "FAIL" : "pass", format_name(c->format),
          c->depth_mode == VK_RESOLVE_MODE_SAMPLE_ZERO_BIT ? "SAMPLE_ZERO" : "AVERAGE",
          stencil ? ", stencil SAMPLE_ZERO" : "", c->partial ? ", partial area" : "",
          bad, WIDTH * HEIGHT);

out:
   vkDestroyPipeline(e->dev, pipe_even, NULL);
   vkDestroyPipeline(e->dev, pipe_all, NULL);
   vkDestroyFramebuffer(e->dev, fb, NULL);
   vkDestroyRenderPass(e->dev, rp, NULL);
   destroy_image(e->dev, &rs_img);
   destroy_image(e->dev, &ms_img);
   return bad;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_dsresolve",
      .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t npd = 1;
   struct env e = { 0 };
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &e.pd));
   if (npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceDepthStencilResolveProperties dsrp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dsrp,
   };
   vkGetPhysicalDeviceProperties2(e.pd, &props2);
   const VkPhysicalDeviceProperties *props = &props2.properties;
   printf("device: %s\n", props->deviceName);
   if (props->apiVersion < VK_API_VERSION_1_2) {
      printf("FAILED (device API version below 1.2)\n");
      return 1;
   }
   if (!props->limits.standardSampleLocations) {
      printf("FAILED (device does not use the standard sample locations)\n");
      return 1;
   }
   if (!(dsrp.supportedDepthResolveModes & VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) ||
       !(dsrp.supportedStencilResolveModes & VK_RESOLVE_MODE_SAMPLE_ZERO_BIT)) {
      printf("FAILED (SAMPLE_ZERO depth/stencil resolve not supported: depth 0x%x "
             "stencil 0x%x)\n", dsrp.supportedDepthResolveModes,
             dsrp.supportedStencilResolveModes);
      return 1;
   }
   const bool average = (dsrp.supportedDepthResolveModes & VK_RESOLVE_MODE_AVERAGE_BIT) != 0;
   printf("resolve modes: depth 0x%x stencil 0x%x independent %u/%u\n",
          dsrp.supportedDepthResolveModes, dsrp.supportedStencilResolveModes,
          (unsigned)dsrp.independentResolveNone, (unsigned)dsrp.independentResolve);

   if (!format_ok(e.pd, VK_FORMAT_D32_SFLOAT)) {
      printf("FAILED (D32_SFLOAT is not a sampled 4x depth attachment)\n");
      return 1;
   }
   VkFormat unorm_fmt = VK_FORMAT_X8_D24_UNORM_PACK32;
   if (!format_ok(e.pd, unorm_fmt)) {
      unorm_fmt = VK_FORMAT_D16_UNORM;
   }
   VkFormat ds_fmt = VK_FORMAT_D24_UNORM_S8_UINT;
   if (!format_ok(e.pd, ds_fmt)) {
      ds_fmt = VK_FORMAT_D32_SFLOAT_S8_UINT;
      if (!format_ok(e.pd, ds_fmt)) {
         printf("FAILED (neither D24S8 nor D32FS8 is a sampled 4x depth/stencil "
                "attachment)\n");
         return 1;
      }
   }
   printf("formats: D32F, %s, %s\n", format_name(unorm_fmt), format_name(ds_fmt));

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qf = i;
         break;
      }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      fprintf(stderr, "FAILED: no graphics queue\n");
      return 1;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   CHECK(vkCreateDevice(e.pd, &dci, NULL, &e.dev));
   VkDevice dev = e.dev;
   vkGetDeviceQueue(dev, qf, 0, &e.queue);
   vkGetPhysicalDeviceMemoryProperties(e.pd, &e.mp);

   /* Four bytes of depth per pixel, then one of stencil. */
   const VkDeviceSize bytes = STENCIL_OFFSET + (VkDeviceSize)WIDTH * HEIGHT;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &e.buf));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, e.buf, &bmr);
   VkDeviceMemory bmem;
   CHECK(bind_memory(dev, &e.mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &bmem));
   CHECK(vkBindBufferMemory(dev, e.buf, bmem, 0));
   void *map;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, &map));
   e.data = map;

   e.vs = load_module(dev, "plane.vert.spv");
   e.fs_all = load_module(dev, "plane.frag.spv");
   e.fs_even = load_module(dev, "plane_even.frag.spv");
   if (!e.vs || !e.fs_all || !e.fs_even) {
      return 1;
   }
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &e.layout));

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &e.cmd));

   const struct ds_case cases[] = {
      { "d32_zero", VK_FORMAT_D32_SFLOAT, VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, false },
      { "d32_avg",  VK_FORMAT_D32_SFLOAT, VK_RESOLVE_MODE_AVERAGE_BIT,     true  },
      { "unorm_avg", unorm_fmt,           VK_RESOLVE_MODE_AVERAGE_BIT,     false },
      { "ds_zero",  ds_fmt,               VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, false },
      { "ds_part",  ds_fmt,               VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, true  },
   };
   const unsigned ncases = sizeof(cases) / sizeof(cases[0]);
   unsigned failed = 0, ran = 0;
   bool api_error = false;
   for (unsigned i = 0; i < ncases; i++) {
      if (cases[i].depth_mode == VK_RESOLVE_MODE_AVERAGE_BIT && !average) {
         printf("%-8s: skipped, AVERAGE depth resolve not supported\n", cases[i].name);
         continue;
      }
      int bad = run_case(&e, &cases[i]);
      if (bad < 0) {
         api_error = true;
         break;
      }
      ran++;
      if (bad > 0) {
         failed++;
      }
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipelineLayout(dev, e.layout, NULL);
   vkDestroyShaderModule(dev, e.fs_even, NULL);
   vkDestroyShaderModule(dev, e.fs_all, NULL);
   vkDestroyShaderModule(dev, e.vs, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, e.buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (api_error || failed) {
      printf("FAILED (%u of %u depth/stencil resolve cases%s)\n", failed, ran,
             api_error ? ", stopped on an API error" : "");
      return 1;
   }
   printf("PASSED (%u depth/stencil resolve cases: D32F sample zero and average, "
          "%s average, %s sample zero full and partial area)\n", ran,
          format_name(unorm_fmt), format_name(ds_fmt));
   return 0;
}
