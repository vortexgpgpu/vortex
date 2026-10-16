/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Precise occlusion queries around draws the device runs, against a depth
 * buffer, so the count is the samples that passed the depth test and not
 * merely the samples covered.
 *
 * The query is counted on the device path -- by the fixed-function merger's
 * passing-fragment counter or by the software merger kernels -- and a draw
 * whose count never reaches the query leaves it short with no error. Each
 * configuration renders the same scene, so the two mergers are held to the
 * same answers:
 *
 *   ff      R8G8B8A8_UNORM, 1 sample        fixed-function merger
 *   msaa4   R8G8B8A8_UNORM, 4 samples       software merger, counts samples
 *   rgba32f R32G32B32A32_SFLOAT, 1 sample   software merger
 *
 * The depth buffer is cleared to 1.0 and every test is LESS. With R the
 * 20x44-pixel rectangle at the top-left corner:
 *
 *   occluder  R at depth 0.25, depth write on         -> all of R passes
 *   probe     full screen at 0.5, depth write off     -> all but R passes
 *   hidden    R at 0.75                               -> nothing passes
 *   proxy     probe with colour writes masked off     -> as probe: a query
 *             proxy draw writes nothing, yet still counts
 *   sum       probe, hidden, probe in one query       -> twice the probe
 *
 * Rectangle edges lie on pixel boundaries, so a pixel is wholly in or out and
 * each multisampled pixel contributes all of its samples or none. The resolved
 * colour is checked first -- R red, the rest green; blue means the hidden draw
 * wrote, white that the masked draw did -- so a draw that went wrong is
 * reported as such instead of being blamed on the query.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define NPX       (WIDTH * HEIGHT)
#define DEPTH_FMT VK_FORMAT_D32_SFLOAT

/* The occluder rectangle R, in pixels from the top-left corner. */
#define R_W       20u
#define R_H       44u
#define R_PX      (R_W * R_H)

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return -1;                                                   \
   }                                                               \
} while (0)

struct config {
   const char           *name;
   VkFormat              format;
   uint32_t              bpp;
   VkSampleCountFlagBits samples;
};

static const struct config configs[] = {
   { "ff",      VK_FORMAT_R8G8B8A8_UNORM,      4,  VK_SAMPLE_COUNT_1_BIT },
   { "msaa4",   VK_FORMAT_R8G8B8A8_UNORM,      4,  VK_SAMPLE_COUNT_4_BIT },
   { "rgba32f", VK_FORMAT_R32G32B32A32_SFLOAT, 16, VK_SAMPLE_COUNT_1_BIT },
};
#define NCONFIGS (sizeof configs / sizeof configs[0])

enum { Q_OCCLUDER, Q_PROBE, Q_HIDDEN, Q_PROXY, Q_SUM, NQUERIES };

static const char *const query_names[NQUERIES] = {
   "occluder", "probe", "hidden", "proxy", "sum",
};

/* Passing pixels per query; a multisampled pixel adds all of its samples. */
static const uint64_t want_px[NQUERIES] = {
   [Q_OCCLUDER] = R_PX,
   [Q_PROBE]    = NPX - R_PX,
   [Q_HIDDEN]   = 0,
   [Q_PROXY]    = NPX - R_PX,
   [Q_SUM]      = 2 * (NPX - R_PX),
};

struct push {
   float rect[4];
   float color[4];
   float depth;
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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
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

static int
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
           VkFormat format, VkSampleCountFlagBits samples,
           VkImageUsageFlags usage, VkImageAspectFlags aspect,
           VkImage *out_img, VkDeviceMemory *out_mem, VkImageView *out_view)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(dev, &imci, NULL, out_img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, *out_img, &imr);
   uint32_t imt = find_mem(mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no image memory\n");
      return -1;
   }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   CHECK(vkAllocateMemory(dev, &imai, NULL, out_mem));
   CHECK(vkBindImageMemory(dev, *out_img, *out_mem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *out_img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1,
                            .layerCount = 1 },
   };
   CHECK(vkCreateImageView(dev, &ivci, NULL, out_view));
   return 0;
}

static int
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl,
              VkShaderModule vs, VkShaderModule fs,
              VkSampleCountFlagBits samples, VkBool32 depth_write,
              VkColorComponentFlags color_mask, VkPipeline *out)
{
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
   VkViewport vp = { 0, 0, (float)WIDTH, (float)HEIGHT, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { WIDTH, HEIGHT } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = samples,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = depth_write,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE, .colorWriteMask = color_mask,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &dss,
      .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out));
   return 0;
}

/* Pixel x or y to clip space; exact in float for a power-of-two target. */
static float
to_clip(uint32_t p, uint32_t extent)
{
   return 2.0f * (float)p / (float)extent - 1.0f;
}

static void
draw_quad(VkCommandBuffer cmd, VkPipelineLayout pl, bool full_screen,
          float depth, const float color[4])
{
   struct push pc = {
      .rect = { -1.0f, -1.0f, 1.0f, 1.0f },
      .depth = depth,
   };
   if (!full_screen) {
      pc.rect[2] = to_clip(R_W, WIDTH);
      pc.rect[3] = to_clip(R_H, HEIGHT);
   }
   memcpy(pc.color, color, sizeof pc.color);
   vkCmdPushConstants(cmd, pl,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof pc, &pc);
   vkCmdDraw(cmd, 6, 1, 0, 0);
}

/* Which draw's colour a pixel holds, from its channels thresholded at half. */
enum px_kind { PX_CLEAR, PX_RED, PX_GREEN, PX_BLUE, PX_WHITE, PX_OTHER };

static enum px_kind
classify(bool r, bool g, bool b)
{
   if (!r && !g && !b) {
      return PX_CLEAR;
   }
   if (r && !g && !b) {
      return PX_RED;
   }
   if (!r && g && !b) {
      return PX_GREEN;
   }
   if (!r && !g && b) {
      return PX_BLUE;
   }
   if (r && g && b) {
      return PX_WHITE;
   }
   return PX_OTHER;
}

/* Render the scene in one configuration, returning the query results and the
 * number of pixels that do not hold the colour the scene leaves there.
 * Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp, VkShaderModule vs,
       VkShaderModule fs, const struct config *cfg, uint64_t got[NQUERIES],
       uint32_t *out_bad_px)
{
   const bool ms = cfg->samples != VK_SAMPLE_COUNT_1_BIT;

   VkImage cimg, dimg, rimg = VK_NULL_HANDLE;
   VkDeviceMemory cmem, dmem, rmem = VK_NULL_HANDLE;
   VkImageView cview, dview, rview = VK_NULL_HANDLE;
   if (make_image(dev, mp, cfg->format, cfg->samples,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  (ms ? 0 : VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
                  VK_IMAGE_ASPECT_COLOR_BIT, &cimg, &cmem, &cview) < 0 ||
       make_image(dev, mp, DEPTH_FMT, cfg->samples,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, &dimg, &dmem, &dview) < 0) {
      return -1;
   }
   if (ms && make_image(dev, mp, cfg->format, VK_SAMPLE_COUNT_1_BIT,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, &rimg, &rmem, &rview) < 0) {
      return -1;
   }

   VkAttachmentDescription atts[3] = {
      { .format = cfg->format, .samples = cfg->samples,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                      : VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                          : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = DEPTH_FMT, .samples = cfg->samples,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
      { .format = cfg->format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference dref = {
      .attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference rref = {
      .attachment = 2, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pResolveAttachments = ms ? &rref : NULL,
      .pDepthStencilAttachment = &dref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = ms ? 3u : 2u, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView views[3] = { cview, dview, rview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = ms ? 3u : 2u, .pAttachments = views,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0, .size = sizeof(struct push),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   const VkColorComponentFlags rgba =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   VkPipeline p_write, p_test, p_proxy;
   if (make_pipeline(dev, rp, pl, vs, fs, cfg->samples, VK_TRUE, rgba,
                     &p_write) < 0 ||
       make_pipeline(dev, rp, pl, vs, fs, cfg->samples, VK_FALSE, rgba,
                     &p_test) < 0 ||
       make_pipeline(dev, rp, pl, vs, fs, cfg->samples, VK_FALSE, 0,
                     &p_proxy) < 0) {
      return -1;
   }

   const VkDeviceSize bytes = (VkDeviceSize)NPX * cfg->bpp;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   uint32_t bmt = find_mem(mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no host memory\n");
      return -1;
   }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, rb, bmem, 0));

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   VkQueryPoolCreateInfo qpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = NQUERIES,
   };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));

   static const float red[4]   = { 1.0f, 0.0f, 0.0f, 1.0f };
   static const float green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
   static const float blue[4]  = { 0.0f, 0.0f, 1.0f, 1.0f };
   static const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   vkCmdResetQueryPool(cmd, qp, 0, NQUERIES);
   VkClearValue clears[3] = {
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
      { .depthStencil = { .depth = 1.0f } },
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = ms ? 3u : 2u, .pClearValues = clears,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   const VkQueryControlFlags precise = VK_QUERY_CONTROL_PRECISE_BIT;

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_write);
   vkCmdBeginQuery(cmd, qp, Q_OCCLUDER, precise);
   draw_quad(cmd, pl, false, 0.25f, red);
   vkCmdEndQuery(cmd, qp, Q_OCCLUDER);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_test);
   vkCmdBeginQuery(cmd, qp, Q_PROBE, precise);
   draw_quad(cmd, pl, true, 0.5f, green);
   vkCmdEndQuery(cmd, qp, Q_PROBE);

   vkCmdBeginQuery(cmd, qp, Q_HIDDEN, precise);
   draw_quad(cmd, pl, false, 0.75f, blue);
   vkCmdEndQuery(cmd, qp, Q_HIDDEN);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_proxy);
   vkCmdBeginQuery(cmd, qp, Q_PROXY, precise);
   draw_quad(cmd, pl, true, 0.5f, white);
   vkCmdEndQuery(cmd, qp, Q_PROXY);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_test);
   vkCmdBeginQuery(cmd, qp, Q_SUM, precise);
   draw_quad(cmd, pl, true, 0.5f, green);
   draw_quad(cmd, pl, false, 0.75f, blue);
   draw_quad(cmd, pl, true, 0.5f, green);
   vkCmdEndQuery(cmd, qp, Q_SUM);

   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, ms ? rimg : cimg,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* WAIT so each result is the finished count and not a partial one. */
   CHECK(vkGetQueryPoolResults(dev, qp, 0, NQUERIES,
                               NQUERIES * sizeof(uint64_t), got,
                               sizeof(uint64_t),
                               VK_QUERY_RESULT_64_BIT |
                               VK_QUERY_RESULT_WAIT_BIT));
   vkDestroyQueryPool(dev, qp, NULL);

   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
   *out_bad_px = 0;
   unsigned kinds[PX_OTHER + 1] = { 0 };
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * cfg->bpp;
         bool c[3];
         for (unsigned k = 0; k < 3; k++) {
            if (cfg->bpp == 16) {
               float f;
               memcpy(&f, p + k * 4, sizeof f);
               c[k] = f > 0.5f;
            } else {
               c[k] = p[k] > 127;
            }
         }
         const enum px_kind got_kind = classify(c[0], c[1], c[2]);
         const enum px_kind want_kind = (x < R_W && y < R_H) ? PX_RED : PX_GREEN;
         kinds[got_kind]++;
         if (got_kind != want_kind) {
            (*out_bad_px)++;
         }
      }
   }
   vkUnmapMemory(dev, bmem);
   if (*out_bad_px) {
      printf("  %s: footprint clear=%u red=%u green=%u blue=%u white=%u "
             "other=%u (want red=%u green=%u)\n", cfg->name, kinds[PX_CLEAR],
             kinds[PX_RED], kinds[PX_GREEN], kinds[PX_BLUE], kinds[PX_WHITE],
             kinds[PX_OTHER], R_PX, NPX - R_PX);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, p_write, NULL);
   vkDestroyPipeline(dev, p_test, NULL);
   vkDestroyPipeline(dev, p_proxy, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, cview, NULL);
   vkDestroyImage(dev, cimg, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkFreeMemory(dev, dmem, NULL);
   if (ms) {
      vkDestroyImageView(dev, rview, NULL);
      vkDestroyImage(dev, rimg, NULL);
      vkFreeMemory(dev, rmem, NULL);
   }
   return 0;
}

static bool
config_supported(VkPhysicalDevice pd, const struct config *cfg)
{
   VkImageFormatProperties ifp;
   if (vkGetPhysicalDeviceImageFormatProperties(pd, cfg->format,
          VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &ifp) != VK_SUCCESS ||
       !(ifp.sampleCounts & cfg->samples)) {
      return false;
   }
   if (vkGetPhysicalDeviceImageFormatProperties(pd, DEPTH_FMT,
          VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0, &ifp) != VK_SUCCESS ||
       !(ifp.sampleCounts & cfg->samples)) {
      return false;
   }
   return true;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "occlusion.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "occlusion.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-occlusion",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
      return 1;
   }

   uint32_t npd = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &npd, &pd) != VK_SUCCESS || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* Precise occlusion queries are an opt-in core feature; without it the
    * result is only guaranteed to be non-zero when any sample passed. */
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!have.occlusionQueryPrecise) {
      printf("FAILED (occlusion: occlusionQueryPrecise not advertised)\n");
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
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
   VkPhysicalDeviceFeatures feats = { .occlusionQueryPrecise = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pEnabledFeatures = &feats,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: cannot create device\n");
      return 1;
   }
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      return 1;
   }

   unsigned bad = 0;
   for (unsigned c = 0; c < NCONFIGS; c++) {
      const struct config *cfg = &configs[c];
      /* Every configuration is one the device is required to render, so a
       * missing one is a lost capability, not an opt-out. */
      if (!config_supported(pd, cfg)) {
         printf("  %s: format/sample count not supported\n", cfg->name);
         bad++;
         continue;
      }
      uint64_t got[NQUERIES];
      uint32_t bad_px = 0;
      if (render(dev, queue, qf, &mp, vs, fs, cfg, got, &bad_px) < 0) {
         return 1;
      }
      printf("%s:", cfg->name);
      for (unsigned q = 0; q < NQUERIES; q++) {
         printf(" %s=%llu", query_names[q], (unsigned long long)got[q]);
      }
      printf("\n");
      if (bad_px) {
         printf("  %s: %u px hold the wrong draw, so its query results cannot "
                "be judged\n", cfg->name, bad_px);
         bad++;
         continue;
      }
      for (unsigned q = 0; q < NQUERIES; q++) {
         const uint64_t want = want_px[q] * (uint64_t)cfg->samples;
         if (got[q] != want) {
            printf("  %s %s: counted %llu samples, want %llu\n", cfg->name,
                   query_names[q], (unsigned long long)got[q],
                   (unsigned long long)want);
            bad++;
         }
      }
   }

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad) {
      printf("FAILED (occlusion: %u checks wrong; zero means the counter was "
             "never updated, and a count of covered rather than passing "
             "samples means the depth test was not applied to it)\n", bad);
      return 1;
   }
   printf("PASSED (occlusion: ff, msaa4 and rgba32f each counted occluder "
          "%u, probe %u, hidden 0, proxy %u and sum %u px worth of samples)\n",
          R_PX, NPX - R_PX, NPX - R_PX, 2 * (NPX - R_PX));
   return 0;
}
