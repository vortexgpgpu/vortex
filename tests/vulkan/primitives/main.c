/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Point and line rasterization test for the vortexpipe driver.
 *
 * Each pass draws into a 32x32 RGBA8 target and checks every pixel against the
 * coverage Vulkan defines for it, computed here:
 *   points      a pixel is covered when its centre lies inside the point's
 *               square; gl_PointCoord is its position in that square;
 *   rectangular a pixel is covered when its centre lies inside the rectangle
 *   lines       the line sweeps at its width (strictLines, and wide lines);
 *   Bresenham   a pixel is covered when the segment leaves its diamond
 *   lines       |x - xc| + |y - yc| < 1/2 (VK_EXT_line_rasterization);
 *   polygon     a triangle drawn in LINE mode covers its three edges as
 *   mode line   rectangular lines;
 *   smooth      a line drawn half a pixel larger all round (at least a pixel
 *   lines       each side) writes its coverage falloff
 *               sat(hw - |across|) * min(sat(hl - |along|), length), into alpha;
 *   stipple     a rectangular line keeps the pixels whose counter -- the
 *               distance along it, run on across a strip's segments and
 *               restarted where primitive restart cuts the strip -- picks a
 *               set bit of the pattern repeated `factor` times.
 * Pixels whose centre lies within a hair of an edge go unchecked: either
 * answer is right there.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 32u
#define TIE  1e-3f

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

struct ctx {
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkShaderModule vs, fs;
   bool bresenham;
};

struct pc {
   float v[4][4];
   float size[2];
   int32_t use_coord;
   int32_t pad;
   float inst[2];
};

enum shape {
   POINTS, RECT_LINES, BRESENHAM_LINES, POLY_LINES, SMOOTH_LINES, STIPPLE_LIST, STIPPLE_STRIP,
   STIPPLE_RESTART, POLY_POINTS,
};

#define STIPPLE_PATTERN 0x0f0fu
#define STIPPLE_FACTOR  2u

struct pass {
   const char *name;
   enum shape shape;
   VkPrimitiveTopology topology;
   uint32_t nverts;
   float width;
   float v[4][3];   /* window x, y, point size */
   uint32_t instances;   /* 0: one */
   float inst[2];        /* window offset per instance */
};

/* 1 covered, 0 not, -1 too close to an edge to tell. */
static int
rect_cover(float px, float py, float ax, float ay, float bx, float by, float w)
{
   const float dx = bx - ax, dy = by - ay, len = sqrtf(dx * dx + dy * dy);
   const float t = ((px - ax) * dx + (py - ay) * dy) / len;
   const float d = fabsf((px - ax) * dy - (py - ay) * dx) / len;
   if (fabsf(t) < TIE || fabsf(t - len) < TIE || fabsf(d - 0.5f * w) < TIE)
      return -1;
   return t > 0.0f && t < len && d < 0.5f * w;
}

static float
l1(float x, float y, float cx, float cy)
{
   return fabsf(x - cx) + fabsf(y - cy);
}

/* The smallest L1 distance from the diamond centre to the segment: the
 * distance is convex and piecewise linear in t, so its minimum sits at an end
 * or where the segment crosses the centre's row or column. */
static int
bresenham_cover(float cx, float cy, float ax, float ay, float bx, float by)
{
   float ts[4] = { 0.0f, 1.0f, -1.0f, -1.0f };
   if (bx != ax) ts[2] = (cx - ax) / (bx - ax);
   if (by != ay) ts[3] = (cy - ay) / (by - ay);
   float best = 1e9f;
   for (int i = 0; i < 4; i++) {
      if (ts[i] < 0.0f || ts[i] > 1.0f)
         continue;
      const float d = l1(ax + ts[i] * (bx - ax), ay + ts[i] * (by - ay), cx, cy);
      if (d < best) best = d;
   }
   const float end = l1(bx, by, cx, cy);
   if (fabsf(best - 0.5f) < TIE || fabsf(end - 0.5f) < TIE)
      return -1;
   return best < 0.5f && end >= 0.5f;   /* it must leave the diamond */
}

static int
expected(const struct pass *p, uint32_t x, uint32_t y, float *coord)
{
   const float px = (float)x + 0.5f, py = (float)y + 0.5f;
   int any = 0;
   switch (p->shape) {
   case POINTS:
   case POLY_POINTS:
      for (uint32_t i = 0; i < p->nverts; i++) {
         const float h = 0.5f * p->v[i][2];
         const float u = px - (p->v[i][0] - h), v = py - (p->v[i][1] - h);
         if (fabsf(u) < TIE || fabsf(v) < TIE || fabsf(u - 2 * h) < TIE || fabsf(v - 2 * h) < TIE)
            return -1;
         if (u > 0 && v > 0 && u < 2 * h && v < 2 * h) {
            coord[0] = u / (2 * h);
            coord[1] = v / (2 * h);
            return 1;
         }
      }
      return 0;
   case RECT_LINES:
   case BRESENHAM_LINES:
      for (uint32_t i = 0; i + 1 < p->nverts; i += 2) {
         const int c = p->shape == RECT_LINES
            ? rect_cover(px, py, p->v[i][0], p->v[i][1], p->v[i + 1][0], p->v[i + 1][1], p->width)
            : bresenham_cover(px, py, p->v[i][0], p->v[i][1], p->v[i + 1][0], p->v[i + 1][1]);
         if (c < 0) return -1;
         any |= c;
      }
      return any;
   case SMOOTH_LINES: {
      /* One segment; coord[0] carries the alpha it writes. */
      const float ax = p->v[0][0], ay = p->v[0][1], bx = p->v[1][0], by = p->v[1][1];
      const float dx = bx - ax, dy = by - ay, len = sqrtf(dx * dx + dy * dy);
      const float hw = p->width > 1.0f ? 0.5f * p->width + 0.5f : 1.0f, hl = 0.5f * len + 0.5f;
      const float mx = 0.5f * (ax + bx), my = 0.5f * (ay + by);
      const float across = fabsf((-(px - mx) * dy + (py - my) * dx) / len);
      const float along = fabsf(((px - mx) * dx + (py - my) * dy) / len);
      if (fabsf(across - hw) < TIE || fabsf(along - hl) < TIE)
         return -1;
      const float sa = fminf(fmaxf(hw - across, 0.0f), 1.0f);
      const float sl = fminf(fminf(fmaxf(hl - along, 0.0f), 1.0f), len);
      coord[0] = sa * sl;
      return across < hw && along < hl;
   }
   case STIPPLE_LIST:
   case STIPPLE_STRIP:
   case STIPPLE_RESTART: {
      /* The counter starts each list segment, and each segment that begins a
       * restarted run, at 0, and runs on along a strip by each earlier
       * segment's length rounded up. STIPPLE_RESTART draws v0-v1 and v2-v3 as
       * one strip cut between them. */
      const bool strip = p->shape == STIPPLE_STRIP;
      const uint32_t step = strip ? 1u : 2u;
      float start = 0.0f;
      for (uint32_t i = 0; i + 1 < p->nverts; i += step) {
         const float ax = p->v[i][0], ay = p->v[i][1], bx = p->v[i + 1][0], by = p->v[i + 1][1];
         const float dx = bx - ax, dy = by - ay, len = sqrtf(dx * dx + dy * dy);
         const int c = rect_cover(px, py, ax, ay, bx, by, p->width);
         if (c < 0) return -1;
         if (c) {
            const float count = start + ((px - ax) * dx + (py - ay) * dy) / len;
            if (fabsf(count - roundf(count)) < TIE) return -1;
            const uint32_t bit = ((uint32_t)count / STIPPLE_FACTOR) & 15u;
            any |= (STIPPLE_PATTERN >> bit) & 1u;
         }
         if (strip) start += ceilf(len);
      }
      return any;
   }
   case POLY_LINES:
      for (uint32_t i = 0; i < 3; i++) {
         const uint32_t j = (i + 1) % 3;
         const int c = rect_cover(px, py, p->v[i][0], p->v[i][1], p->v[j][0], p->v[j][1], p->width);
         if (c < 0) return -1;
         any |= c;
      }
      return any;
   }
   return 0;
}

/* Over every instance, each drawn at its own offset. */
static int
expected_all(const struct pass *p, uint32_t x, uint32_t y, float *coord)
{
   const uint32_t n = p->instances ? p->instances : 1u;
   int any = 0;
   for (uint32_t k = 0; k < n; k++) {
      struct pass q = *p;
      for (uint32_t i = 0; i < q.nverts; i++) {
         q.v[i][0] += (float)k * p->inst[0];
         q.v[i][1] += (float)k * p->inst[1];
      }
      const int c = expected(&q, x, y, coord);
      if (c < 0)
         return -1;
      any |= c;
   }
   return any;
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

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) return VK_NULL_HANDLE;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS)
      sm = VK_NULL_HANDLE;
   free(spv);
   return sm;
}

static bool
run_pass(const struct ctx *x, const struct pass *p)
{
   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { SIZE, SIZE, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(x->dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(x->dev, img, &imr);
   uint32_t imt = find_mem(&x->mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return false; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(x->dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(x->dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(x->dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
   CHECK(vkCreateRenderPass(x->dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = SIZE, .height = SIZE, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(x->dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = {
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct pc),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(x->dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = x->vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = x->fs, .pName = "main" },
   };
   const bool restart = p->shape == STIPPLE_RESTART;
   const bool stippled = p->shape == STIPPLE_LIST || p->shape == STIPPLE_STRIP || restart;
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = p->topology, .primitiveRestartEnable = restart,
   };
   VkViewport vp = { 0, 0, (float)SIZE, (float)SIZE, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { SIZE, SIZE } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationLineStateCreateInfoEXT ls = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
      .lineRasterizationMode = p->shape == BRESENHAM_LINES ? VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT
         : p->shape == SMOOTH_LINES || p->shape == POLY_POINTS
            ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
         : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
      .stippledLineEnable = stippled,
      .lineStippleFactor = STIPPLE_FACTOR, .lineStipplePattern = STIPPLE_PATTERN,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = x->bresenham ? &ls : NULL,
      .polygonMode = p->shape == POLY_LINES ? VK_POLYGON_MODE_LINE
                   : p->shape == POLY_POINTS ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = p->width,
   };
   VkPipelineMultisampleStateCreateInfo mss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
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
      .pMultisampleState = &mss, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp,
   };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(x->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize bytes = (VkDeviceSize)SIZE * SIZE * 4u;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(x->dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(x->dev, rb, &bmr);
   uint32_t bmt = find_mem(&x->mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return false; }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(x->dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(x->dev, rb, bmem, 0));

   static const uint16_t restart_idx[] = { 0, 1, 0xffff, 2, 3 };
   VkBuffer ib = VK_NULL_HANDLE;
   VkDeviceMemory imm = VK_NULL_HANDLE;
   if (restart) {
      VkBufferCreateInfo ibci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = sizeof(restart_idx), .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      };
      CHECK(vkCreateBuffer(x->dev, &ibci, NULL, &ib));
      VkMemoryRequirements imr2;
      vkGetBufferMemoryRequirements(x->dev, ib, &imr2);
      VkMemoryAllocateInfo imai2 = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = imr2.size,
         .memoryTypeIndex = find_mem(&x->mp, imr2.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      if (imai2.memoryTypeIndex == UINT32_MAX) {
         fprintf(stderr, "FAILED: no host memory\n");
         return false;
      }
      CHECK(vkAllocateMemory(x->dev, &imai2, NULL, &imm));
      CHECK(vkBindBufferMemory(x->dev, ib, imm, 0));
      void *map;
      CHECK(vkMapMemory(x->dev, imm, 0, VK_WHOLE_SIZE, 0, &map));
      memcpy(map, restart_idx, sizeof(restart_idx));
      vkUnmapMemory(x->dev, imm);
   }

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = x->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(x->dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(x->dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb, .renderArea = { { 0, 0 }, { SIZE, SIZE } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   struct pc pcv = { .size = { (float)SIZE, (float)SIZE }, .use_coord = p->shape == POINTS,
                     .inst = { p->inst[0], p->inst[1] } };
   for (uint32_t i = 0; i < p->nverts; i++) {
      pcv.v[i][0] = p->v[i][0];
      pcv.v[i][1] = p->v[i][1];
      pcv.v[i][2] = p->v[i][2];
      pcv.v[i][3] = 0.0f;
   }
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(pcv), &pcv);
   if (restart) {
      vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, sizeof(restart_idx) / sizeof(restart_idx[0]), 1, 0, 0, 0);
   } else {
      vkCmdDraw(cmd, p->nverts, p->instances ? p->instances : 1u, 0, 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .imageExtent = { SIZE, SIZE, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(x->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(x->queue));

   const uint8_t *px;
   CHECK(vkMapMemory(x->dev, bmem, 0, bytes, 0, (void **)&px));
   unsigned bad = 0, checked = 0;
   for (uint32_t y = 0; y < SIZE; y++) {
      for (uint32_t xx = 0; xx < SIZE; xx++) {
         float coord[2] = { 0, 0 };
         const int want = expected_all(p, xx, y, coord);
         if (want < 0)
            continue;
         checked++;
         const uint8_t *c = &px[(y * SIZE + xx) * 4];
         const bool got = c[1] > 128;
         bool ok = got == (want == 1);
         if (ok && got && p->shape == SMOOTH_LINES) {
            /* alpha to within one 8-bit step and a little rounding */
            ok = fabsf(c[3] / 255.0f - coord[0]) < 0.006f;
         }
         if (ok && got && p->shape == POINTS) {
            /* gl_PointCoord to within one 8-bit step and a little rounding */
            ok = fabsf(c[0] / 255.0f - coord[0]) < 0.01f && fabsf(c[2] / 255.0f - coord[1]) < 0.01f;
         }
         if (ok && got && p->shape == POLY_POINTS) {
            /* line state, smooth here, leaves points alone */
            ok = c[3] == 255;
         }
         if (!ok) {
            if (bad < 4)
               printf("  %s: pixel (%u,%u) %s (rgb %u %u %u)\n", p->name, xx, y,
                      want ? "expected covered" : "expected empty", c[0], c[1], c[2]);
            bad++;
         }
      }
   }
   if (bad) {
      /* the coverage drawn ('#') against the coverage expected ('o': missing,
       * 'x': extra), to read the failure at a glance */
      for (uint32_t y = 0; y < SIZE; y++) {
         char row[SIZE + 1];
         for (uint32_t xx = 0; xx < SIZE; xx++) {
            float coord[2];
            const int want = expected_all(p, xx, y, coord);
            const bool got = px[(y * SIZE + xx) * 4 + 1] > 128;
            row[xx] = want < 0 ? (got ? '+' : ',') : got ? (want ? '#' : 'x') : (want ? 'o' : '.');
         }
         row[SIZE] = 0;
         printf("    %s\n", row);
      }
   }
   vkUnmapMemory(x->dev, bmem);

   vkDestroyCommandPool(x->dev, cp, NULL);
   vkFreeMemory(x->dev, bmem, NULL);
   vkDestroyBuffer(x->dev, rb, NULL);
   if (restart) {
      vkFreeMemory(x->dev, imm, NULL);
      vkDestroyBuffer(x->dev, ib, NULL);
   }
   vkDestroyPipeline(x->dev, pipe, NULL);
   vkDestroyPipelineLayout(x->dev, pl, NULL);
   vkDestroyFramebuffer(x->dev, fb, NULL);
   vkDestroyRenderPass(x->dev, rp, NULL);
   vkDestroyImageView(x->dev, view, NULL);
   vkFreeMemory(x->dev, imem, NULL);
   vkDestroyImage(x->dev, img, NULL);

   printf("%-22s %s (%u wrong of %u checked)\n", p->name, bad ? "FAIL" : "ok", bad, checked);
   return bad == 0;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "primitives.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "primitives.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-primitives", .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
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
   VkPhysicalDeviceLineRasterizationFeaturesEXT lf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &lf,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.wideLines || !f2.features.largePoints || !f2.features.fillModeNonSolid ||
       !lf.rectangularLines || !lf.bresenhamLines || !lf.smoothLines ||
       !lf.stippledRectangularLines) {
      fprintf(stderr, "FAILED: wide lines, large points, polygon modes or line modes missing\n");
      return 1;
   }

   struct ctx x = { 0 };
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   x.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { x.qf = i; break; }
   free(qfp);
   if (x.qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = x.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceLineRasterizationFeaturesEXT lfe = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
      .rectangularLines = VK_TRUE, .bresenhamLines = VK_TRUE, .smoothLines = VK_TRUE,
      .stippledRectangularLines = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 want = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &lfe,
      .features = { .wideLines = VK_TRUE, .largePoints = VK_TRUE, .fillModeNonSolid = VK_TRUE },
   };
   const char *exts[] = { VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts,
   };
   if (vkCreateDevice(pd, &dci, NULL, &x.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   x.bresenham = true;
   vkGetDeviceQueue(x.dev, x.qf, 0, &x.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &x.mp);
   x.vs = load_module(x.dev, vs_path);
   x.fs = load_module(x.dev, fs_path);
   if (!x.vs || !x.fs) return 1;

   const struct pass passes[] = {
      { "points", POINTS, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 3, 1.0f,
        { { 5.5f, 5.5f, 1.0f }, { 20.3f, 10.7f, 4.0f }, { 9.2f, 26.6f, 7.0f } } },
      { "rectangular lines", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 4, 1.0f,
        { { 3.2f, 25.7f, 0 }, { 28.6f, 27.3f, 0 }, { 4.3f, 2.1f, 0 }, { 9.7f, 20.9f, 0 } } },
      { "wide lines", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 2, 3.0f,
        { { 2.6f, 4.4f, 0 }, { 29.3f, 21.8f, 0 } } },
      { "bresenham lines", BRESENHAM_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 4, 1.0f,
        { { 2.3f, 3.6f, 0 }, { 29.1f, 18.4f, 0 }, { 26.7f, 2.2f, 0 }, { 21.4f, 29.8f, 0 } } },
      /* Wider than the target: the rectangles overhang it on every side. */
      { "overhanging lines", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 4, 31.0f,
        { { 30.4f, 27.2f, 0 }, { 1.6f, 27.2f, 0 }, { 1.6f, 11.2f, 0 }, { 33.6f, 1.6f, 0 } } },
      { "overhanging line", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 2, 31.0f,
        { { 30.4f, 27.2f, 0 }, { 1.6f, 27.2f, 0 } } },
      { "overhanging lines 2", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 4, 31.0f,
        { { 27.2f, 14.4f, 0 }, { 17.76f, 19.2f, 0 }, { 30.08f, 27.2f, 0 }, { 28.8f, 4.8f, 0 } } },
      { "smooth line", SMOOTH_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 2, 1.0f,
        { { 3.3f, 5.2f, 0 }, { 27.6f, 13.9f, 0 } } },
      { "wide smooth line", SMOOTH_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 2, 4.0f,
        { { 6.4f, 27.1f, 0 }, { 24.9f, 8.3f, 0 } } },
      { "polygon mode point", POLY_POINTS, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 1.0f,
        { { 5.5f, 5.5f, 1.0f }, { 20.3f, 10.7f, 4.0f }, { 9.2f, 26.6f, 7.0f } } },
      /* The third vertex of each instance is a partial line, never drawn. */
      { "instanced lines", RECT_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 3, 1.0f,
        { { 3.2f, 4.7f, 0 }, { 27.6f, 9.3f, 0 }, { 14.1f, 13.8f, 0 } }, 2, { 0.0f, 14.0f } },
      { "instanced stipple strip", STIPPLE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 3, 1.0f,
        { { 1.6f, 4.3f, 0 }, { 15.3f, 7.7f, 0 }, { 30.4f, 5.2f, 0 } }, 2, { 0.0f, 16.0f } },
      { "polygon mode line", POLY_LINES, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 3, 1.0f,
        { { 3.3f, 3.7f, 0 }, { 28.4f, 6.2f, 0 }, { 12.6f, 28.1f, 0 } } },
      { "stipple list", STIPPLE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 4, 1.0f,
        { { 1.3f, 4.6f, 0 }, { 30.2f, 7.1f, 0 }, { 2.4f, 24.3f, 0 }, { 29.7f, 28.8f, 0 } } },
      { "stipple strip", STIPPLE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 3, 1.0f,
        { { 1.6f, 12.3f, 0 }, { 15.3f, 15.7f, 0 }, { 30.4f, 13.2f, 0 } } },
      { "stipple restart", STIPPLE_RESTART, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 4, 1.0f,
        { { 1.3f, 4.6f, 0 }, { 30.2f, 7.1f, 0 }, { 2.4f, 24.3f, 0 }, { 29.7f, 28.8f, 0 } } },
   };
   unsigned failed = 0;
   for (unsigned i = 0; i < sizeof(passes) / sizeof(passes[0]); i++)
      failed += !run_pass(&x, &passes[i]);

   vkDestroyShaderModule(x.dev, x.vs, NULL);
   vkDestroyShaderModule(x.dev, x.fs, NULL);
   vkDestroyDevice(x.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (primitives: %u pass(es) wrong)\n", failed);
      return 1;
   }
   printf("PASSED (primitives: points, rectangular/wide/Bresenham lines, polygon "
          "modes, smooth lines, stipple, stipple with restart, instancing)\n");
   return 0;
}
