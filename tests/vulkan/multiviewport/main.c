/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Viewport arrays, per-viewport scissors, gl_ViewportIndex and gl_Layer for the
 * vortexpipe driver. Each primitive selects its viewport (and with it the
 * transform and the scissor) through the gl_ViewportIndex its vertex shader
 * writes, or its framebuffer layer through gl_Layer; the fragment shader reads
 * both back and stamps them into the colour:
 *
 *   red   = 40 + 50 * gl_ViewportIndex
 *   green = (10 + 60 * gl_Layer) mod 256
 *
 * Cases, every pixel compared exactly against a host model:
 *
 *   viewports  four viewports in the quadrants of a 64x64 target, the last
 *              one y-flipped, each with its own scissor crossing its quad's
 *              edges at odd offsets. The quad each draws covers the lower left
 *              three quarters of its viewport, so geometry clips two sides and
 *              the scissor the others.
 *   overlap    four full-target viewports, one full-target quad each, and four
 *              overlapping scissors: only the per-primitive scissor separates
 *              the primitives, and the last one drawn wins where they overlap.
 *   msaa       the viewports case at 4 samples, resolved: the software
 *              rasterizer's per-primitive rectangle.
 *   layers     two quads into a two-layer target, routed by gl_Layer.
 *
 * The remaining cases take every vertex from a vertex buffer that also carries
 * the viewport index, layer and point size it writes:
 *
 *   depthclip  three viewports with depth ranges [0, 0.5], [0.5, 1] and the
 *              single value 0.25 over a D32_SFLOAT attachment; each draws a
 *              quad whose clip z runs -0.5..1.5 across it, so the part past
 *              the near plane and the part past the far plane are clipped.
 *              Viewport 0's far part maps to depths inside [0, 1]: only its
 *              own range drops it.
 *   depthclamp the same with depth clamp: every pixel is drawn, the ones past
 *              either plane at that viewport's own bound.
 *   vpclamp    gl_ViewportIndex 7, 2 (the count) and -1 with two viewports
 *              bound: drawn through viewport 0 and its scissor.
 *   layerclamp gl_Layer 5 and 3 into a three-layer target: the last layer.
 *   linepoint  lines whose second vertex writes a different viewport index
 *              than the first (provoking) one, and points of several sizes,
 *              each through its own quadrant viewport.
 *   layerblend two layered draws into a three-layer target with a D32_SFLOAT
 *              attachment, LESS and additive blending: each pixel's red counts
 *              the fragments blended into it, so a layer replayed twice, a
 *              primitive drawn into another layer, or a layer tested against
 *              another layer's depth shows.
 *   layers6    the same over six layers of an eight-layer image, the
 *              framebuffer starting at its layer 1: the image's other layers
 *              must keep their clear.
 *   layers64   64 layers of a 16x16 target, depth tested and blended, every
 *              fifth layer selected by no primitive and one primitive's
 *              gl_Layer past the last layer.
 *   layersR    a colour image taller than the framebuffer and a depth image
 *              taller still, so the two step a layer by different row counts.
 *   layers3d   a 2D array view of slices 1..4 of a six-slice 3D image.
 *   layersms   four layers at 4 samples, blended and resolved, with every
 *              edge on a pixel boundary so the resolve is exact.
 *
 * A driver that draws every primitive through viewport 0, scissors them all
 * with scissor 0, or renders every layer into layer 0 fails each case on most
 * of its pixels.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FORMAT       VK_FORMAT_R8G8B8A8_UNORM
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT
#define MAX_VP       4u
#define MAX_PRIMS    96u
#define MAX_DRAWS    2u
/* The interpolated depth plane is float arithmetic on the device; every
 * expectation that matters differs by at least 1/64. */
#define DEPTH_TOL    1e-5

/* The clear: blue, which no fragment writes. */
static const uint8_t k_clear[4] = { 0, 0, 255, 255 };

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

struct ctx {
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkShaderModule vs_viewport, vs_layer, vs_attr, fs;
};

/* One case: the target, the viewport array, and the draw. */
struct mv_case {
   const char *name;
   uint32_t width, height, layers, samples;
   uint32_t vp_count;
   VkViewport vp[MAX_VP];
   VkRect2D sc[MAX_VP];
   bool layered;      /* gl_Layer routing, else gl_ViewportIndex */
   float quad_hi;     /* viewport cases: each quad spans NDC [-1, quad_hi] */
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
   if (out_size) *out_size = (size_t)sz;
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

static int
alloc_image(struct ctx *c, uint32_t w, uint32_t h, uint32_t layers, uint32_t samples,
            VkFormat format, VkImageAspectFlags aspect, VkImageUsageFlags usage,
            VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = layers,
      .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, *img, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&c->mp, mr.memoryTypeBits, 0),
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(c->dev, *img, *mem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img,
      .viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1, .layerCount = layers },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, view));
   return 0;
}

/* A window-space interval [lo, hi) a quad spanning NDC [-1, ndc_hi] covers
 * through a viewport axis (origin, extent); the extent may be negative. */
static void
quad_span(float origin, float extent, float ndc_hi, int32_t *lo, int32_t *hi)
{
   const float a = origin, b = origin + (ndc_hi + 1.0f) * 0.5f * extent;
   *lo = (int32_t)(a < b ? a : b);
   *hi = (int32_t)(a < b ? b : a);
}

/* The colour the case leaves at (layer, x, y). */
static void
expect_pixel(const struct mv_case *k, uint32_t layer, int32_t x, int32_t y, uint8_t out[4])
{
   memcpy(out, k_clear, 4);
   if (k->layered) {
      /* Quad 0 in layer 0 covers NDC [-1,0]^2, quad 1 in layer 1 [0,1]^2. */
      const int32_t hw = (int32_t)k->width / 2, hh = (int32_t)k->height / 2;
      const bool in0 = x < hw && y < hh, in1 = x >= hw && y >= hh;
      if ((layer == 0 && in0) || (layer == 1 && in1)) {
         out[0] = 40;
         out[1] = (uint8_t)(10 + 60 * layer);
         out[2] = 0;
         out[3] = 255;
      }
      return;
   }
   /* Primitives are drawn in viewport order, so the last one covering the
    * pixel decides it. */
   for (uint32_t i = 0; i < k->vp_count; i++) {
      int32_t x0, x1, y0, y1;
      quad_span(k->vp[i].x, k->vp[i].width, k->quad_hi, &x0, &x1);
      quad_span(k->vp[i].y, k->vp[i].height, k->quad_hi, &y0, &y1);
      const VkRect2D *s = &k->sc[i];
      const bool in = x >= x0 && x < x1 && y >= y0 && y < y1 &&
                      x >= s->offset.x && x < s->offset.x + (int32_t)s->extent.width &&
                      y >= s->offset.y && y < s->offset.y + (int32_t)s->extent.height;
      if (in) {
         out[0] = (uint8_t)(40 + 50 * i);
         out[1] = 10;
         out[2] = 0;
         out[3] = 255;
      }
   }
}

static int
run_case(struct ctx *c, const struct mv_case *k, unsigned *failed)
{
   const bool ms = k->samples > 1;
   VkImage img, rimg = VK_NULL_HANDLE;
   VkDeviceMemory imem, rmem = VK_NULL_HANDLE;
   VkImageView view, rview = VK_NULL_HANDLE;
   if (alloc_image(c, k->width, k->height, k->layers, k->samples,
                   FORMAT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                   (ms ? 0 : VK_IMAGE_USAGE_TRANSFER_SRC_BIT), &img, &imem, &view))
      return 1;
   if (ms && alloc_image(c, k->width, k->height, k->layers, 1, FORMAT, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &rimg, &rmem, &rview))
      return 1;

   VkAttachmentDescription att[2] = {
      { .format = FORMAT, .samples = (VkSampleCountFlagBits)k->samples,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                          : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference rref = { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pResolveAttachments = ms ? &rref : NULL,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = ms ? 2 : 1, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &rp));
   VkImageView fbviews[2] = { view, rview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = ms ? 2 : 1, .pAttachments = fbviews,
      .width = k->width, .height = k->height, .layers = k->layers,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, &pl));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = k->layered ? c->vs_layer : c->vs_viewport, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = c->fs, .pName = "main" },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = k->vp_count, .pViewports = k->vp,
      .scissorCount = k->vp_count, .pScissors = k->sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo msi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
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
      .pMultisampleState = &msi, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize layer_bytes = (VkDeviceSize)k->width * k->height * 4;
   const VkDeviceSize bytes = layer_bytes * k->layers;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(c->dev, rb, &bmr);
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size,
      .memoryTypeIndex = find_mem(&c->mp, bmr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(c->dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(c->dev, rb, bmem, 0));

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(c->dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkClearValue clear = { .color = { .float32 = { k_clear[0] / 255.0f, k_clear[1] / 255.0f,
                                                   k_clear[2] / 255.0f, k_clear[3] / 255.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { k->width, k->height } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &k->quad_hi);
   vkCmdDraw(cmd, 6u * (k->layered ? k->layers : k->vp_count), 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = k->layers },
      .imageExtent = { k->width, k->height, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, ms ? rimg : img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   uint8_t *px;
   CHECK(vkMapMemory(c->dev, bmem, 0, bytes, 0, (void **)&px));
   uint32_t bad = 0, drawn = 0;
   for (uint32_t l = 0; l < k->layers; l++) {
      for (uint32_t y = 0; y < k->height; y++) {
         for (uint32_t x = 0; x < k->width; x++) {
            const uint8_t *p = px + l * layer_bytes + ((size_t)y * k->width + x) * 4;
            uint8_t want[4];
            expect_pixel(k, l, (int32_t)x, (int32_t)y, want);
            drawn += memcmp(want, k_clear, 4) != 0;
            if (memcmp(p, want, 4) != 0) {
               if (bad < 8)
                  printf("  %s: layer %u (%u,%u) = %u,%u,%u,%u, want %u,%u,%u,%u\n",
                         k->name, l, x, y, p[0], p[1], p[2], p[3],
                         want[0], want[1], want[2], want[3]);
               bad++;
            }
         }
      }
   }
   vkUnmapMemory(c->dev, bmem);
   const uint32_t total = k->width * k->height * k->layers;
   printf("%-9s: %s  %u/%u px match (%u drawn)\n", k->name, bad ? "FAIL" : "pass",
          total - bad, total, drawn);
   if (bad)
      (*failed)++;

   vkDestroyCommandPool(c->dev, cp, NULL);
   vkFreeMemory(c->dev, bmem, NULL);
   vkDestroyBuffer(c->dev, rb, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyFramebuffer(c->dev, fb, NULL);
   vkDestroyRenderPass(c->dev, rp, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkFreeMemory(c->dev, imem, NULL);
   vkDestroyImage(c->dev, img, NULL);
   if (ms) {
      vkDestroyImageView(c->dev, rview, NULL);
      vkFreeMemory(c->dev, rmem, NULL);
      vkDestroyImage(c->dev, rimg, NULL);
   }
   return 0;
}

/* A primitive of the vertex-buffer cases, in clip space at w = 1. Its depth runs
 * along x only, so a quad's two triangles share one plane. */
enum shape { QUAD, LINE, POINT };

struct prim {
   enum shape shape;
   float x0, y0, x1, y1;   /* quad corners, line ends, or the point's centre (x0, y0) */
   float z0, z1;           /* clip z at x0 and at x1 */
   float size;             /* point size */
   int32_t vpi, layer;     /* written by the first (provoking) vertex */
   int32_t vpi_end;        /* written by a line's second vertex */
};

/* A draw: `count` primitives of one shape from `first`. */
struct draw_range {
   uint32_t first, count;
};

struct scene {
   const char *name;
   uint32_t width, height, layers;
   /* The images behind the framebuffer: img_layers layers (3D slices when
    * is_3d), of which the framebuffer renders `layers` from base_layer; the
    * colour image img_h rows tall and the depth image zimg_h (0: height). */
   uint32_t img_layers, base_layer, img_h, zimg_h;
   bool is_3d;
   uint32_t samples;       /* 0 or 1, or 4: resolved into a single-sample image */
   uint32_t vp_count;
   VkViewport vp[MAX_VP];
   VkRect2D sc[MAX_VP];
   bool depth;             /* a D32_SFLOAT attachment, tested with depth_op and written */
   float depth_clear;
   VkCompareOp depth_op;
   bool depth_clamp;
   bool blend_add;         /* ONE + ONE on every channel */
   uint32_t ndraws;
   struct draw_range draws[MAX_DRAWS];
   uint32_t nprims;
   struct prim p[MAX_PRIMS];
};

struct vtx {
   float pos[4];
   float sel[4];           /* viewport index, layer, point size */
};

static uint32_t
prim_vertices(const struct prim *p, struct vtx *out)
{
   const float sel[4] = { (float)p->vpi, (float)p->layer, p->size, 0.0f };
   switch (p->shape) {
   case QUAD: {
      const float c[6][2] = { { p->x0, p->y0 }, { p->x1, p->y0 }, { p->x1, p->y1 },
                              { p->x0, p->y0 }, { p->x1, p->y1 }, { p->x0, p->y1 } };
      for (unsigned i = 0; i < 6; i++) {
         const float pos[4] = { c[i][0], c[i][1], c[i][0] == p->x0 ? p->z0 : p->z1, 1.0f };
         memcpy(out[i].pos, pos, sizeof pos);
         memcpy(out[i].sel, sel, sizeof sel);
      }
      return 6;
   }
   case LINE: {
      const float a[4] = { p->x0, p->y0, p->z0, 1.0f }, b[4] = { p->x1, p->y1, p->z1, 1.0f };
      memcpy(out[0].pos, a, sizeof a);
      memcpy(out[0].sel, sel, sizeof sel);
      memcpy(out[1].pos, b, sizeof b);
      memcpy(out[1].sel, sel, sizeof sel);
      out[1].sel[0] = (float)p->vpi_end;
      return 2;
   }
   default: {
      const float a[4] = { p->x0, p->y0, p->z0, 1.0f };
      memcpy(out[0].pos, a, sizeof a);
      memcpy(out[0].sel, sel, sizeof sel);
      return 1;
   }
   }
}

static double dmin(double a, double b) { return a < b ? a : b; }
static double dmax(double a, double b) { return a > b ? a : b; }

static double
to_window(float ndc, float origin, float extent)
{
   return origin + ((double)ndc + 1.0) * 0.5 * extent;
}

/* Whether the primitive covers the pixel centre (xc, yc) in window space, and
 * where along x it lies (0 at x0, 1 at x1). The cases keep every centre off the
 * edges, so no tie-break rule matters; lines are horizontal, through a row of
 * centres, from a quarter pixel past one centre to a quarter pixel past
 * another, which the rectangle and the diamond-exit rules cover alike. */
static bool
prim_covers(const struct prim *p, double ax, double ay, double bx, double by,
            double xc, double yc, double *t)
{
   *t = bx != ax ? (xc - ax) / (bx - ax) : 0.0;
   switch (p->shape) {
   case QUAD:
      return xc >= dmin(ax, bx) && xc < dmax(ax, bx) && yc >= dmin(ay, by) && yc < dmax(ay, by);
   case LINE:
      return yc == ay && xc >= dmin(ax, bx) && xc < dmax(ax, bx);
   default:
      return fabs(xc - ax) < 0.5 * p->size && fabs(yc - ay) < 0.5 * p->size;
   }
}

static bool
depth_passes(VkCompareOp op, float z, float stored)
{
   switch (op) {
   case VK_COMPARE_OP_LESS:          return z < stored;
   case VK_COMPARE_OP_LESS_OR_EQUAL: return z <= stored;
   default:                          return true;
   }
}

/* The layers (3D slices) of the images behind a scene's framebuffer. */
static uint32_t
scene_img_layers(const struct scene *s)
{
   return s->img_layers ? s->img_layers : s->layers;
}

/* The colour and depth planes the scene leaves, layer after layer. A viewport
 * index past the array selects viewport 0 and a layer past the framebuffer its
 * last layer; the fragment shader then reads those. */
static void
model_scene(const struct scene *s, uint8_t *col, float *dep)
{
   const size_t plane = (size_t)s->width * s->height;
   for (size_t i = 0; i < plane * scene_img_layers(s); i++) {
      memcpy(col + 4 * i, k_clear, 4);
      dep[i] = s->depth_clear;
   }
   for (uint32_t n = 0; n < s->nprims; n++) {
      const struct prim *p = &s->p[n];
      const uint32_t vi = (uint32_t)p->vpi < s->vp_count ? (uint32_t)p->vpi : 0u;
      const uint32_t li = (uint32_t)p->layer < s->layers ? (uint32_t)p->layer : s->layers - 1;
      const VkViewport *v = &s->vp[vi];
      const VkRect2D *sc = &s->sc[vi];
      const uint8_t src[4] = { (uint8_t)(40 + 50 * vi), (uint8_t)(10 + 60 * li), 0, 255 };
      const double ax = to_window(p->x0, v->x, v->width), bx = to_window(p->x1, v->x, v->width);
      const double ay = to_window(p->y0, v->y, v->height), by = to_window(p->y1, v->y, v->height);
      const double zlo = dmin(v->minDepth, v->maxDepth), zhi = dmax(v->minDepth, v->maxDepth);
      for (uint32_t y = 0; y < s->height; y++) {
         for (uint32_t x = 0; x < s->width; x++) {
            if ((int32_t)x < sc->offset.x || (int32_t)x >= sc->offset.x + (int32_t)sc->extent.width ||
                (int32_t)y < sc->offset.y || (int32_t)y >= sc->offset.y + (int32_t)sc->extent.height)
               continue;
            double t;
            if (!prim_covers(p, ax, ay, bx, by, x + 0.5, y + 0.5, &t))
               continue;
            const double z = p->z0 + t * (p->z1 - p->z0);
            if (!s->depth_clamp && (z < 0.0 || z > 1.0))
               continue;
            double d = v->minDepth + z * (v->maxDepth - v->minDepth);
            if (s->depth_clamp)
               d = dmin(dmax(d, zlo), zhi);
            const size_t i = (s->base_layer + li) * plane + (size_t)y * s->width + x;
            if (s->depth) {
               if (!depth_passes(s->depth_op, (float)d, dep[i]))
                  continue;
               dep[i] = (float)d;
            }
            for (unsigned ch = 0; ch < 4; ch++) {
               const unsigned sum = s->blend_add ? col[4 * i + ch] + src[ch] : src[ch];
               col[4 * i + ch] = (uint8_t)(sum > 255 ? 255 : sum);
            }
         }
      }
   }
}

static VkPrimitiveTopology
shape_topology(enum shape sh)
{
   return sh == QUAD ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        : sh == LINE ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
}

static int
host_buffer(struct ctx *c, VkDeviceSize bytes, VkBufferUsageFlags usage,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&c->mp, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(c->dev, *buf, *mem, 0));
   return 0;
}

/* One image of a scene and the view the framebuffer renders through: `layers`
 * of its layers (3D slices) from base_layer, of an image `h` rows tall. */
static int
alloc_scene_image(struct ctx *c, const struct scene *s, uint32_t h, uint32_t samples,
                  VkFormat format, VkImageAspectFlags aspect, VkImageUsageFlags usage,
                  VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   const bool is_3d = s->is_3d && aspect == VK_IMAGE_ASPECT_COLOR_BIT;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = is_3d ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0,
      .imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, .format = format,
      .extent = { s->width, h, is_3d ? scene_img_layers(s) : 1u }, .mipLevels = 1,
      .arrayLayers = is_3d ? 1u : scene_img_layers(s),
      .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, *img, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&c->mp, mr.memoryTypeBits, 0),
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(c->dev, *img, *mem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img,
      .viewType = s->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1,
                            .baseArrayLayer = s->base_layer, .layerCount = s->layers },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, view));
   return 0;
}

/* Every layer (slice) of an image to TRANSFER_DST, cleared to the scene's
 * clear, so the layers the framebuffer leaves out read back as cleared. */
static void
preclear(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect, const VkClearValue *v)
{
   const VkImageSubresourceRange all = { aspect, 0, VK_REMAINING_MIP_LEVELS, 0,
                                         VK_REMAINING_ARRAY_LAYERS };
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = all,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &b);
   if (aspect == VK_IMAGE_ASPECT_COLOR_BIT)
      vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v->color, 1, &all);
   else
      vkCmdClearDepthStencilImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  &v->depthStencil, 1, &all);
   b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
   b.oldLayout = b.newLayout;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                        0, 0, NULL, 0, NULL, 1, &b);
}

static int
run_scene(struct ctx *c, const struct scene *s, unsigned *failed)
{
   const uint32_t S = s->samples > 1 ? s->samples : 1u;
   VkImage img, zimg = VK_NULL_HANDLE, rimg = VK_NULL_HANDLE;
   VkDeviceMemory imem, zmem = VK_NULL_HANDLE, rmem = VK_NULL_HANDLE;
   VkImageView view, zview = VK_NULL_HANDLE, rview = VK_NULL_HANDLE;
   if (alloc_scene_image(c, s, s->img_h ? s->img_h : s->height, S, FORMAT,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &img, &imem, &view))
      return 1;
   if (S > 1 && alloc_scene_image(c, s, s->img_h ? s->img_h : s->height, 1, FORMAT,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &rimg, &rmem, &rview))
      return 1;
   if (s->depth && alloc_scene_image(c, s, s->zimg_h ? s->zimg_h : s->height, S, DEPTH_FORMAT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &zimg, &zmem, &zview))
      return 1;

   VkAttachmentDescription att[3] = {
      { .format = FORMAT, .samples = (VkSampleCountFlagBits)S,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = DEPTH_FORMAT, .samples = (VkSampleCountFlagBits)S,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference zref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkAttachmentReference rref = { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pResolveAttachments = S > 1 ? &rref : NULL,
      .pDepthStencilAttachment = s->depth ? &zref : NULL,
   };
   if (!s->depth && S > 1) {
      att[1] = att[2];
      rref.attachment = 1;
   }
   const uint32_t natt = 1u + (s->depth ? 1u : 0u) + (S > 1 ? 1u : 0u);
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = natt, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &rp));
   VkImageView fbviews[3] = { view, s->depth ? zview : rview, rview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = natt, .pAttachments = fbviews,
      .width = s->width, .height = s->height, .layers = s->layers,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &fb));

   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, &pl));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = c->vs_attr, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = c->fs, .pName = "main" },
   };
   VkVertexInputBindingDescription vib = { 0, sizeof(struct vtx), VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription via[2] = {
      { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct vtx, pos) },
      { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct vtx, sel) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vib,
      .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = via,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
   };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = s->vp_count, .pViewports = s->vp,
      .scissorCount = s->vp_count, .pScissors = s->sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = s->depth_clamp,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo msi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)S,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = s->depth, .depthWriteEnable = s->depth, .depthCompareOp = s->depth_op,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = s->blend_add,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
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
      .pMultisampleState = &msi, .pDepthStencilState = &dss, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline pipes[MAX_DRAWS];
   uint32_t first_vtx[MAX_DRAWS], nvtx[MAX_DRAWS];
   struct vtx verts[MAX_PRIMS * 6];
   uint32_t total_vtx = 0;
   for (uint32_t d = 0; d < s->ndraws; d++) {
      const struct draw_range *r = &s->draws[d];
      ia.topology = shape_topology(s->p[r->first].shape);
      CHECK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipes[d]));
      first_vtx[d] = total_vtx;
      for (uint32_t n = r->first; n < r->first + r->count; n++)
         total_vtx += prim_vertices(&s->p[n], &verts[total_vtx]);
      nvtx[d] = total_vtx - first_vtx[d];
   }

   VkBuffer vb;
   VkDeviceMemory vbmem;
   if (host_buffer(c, sizeof(struct vtx) * total_vtx, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vbmem))
      return 1;
   void *vmap;
   CHECK(vkMapMemory(c->dev, vbmem, 0, sizeof(struct vtx) * total_vtx, 0, &vmap));
   memcpy(vmap, verts, sizeof(struct vtx) * total_vtx);
   vkUnmapMemory(c->dev, vbmem);

   const size_t plane = (size_t)s->width * s->height;
   const uint32_t nl = scene_img_layers(s);
   const VkDeviceSize cbytes = plane * nl * 4, zbytes = plane * nl * sizeof(float);
   VkBuffer rb;
   VkDeviceMemory rbmem;
   if (host_buffer(c, cbytes + zbytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rbmem))
      return 1;

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(c->dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkClearValue clear[3] = {
      { .color = { .float32 = { k_clear[0] / 255.0f, k_clear[1] / 255.0f,
                                k_clear[2] / 255.0f, k_clear[3] / 255.0f } } },
      { .depthStencil = { s->depth_clear, 0 } },
   };
   clear[2] = clear[0];
   preclear(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT, &clear[0]);
   if (rimg)
      preclear(cmd, rimg, VK_IMAGE_ASPECT_COLOR_BIT, &clear[0]);
   if (s->depth)
      preclear(cmd, zimg, VK_IMAGE_ASPECT_DEPTH_BIT, &clear[1]);
   if (!s->depth)
      clear[1] = clear[0];
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { s->width, s->height } },
      .clearValueCount = natt, .pClearValues = clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   const VkDeviceSize vb_off = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
   for (uint32_t d = 0; d < s->ndraws; d++) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[d]);
      vkCmdDraw(cmd, nvtx[d], 1, first_vtx[d], 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy cregion = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = s->is_3d ? 1u : nl },
      .imageExtent = { s->width, s->height, s->is_3d ? nl : 1u },
   };
   vkCmdCopyImageToBuffer(cmd, rimg ? rimg : img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1,
                          &cregion);
   if (s->depth && S == 1) {
      VkBufferImageCopy zregion = {
         .bufferOffset = cbytes,
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .layerCount = nl },
         .imageExtent = { s->width, s->height, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, zimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &zregion);
   }
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   uint8_t *want_c = malloc(cbytes);
   float *want_z = malloc(zbytes);
   model_scene(s, want_c, want_z);
   uint8_t *px;
   CHECK(vkMapMemory(c->dev, rbmem, 0, cbytes + zbytes, 0, (void **)&px));
   const float *gz = (const float *)(px + cbytes);
   uint32_t bad = 0, drawn = 0;
   for (size_t i = 0; i < plane * nl; i++) {
      const uint8_t *p = px + 4 * i, *w = want_c + 4 * i;
      drawn += memcmp(w, k_clear, 4) != 0;
      const bool zbad = s->depth && S == 1 && !(fabs((double)gz[i] - want_z[i]) <= DEPTH_TOL);
      if (memcmp(p, w, 4) != 0 || zbad) {
         if (bad < 8) {
            const uint32_t l = (uint32_t)(i / plane), x = (uint32_t)(i % plane % s->width),
                           y = (uint32_t)(i % plane / s->width);
            printf("  %s: layer %u (%u,%u) = %u,%u,%u,%u z %.7f, want %u,%u,%u,%u z %.7f\n",
                   s->name, l, x, y, p[0], p[1], p[2], p[3], s->depth ? gz[i] : 0.0f,
                   w[0], w[1], w[2], w[3], s->depth ? want_z[i] : 0.0f);
         }
         bad++;
      }
   }
   vkUnmapMemory(c->dev, rbmem);
   free(want_c);
   free(want_z);
   const uint32_t total = (uint32_t)(plane * nl);
   printf("%-10s: %s  %u/%u px match (%u drawn)\n", s->name, bad ? "FAIL" : "pass",
          total - bad, total, drawn);
   if (bad)
      (*failed)++;

   vkDestroyCommandPool(c->dev, cp, NULL);
   vkFreeMemory(c->dev, rbmem, NULL);
   vkDestroyBuffer(c->dev, rb, NULL);
   vkFreeMemory(c->dev, vbmem, NULL);
   vkDestroyBuffer(c->dev, vb, NULL);
   for (uint32_t d = 0; d < s->ndraws; d++)
      vkDestroyPipeline(c->dev, pipes[d], NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyFramebuffer(c->dev, fb, NULL);
   vkDestroyRenderPass(c->dev, rp, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkFreeMemory(c->dev, imem, NULL);
   vkDestroyImage(c->dev, img, NULL);
   if (rimg) {
      vkDestroyImageView(c->dev, rview, NULL);
      vkFreeMemory(c->dev, rmem, NULL);
      vkDestroyImage(c->dev, rimg, NULL);
   }
   if (s->depth) {
      vkDestroyImageView(c->dev, zview, NULL);
      vkFreeMemory(c->dev, zmem, NULL);
      vkDestroyImage(c->dev, zimg, NULL);
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const char *vsv_path = (argc > 1) ? argv[1] : "mvp_viewport.vert.spv";
   const char *vsl_path = (argc > 2) ? argv[2] : "mvp_layer.vert.spv";
   const char *fs_path  = (argc > 3) ? argv[3] : "mvp.frag.spv";
   const char *vsa_path = (argc > 4) ? argv[4] : "mvp_attr.vert.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-multiviewport",
      .apiVersion = VK_API_VERSION_1_2,
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

   /* multiViewport for the arrays and gl_ViewportIndex in the fragment shader;
    * shaderOutputViewportIndex / shaderOutputLayer for the vertex shaders and
    * gl_Layer in the fragment shader (the ShaderLayer capability); depthClamp
    * and largePoints for the depth-clamp and point cases. */
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!f2.features.multiViewport || !f12.shaderOutputViewportIndex || !f12.shaderOutputLayer ||
       !f2.features.depthClamp || !f2.features.largePoints) {
      fprintf(stderr, "FAILED: multiViewport=%u shaderOutputViewportIndex=%u "
              "shaderOutputLayer=%u depthClamp=%u largePoints=%u\n", f2.features.multiViewport,
              f12.shaderOutputViewportIndex, f12.shaderOutputLayer, f2.features.depthClamp,
              f2.features.largePoints);
      return 1;
   }
   if (props.limits.maxViewports < MAX_VP) {
      fprintf(stderr, "FAILED: maxViewports %u\n", props.limits.maxViewports);
      return 1;
   }

   struct ctx c = { 0 };
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   c.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { c.qf = i; break; }
   free(qfp);
   if (c.qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan12Features e12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .shaderOutputViewportIndex = VK_TRUE, .shaderOutputLayer = VK_TRUE,
   };
   VkPhysicalDeviceFeatures en = {
      .multiViewport = VK_TRUE, .depthClamp = VK_TRUE, .largePoints = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &e12,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &en,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &c.dev));
   vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c.mp);
   c.vs_viewport = load_module(c.dev, vsv_path);
   c.vs_layer = load_module(c.dev, vsl_path);
   c.fs = load_module(c.dev, fs_path);
   c.vs_attr = load_module(c.dev, vsa_path);
   if (!c.vs_viewport || !c.vs_layer || !c.fs || !c.vs_attr) return 1;

   /* Quadrant viewports, the last y-flipped; each scissor crosses its quad's
    * edges at odd offsets, scissor 0 reaching into viewport 1's quadrant. */
   const struct mv_case viewports = {
      .name = "viewports", .width = 64, .height = 64, .layers = 1, .samples = 1,
      .vp_count = 4,
      .vp = { {  0,  0, 32,  32, 0, 1 }, { 32,  0, 32, 32, 0, 1 },
              {  0, 32, 32,  32, 0, 1 }, { 32, 64, 32, -32, 0, 1 } },
      .sc = { { {  3,  5 }, { 40, 13 } }, { { 37,  2 }, { 17, 27 } },
              { {  1, 39 }, { 30, 20 } }, { { 45, 33 }, { 19, 31 } } },
      .quad_hi = 0.5f,
   };
   const struct mv_case overlap = {
      .name = "overlap", .width = 64, .height = 64, .layers = 1, .samples = 1,
      .vp_count = 4,
      .vp = { { 0, 0, 64, 64, 0, 1 }, { 0, 0, 64, 64, 0, 1 },
              { 0, 0, 64, 64, 0, 1 }, { 0, 0, 64, 64, 0, 1 } },
      .sc = { { {  5,  3 }, { 41, 30 } }, { { 22, 11 }, { 27, 41 } },
              { {  0, 27 }, { 35, 19 } }, { { 33, 37 }, { 31, 23 } } },
      .quad_hi = 1.0f,
   };
   struct mv_case msaa = viewports;
   msaa.name = "msaa";
   msaa.samples = 4;
   const struct mv_case layers = {
      .name = "layers", .width = 32, .height = 32, .layers = 2, .samples = 1,
      .vp_count = 1,
      .vp = { { 0, 0, 32, 32, 0, 1 } },
      .sc = { { { 0, 0 }, { 32, 32 } } },
      .layered = true,
   };
   const struct mv_case *cases[] = { &viewports, &overlap, &msaa, &layers };

   unsigned failed = 0;
   for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++)
      if (run_case(&c, cases[i], &failed))
         return 1;

   /* NDC of a window coordinate in a 32-pixel viewport axis. */
#define N32(w) ((float)(w) / 16.0f - 1.0f)
#define RECT(xa, ya, xb, yb, z, vi, li) \
   { .shape = QUAD, .x0 = (xa), .y0 = (ya), .x1 = (xb), .y1 = (yb), .z0 = (z), .z1 = (z), \
     .vpi = (vi), .layer = (li) }
#define HLINE(xa, xb, y, vi, vend) \
   { .shape = LINE, .x0 = N32(xa), .y0 = N32(y), .x1 = N32(xb), .y1 = N32(y), \
     .z0 = 0.5f, .z1 = 0.5f, .vpi = (vi), .vpi_end = (vend) }
#define POINTAT(x, y, sz, vi) \
   { .shape = POINT, .x0 = N32(x), .y0 = N32(y), .z0 = 0.5f, .size = (sz), .vpi = (vi) }

   /* Clip z runs -0.5..1.5 across each quad, crossing 0 and 1 on the pixel
    * edges a quarter and three quarters of the way across its viewport. */
   struct scene depthclip = {
      .name = "depthclip", .width = 96, .height = 32, .layers = 1,
      .vp_count = 3,
      .vp = { {  0, 0, 32, 32, 0.0f, 0.5f }, { 32, 0, 32, 32, 0.5f, 1.0f },
              { 64, 0, 32, 32, 0.25f, 0.25f } },
      .sc = { { {  0, 0 }, { 32, 32 } }, { { 32, 0 }, { 32, 32 } }, { { 64, 0 }, { 32, 32 } } },
      .depth = true, .depth_clear = 0.75f, .depth_op = VK_COMPARE_OP_ALWAYS,
      .ndraws = 1, .draws = { { 0, 3 } }, .nprims = 3,
   };
   for (int i = 0; i < 3; i++)
      depthclip.p[i] = (struct prim){ .shape = QUAD, .x0 = -1, .y0 = -1, .x1 = 1, .y1 = 1,
                                      .z0 = -0.5f, .z1 = 1.5f, .vpi = i };
   struct scene depthclamp = depthclip;
   depthclamp.name = "depthclamp";
   depthclamp.depth_clamp = true;

   /* Viewport 0's scissor is narrower than its quads, so a primitive that kept
    * an out-of-range index's scissor, or none, shows past it. */
   const struct scene vpclamp = {
      .name = "vpclamp", .width = 64, .height = 32, .layers = 1,
      .vp_count = 2,
      .vp = { { 0, 0, 32, 32, 0, 1 }, { 32, 0, 32, 32, 0, 1 } },
      .sc = { { { 3, 5 }, { 22, 20 } }, { { 36, 2 }, { 25, 27 } } },
      .ndraws = 1, .draws = { { 0, 4 } }, .nprims = 4,
      .p = { RECT(-1, -1, 0, 0, 0.5f, 1, 0), RECT(-1, -1, 0, 1, 0.5f, 7, 0),
             RECT(0, -1, 1, 0, 0.5f, 2, 0), RECT(0, 0, 1, 1, 0.5f, -1, 0) },
   };
   const struct scene layerclamp = {
      .name = "layerclamp", .width = 32, .height = 32, .layers = 3,
      .vp_count = 1, .vp = { { 0, 0, 32, 32, 0, 1 } }, .sc = { { { 0, 0 }, { 32, 32 } } },
      .ndraws = 1, .draws = { { 0, 4 } }, .nprims = 4,
      .p = { RECT(-1, -1, 0, 0, 0.5f, 0, 0), RECT(0, -1, 1, 0, 0.5f, 0, 1),
             RECT(-1, 0, 0, 1, 0.5f, 0, 5), RECT(0, 0, 1, 1, 0.5f, 0, 3) },
   };
   /* Quadrant viewports; scissors 0 and 3 cut a line and a point. Each line's
    * second vertex names another quadrant, where it would land in another
    * row and colour. */
   const struct scene linepoint = {
      .name = "linepoint", .width = 64, .height = 64, .layers = 1,
      .vp_count = 4,
      .vp = { { 0, 0, 32, 32, 0, 1 }, { 32, 0, 32, 32, 0, 1 },
              { 0, 32, 32, 32, 0, 1 }, { 32, 32, 32, 32, 0, 1 } },
      .sc = { { { 0, 10 }, { 12, 22 } }, { { 32, 0 }, { 32, 32 } },
              { { 0, 32 }, { 32, 32 } }, { { 40, 32 }, { 24, 32 } } },
      .ndraws = 2, .draws = { { 0, 4 }, { 4, 4 } }, .nprims = 8,
      .p = { HLINE(3.25f, 27.25f, 5.5f, 1, 0), HLINE(6.25f, 30.25f, 20.5f, 2, 3),
             HLINE(1.25f, 17.25f, 12.5f, 0, 2), HLINE(10.25f, 20.25f, 27.5f, 3, 1),
             POINTAT(8, 8, 4.0f, 3), POINTAT(8.5f, 20.5f, 3.0f, 0),
             POINTAT(24.5f, 20.5f, 5.0f, 1), POINTAT(4, 28, 2.0f, 2) },
   };
   /* The first draw puts one quad at depth 0.5 into each layer; the second
    * overlaps them from in front and from behind, one of its quads behind
    * everything already in its layer but in front of the clear. */
   const struct scene layerblend = {
      .name = "layerblend", .width = 32, .height = 32, .layers = 3,
      .vp_count = 1, .vp = { { 0, 0, 32, 32, 0, 1 } }, .sc = { { { 0, 0 }, { 32, 32 } } },
      .depth = true, .depth_clear = 1.0f, .depth_op = VK_COMPARE_OP_LESS, .blend_add = true,
      .ndraws = 2, .draws = { { 0, 3 }, { 3, 5 } }, .nprims = 8,
      .p = { RECT(N32(0), N32(0), N32(20), N32(20), 0.5f, 0, 0),
             RECT(N32(8), N32(4), N32(28), N32(24), 0.5f, 0, 1),
             RECT(N32(12), N32(12), N32(32), N32(32), 0.5f, 0, 2),
             RECT(N32(10), N32(10), N32(30), N32(30), 0.25f, 0, 0),
             RECT(N32(0), N32(0), N32(16), N32(16), 0.75f, 0, 1),
             RECT(N32(0), N32(0), N32(16), N32(16), 0.625f, 0, 2),
             RECT(N32(8), N32(8), N32(24), N32(24), 0.375f, 0, 2),
             RECT(N32(20), N32(20), N32(28), N32(28), 0.875f, 0, 2) },
   };
   /* Six layers of an eight-layer image from its layer 1: one quad per layer,
    * then a second per layer from in front or behind, depth tested and
    * blended. */
   struct scene layers6 = {
      .name = "layers6", .width = 32, .height = 32, .layers = 6,
      .img_layers = 8, .base_layer = 1,
      .vp_count = 1, .vp = { { 0, 0, 32, 32, 0, 1 } }, .sc = { { { 0, 0 }, { 32, 32 } } },
      .depth = true, .depth_clear = 1.0f, .depth_op = VK_COMPARE_OP_LESS, .blend_add = true,
      .ndraws = 2, .draws = { { 0, 6 }, { 6, 6 } }, .nprims = 12,
   };
   for (int l = 0; l < 6; l++) {
      layers6.p[l] = (struct prim)RECT(N32(2 * l), N32(l), N32(2 * l + 14), N32(l + 16),
                                       0.5f, 0, l);
      layers6.p[6 + l] = (struct prim)RECT(N32(8), N32(8 + l), N32(24), N32(20 + l),
                                           (l & 1) ? 0.25f : 0.75f, 0, l);
   }
   /* 64 layers of 16x16: every fifth layer gets nothing, every seventh a
    * second quad in front, and one quad names layer 70, past the last. */
   struct scene layers64 = {
      .name = "layers64", .width = 16, .height = 16, .layers = 64,
      .vp_count = 1, .vp = { { 0, 0, 16, 16, 0, 1 } }, .sc = { { { 0, 0 }, { 16, 16 } } },
      .depth = true, .depth_clear = 1.0f, .depth_op = VK_COMPARE_OP_LESS, .blend_add = true,
      .ndraws = 2,
   };
   {
      uint32_t n = 0;
      for (int l = 0; l < 64; l++) {
         if (l % 5 == 3)
            continue;
         const float x = (float)(l % 8), y = (float)((l / 8) % 8);
         layers64.p[n++] = (struct prim)RECT(x / 8 - 1, y / 8 - 1, (x + 8) / 8 - 1,
                                             (y + 7) / 8 - 1, 0.5f, 0, l);
      }
      layers64.draws[0] = (struct draw_range){ 0, n };
      const uint32_t first = n;
      for (int l = 0; l < 64; l += 7)
         layers64.p[n++] = (struct prim)RECT(-0.75f, -0.75f, 0.25f, 0.5f, 0.25f, 0, l);
      layers64.p[n++] = (struct prim)RECT(0.0f, 0.0f, 1.0f, 1.0f, 0.125f, 0, 70);
      layers64.draws[1] = (struct draw_range){ first, n - first };
      layers64.nprims = n;
   }
   /* The colour image is 40 rows tall and the depth image 52, so a layer is a
    * different number of rows in each. */
   struct scene layersR = layers6;
   layersR.name = "layersR";
   layersR.layers = 4;
   layersR.img_layers = 4;
   layersR.base_layer = 0;
   layersR.img_h = 40;
   layersR.zimg_h = 52;
   layersR.draws[0] = (struct draw_range){ 0, 4 };
   layersR.draws[1] = (struct draw_range){ 4, 4 };
   layersR.nprims = 8;
   for (int l = 0; l < 4; l++) {
      layersR.p[l] = layers6.p[l];
      layersR.p[4 + l] = layers6.p[6 + l];
   }
   /* Slices 1..4 of a six-slice 3D image, as a 2D array. */
   struct scene layers3d = layersR;
   layers3d.name = "layers3d";
   layers3d.img_layers = 6;
   layers3d.base_layer = 1;
   layers3d.img_h = 0;
   layers3d.zimg_h = 0;
   layers3d.is_3d = true;
   layers3d.depth = false;
   /* Four layers at 4 samples, every edge on a pixel boundary. */
   struct scene layersms = layers3d;
   layersms.name = "layersms";
   layersms.img_layers = 4;
   layersms.base_layer = 0;
   layersms.is_3d = false;
   layersms.samples = 4;
#undef POINTAT
#undef HLINE
#undef RECT
#undef N32
   const struct scene *scenes[] = {
      &depthclip, &depthclamp, &vpclamp, &layerclamp, &linepoint, &layerblend,
      &layers6, &layers64, &layersR, &layers3d, &layersms,
   };
   for (unsigned i = 0; i < sizeof scenes / sizeof scenes[0]; i++)
      if (run_scene(&c, scenes[i], &failed))
         return 1;

   vkDestroyShaderModule(c.dev, c.vs_attr, NULL);
   vkDestroyShaderModule(c.dev, c.fs, NULL);
   vkDestroyShaderModule(c.dev, c.vs_layer, NULL);
   vkDestroyShaderModule(c.dev, c.vs_viewport, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u case(s))\n", failed);
      return 1;
   }
   printf("PASSED (viewport arrays, per-viewport scissors and depth ranges, gl_ViewportIndex, "
          "gl_Layer)\n");
   return 0;
}
