/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Colour attachment formats the software merger stores field by field: snorm
 * at 8 and 16 bits, unorm 16, the BGRA byte order in snorm and integer, the
 * 2-10-10-10 packings, 12-byte RGB32 texels, the packed 16-bit orders, A8 and
 * the X6/X4 single- and two-plane formats.
 *
 * Each format takes one constant colour -- (1.0, -0.6, 0.3, 0.8) for the float
 * and normalized ones, (200, 3, 70000, 1) or (-300, 100, -3, 1) for the integer
 * ones -- over a zero clear. Every field of the stored texel is checked against
 * the exact value: within one step of its code for a normalized field (a value
 * that lands halfway may round either way), exactly for a float or integer
 * one. The field layouts below restate the Vulkan format definitions.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u

enum kind { UNORM, SNORM, UINT, SINT, FLOAT };

struct phase {
   const char *name;
   VkFormat    fmt;
   unsigned    bpp;
   enum kind   kind;
   uint8_t     off[4];     /* bit offset of R, G, B, A in the texel */
   uint8_t     width[4];   /* 0: the format has no such channel */
};



static const struct phase PHASES[] = {
   { "r8_snorm",       VK_FORMAT_R8_SNORM,        1, SNORM, { 0 },            { 8 } },
   { "rg8_snorm",      VK_FORMAT_R8G8_SNORM,      2, SNORM, { 0, 8 },         { 8, 8 } },
   { "rgba8_snorm",    VK_FORMAT_R8G8B8A8_SNORM,  4, SNORM, { 0, 8, 16, 24 }, { 8, 8, 8, 8 } },
   { "abgr8_snorm",    VK_FORMAT_A8B8G8R8_SNORM_PACK32, 4, SNORM, { 0, 8, 16, 24 }, { 8, 8, 8, 8 } },
   { "bgra8_snorm",    VK_FORMAT_B8G8R8A8_SNORM,  4, SNORM, { 16, 8, 0, 24 }, { 8, 8, 8, 8 } },
   { "bgra8_uint",     VK_FORMAT_B8G8R8A8_UINT,   4, UINT,  { 16, 8, 0, 24 }, { 8, 8, 8, 8 } },
   { "bgra8_sint",     VK_FORMAT_B8G8R8A8_SINT,   4, SINT,  { 16, 8, 0, 24 }, { 8, 8, 8, 8 } },
   { "a2r10g10b10",    VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4, UNORM, { 20, 10, 0, 30 }, { 10, 10, 10, 2 } },
   { "a2r10g10b10_sn", VK_FORMAT_A2R10G10B10_SNORM_PACK32, 4, SNORM, { 20, 10, 0, 30 }, { 10, 10, 10, 2 } },
   { "a2r10g10b10_ui", VK_FORMAT_A2R10G10B10_UINT_PACK32,  4, UINT,  { 20, 10, 0, 30 }, { 10, 10, 10, 2 } },
   { "a2b10g10r10_sn", VK_FORMAT_A2B10G10R10_SNORM_PACK32, 4, SNORM, { 0, 10, 20, 30 }, { 10, 10, 10, 2 } },
   { "r16_unorm",      VK_FORMAT_R16_UNORM,       2, UNORM, { 0 },            { 16 } },
   { "r16_snorm",      VK_FORMAT_R16_SNORM,       2, SNORM, { 0 },            { 16 } },
   { "rg16_unorm",     VK_FORMAT_R16G16_UNORM,    4, UNORM, { 0, 16 },        { 16, 16 } },
   { "rg16_snorm",     VK_FORMAT_R16G16_SNORM,    4, SNORM, { 0, 16 },        { 16, 16 } },
   { "rgba16_unorm",   VK_FORMAT_R16G16B16A16_UNORM, 8, UNORM, { 0, 16, 32, 48 }, { 16, 16, 16, 16 } },
   { "rgba16_snorm",   VK_FORMAT_R16G16B16A16_SNORM, 8, SNORM, { 0, 16, 32, 48 }, { 16, 16, 16, 16 } },
   { "rgb32_sfloat",   VK_FORMAT_R32G32B32_SFLOAT, 12, FLOAT, { 0, 32, 64 },  { 32, 32, 32 } },
   { "rgb32_uint",     VK_FORMAT_R32G32B32_UINT,  12, UINT,  { 0, 32, 64 },   { 32, 32, 32 } },
   { "rgb32_sint",     VK_FORMAT_R32G32B32_SINT,  12, SINT,  { 0, 32, 64 },   { 32, 32, 32 } },
   { "b4g4r4a4",       VK_FORMAT_B4G4R4A4_UNORM_PACK16, 2, UNORM, { 4, 8, 12, 0 }, { 4, 4, 4, 4 } },
   { "b5g6r5",         VK_FORMAT_B5G6R5_UNORM_PACK16,   2, UNORM, { 0, 5, 11 },    { 5, 6, 5 } },
   { "r5g5b5a1",       VK_FORMAT_R5G5B5A1_UNORM_PACK16, 2, UNORM, { 11, 6, 1, 0 }, { 5, 5, 5, 1 } },
   { "b5g5r5a1",       VK_FORMAT_B5G5R5A1_UNORM_PACK16, 2, UNORM, { 1, 6, 11, 0 }, { 5, 5, 5, 1 } },
   { "a4r4g4b4",       VK_FORMAT_A4R4G4B4_UNORM_PACK16, 2, UNORM, { 8, 4, 0, 12 }, { 4, 4, 4, 4 } },
   { "a4b4g4r4",       VK_FORMAT_A4B4G4R4_UNORM_PACK16, 2, UNORM, { 0, 4, 8, 12 }, { 4, 4, 4, 4 } },
   { "a1b5g5r5",       VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR, 2, UNORM, { 0, 5, 10, 15 }, { 5, 5, 5, 1 } },
   { "a8",             VK_FORMAT_A8_UNORM_KHR,    1, UNORM, { 0, 0, 0, 0 },   { 0, 0, 0, 8 } },
   { "r10x6",          VK_FORMAT_R10X6_UNORM_PACK16, 2, UNORM, { 6 },         { 10 } },
   { "r10x6g10x6",     VK_FORMAT_R10X6G10X6_UNORM_2PACK16, 4, UNORM, { 6, 22 }, { 10, 10 } },
   { "r12x4",          VK_FORMAT_R12X4_UNORM_PACK16, 2, UNORM, { 4 },         { 12 } },
   { "r12x4g12x4",     VK_FORMAT_R12X4G12X4_UNORM_2PACK16, 4, UNORM, { 4, 20 }, { 12, 12 } },
};
#define NUM_PHASES (sizeof(PHASES) / sizeof(PHASES[0]))

static const float    COLOR_F[4] = { 1.0f, -0.6f, 0.3f, 0.8f };
static const uint32_t COLOR_U[4] = { 200u, 3u, 70000u, 1u };
static const int32_t  COLOR_I[4] = { -300, 100, -3, 1 };

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return -1;                                                   \
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

/* Draw the triangle into an attachment of this phase's format and report its
 * centre texel and its footprint. Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, const struct phase *ph,
       uint8_t centre[16], long *out_covered)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = ph->fmt,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   uint32_t imt = find_mem(mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return -1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ph->fmt,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = ph->fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference attref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &attref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };

   /* No vertex input: the vertex shader builds its positions from
    * gl_VertexIndex, so nothing but the fragment colour is under test. */
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
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
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
      .pMultisampleState = &ms, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * ph->bpp;
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
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return -1; }
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

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkClearValue clear;
   memset(&clear, 0, sizeof clear);
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
   /* A texel the draw changed differs from the corner, which it never covers. */
   uint8_t bg[16];
   memcpy(bg, px + (size_t)(WIDTH * HEIGHT - 1) * ph->bpp, ph->bpp);
   *out_covered = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         if (memcmp(px + ((size_t)y * WIDTH + x) * ph->bpp, bg, ph->bpp)) {
            (*out_covered)++;
         }
      }
   }
   /* The triangle's lower-left corner is at (-0.9,-0.9) and it spans 1.8 NDC on
    * each axis, so a quarter of the way in on both axes is comfortably inside
    * it and clear of every edge. */
   memset(centre, 0, 16);
   memcpy(centre, px + ((size_t)(HEIGHT / 4) * WIDTH + WIDTH / 4) * ph->bpp,
          ph->bpp);
   vkUnmapMemory(dev, bmem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   return 0;
}

/* One field of the stored texel against the exact value of its channel. */
static bool
check_field(const struct phase *ph, unsigned c, uint64_t lo, uint64_t hi, double *got,
            double *want)
{
   const unsigned w = ph->width[c], off = ph->off[c];
   const uint64_t word = off >= 64 ? hi : lo;
   const uint64_t raw = (word >> (off & 63)) & ((w >= 64) ? ~0ull : ((1ull << w) - 1));
   switch (ph->kind) {
   case UNORM: {
      const double max = (double)((1u << w) - 1);
      *got = (double)raw;
      *want = fmin(fmax(COLOR_F[c], 0.0), 1.0) * max;
      return fabs(*got - *want) <= 1.0;
   }
   case SNORM: {
      const double max = (double)((1u << (w - 1)) - 1);
      *got = (double)(((int64_t)(raw << (64 - w))) >> (64 - w));
      *want = fmin(fmax(COLOR_F[c], -1.0), 1.0) * max;
      return fabs(*got - *want) <= 1.0;
   }
   case UINT: {
      const double max = (w >= 32) ? 4294967295.0 : (double)((1u << w) - 1);
      *got = (double)raw;
      *want = fmin((double)COLOR_U[c], max);
      return *got == *want;
   }
   case SINT: {
      const double max = (w >= 32) ? 2147483647.0 : (double)((1u << (w - 1)) - 1);
      *got = (double)(((int64_t)(raw << (64 - w))) >> (64 - w));
      *want = fmin(fmax((double)COLOR_I[c], -max - 1.0), max);
      return *got == *want;
   }
   default: {
      float f;
      const uint32_t bits = (uint32_t)raw;
      memcpy(&f, &bits, 4);
      *got = f;
      *want = COLOR_F[c];
      return *got == *want;
   }
   }
}

static bool
check_phase(const struct phase *ph, const uint8_t centre[16], long covered)
{
   const long EXPECT_PX = 1659;
   bool ok = covered >= EXPECT_PX - 200 && covered <= EXPECT_PX + 200;
   uint64_t lo = 0, hi = 0;
   memcpy(&lo, centre, ph->bpp < 8 ? ph->bpp : 8);
   if (ph->bpp > 8)
      memcpy(&hi, centre + 8, ph->bpp - 8);
   char detail[160];
   int k = 0;
   for (unsigned c = 0; c < 4; c++) {
      if (!ph->width[c])
         continue;
      double got, want;
      if (!check_field(ph, c, lo, hi, &got, &want)) {
         ok = false;
         k += snprintf(detail + k, sizeof detail - (size_t)k, " %c=%g (want %g)",
                       "RGBA"[c], got, want);
      }
   }
   if (ok) {
      printf("PASSED (rtfield %s: %u-byte texel over %ld px)\n", ph->name, ph->bpp, covered);
      return true;
   }
   printf("FAILED (rtfield %s: %ld px covered, expected ~%ld;%s)\n", ph->name, covered,
          EXPECT_PX, k ? detail : " fields right");
   return false;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "rtfield.vert.spv";
   const char *ff_path = (argc > 2) ? argv[2] : "rtfield_f.frag.spv";
   const char *fu_path = (argc > 3) ? argv[3] : "rtfield_u.frag.spv";
   const char *fi_path = (argc > 4) ? argv[4] : "rtfield_i.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-rtfield",
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
      fprintf(stderr, "FAILED: no physical device\n"); return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
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
   VkShaderModule ff = load_module(dev, ff_path);
   VkShaderModule fu = load_module(dev, fu_path);
   VkShaderModule fi = load_module(dev, fi_path);
   if (!vs || !ff || !fu || !fi) {
      return 1;
   }

   bool ok = true;
   for (unsigned i = 0; i < NUM_PHASES; i++) {
      /* Lavapipe renders every one of these, so a format left unadvertised is a
       * failure, not a skip. */
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, PHASES[i].fmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
         printf("FAILED (rtfield %s: not advertised as a colour attachment)\n",
                PHASES[i].name);
         ok = false;
         continue;
      }
      uint8_t centre[16];
      long covered = 0;
      VkShaderModule fs = PHASES[i].kind == UINT ? fu : (PHASES[i].kind == SINT ? fi : ff);
      if (render(dev, queue, qf, &mp, vs, fs, &PHASES[i], centre, &covered) < 0) {
         return 1;
      }
      if (!check_phase(&PHASES[i], centre, covered)) {
         ok = false;
      }
   }

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, ff, NULL);
   vkDestroyShaderModule(dev, fu, NULL);
   vkDestroyShaderModule(dev, fi, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return ok ? 0 : 1;
}
