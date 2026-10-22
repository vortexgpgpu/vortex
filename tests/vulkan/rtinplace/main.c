/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Rendering into one level and layer of an array image.
 *
 * The colour image is RGBA8, 100x76 with three levels and five layers; the
 * depth/stencil image has the same shape in each format the output merger
 * stores (D32F_S8, S8, D32F, D24S8, D16). The pass renders into level 2 (25x19)
 * of layer 3. At that size no row is a multiple of 16 texels, so each format's
 * row stride carries padding, and the level and layer sit at offsets of their
 * own inside the image.
 *
 * Draw A covers the left 12 columns at depth 0.25 and replaces the stencil with
 * 0x5A; draw B covers everything at depth 0.5, admitted where the stencil is not
 * 0x5A and the depth test passes, and increments it. Every pixel's colour holds
 * its own coordinates, so a texel written through the wrong row stride, level
 * or layer lands where another one is expected. The target is copied out and
 * checked texel by texel in each aspect, and two neighbouring subresources --
 * the same level of layer 2 and level 1 of layer 3 -- must keep what a clear
 * put there before the pass.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BASE_W     100u
#define BASE_H     76u
#define LEVELS     3u
#define LAYERS     5u
#define LEVEL      2u
#define LAYER      3u
#define W          (BASE_W >> LEVEL)
#define H          (BASE_H >> LEVEL)
#define SPLIT      12u                 /* columns draw A covers */
#define COLOR_FMT  VK_FORMAT_R8G8B8A8_UNORM
#define MARK       0x5Au
#define S_CLEAR    0x10u
#define S_KEEP     0x33u               /* the neighbours' stencil */
#define Z_CLEAR    0.75f
#define Z_KEEP     0.125f              /* the neighbours' depth */
#define C_KEEP     7u                  /* the neighbours' colour, every channel */
#define BLUE_A     51u
#define BLUE_B     153u

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

static VkResult
alloc_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkImage img, VkDeviceMemory *out)
{
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits, 0);
   if (mt == UINT32_MAX) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkResult r = vkAllocateMemory(dev, &mai, NULL, out);
   if (r == VK_SUCCESS)
      r = vkBindImageMemory(dev, img, *out, 0);
   return r;
}

struct zs_format {
   VkFormat    format;
   const char *name;
   bool        depth, stencil;
   uint32_t    depth_bytes;     /* bytes per texel of a depth-aspect copy */
   uint32_t    depth_max;       /* unorm range, 0 for float */
};

static const struct zs_format kFormats[] = {
   { VK_FORMAT_D32_SFLOAT_S8_UINT, "D32F_S8", true,  true,  4, 0 },
   { VK_FORMAT_S8_UINT,            "S8",      false, true,  0, 0 },
   { VK_FORMAT_D32_SFLOAT,         "D32F",    true,  false, 4, 0 },
   { VK_FORMAT_D24_UNORM_S8_UINT,  "D24S8",   true,  true,  4, 0xffffffu },
   { VK_FORMAT_D16_UNORM,          "D16",     true,  false, 2, 0xffffu },
};

/* One subresource's aspects as copied out: tight rows. */
struct readback {
   VkDeviceSize color_off, depth_off, stencil_off;
};

/* The depth a copy of the depth aspect holds at texel i, as a float. */
static float
depth_at(const struct zs_format *f, const uint8_t *p, uint32_t i)
{
   if (!f->depth_max) {
      float d;
      memcpy(&d, p + (size_t)i * 4, 4);
      return d;
   }
   uint32_t v = 0;
   memcpy(&v, p + (size_t)i * f->depth_bytes, f->depth_bytes);
   return (float)(v & f->depth_max) / (float)f->depth_max;
}

static bool
depth_ok(const struct zs_format *f, float got, float want)
{
   if (!f->depth_max)
      return got == want;
   return fabsf(got - want) * (float)f->depth_max <= 1.0f;
}

static int
run(const char *vs_path, const char *fs_path, const struct zs_format *zf)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-rtinplace", .apiVersion = VK_API_VERSION_1_1,
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

   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pd, zf->format, &fp);
   if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
      printf("FAILED (%s is not a depth/stencil attachment format)\n", zf->name);
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
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
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* --- images and the target's views ----------------------------- */
   VkImageCreateInfo cimci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = COLOR_FMT,
      .extent = { BASE_W, BASE_H, 1 }, .mipLevels = LEVELS, .arrayLayers = LAYERS,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   VkImage cimg;
   CHECK(vkCreateImage(dev, &cimci, NULL, &cimg));
   VkDeviceMemory cmem;
   CHECK(alloc_image(dev, &mp, cimg, &cmem));
   VkImageCreateInfo dimci = cimci;
   dimci.format = zf->format;
   dimci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   VkImage dimg;
   CHECK(vkCreateImage(dev, &dimci, NULL, &dimg));
   VkDeviceMemory dmem;
   CHECK(alloc_image(dev, &mp, dimg, &dmem));

   const VkImageAspectFlags zs_aspects =
      (zf->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (zf->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
   VkImageViewCreateInfo civci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = cimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = COLOR_FMT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = LEVEL, .levelCount = 1,
                            .baseArrayLayer = LAYER, .layerCount = 1 },
   };
   VkImageView cview;
   CHECK(vkCreateImageView(dev, &civci, NULL, &cview));
   VkImageViewCreateInfo divci = civci;
   divci.image = dimg;
   divci.format = zf->format;
   divci.subresourceRange.aspectMask = zs_aspects;
   VkImageView dview;
   CHECK(vkCreateImageView(dev, &divci, NULL, &dview));

   /* --- render pass and pipelines --------------------------------- */
   VkAttachmentDescription att[2] = {
      { .format = COLOR_FMT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = zf->format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &dref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkImageView fbviews[2] = { cview, dview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = fbviews,
      .width = W, .height = H, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) return 1;
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0, .size = 16,
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
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
   VkViewport vp = { 0, 0, (float)W, (float)H, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
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
   const VkStencilOpState mark = {
      .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff, .writeMask = 0xff, .reference = MARK,
   };
   const VkStencilOpState admit = {
      .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_INCREMENT_AND_CLAMP,
      .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_NOT_EQUAL,
      .compareMask = 0xff, .writeMask = 0xff, .reference = MARK,
   };
   VkPipelineDepthStencilStateCreateInfo ds_a = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .stencilTestEnable = VK_TRUE, .front = mark, .back = mark,
   };
   VkPipelineDepthStencilStateCreateInfo ds_b = ds_a;
   ds_b.front = admit;
   ds_b.back = admit;
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
      .pMultisampleState = &ms, .pDepthStencilState = &ds_a,
      .pColorBlendState = &cb, .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline pipe_a, pipe_b;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe_a));
   gpci.pDepthStencilState = &ds_b;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe_b));

   /* --- readback buffer: three subresources, each aspect tight ----- */
   struct { uint32_t level, layer, w, h; struct readback rb; } sub_rb[3] = {
      { LEVEL, LAYER, W, H }, { LEVEL, LAYER - 1, W, H },
      { LEVEL - 1, LAYER, BASE_W >> (LEVEL - 1), BASE_H >> (LEVEL - 1) },
   };
   VkDeviceSize bytes = 0;
   for (unsigned i = 0; i < 3; i++) {
      const VkDeviceSize n = (VkDeviceSize)sub_rb[i].w * sub_rb[i].h;
      sub_rb[i].rb.color_off = bytes;   bytes += n * 4;
      sub_rb[i].rb.depth_off = bytes;   bytes += zf->depth ? n * zf->depth_bytes : 0;
      bytes = (bytes + 3) & ~(VkDeviceSize)3;
      sub_rb[i].rb.stencil_off = bytes; bytes += zf->stencil ? n : 0;
      bytes = (bytes + 3) & ~(VkDeviceSize)3;
   }
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rbuf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rbuf));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rbuf, &bmr);
   uint32_t bmt = find_mem(&mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, rbuf, bmem, 0));

   /* --- commands ---------------------------------------------------- */
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

   /* Every subresource of both images starts out as a known clear. */
   VkImageMemoryBarrier to_dst[2] = {
      { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cimg,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, LAYERS } },
      { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dimg,
        .subresourceRange = { zs_aspects, 0, LEVELS, 0, LAYERS } },
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 2, to_dst);
   const VkClearColorValue keep_c = { .float32 = { C_KEEP / 255.0f, C_KEEP / 255.0f,
                                                   C_KEEP / 255.0f, C_KEEP / 255.0f } };
   const VkClearDepthStencilValue keep_zs = { Z_KEEP, S_KEEP };
   vkCmdClearColorImage(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &keep_c, 1,
                        &to_dst[0].subresourceRange);
   vkCmdClearDepthStencilImage(cmd, dimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &keep_zs, 1,
                               &to_dst[1].subresourceRange);
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);

   VkClearValue clears[2] = {
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
      { .depthStencil = { Z_CLEAR, S_CLEAR } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb, .renderArea = { { 0, 0 }, { W, H } },
      .clearValueCount = 2, .pClearValues = clears,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   const float rect_a[4] = { -1.0f, -1.0f + 2.0f * SPLIT / W, 0.25f, BLUE_A / 255.0f };
   const float rect_b[4] = { -1.0f, 1.0f, 0.5f, BLUE_B / 255.0f };
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_a);
   vkCmdPushConstants(cmd, pl, pcr.stageFlags, 0, sizeof rect_a, rect_a);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_b);
   vkCmdPushConstants(cmd, pl, pcr.stageFlags, 0, sizeof rect_b, rect_b);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   /* The neighbours were cleared in TRANSFER_DST; move every subresource to
    * TRANSFER_SRC for the copies. */
   VkImageMemoryBarrier to_src[2] = { to_dst[0], to_dst[1] };
   for (unsigned i = 0; i < 2; i++) {
      to_src[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      to_src[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      to_src[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      to_src[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   }
   /* Only the neighbours are transitioned: the target is already
    * TRANSFER_SRC, and an UNDEFINED transition would discard it. */
   VkImageMemoryBarrier nb[4];
   unsigned nnb = 0;
   for (unsigned i = 1; i < 3; i++) {
      for (unsigned j = 0; j < 2; j++) {
         nb[nnb] = to_src[j];
         nb[nnb].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
         nb[nnb].subresourceRange.baseMipLevel = sub_rb[i].level;
         nb[nnb].subresourceRange.levelCount = 1;
         nb[nnb].subresourceRange.baseArrayLayer = sub_rb[i].layer;
         nb[nnb].subresourceRange.layerCount = 1;
         nnb++;
      }
   }
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, nnb, nb);
   for (unsigned i = 0; i < 3; i++) {
      VkBufferImageCopy r = {
         .bufferOffset = sub_rb[i].rb.color_off,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, sub_rb[i].level, sub_rb[i].layer, 1 },
         .imageExtent = { sub_rb[i].w, sub_rb[i].h, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &r);
      if (zf->depth) {
         r.bufferOffset = sub_rb[i].rb.depth_off;
         r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
         vkCmdCopyImageToBuffer(cmd, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &r);
      }
      if (zf->stencil) {
         r.bufferOffset = sub_rb[i].rb.stencil_off;
         r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
         vkCmdCopyImageToBuffer(cmd, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &r);
      }
   }
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify -------------------------------------------------------- */
   uint8_t *m;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&m));
   unsigned fails = 0;
   for (unsigned i = 0; i < 3; i++) {
      const uint32_t sw = sub_rb[i].w, sh = sub_rb[i].h;
      const uint8_t *c = m + sub_rb[i].rb.color_off;
      const uint8_t *d = m + sub_rb[i].rb.depth_off;
      const uint8_t *s = m + sub_rb[i].rb.stencil_off;
      for (uint32_t y = 0; y < sh; y++) {
         for (uint32_t x = 0; x < sw; x++) {
            const uint32_t t = y * sw + x;
            uint8_t wc[4] = { C_KEEP, C_KEEP, C_KEEP, C_KEEP };
            float wz = Z_KEEP;
            uint32_t ws = S_KEEP;
            if (i == 0) {
               const bool a = x < SPLIT;
               wc[0] = (uint8_t)x; wc[1] = (uint8_t)y;
               wc[2] = a ? BLUE_A : BLUE_B; wc[3] = 255;
               wz = a ? 0.25f : 0.5f;
               ws = a ? MARK : S_CLEAR + 1u;
            }
            const bool cbad = memcmp(c + (size_t)t * 4, wc, 4) != 0;
            const float gz = zf->depth ? depth_at(zf, d, t) : wz;
            const bool zbad = zf->depth && !depth_ok(zf, gz, wz);
            const bool sbad = zf->stencil && s[t] != ws;
            if (cbad || zbad || sbad) {
               if (fails < 6)
                  fprintf(stderr, "  %s level %u layer %u (%u,%u): colour %u,%u,%u,%u want "
                          "%u,%u,%u,%u; depth %g want %g; stencil 0x%02x want 0x%02x\n",
                          zf->name, sub_rb[i].level, sub_rb[i].layer, x, y,
                          c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3],
                          wc[0], wc[1], wc[2], wc[3], gz, wz,
                          zf->stencil ? s[t] : ws, ws);
               fails++;
            }
         }
      }
   }
   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (fails) {
      printf("FAILED (%s: %u texels differ in the target or its neighbours)\n", zf->name, fails);
      return 1;
   }
   printf("%s: level %u layer %u (%ux%u) and both neighbours match\n", zf->name, LEVEL, LAYER,
          W, H);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "rtinplace.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "rtinplace.frag.spv";
   for (unsigned i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++) {
      if (run(vs_path, fs_path, &kFormats[i]))
         return 1;
   }
   printf("PASSED (rtinplace: %u depth/stencil formats, level %u of layer %u, padded rows)\n",
          (unsigned)(sizeof kFormats / sizeof kFormats[0]), LEVEL, LAYER);
   return 0;
}
