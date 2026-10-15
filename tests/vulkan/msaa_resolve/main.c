/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Resolves of 4x multisample images, from sources whose four samples all hold
 * different values, so a resolve that took one sample where it should average,
 * or averaged where it should take sample 0, cannot match by accident.
 *
 * The sources are filled by sample-rate shading (gl_SampleID), then resolved:
 *
 *   rpass   RGBA8 through the render pass's own resolve attachment: the
 *           per-channel average, within one code (rounding is the device's).
 *   box     RGBA8 through vkCmdResolveImage, whole image: the same average.
 *   float   R32F through vkCmdResolveImage: the average, exactly -- every value
 *           involved is a short dyadic fraction.
 *   uint    RGBA8UI through vkCmdResolveImage: sample 0, exactly.
 *   sub     RGBA8, source offset (4,12), destination offset (8,8), 16x16, into
 *           a 32x32 target cleared beforehand: the region holds the average of
 *           the offset source and every pixel outside it keeps the clear.
 *   layer   layer 1 of a two-layer RGBA8 source whose layers hold different
 *           values: the average of layer 1, not of layer 0.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH    32u
#define HEIGHT   32u
#define SAMPLES  4u

#define UNORM_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define FLOAT_FORMAT VK_FORMAT_R32_SFLOAT
#define UINT_FORMAT  VK_FORMAT_R8G8B8A8_UINT

/* Seeds the fill shaders add to their first channel: one per source layer, so
 * the two layers of the layered source hold different values. */
#define SEED_PLAIN   0
#define SEED_LAYER0  128
#define SEED_LAYER1  64

/* The sub-rectangle resolve. */
#define SUB_SRC_X    4
#define SUB_SRC_Y    12
#define SUB_DST_X    8
#define SUB_DST_Y    8
#define SUB_W        16u
#define SUB_H        16u
static const float   k_sub_clear[4] = { 0.2f, 0.4f, 0.6f, 0.8f };
static const uint8_t k_sub_clear8[4] = { 51, 102, 153, 204 };

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* What fill_unorm.frag writes into channel c of sample k of pixel (x, y). */
static int
expect_unorm(uint32_t x, uint32_t y, int seed, uint32_t k, uint32_t c)
{
   switch (c) {
   case 0:  return 16 * (int)k + (int)x + seed;
   case 1:  return 50 * (int)k + (int)y;
   case 2:  return k == 0 ? 255 : 0;
   default: return 255 - 60 * (int)k;
   }
}

/* What fill_float.frag writes into sample k of pixel (x, y). */
static float
expect_float(uint32_t x, uint32_t y, uint32_t k)
{
   return (float)(x + 32 * y) + 0.25f * (float)k;
}

/* What fill_uint.frag writes into channel c of sample k of pixel (x, y). */
static uint32_t
expect_uint(uint32_t x, uint32_t y, uint32_t k, uint32_t c)
{
   const uint32_t v[4] = { 10 + k, 20 + 3 * k, x, y };
   return v[c];
}

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

struct image {
   VkImage        img;
   VkDeviceMemory mem;
};

static VkResult
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
           VkFormat fmt, uint32_t layers, VkSampleCountFlagBits samples,
           VkImageUsageFlags usage, struct image *out)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = layers,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = vkCreateImage(dev, &ici, NULL, &out->img);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, out->img, &mr);
   r = bind_memory(dev, mp, &mr, 0, &out->mem);
   if (r != VK_SUCCESS) {
      return r;
   }
   return vkBindImageMemory(dev, out->img, out->mem, 0);
}

static VkResult
make_view(VkDevice dev, VkImage img, VkFormat fmt, uint32_t layer, VkImageView *out)
{
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .baseArrayLayer = layer,
                            .layerCount = 1 },
   };
   return vkCreateImageView(dev, &vci, NULL, out);
}

/* A one-subpass pass writing a 4x attachment that ends ready to be resolved,
 * optionally with a single-sample resolve attachment of its own. */
static VkResult
make_fill_pass(VkDevice dev, VkFormat fmt, bool with_resolve, VkRenderPass *out)
{
   VkAttachmentDescription atts[2] = {
      { .format = fmt, .samples = (VkSampleCountFlagBits)SAMPLES,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
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
   VkAttachmentReference rref = {
      .attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pResolveAttachments = with_resolve ? &rref : NULL,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = with_resolve ? 2 : 1, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   return vkCreateRenderPass(dev, &rpci, NULL, out);
}

static VkResult
make_pipeline(VkDevice dev, VkShaderModule vs, VkShaderModule fs,
              VkRenderPass rp, VkPipelineLayout layout, VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo st[2] = {
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
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)SAMPLES,
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
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb,
      .layout = layout, .renderPass = rp, .subpass = 0,
   };
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static void
fill(VkCommandBuffer cmd, VkRenderPass rp, VkFramebuffer fb, VkPipeline pipe,
     VkPipelineLayout layout, int32_t seed)
{
   VkClearValue clear = { .color = { .uint32 = { 0, 0, 0, 0 } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(seed), &seed);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
}

static void
layout_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to,
               VkAccessFlags src_access, VkAccessFlags dst_access,
               VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
   VkImageMemoryBarrier imb = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access, .dstAccessMask = dst_access,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &imb);
}

static void
resolve(VkCommandBuffer cmd, VkImage src, uint32_t src_layer, VkImage dst,
        int32_t sx, int32_t sy, int32_t dx, int32_t dy, uint32_t w, uint32_t h)
{
   VkImageResolve region = {
      .srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                          .baseArrayLayer = src_layer, .layerCount = 1 },
      .srcOffset = { sx, sy, 0 },
      .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .dstOffset = { dx, dy, 0 },
      .extent = { w, h, 1 },
   };
   vkCmdResolveImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

/* An RGBA8 resolve of a source filled with `seed`: each channel within one code
 * of the exact average of the four samples. (sx, sy) is the source pixel. */
static bool
unorm_avg_ok(const uint8_t *px, uint32_t sx, uint32_t sy, int seed)
{
   for (uint32_t c = 0; c < 4; c++) {
      int sum = 0;
      for (uint32_t k = 0; k < SAMPLES; k++) {
         sum += expect_unorm(sx, sy, seed, k, c);
      }
      if (fabs((double)px[c] - (double)sum / SAMPLES) > 1.0) {
         return false;
      }
   }
   return true;
}

static int
report(const char *name, unsigned bad, unsigned total, const char *first)
{
   printf("%-6s: %s  bad pixels=%u of %u%s%s\n", name, bad ? "FAIL" : "pass", bad,
          total, bad ? "  first: " : "", bad ? first : "");
   return bad ? 1 : 0;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_resolve",
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
   if (npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   const VkFormat src_formats[3] = { UNORM_FORMAT, FLOAT_FORMAT, UINT_FORMAT };
   VkSampleCountFlags counts = ~(VkSampleCountFlags)0;
   for (unsigned i = 0; i < 3; i++) {
      VkImageFormatProperties ifp;
      if (vkGetPhysicalDeviceImageFormatProperties(pd, src_formats[i],
             VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
             0, &ifp) != VK_SUCCESS) {
         ifp.sampleCounts = 0;
      }
      counts &= ifp.sampleCounts;
   }
   if (!(counts & SAMPLES) || !have.sampleRateShading) {
      printf("FAILED (device lacks 4x RGBA8/R32F/RGBA8UI attachments or "
             "sampleRateShading: counts=0x%x ss=%u)\n", (unsigned)counts,
             (unsigned)have.sampleRateShading);
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
   VkPhysicalDeviceFeatures want = { .sampleRateShading = VK_TRUE };
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

   /* --- sources ------------------------------------------------------ */
   const VkSampleCountFlagBits S = (VkSampleCountFlagBits)SAMPLES;
   const VkImageUsageFlags src_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   struct image src_unorm, src_float, src_uint, src_layers;
   CHECK(make_image(dev, &mp, UNORM_FORMAT, 1, S, src_usage, &src_unorm));
   CHECK(make_image(dev, &mp, FLOAT_FORMAT, 1, S, src_usage, &src_float));
   CHECK(make_image(dev, &mp, UINT_FORMAT, 1, S, src_usage, &src_uint));
   CHECK(make_image(dev, &mp, UNORM_FORMAT, 2, S, src_usage, &src_layers));
   VkImageView v_unorm, v_float, v_uint, v_layer[2];
   CHECK(make_view(dev, src_unorm.img, UNORM_FORMAT, 0, &v_unorm));
   CHECK(make_view(dev, src_float.img, FLOAT_FORMAT, 0, &v_float));
   CHECK(make_view(dev, src_uint.img, UINT_FORMAT, 0, &v_uint));
   CHECK(make_view(dev, src_layers.img, UNORM_FORMAT, 0, &v_layer[0]));
   CHECK(make_view(dev, src_layers.img, UNORM_FORMAT, 1, &v_layer[1]));

   /* --- single-sample destinations, in read-back order --------------- */
   enum { D_RPASS, D_BOX, D_FLOAT, D_UINT, D_SUB, D_LAYER, D_COUNT };
   static const char *const d_name[D_COUNT] = {
      "rpass", "box", "float", "uint", "sub", "layer",
   };
   const VkFormat d_format[D_COUNT] = {
      UNORM_FORMAT, UNORM_FORMAT, FLOAT_FORMAT, UINT_FORMAT, UNORM_FORMAT, UNORM_FORMAT,
   };
   struct image dst[D_COUNT];
   for (unsigned i = 0; i < D_COUNT; i++) {
      VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                (i == D_RPASS ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                              : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      CHECK(make_image(dev, &mp, d_format[i], 1, VK_SAMPLE_COUNT_1_BIT, usage, &dst[i]));
   }
   VkImageView v_rpass;
   CHECK(make_view(dev, dst[D_RPASS].img, UNORM_FORMAT, 0, &v_rpass));

   /* --- passes and framebuffers -------------------------------------- */
   VkRenderPass rp_resolve, rp_unorm, rp_float, rp_uint;
   CHECK(make_fill_pass(dev, UNORM_FORMAT, true, &rp_resolve));
   CHECK(make_fill_pass(dev, UNORM_FORMAT, false, &rp_unorm));
   CHECK(make_fill_pass(dev, FLOAT_FORMAT, false, &rp_float));
   CHECK(make_fill_pass(dev, UINT_FORMAT, false, &rp_uint));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkImageView rs_views[2] = { v_unorm, v_rpass };
   VkFramebuffer fb_resolve, fb_float, fb_uint, fb_layer[2];
   fbci.renderPass = rp_resolve;
   fbci.attachmentCount = 2;
   fbci.pAttachments = rs_views;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_resolve));
   fbci.attachmentCount = 1;
   fbci.renderPass = rp_float;
   fbci.pAttachments = &v_float;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_float));
   fbci.renderPass = rp_uint;
   fbci.pAttachments = &v_uint;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_uint));
   fbci.renderPass = rp_unorm;
   for (unsigned l = 0; l < 2; l++) {
      fbci.pAttachments = &v_layer[l];
      CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_layer[l]));
   }

   /* --- pipelines ---------------------------------------------------- */
   VkShaderModule vs = load_module(dev, "resolve.vert.spv");
   VkShaderModule fs_unorm = load_module(dev, "fill_unorm.frag.spv");
   VkShaderModule fs_float = load_module(dev, "fill_float.frag.spv");
   VkShaderModule fs_uint = load_module(dev, "fill_uint.frag.spv");
   if (!vs || !fs_unorm || !fs_float || !fs_uint) {
      return 1;
   }
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4,
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   /* A pass with a resolve attachment is not compatible with one without, so
    * the render-pass resolve gets a pipeline of its own. */
   VkPipeline p_resolve, p_unorm, p_float, p_uint;
   CHECK(make_pipeline(dev, vs, fs_unorm, rp_resolve, pl, &p_resolve));
   CHECK(make_pipeline(dev, vs, fs_unorm, rp_unorm, pl, &p_unorm));
   CHECK(make_pipeline(dev, vs, fs_float, rp_float, pl, &p_float));
   CHECK(make_pipeline(dev, vs, fs_uint, rp_uint, pl, &p_uint));

   /* --- read-back buffer: one 4-byte texel per pixel per destination - */
   const VkDeviceSize plane = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = plane * D_COUNT, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   VkDeviceMemory rbmem;
   CHECK(bind_memory(dev, &mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &rbmem));
   CHECK(vkBindBufferMemory(dev, rb, rbmem, 0));

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

   /* --- record: fill, resolve, read back ----------------------------- */
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   fill(cmd, rp_resolve, fb_resolve, p_resolve, pl, SEED_PLAIN);
   fill(cmd, rp_float, fb_float, p_float, pl, SEED_PLAIN);
   fill(cmd, rp_uint, fb_uint, p_uint, pl, SEED_PLAIN);
   fill(cmd, rp_unorm, fb_layer[0], p_unorm, pl, SEED_LAYER0);
   fill(cmd, rp_unorm, fb_layer[1], p_unorm, pl, SEED_LAYER1);

   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   for (unsigned i = 0; i < D_COUNT; i++) {
      if (i != D_RPASS) {
         layout_barrier(cmd, dst[i].img, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT);
      }
   }

   VkClearColorValue sub_clear;
   memcpy(sub_clear.float32, k_sub_clear, sizeof(k_sub_clear));
   VkImageSubresourceRange whole = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1,
   };
   vkCmdClearColorImage(cmd, dst[D_SUB].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        &sub_clear, 1, &whole);
   layout_barrier(cmd, dst[D_SUB].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT);

   resolve(cmd, src_unorm.img, 0, dst[D_BOX].img, 0, 0, 0, 0, WIDTH, HEIGHT);
   resolve(cmd, src_float.img, 0, dst[D_FLOAT].img, 0, 0, 0, 0, WIDTH, HEIGHT);
   resolve(cmd, src_uint.img, 0, dst[D_UINT].img, 0, 0, 0, 0, WIDTH, HEIGHT);
   resolve(cmd, src_unorm.img, 0, dst[D_SUB].img, SUB_SRC_X, SUB_SRC_Y,
           SUB_DST_X, SUB_DST_Y, SUB_W, SUB_H);
   resolve(cmd, src_layers.img, 1, dst[D_LAYER].img, 0, 0, 0, 0, WIDTH, HEIGHT);

   for (unsigned i = 0; i < D_COUNT; i++) {
      if (i != D_RPASS) {
         layout_barrier(cmd, dst[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
      }
   }
   for (unsigned i = 0; i < D_COUNT; i++) {
      VkBufferImageCopy region = {
         .bufferOffset = plane * i,
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, dst[i].img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb, 1, &region);
   }
   VkMemoryBarrier hb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &hb, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   void *map;
   CHECK(vkMapMemory(dev, rbmem, 0, plane * D_COUNT, 0, &map));
   const uint8_t *base = map;

   unsigned failed = 0;
   for (unsigned i = 0; i < D_COUNT; i++) {
      const uint8_t *img = base + plane * i;
      unsigned bad = 0;
      char first[160] = "";
      for (uint32_t y = 0; y < HEIGHT; y++) {
         for (uint32_t x = 0; x < WIDTH; x++) {
            const uint8_t *px = img + ((size_t)y * WIDTH + x) * 4;
            bool ok = true;
            switch (i) {
            case D_RPASS:
            case D_BOX:
               ok = unorm_avg_ok(px, x, y, SEED_PLAIN);
               break;
            case D_LAYER:
               ok = unorm_avg_ok(px, x, y, SEED_LAYER1);
               break;
            case D_FLOAT: {
               float v;
               memcpy(&v, px, sizeof(v));
               float sum = 0.0f;
               for (uint32_t k = 0; k < SAMPLES; k++) {
                  sum += expect_float(x, y, k);
               }
               ok = v == sum / SAMPLES;
               break;
            }
            case D_UINT:
               for (uint32_t c = 0; c < 4; c++) {
                  ok = ok && px[c] == expect_uint(x, y, 0, c);
               }
               break;
            case D_SUB: {
               const bool inside = x >= SUB_DST_X && x < SUB_DST_X + SUB_W &&
                                   y >= SUB_DST_Y && y < SUB_DST_Y + SUB_H;
               if (inside) {
                  ok = unorm_avg_ok(px, x - SUB_DST_X + SUB_SRC_X,
                                    y - SUB_DST_Y + SUB_SRC_Y, SEED_PLAIN);
               } else {
                  ok = memcmp(px, k_sub_clear8, 4) == 0;
               }
               break;
            }
            }
            if (!ok) {
               if (bad == 0) {
                  if (i == D_FLOAT) {
                     float v;
                     memcpy(&v, px, sizeof(v));
                     snprintf(first, sizeof(first), "(%u,%u)=%.4f", x, y, v);
                  } else {
                     snprintf(first, sizeof(first), "(%u,%u)=(%u,%u,%u,%u)", x, y,
                              px[0], px[1], px[2], px[3]);
                  }
               }
               bad++;
            }
         }
      }
      failed += (unsigned)report(d_name[i], bad, WIDTH * HEIGHT, first);
   }
   vkUnmapMemory(dev, rbmem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, rbmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, p_uint, NULL);
   vkDestroyPipeline(dev, p_float, NULL);
   vkDestroyPipeline(dev, p_unorm, NULL);
   vkDestroyPipeline(dev, p_resolve, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyShaderModule(dev, fs_uint, NULL);
   vkDestroyShaderModule(dev, fs_float, NULL);
   vkDestroyShaderModule(dev, fs_unorm, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   for (unsigned l = 0; l < 2; l++) {
      vkDestroyFramebuffer(dev, fb_layer[l], NULL);
   }
   vkDestroyFramebuffer(dev, fb_uint, NULL);
   vkDestroyFramebuffer(dev, fb_float, NULL);
   vkDestroyFramebuffer(dev, fb_resolve, NULL);
   vkDestroyRenderPass(dev, rp_uint, NULL);
   vkDestroyRenderPass(dev, rp_float, NULL);
   vkDestroyRenderPass(dev, rp_unorm, NULL);
   vkDestroyRenderPass(dev, rp_resolve, NULL);
   vkDestroyImageView(dev, v_rpass, NULL);
   for (unsigned i = 0; i < D_COUNT; i++) {
      vkDestroyImage(dev, dst[i].img, NULL);
      vkFreeMemory(dev, dst[i].mem, NULL);
   }
   vkDestroyImageView(dev, v_layer[1], NULL);
   vkDestroyImageView(dev, v_layer[0], NULL);
   vkDestroyImageView(dev, v_uint, NULL);
   vkDestroyImageView(dev, v_float, NULL);
   vkDestroyImageView(dev, v_unorm, NULL);
   const struct image *srcs[4] = { &src_unorm, &src_float, &src_uint, &src_layers };
   for (unsigned i = 0; i < 4; i++) {
      vkDestroyImage(dev, srcs[i]->img, NULL);
      vkFreeMemory(dev, srcs[i]->mem, NULL);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u of %u resolve cases)\n", failed, (unsigned)D_COUNT);
      return 1;
   }
   printf("PASSED (%u resolve cases: render pass and vkCmdResolveImage, RGBA8, "
          "R32F, RGBA8UI, sub-rectangle, layer)\n", (unsigned)D_COUNT);
   return 0;
}
