/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Interpolation functions and sample built-ins of a 4x multisample pass.
 *
 * Each case draws one triangle covering every pixel of an 8x8 4x float
 * attachment; its fragment shader writes what it observed into a storage
 * buffer of four vec4 slots per pixel, filled with -1 beforehand, and the host
 * checks every slot. The input v is the window position, without perspective,
 * so an interpolated value is the point it was evaluated at.
 *
 *   offset    interpolateAtOffset(v, (-0.5, -0.5)) and (0.4375, 0.4375): the
 *             pixel's corner (x, y) and (x + 0.9375, y + 0.9375).
 *   sample    interpolateAtSample(v, k), k = 0..3: the standard 4x positions.
 *   centroid  interpolateAtCentroid(v) on a fully covered pixel: the centre.
 *   cmask     the same with pipeline pSampleMask = 0b0101: still the centre,
 *             the primitive covering the whole pixel.
 *   msample   gl_SampleMaskIn with minSampleShading = 1: 1 << gl_SampleID in
 *             the slot of each sample.
 *   mpixel    gl_SampleMaskIn at pixel rate: 0xF.
 *   samples   textureSamples of the 4x attachment through a sampler2DMS, in a
 *             second, single-sample pass: 4.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH    8u
#define HEIGHT   8u
#define SAMPLES  4u
#define SLOTS    4u
#define READ_FORMAT VK_FORMAT_R8G8B8A8_UNORM

/* Every expected position is a multiple of 1/16 below 16, exact in float; the
 * bound covers the interpolation arithmetic only. */
#define TOL 1e-4f

#define UNWRITTEN -1.0f

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static const float k_pos[SAMPLES][2] = {
   { 0.375f, 0.125f }, { 0.875f, 0.375f }, { 0.125f, 0.625f }, { 0.625f, 0.875f },
};

enum interp_kind {
   KIND_OFFSET, KIND_SAMPLE, KIND_CENTROID, KIND_MASK_SAMPLE, KIND_MASK_PIXEL,
   KIND_SAMPLES,
};

struct interp_case {
   const char      *name;
   const char      *fs;
   enum interp_kind kind;
   bool             sample_shading;
   VkSampleMask     pipe_mask;
};

struct env {
   VkDevice                         dev;
   VkQueue                          queue;
   VkCommandBuffer                  cmd;
   VkRenderPass                     rp_draw, rp_read;
   VkFramebuffer                    fb_draw, fb_read;
   VkPipelineLayout                 layout;
   VkDescriptorSet                  dset;
   VkShaderModule                   vs;
   VkBuffer                         buf;
   const float                     *res;   /* mapped result buffer */
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

static VkResult
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
           VkFormat fmt, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
           VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = vkCreateImage(dev, &ici, NULL, img);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, *img, &mr);
   r = bind_memory(dev, mp, &mr, 0, mem);
   if (r == VK_SUCCESS) {
      r = vkBindImageMemory(dev, *img, *mem, 0);
   }
   if (r != VK_SUCCESS) {
      return r;
   }
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   return vkCreateImageView(dev, &vci, NULL, view);
}

static VkResult
make_render_pass(VkDevice dev, VkFormat fmt, VkSampleCountFlagBits samples,
                 VkAttachmentStoreOp store, VkImageLayout final_layout,
                 VkRenderPass *out)
{
   VkAttachmentDescription att = {
      .format = fmt, .samples = samples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = store,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = final_layout,
   };
   VkAttachmentReference ref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   return vkCreateRenderPass(dev, &rpci, NULL, out);
}

static VkResult
make_pipeline(VkDevice dev, VkShaderModule vs, VkShaderModule fs,
              VkRenderPass rp, VkPipelineLayout layout,
              const VkPipelineMultisampleStateCreateInfo *ms, VkPipeline *out)
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
      .pMultisampleState = ms, .pColorBlendState = &cb,
      .layout = layout, .renderPass = rp, .subpass = 0,
   };
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static void
memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage,
               VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
               VkAccessFlags dst_access)
{
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = src_access, .dstAccessMask = dst_access,
   };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

static bool
near(float a, float b)
{
   return fabsf(a - b) <= TOL;
}

static bool
is_vec(const float *s, float a, float b, float c, float d)
{
   return near(s[0], a) && near(s[1], b) && near(s[2], c) && near(s[3], d);
}

static bool
is_unwritten(const float *s)
{
   return s[0] == UNWRITTEN && s[1] == UNWRITTEN && s[2] == UNWRITTEN && s[3] == UNWRITTEN;
}

/* Whether the four slots of pixel (x, y) hold what the case writes there. */
static bool
check_pixel(const struct interp_case *c, uint32_t px, uint32_t py, const float *s)
{
   const float x = (float)px, y = (float)py;
   switch (c->kind) {
   case KIND_OFFSET:
      return is_vec(s, x, y, x + 0.9375f, y + 0.9375f) &&
             is_unwritten(s + 4) && is_unwritten(s + 8) && is_unwritten(s + 12);
   case KIND_SAMPLE:
      for (uint32_t k = 0; k < SAMPLES; k++) {
         if (!is_vec(s + k * 4, x + k_pos[k][0], y + k_pos[k][1], 0.0f, 1.0f)) {
            return false;
         }
      }
      return true;
   case KIND_CENTROID:
      return is_vec(s, x + 0.5f, y + 0.5f, 0.0f, 1.0f) &&
             is_unwritten(s + 4) && is_unwritten(s + 8) && is_unwritten(s + 12);
   case KIND_MASK_SAMPLE:
      for (uint32_t k = 0; k < SAMPLES; k++) {
         const float *v = s + k * 4;
         if (v[0] != (float)(1u << k) || v[1] != (float)k || v[2] != 0.0f || v[3] != 1.0f) {
            return false;
         }
      }
      return true;
   case KIND_MASK_PIXEL:
      return s[0] == (float)((1u << SAMPLES) - 1) && s[1] == 0.0f && s[2] == 0.0f &&
             s[3] == 1.0f && is_unwritten(s + 4) && is_unwritten(s + 8) &&
             is_unwritten(s + 12);
   case KIND_SAMPLES:
      return s[0] == (float)SAMPLES && s[1] == 0.0f && s[2] == 0.0f && s[3] == 1.0f &&
             is_unwritten(s + 4) && is_unwritten(s + 8) && is_unwritten(s + 12);
   }
   return false;
}

/* Runs the case's shader over every pixel, in the 4x pass or, for the sample
 * count query, in the single-sample pass reading the 4x attachment, and checks
 * the result buffer. Returns the number of failing pixels, or -1 on an API
 * error. */
static int
run_case(struct env *e, const struct interp_case *c)
{
   VkShaderModule fs = load_module(e->dev, c->fs);
   if (!fs) {
      return -1;
   }
   const bool read_pass = c->kind == KIND_SAMPLES;
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = read_pass ? VK_SAMPLE_COUNT_1_BIT
                                        : (VkSampleCountFlagBits)SAMPLES,
      .sampleShadingEnable = c->sample_shading,
      .minSampleShading = c->sample_shading ? 1.0f : 0.0f,
      .pSampleMask = (c->pipe_mask != ~0u) ? &c->pipe_mask : NULL,
   };
   VkPipeline pipe;
   if (make_pipeline(e->dev, e->vs, fs, read_pass ? e->rp_read : e->rp_draw, e->layout,
                     &ms, &pipe) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: %s pipeline\n", c->name);
      vkDestroyShaderModule(e->dev, fs, NULL);
      return -1;
   }

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   vkResetCommandBuffer(e->cmd, 0);
   vkBeginCommandBuffer(e->cmd, &cbbi);

   float unwritten = UNWRITTEN;
   uint32_t fill;
   memcpy(&fill, &unwritten, sizeof(fill));
   vkCmdFillBuffer(e->cmd, e->buf, 0, VK_WHOLE_SIZE, fill);
   memory_barrier(e->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);

   VkClearValue clear = { .color = { .float32 = { -1.0f, -1.0f, -1.0f, -1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = read_pass ? e->rp_read : e->rp_draw,
      .framebuffer = read_pass ? e->fb_read : e->fb_draw,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(e->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, e->layout,
                           0, 1, &e->dset, 0, NULL);
   vkCmdDraw(e->cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(e->cmd);

   memory_barrier(e->cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
   vkEndCommandBuffer(e->cmd);

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &e->cmd,
   };
   VkResult r = vkQueueSubmit(e->queue, 1, &si, VK_NULL_HANDLE);
   if (r == VK_SUCCESS) {
      r = vkQueueWaitIdle(e->queue);
   }
   vkDestroyPipeline(e->dev, pipe, NULL);
   vkDestroyShaderModule(e->dev, fs, NULL);
   if (r != VK_SUCCESS) {
      fprintf(stderr, "FAILED: %s submit -> VkResult %d\n", c->name, (int)r);
      return -1;
   }

   int bad = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const float *s = e->res + (size_t)(y * WIDTH + x) * SLOTS * 4;
         if (check_pixel(c, x, y, s)) {
            continue;
         }
         if (bad < 4) {
            printf("  %s (%u,%u):", c->name, x, y);
            for (uint32_t k = 0; k < SLOTS; k++) {
               printf(" [%u]=(%.4f,%.4f,%.4f,%.4f)", k, s[k * 4 + 0], s[k * 4 + 1],
                      s[k * 4 + 2], s[k * 4 + 3]);
            }
            printf("\n");
         }
         bad++;
      }
   }
   printf("%-8s: %s  bad pixels=%d of %u\n", c->name, bad ? "FAIL" : "pass", bad,
          WIDTH * HEIGHT);
   return bad;
}

static bool
format_ok(VkPhysicalDevice pd, VkFormat fmt)
{
   VkImageFormatProperties ifp;
   if (vkGetPhysicalDeviceImageFormatProperties(pd, fmt, VK_IMAGE_TYPE_2D,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0,
          &ifp) != VK_SUCCESS) {
      return false;
   }
   return (ifp.sampleCounts & SAMPLES) != 0;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_interp",
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
   if (!have.sampleRateShading || !have.fragmentStoresAndAtomics) {
      printf("FAILED (device lacks sampleRateShading=%u or fragment stores=%u)\n",
             (unsigned)have.sampleRateShading, (unsigned)have.fragmentStoresAndAtomics);
      return 1;
   }
   if (!props.limits.standardSampleLocations) {
      printf("FAILED (device does not use the standard sample locations)\n");
      return 1;
   }
   if (props.limits.minInterpolationOffset > -0.5f ||
       props.limits.maxInterpolationOffset < 0.4375f) {
      printf("FAILED (interpolation offsets [%g, %g] do not reach -0.5 and 0.4375)\n",
             props.limits.minInterpolationOffset, props.limits.maxInterpolationOffset);
      return 1;
   }
   /* The attachment's values are never checked, so RGBA16F is an equal
    * substitute where RGBA32F cannot be a 4x attachment. */
   VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
   if (!format_ok(pd, format)) {
      format = VK_FORMAT_R16G16B16A16_SFLOAT;
      if (!format_ok(pd, format)) {
         printf("FAILED (neither RGBA32F nor RGBA16F is a sampled 4x attachment)\n");
         return 1;
      }
   }
   printf("format: %s\n", format == VK_FORMAT_R32G32B32A32_SFLOAT ? "RGBA32F" : "RGBA16F");

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
   VkPhysicalDeviceFeatures want = {
      .sampleRateShading = VK_TRUE, .fragmentStoresAndAtomics = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
   };
   struct env e = { 0 };
   CHECK(vkCreateDevice(pd, &dci, NULL, &e.dev));
   VkDevice dev = e.dev;
   vkGetDeviceQueue(dev, qf, 0, &e.queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   VkImage msimg, ssimg;
   VkDeviceMemory msmem, ssmem;
   VkImageView msview, ssview;
   CHECK(make_image(dev, &mp, format, (VkSampleCountFlagBits)SAMPLES,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    &msimg, &msmem, &msview));
   CHECK(make_image(dev, &mp, READ_FORMAT, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &ssimg, &ssmem, &ssview));

   CHECK(make_render_pass(dev, format, (VkSampleCountFlagBits)SAMPLES,
                          VK_ATTACHMENT_STORE_OP_STORE,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &e.rp_draw));
   CHECK(make_render_pass(dev, READ_FORMAT, VK_SAMPLE_COUNT_1_BIT,
                          VK_ATTACHMENT_STORE_OP_DONT_CARE,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &e.rp_read));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = e.rp_draw, .attachmentCount = 1, .pAttachments = &msview,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &e.fb_draw));
   fbci.renderPass = e.rp_read;
   fbci.pAttachments = &ssview;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &e.fb_read));

   const VkDeviceSize rbytes = (VkDeviceSize)WIDTH * HEIGHT * SLOTS * 16;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = rbytes,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &e.buf));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, e.buf, &bmr);
   VkDeviceMemory bmem;
   CHECK(bind_memory(dev, &mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &bmem));
   CHECK(vkBindBufferMemory(dev, e.buf, bmem, 0));
   void *rmap;
   CHECK(vkMapMemory(dev, bmem, 0, rbytes, 0, &rmap));
   e.res = rmap;

   e.vs = load_module(dev, "interp.vert.spv");
   if (!e.vs) {
      return 1;
   }

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &e.layout));

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &e.dset));
   VkDescriptorBufferInfo dbi = { .buffer = e.buf, .offset = 0, .range = rbytes };
   VkDescriptorImageInfo dii = {
      .sampler = sampler, .imageView = msview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e.dset,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e.dset,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

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

   /* The sample count query runs last: it reads the 4x attachment, which the
    * cases before it leave written and in the shader-read layout. */
   const struct interp_case cases[] = {
      { "offset",   "interp_offset.frag.spv",   KIND_OFFSET,      false, ~0u },
      { "sample",   "interp_sample.frag.spv",   KIND_SAMPLE,      false, ~0u },
      { "centroid", "interp_centroid.frag.spv", KIND_CENTROID,    false, ~0u },
      { "cmask",    "interp_centroid.frag.spv", KIND_CENTROID,    false, 0x5u },
      { "msample",  "mask_sample.frag.spv",     KIND_MASK_SAMPLE, true,  ~0u },
      { "mpixel",   "mask_pixel.frag.spv",      KIND_MASK_PIXEL,  false, ~0u },
      { "samples",  "tex_samples.frag.spv",     KIND_SAMPLES,     false, ~0u },
   };
   const unsigned ncases = sizeof(cases) / sizeof(cases[0]);
   unsigned failed = 0;
   bool api_error = false;
   for (unsigned i = 0; i < ncases; i++) {
      int bad = run_case(&e, &cases[i]);
      if (bad < 0) {
         api_error = true;
         break;
      }
      if (bad > 0) {
         failed++;
      }
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipelineLayout(dev, e.layout, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, e.vs, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, e.buf, NULL);
   vkDestroyFramebuffer(dev, e.fb_read, NULL);
   vkDestroyFramebuffer(dev, e.fb_draw, NULL);
   vkDestroyRenderPass(dev, e.rp_read, NULL);
   vkDestroyRenderPass(dev, e.rp_draw, NULL);
   vkDestroyImageView(dev, ssview, NULL);
   vkFreeMemory(dev, ssmem, NULL);
   vkDestroyImage(dev, ssimg, NULL);
   vkDestroyImageView(dev, msview, NULL);
   vkFreeMemory(dev, msmem, NULL);
   vkDestroyImage(dev, msimg, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (api_error || failed) {
      printf("FAILED (%u of %u interpolation cases%s)\n", failed, ncases,
             api_error ? ", stopped on an API error" : "");
      return 1;
   }
   printf("PASSED (%u interpolation cases: at offset, at sample, at centroid with and "
          "without a sample mask, sample mask in at sample and pixel rate, "
          "textureSamples)\n", ncases);
   return 0;
}
