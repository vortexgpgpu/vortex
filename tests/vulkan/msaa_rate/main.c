/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Where a 4x multisample pass evaluates its fragment shader.
 *
 * Each case draws into an 8x8 4x float attachment cleared to -1, then a second,
 * single-sample pass copies every sample of it into a buffer through texelFetch
 * on a sampler2DMS. The fragment inputs are window positions, so every stored
 * value says where in the pixel it was computed.
 *
 *   sid       gl_SampleID / gl_SamplePosition: sample k holds (k, position of
 *             k, 1), exactly. Positions are the standard 4x pattern.
 *   minss     minSampleShading = 1.0, no per-sample built-in read: sample k
 *             holds the window position of sample k, so the four samples of
 *             every pixel hold four distinct values.
 *   centroid  one invocation per pixel, centroid input, a triangle edge at
 *             x = 4.5: fully covered pixels hold their centre, column 4 holds
 *             the position of one of its covered samples (0 or 2) in both of
 *             them, and its uncovered samples keep the clear.
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
#define READ_FORMAT VK_FORMAT_R8G8B8A8_UNORM

/* Window positions are multiples of 1/8 below 16, exact in either format;
 * the bound covers the interpolation arithmetic only. */
#define TOL 1e-3f

/* The x of the one triangle edge inside the target, in window coordinates. */
#define EDGE_X 4.5f

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

enum rate_kind { RATE_SID, RATE_MINSS, RATE_CENTROID };

struct rate_case {
   const char    *name;
   const char    *fs;
   enum rate_kind kind;
   bool           sample_shading;
   uint32_t       first_vertex;
   uint32_t       vertex_count;
};

struct env {
   VkDevice                         dev;
   VkQueue                          queue;
   VkCommandBuffer                  cmd;
   VkRenderPass                     rp_draw, rp_read;
   VkFramebuffer                    fb_draw, fb_read;
   VkPipelineLayout                 pl_draw, pl_read;
   VkPipeline                       pipe_read;
   VkDescriptorSet                  dset;
   VkShaderModule                   vs;
   const float                     *samples;   /* mapped read-back buffer */
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
is_clear(const float *s)
{
   return s[0] == -1.0f && s[1] == -1.0f && s[2] == -1.0f && s[3] == -1.0f;
}

/* Whether a stored sample is the window position (x, y) as the shaders write
 * it, (x, y, 0, 1). */
static bool
is_position(const float *s, float x, float y)
{
   return near(s[0], x) && near(s[1], y) && s[2] == 0.0f && s[3] == 1.0f;
}

static bool
check_pixel(const struct rate_case *c, uint32_t px, uint32_t py, const float *s)
{
   switch (c->kind) {
   case RATE_SID:
      for (uint32_t k = 0; k < SAMPLES; k++) {
         const float *v = s + k * 4;
         if (v[0] != (float)k || v[1] != k_pos[k][0] || v[2] != k_pos[k][1] ||
             v[3] != 1.0f) {
            return false;
         }
      }
      return true;
   case RATE_MINSS:
      for (uint32_t k = 0; k < SAMPLES; k++) {
         if (!is_position(s + k * 4, (float)px + k_pos[k][0], (float)py + k_pos[k][1])) {
            return false;
         }
      }
      return true;
   case RATE_CENTROID: {
      uint32_t covered = 0;
      for (uint32_t k = 0; k < SAMPLES; k++) {
         if ((float)px + k_pos[k][0] < EDGE_X) {
            covered |= 1u << k;
         }
      }
      int first = -1;
      for (uint32_t k = 0; k < SAMPLES; k++) {
         const float *v = s + k * 4;
         if (!(covered & (1u << k))) {
            if (!is_clear(v)) {
               return false;
            }
            continue;
         }
         /* One invocation shades the whole pixel, so its covered samples agree. */
         if (first < 0) {
            first = (int)k;
         } else if (memcmp(v, s + first * 4, 4 * sizeof(float)) != 0) {
            return false;
         }
      }
      if (covered == 0) {
         return true;
      }
      const float *v = s + first * 4;
      if (covered == (1u << SAMPLES) - 1) {
         return is_position(v, (float)px + 0.5f, (float)py + 0.5f);
      }
      for (uint32_t k = 0; k < SAMPLES; k++) {
         if ((covered & (1u << k)) &&
             is_position(v, (float)px + k_pos[k][0], (float)py + k_pos[k][1])) {
            return true;
         }
      }
      return false;
   }
   }
   return false;
}

/* Draws the case into the multisample attachment, copies its samples out and
 * checks them. Returns the number of failing pixels, or -1 on an API error. */
static int
run_case(struct env *e, const struct rate_case *c)
{
   VkShaderModule fs = load_module(e->dev, c->fs);
   if (!fs) {
      return -1;
   }
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)SAMPLES,
      .sampleShadingEnable = c->sample_shading,
      .minSampleShading = c->sample_shading ? 1.0f : 0.0f,
   };
   VkPipeline pipe;
   if (make_pipeline(e->dev, e->vs, fs, e->rp_draw, e->pl_draw, &ms, &pipe) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: %s pipeline\n", c->name);
      return -1;
   }

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   vkResetCommandBuffer(e->cmd, 0);
   vkBeginCommandBuffer(e->cmd, &cbbi);

   VkClearValue clear = { .color = { .float32 = { -1.0f, -1.0f, -1.0f, -1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = e->rp_draw, .framebuffer = e->fb_draw,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(e->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(e->cmd, c->vertex_count, 1, c->first_vertex, 0);
   vkCmdEndRenderPass(e->cmd);

   memory_barrier(e->cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

   VkClearValue black = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } };
   rpbi.renderPass = e->rp_read;
   rpbi.framebuffer = e->fb_read;
   rpbi.pClearValues = &black;
   vkCmdBeginRenderPass(e->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, e->pipe_read);
   vkCmdBindDescriptorSets(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, e->pl_read,
                           0, 1, &e->dset, 0, NULL);
   int32_t width = (int32_t)WIDTH;
   vkCmdPushConstants(e->cmd, e->pl_read, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(width), &width);
   vkCmdDraw(e->cmd, 6, 1, 0, 0);
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
         const float *s = e->samples + (size_t)(y * WIDTH + x) * SAMPLES * 4;
         if (check_pixel(c, x, y, s)) {
            continue;
         }
         if (bad < 4) {
            printf("  %s (%u,%u):", c->name, x, y);
            for (uint32_t k = 0; k < SAMPLES; k++) {
               printf(" s%u=(%.4f,%.4f,%.4f,%.4f)", k, s[k * 4 + 0], s[k * 4 + 1],
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
      .pApplicationName = "vortexpipe-msaa_rate",
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
   if (!(props.limits.standardSampleLocations)) {
      printf("FAILED (device does not use the standard sample locations)\n");
      return 1;
   }
   /* RGBA16F holds every value written here exactly, so it is an equal
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

   /* Four vec4 per pixel, one per sample. */
   const VkDeviceSize sbytes = (VkDeviceSize)WIDTH * HEIGHT * SAMPLES * 16;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sbytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer sbuf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &sbuf));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, sbuf, &bmr);
   VkDeviceMemory sbmem;
   CHECK(bind_memory(dev, &mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sbmem));
   CHECK(vkBindBufferMemory(dev, sbuf, sbmem, 0));
   void *smap;
   CHECK(vkMapMemory(dev, sbmem, 0, sbytes, 0, &smap));
   e.samples = smap;

   e.vs = load_module(dev, "rate.vert.spv");
   VkShaderModule read_fs = load_module(dev, "rate_read.frag.spv");
   if (!e.vs || !read_fs) {
      return 1;
   }

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
   };
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &e.pl_draw));
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4,
   };
   plci.setLayoutCount = 1;
   plci.pSetLayouts = &dsl;
   plci.pushConstantRangeCount = 1;
   plci.pPushConstantRanges = &pcr;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &e.pl_read));

   VkPipelineMultisampleStateCreateInfo ms_single = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   CHECK(make_pipeline(dev, e.vs, read_fs, e.rp_read, e.pl_read, &ms_single,
                       &e.pipe_read));

   /* A multisample image is fetched by integer coordinate, never filtered. */
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
      { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
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
   VkDescriptorImageInfo dii = {
      .sampler = sampler, .imageView = msview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkDescriptorBufferInfo dbi = { .buffer = sbuf, .offset = 0, .range = sbytes };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e.dset,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = e.dset,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
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

   const struct rate_case cases[] = {
      { "sid",      "rate_sid.frag.spv",      RATE_SID,      false, 0, 6 },
      { "minss",    "rate_minss.frag.spv",    RATE_MINSS,    true,  0, 6 },
      { "centroid", "rate_centroid.frag.spv", RATE_CENTROID, false, 6, 3 },
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

   vkUnmapMemory(dev, sbmem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipeline(dev, e.pipe_read, NULL);
   vkDestroyPipelineLayout(dev, e.pl_read, NULL);
   vkDestroyPipelineLayout(dev, e.pl_draw, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, read_fs, NULL);
   vkDestroyShaderModule(dev, e.vs, NULL);
   vkFreeMemory(dev, sbmem, NULL);
   vkDestroyBuffer(dev, sbuf, NULL);
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
      printf("FAILED (%u of %u shading-rate cases%s)\n", failed, ncases,
             api_error ? ", stopped on an API error" : "");
      return 1;
   }
   printf("PASSED (%u shading-rate cases: sample id/position, minSampleShading, "
          "centroid)\n", ncases);
   return 0;
}
