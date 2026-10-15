/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Per-sample coverage controls of a 4x multisample pass.
 *
 * Each case draws a full-screen quad into a 4x RGBA8 attachment cleared to
 * green, then a second, single-sample pass copies every sample of it into a
 * buffer through texelFetch on a sampler2DMS. A sample either holds the draw's
 * red or still holds the clear; anything else is a corrupted sample.
 *
 *   a2c     alpha-to-coverage, alpha 0.0 / 0.3 / 0.6 / 1.0 in the four screen
 *           quadrants -> round(4 * alpha) = 0 / 1 / 2 / 4 samples written per
 *           pixel. Which samples an alpha selects is implementation-defined, so
 *           only the count is asserted; the written samples keep the shader's
 *           alpha.
 *   smask   pipeline pSampleMask = 0b0101 -> exactly samples 0 and 2.
 *   fsmask  gl_SampleMask[0] = 0b1010 from the shader -> exactly samples 1, 3.
 *   a2one   alpha-to-one with alpha 0.25 -> all samples, alpha stored as 1.0.
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
#define FORMAT   VK_FORMAT_R8G8B8A8_UNORM

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* Expected per-sample state: which samples a case writes, either by index or,
 * for alpha-to-coverage, only by how many. */
enum cov_kind { COV_A2C, COV_MASK, COV_ALL };

struct cov_case {
   const char   *name;
   const char   *fs;
   enum cov_kind kind;
   uint32_t      mask;          /* COV_MASK: samples that must hold the draw */
   VkSampleMask  pipe_mask;     /* pipeline pSampleMask, ~0 for none */
   bool          a2c;
   bool          a2one;
   int           alpha;         /* stored alpha the written samples must hold;
                                   -1 = the a2c quadrant's own alpha */
};

static const float k_a2c_alpha[4] = { 0.0f, 0.3f, 0.6f, 1.0f };
static const unsigned k_a2c_count[4] = { 0, 1, 2, 4 };

struct env {
   VkDevice                         dev;
   VkQueue                          queue;
   VkPhysicalDeviceMemoryProperties mp;
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
make_render_pass(VkDevice dev, VkSampleCountFlagBits samples,
                 VkAttachmentStoreOp store, VkImageLayout final_layout,
                 VkRenderPass *out)
{
   VkAttachmentDescription att = {
      .format = FORMAT, .samples = samples,
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

static uint8_t
to_unorm8(float v)
{
   return (uint8_t)lroundf(v * 255.0f);
}

/* Draws the case into the multisample attachment, copies its samples out and
 * checks them. Returns the number of failing pixels, or -1 on an API error. */
static int
run_case(struct env *e, const struct cov_case *c)
{
   VkShaderModule fs = load_module(e->dev, c->fs);
   if (!fs) {
      return -1;
   }
   VkSampleMask pipe_mask = c->pipe_mask;
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)SAMPLES,
      .pSampleMask = (pipe_mask != ~0u) ? &pipe_mask : NULL,
      .alphaToCoverageEnable = c->a2c,
      .alphaToOneEnable = c->a2one,
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

   VkClearValue green = { .color = { .float32 = { 0.0f, 1.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = e->rp_draw, .framebuffer = e->fb_draw,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &green,
   };
   vkCmdBeginRenderPass(e->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(e->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(e->cmd, 6, 1, 0, 0);
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
   unsigned corrupt = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const unsigned q = (y >= HEIGHT / 2 ? 2u : 0u) + (x >= WIDTH / 2 ? 1u : 0u);
         const int want_alpha = c->alpha >= 0 ? c->alpha
                                              : (int)lroundf(k_a2c_alpha[q] * 255.0f);
         uint32_t drawn = 0;
         bool pixel_ok = true;
         for (uint32_t k = 0; k < SAMPLES; k++) {
            const float *s = e->samples + ((size_t)(y * WIDTH + x) * SAMPLES + k) * 4;
            const uint8_t r8 = to_unorm8(s[0]), g8 = to_unorm8(s[1]),
                          b8 = to_unorm8(s[2]), a8 = to_unorm8(s[3]);
            if (r8 == 255 && g8 == 0 && b8 == 0) {
               drawn |= 1u << k;
               if (abs((int)a8 - want_alpha) > 1) {
                  pixel_ok = false;
               }
            } else if (!(r8 == 0 && g8 == 255 && b8 == 0 && a8 == 255)) {
               corrupt++;
               pixel_ok = false;
            }
         }
         if (c->kind == COV_A2C) {
            pixel_ok = pixel_ok &&
                       (unsigned)__builtin_popcount(drawn) == k_a2c_count[q];
         } else {
            const uint32_t want = (c->kind == COV_ALL) ? (1u << SAMPLES) - 1 : c->mask;
            pixel_ok = pixel_ok && drawn == want;
         }
         if (!pixel_ok) {
            if (bad < 4) {
               const float *s = e->samples + (size_t)(y * WIDTH + x) * SAMPLES * 4;
               printf("  %s (%u,%u): drawn=0x%x", c->name, x, y, drawn);
               for (uint32_t k = 0; k < SAMPLES; k++) {
                  printf(" s%u=(%u,%u,%u,%u)", k, to_unorm8(s[k * 4 + 0]),
                         to_unorm8(s[k * 4 + 1]), to_unorm8(s[k * 4 + 2]),
                         to_unorm8(s[k * 4 + 3]));
               }
               printf("\n");
            }
            bad++;
         }
      }
   }
   printf("%-7s: %s  bad pixels=%d of %u, corrupt samples=%u\n", c->name,
          bad ? "FAIL" : "pass", bad, WIDTH * HEIGHT, corrupt);
   return bad;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_cov",
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
   if (!(props.limits.framebufferColorSampleCounts & SAMPLES) ||
       !(props.limits.sampledImageColorSampleCounts & SAMPLES) ||
       !have.alphaToOne || !have.fragmentStoresAndAtomics) {
      printf("FAILED (device lacks 4x colour/sampled counts, alphaToOne or "
             "fragment stores: fb=0x%x sampled=0x%x a2one=%u stores=%u)\n",
             (unsigned)props.limits.framebufferColorSampleCounts,
             (unsigned)props.limits.sampledImageColorSampleCounts,
             (unsigned)have.alphaToOne, (unsigned)have.fragmentStoresAndAtomics);
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
   VkPhysicalDeviceFeatures want = {
      .alphaToOne = VK_TRUE, .fragmentStoresAndAtomics = VK_TRUE,
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
   vkGetPhysicalDeviceMemoryProperties(pd, &e.mp);

   VkImage msimg, ssimg;
   VkDeviceMemory msmem, ssmem;
   VkImageView msview, ssview;
   CHECK(make_image(dev, &e.mp, FORMAT, (VkSampleCountFlagBits)SAMPLES,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    &msimg, &msmem, &msview));
   CHECK(make_image(dev, &e.mp, FORMAT, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &ssimg, &ssmem, &ssview));

   CHECK(make_render_pass(dev, (VkSampleCountFlagBits)SAMPLES,
                          VK_ATTACHMENT_STORE_OP_STORE,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &e.rp_draw));
   CHECK(make_render_pass(dev, VK_SAMPLE_COUNT_1_BIT,
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
   CHECK(bind_memory(dev, &e.mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sbmem));
   CHECK(vkBindBufferMemory(dev, sbuf, sbmem, 0));
   void *smap;
   CHECK(vkMapMemory(dev, sbmem, 0, sbytes, 0, &smap));
   e.samples = smap;

   e.vs = load_module(dev, "cov.vert.spv");
   VkShaderModule read_fs = load_module(dev, "cov_read.frag.spv");
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

   const struct cov_case cases[] = {
      { "a2c",    "cov_a2c.frag.spv",   COV_A2C,  0,   ~0u,  true,  false, -1  },
      { "smask",  "cov_solid.frag.spv", COV_MASK, 0x5, 0x5,  false, false, 255 },
      { "fsmask", "cov_smask.frag.spv", COV_MASK, 0xA, ~0u,  false, false, 255 },
      { "a2one",  "cov_a2one.frag.spv", COV_ALL,  0,   ~0u,  false, true,  255 },
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
      printf("FAILED (%u of %u coverage cases%s)\n", failed, ncases,
             api_error ? ", stopped on an API error" : "");
      return 1;
   }
   printf("PASSED (%u coverage cases: a2c counts, pipeline and shader sample "
          "masks, alpha-to-one)\n", ncases);
   return 0;
}
