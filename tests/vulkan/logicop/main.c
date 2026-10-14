/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Logic-op test for the vortexpipe driver.
 *
 * The target is cleared to yellow and a full-screen quad emits magenta through
 * an XOR logic op. A unorm target merges every pixel to cyan, a value neither
 * operand carries; an sRGB or float target takes no logic op, so the magenta
 * source lands unmodified. Every channel is 0 or 1, so each result is exact.
 * The blend factors are left at their defaults, which the output merger reads
 * as "blending disabled": a logic op has to take effect anyway.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

#define RGBA_MASK (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | \
                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)

struct pass {
   const char *name;
   VkFormat format;
   float expect[4];
};

static const struct pass kPasses[] = {
   { "unorm", VK_FORMAT_R8G8B8A8_UNORM,      { 0.0f, 1.0f, 1.0f, 0.0f } },
   { "srgb",  VK_FORMAT_R8G8B8A8_SRGB,       { 1.0f, 0.0f, 1.0f, 1.0f } },
   { "float", VK_FORMAT_R16G16B16A16_SFLOAT, { 1.0f, 0.0f, 1.0f, 1.0f } },
};

struct ctx {
   VkDevice dev;
   VkQueue queue;
   VkCommandPool cp;
   VkPhysicalDeviceMemoryProperties mp;
   VkShaderModule vs;
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
   if (buf) *out_size = (size_t)sz;
   return buf;
}

/* first memory type satisfying `want`; UINT32_MAX if none. */
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

static VkDeviceMemory
bind_memory(const struct ctx *c, VkMemoryRequirements mr, VkMemoryPropertyFlags want)
{
   uint32_t mt = find_mem(&c->mp, mr.memoryTypeBits, want);
   if (mt == UINT32_MAX)
      return VK_NULL_HANDLE;
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem = VK_NULL_HANDLE;
   if (vkAllocateMemory(c->dev, &mai, NULL, &mem) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return mem;
}

static float
half_to_float(uint16_t h)
{
   const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
   float v;
   if (e == 0) {
      v = (float)m / 16777216.0f;
   } else {
      const uint32_t bits = (e == 31) ? (0x7f800000u | (m << 13))
                                      : (((e + 112u) << 23) | (m << 13));
      memcpy(&v, &bits, sizeof v);
   }
   return (h & 0x8000) ? -v : v;
}

/* Renders one pass and counts the pixels holding the expected colour. */
static int
run_pass(const struct ctx *c, const struct pass *p, unsigned *matched, float got[4])
{
   const bool is_half = p->format == VK_FORMAT_R16G16B16A16_SFLOAT;
   const uint32_t bpp = is_half ? 8u : 4u;

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = p->format,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(c->dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, img, &imr);
   VkDeviceMemory imem = bind_memory(c, imr, 0);
   if (!imem) { fprintf(stderr, "FAILED: no image memory\n"); return 1; }
   CHECK(vkBindImageMemory(c->dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = p->format,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = p->format, .samples = VK_SAMPLE_COUNT_1_BIT,
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
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &fb));

   VkShaderModule fs = load_module(c->dev, "logicop_xor.frag.spv");
   if (!fs) return 1;
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = c->vs, .pName = "main" },
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
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_TRUE, .logicOp = VK_LOGIC_OP_XOR,
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
   CHECK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * bpp;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(c->dev, rb, &bmr);
   VkDeviceMemory bmem = bind_memory(c, bmr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (!bmem) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   CHECK(vkBindBufferMemory(c->dev, rb, bmem, 0));

   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = c->cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkClearValue clear = { .color = { .float32 = { 1.0f, 1.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   uint8_t *px;
   CHECK(vkMapMemory(c->dev, bmem, 0, bytes, 0, (void **)&px));
   *matched = 0;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      float v[4];
      for (uint32_t ch = 0; ch < 4; ch++) {
         v[ch] = is_half ? half_to_float(((const uint16_t *)px)[i * 4 + ch])
                         : (float)px[i * 4 + ch] / 255.0f;
      }
      bool ok = true;
      for (uint32_t ch = 0; ch < 4; ch++) {
         if (v[ch] != p->expect[ch])
            ok = false;
      }
      if (ok)
         (*matched)++;
      if (i == (HEIGHT / 2) * WIDTH + WIDTH / 2)
         memcpy(got, v, sizeof v);
   }
   vkUnmapMemory(c->dev, bmem);

   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyShaderModule(c->dev, fs, NULL);
   vkDestroyFramebuffer(c->dev, fb, NULL);
   vkDestroyRenderPass(c->dev, rp, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyImage(c->dev, img, NULL);
   vkFreeMemory(c->dev, imem, NULL);
   vkDestroyBuffer(c->dev, rb, NULL);
   vkFreeMemory(c->dev, bmem, NULL);
   return 0;
}

int
main(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-logicop",
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
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceFeatures avail;
   vkGetPhysicalDeviceFeatures(pd, &avail);
   if (!avail.logicOp) { printf("FAILED: logicOp is not supported\n"); return 1; }

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
   VkPhysicalDeviceFeatures feats = { .logicOp = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &feats,
   };
   struct ctx c;
   CHECK(vkCreateDevice(pd, &dci, NULL, &c.dev));
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c.mp);
   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
   };
   CHECK(vkCreateCommandPool(c.dev, &cmpci, NULL, &c.cp));
   c.vs = load_module(c.dev, "logicop.vert.spv");
   if (!c.vs) return 1;

   const unsigned total = WIDTH * HEIGHT;
   bool ok = true;
   for (size_t k = 0; k < sizeof kPasses / sizeof kPasses[0]; k++) {
      const struct pass *p = &kPasses[k];
      unsigned matched = 0;
      float got[4] = { 0 };
      if (run_pass(&c, p, &matched, got))
         return 1;
      printf("%s: %u/%u pixels = (%g, %g, %g, %g); centre = (%g, %g, %g, %g)\n",
             p->name, matched, total, p->expect[0], p->expect[1], p->expect[2],
             p->expect[3], got[0], got[1], got[2], got[3]);
      if (matched != total)
         ok = false;
   }

   vkDestroyShaderModule(c.dev, c.vs, NULL);
   vkDestroyCommandPool(c.dev, c.cp, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!ok) {
      printf("FAILED (logicop)\n");
      return 1;
   }
   printf("PASSED (logicop: XOR on unorm, sRGB and float pass-through exact)\n");
   return 0;
}
