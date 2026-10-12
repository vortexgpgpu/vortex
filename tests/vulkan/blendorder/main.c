/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Primitive-order merge test for the vortexpipe driver.
 *
 * Sixteen full-screen layers drawn in ONE instanced draw and blended
 * source-over (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) at half coverage. Every pixel
 * merges sixteen fragments launched in different warps, and the result is the
 * layers' weighted sum with the last one weighted most: two consecutive layers
 * merged in the wrong order move a channel by a quarter of the range or more,
 * while correct-order rounding stays within a couple of steps. Vulkan requires
 * primitive order here, which the warps alone do not give.
 *
 * The target is a single 2x2 quad, so consecutive fragment warps are
 * consecutive layers of the same pixels and are in flight together even on a
 * four-warp core; a larger target would space a pixel's layers further apart
 * than the warps that can be resident at once, and nothing could race.
 *
 * The pass runs twice: on RGBA8, which the fixed-function output merger takes,
 * and on RGBA16F, which takes the software merger's read-modify-write.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   2u
#define HEIGHT  2u
#define LAYERS  16u

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

static float
half_to_float(uint16_t h)
{
   const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
   float v;
   if (e == 0)       v = ldexpf((float)m, -24);
   else if (e == 31) v = m ? NAN : INFINITY;
   else              v = ldexpf((float)(m | 0x400u), (int)e - 25);
   return s ? -v : v;
}

/* Channel c of layer k, as the fragment shader writes it. */
static float
layer_channel(uint32_t k, uint32_t c)
{
   return (float)((k >> c) & 1u);
}

/* Source-over of all layers onto the cleared black, in draw order. `alpha` is
 * the coverage as the attachment stores it: 128/255 for RGBA8, 0.5 for 16F. */
static float
expected_channel(uint32_t c, float alpha)
{
   float v = 0.0f;
   for (uint32_t k = 0; k < LAYERS; k++)
      v = layer_channel(k, c) * alpha + v * (1.0f - alpha);
   return v;
}

static bool
run_pass(const struct ctx *x, VkFormat format, const char *name)
{
   const bool half = (format == VK_FORMAT_R16G16B16A16_SFLOAT);
   const uint32_t bpp = half ? 8u : 4u;

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(x->dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = format, .samples = VK_SAMPLE_COUNT_1_BIT,
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
   CHECK(vkCreateRenderPass(x->dev, &rpci, NULL, &rp));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(x->dev, &fbci, NULL, &fb));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(x->dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = x->vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = x->fs, .pName = "main" },
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
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp        = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp        = VK_BLEND_OP_ADD,
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
   CHECK(vkCreateGraphicsPipelines(x->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * bpp;
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

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = x->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(x->dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
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
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(cmd, 6, LAYERS, 0, 0);
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
   CHECK(vkQueueSubmit(x->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(x->queue));

   /* RGBA8 stores the half coverage as 128/255 and each blend step rounds to
    * a step, so allow two steps; 16F keeps 0.5 exactly and rounds to 11 bits.
    * A swapped pair of the final layers is off by 0.25 or more. */
   const float alpha = half ? 0.5f : 128.0f / 255.0f;
   const float tol   = half ? 0.004f : 2.5f / 255.0f;
   float want[3];
   for (uint32_t c = 0; c < 3; c++)
      want[c] = expected_channel(c, alpha);

   uint8_t *px;
   CHECK(vkMapMemory(x->dev, bmem, 0, bytes, 0, (void **)&px));
   unsigned bad = 0;
   float worst = 0.0f;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      for (uint32_t c = 0; c < 3; c++) {
         float got = half
            ? half_to_float(((const uint16_t *)px)[i * 4 + c])
            : (float)px[i * 4 + c] / 255.0f;
         float d = fabsf(got - want[c]);
         if (d > worst) worst = d;
         if (d > tol) { bad++; break; }
      }
   }
   vkUnmapMemory(x->dev, bmem);

   vkDestroyCommandPool(x->dev, cp, NULL);
   vkFreeMemory(x->dev, bmem, NULL);
   vkDestroyBuffer(x->dev, rb, NULL);
   vkDestroyPipeline(x->dev, pipe, NULL);
   vkDestroyPipelineLayout(x->dev, pl, NULL);
   vkDestroyFramebuffer(x->dev, fb, NULL);
   vkDestroyRenderPass(x->dev, rp, NULL);
   vkDestroyImageView(x->dev, view, NULL);
   vkFreeMemory(x->dev, imem, NULL);
   vkDestroyImage(x->dev, img, NULL);

   printf("%s: %u/%u pixels out of order (worst channel error %.4f, want %.4f %.4f %.4f)\n",
          name, bad, WIDTH * HEIGHT, worst, want[0], want[1], want[2]);
   return bad == 0;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "blendorder.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "blendorder.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-blendorder",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
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
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   if (vkCreateDevice(pd, &dci, NULL, &x.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(x.dev, x.qf, 0, &x.queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &x.mp);

   x.vs = load_module(x.dev, vs_path);
   x.fs = load_module(x.dev, fs_path);
   if (!x.vs || !x.fs) return 1;

   bool ff = run_pass(&x, VK_FORMAT_R8G8B8A8_UNORM, "rgba8 (fixed-function merger)");
   bool sw = run_pass(&x, VK_FORMAT_R16G16B16A16_SFLOAT, "rgba16f (software merger)");

   vkDestroyShaderModule(x.dev, x.vs, NULL);
   vkDestroyShaderModule(x.dev, x.fs, NULL);
   vkDestroyDevice(x.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!ff || !sw) {
      printf("FAILED (blendorder: fixed-function %s, software %s)\n",
             ff ? "ok" : "out of order", sw ? "ok" : "out of order");
      return 1;
   }
   printf("PASSED (blendorder: %u layers merged in draw order on both mergers)\n", LAYERS);
   return 0;
}
