/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Clipping and depth-range test for the vortexpipe driver.
 *
 * Each pass draws one full-screen quad into a 16x16 RGBA8 target with a D24S8
 * depth attachment and checks every pixel against the coverage it must have:
 *   clip distance   gl_ClipDistance = x drops the left half;
 *   cull distance   a distance negative at all three vertices drops the quad's
 *                   second triangle, and negative at one keeps the first;
 *   depth clip      depth running 0.5..1.5 across the quad drops the part past
 *                   the viewport's far depth;
 *   depth clamp     the same quad with depth clamp and a 0..0.5 depth range
 *                   keeps every pixel, clamped to a depth that passes
 *                   LESS_OR_EQUAL against a 0.5 clear (saturating to the
 *                   attachment's [0, 1] alone would not), and GREATER_OR_EQUAL
 *                   passes exactly the half clamped to the upper bound;
 *   near clamp      depth running -0.5..0.5 with clamp keeps the half in front
 *                   of the near plane, at 0, which passes LESS_OR_EQUAL against
 *                   a 0 clear (clipping would drop it);
 *   depth bias      a quad at 0.5 against a 0.5 clear passes LESS only when a
 *                   negative constant bias moves it in front;
 *   msaa scissor    at 4x, which the software rasterizer draws, a scissor rect
 *                   away from the origin bounds the quad on all four sides.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 16u

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

struct pc {
   float z[2];
   int32_t mode;
};

/* One pass: the pipeline state it varies, what it draws, and the coverage it
 * expects at pixel (x, y). */
struct pass {
   const char *name;
   bool depth_clamp;
   float depth_bias;
   VkCompareOp compare;     /* VK_COMPARE_OP_ALWAYS = depth test off */
   float depth_clear;
   float max_depth;         /* the viewport's maxDepth */
   uint32_t samples;        /* 1, or 4 with a resolve attachment */
   VkRect2D scissor;
   struct pc pc;
   bool (*covered)(uint32_t x, uint32_t y);
   bool (*checked)(uint32_t x, uint32_t y);
};

static bool all_px(uint32_t x, uint32_t y)        { (void)x; (void)y; return true; }
static bool no_px(uint32_t x, uint32_t y)         { (void)x; (void)y; return false; }
static bool right_half(uint32_t x, uint32_t y)    { (void)y; return x >= SIZE / 2; }
static bool left_half(uint32_t x, uint32_t y)     { (void)y; return x < SIZE / 2; }
/* The first triangle covers x + y < 0 in NDC; its centres on the shared edge
 * belong to either triangle by the tie-break rule, so they go unchecked. */
static bool first_tri(uint32_t x, uint32_t y)     { return x + y < SIZE - 1; }
static bool inner_rect(uint32_t x, uint32_t y)    { return x >= 4 && x < 12 && y >= 4 && y < 12; }
static bool off_diagonal(uint32_t x, uint32_t y)  { return x + y != SIZE - 1; }

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
make_image(const struct ctx *x, VkFormat format, VkImageUsageFlags usage,
           VkImageAspectFlags aspect, VkSampleCountFlagBits samples,
           VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { SIZE, SIZE, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(x->dev, &imci, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(x->dev, *img, &mr);
   uint32_t mt = find_mem(&x->mp, mr.memoryTypeBits, 0);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return false; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(x->dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(x->dev, *img, *mem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1, .layerCount = 1 },
   };
   CHECK(vkCreateImageView(x->dev, &ivci, NULL, view));
   return true;
}

static bool
run_pass(const struct ctx *x, const struct pass *p)
{
   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM, zfmt = VK_FORMAT_D24_UNORM_S8_UINT;
   const bool ms = p->samples > 1;
   const VkSampleCountFlagBits sc_count = ms ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
   VkImage cimg, zimg, rimg = VK_NULL_HANDLE;
   VkDeviceMemory cmem, zmem, rmem = VK_NULL_HANDLE;
   VkImageView cview, zview, rview = VK_NULL_HANDLE;
   if (!make_image(x, cfmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, sc_count, &cimg, &cmem, &cview) ||
       !make_image(x, zfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                   VK_IMAGE_ASPECT_DEPTH_BIT, sc_count, &zimg, &zmem, &zview))
      return false;
   /* A multisample pass resolves into a single-sample image, which is read. */
   if (ms && !make_image(x, cfmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT, &rimg, &rmem, &rview))
      return false;

   VkAttachmentDescription att[3] = {
      { .format = cfmt, .samples = sc_count,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = zfmt, .samples = sc_count,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
      { .format = cfmt, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference zref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkAttachmentReference rref = { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pResolveAttachments = ms ? &rref : NULL,
      .pDepthStencilAttachment = &zref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = ms ? 3 : 2, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(x->dev, &rpci, NULL, &rp));

   VkImageView views[3] = { cview, zview, rview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = ms ? 3 : 2, .pAttachments = views,
      .width = SIZE, .height = SIZE, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(x->dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct pc) };
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
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport vp = { 0, 0, (float)SIZE, (float)SIZE, 0.0f, p->max_depth };
   VkRect2D sc = p->scissor;
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = p->depth_clamp,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = p->depth_bias != 0.0f,
      .depthBiasConstantFactor = p->depth_bias,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo mss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = sc_count,
   };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = p->compare != VK_COMPARE_OP_ALWAYS,
      .depthWriteEnable = VK_FALSE, .depthCompareOp = p->compare,
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
      .pMultisampleState = &mss, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = pl, .renderPass = rp,
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
   VkClearValue clear[2] = {
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
      { .depthStencil = { p->depth_clear, 0 } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { SIZE, SIZE } },
      .clearValueCount = 2, .pClearValues = clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(p->pc), &p->pc);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .imageExtent = { SIZE, SIZE, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, ms ? rimg : cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(x->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(x->queue));

   const uint8_t *px;
   CHECK(vkMapMemory(x->dev, bmem, 0, bytes, 0, (void **)&px));
   unsigned bad = 0;
   for (uint32_t y = 0; y < SIZE; y++) {
      for (uint32_t xx = 0; xx < SIZE; xx++) {
         if (p->checked && !p->checked(xx, y))
            continue;
         const bool green = px[(y * SIZE + xx) * 4 + 1] > 128;
         if (green != p->covered(xx, y)) {
            if (bad < 4)
               printf("  %s: pixel (%u,%u) %s\n", p->name, xx, y,
                      green ? "drawn, expected empty" : "empty, expected drawn");
            bad++;
         }
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
   if (ms) {
      vkDestroyImageView(x->dev, rview, NULL);
      vkFreeMemory(x->dev, rmem, NULL);
      vkDestroyImage(x->dev, rimg, NULL);
   }
   vkDestroyImageView(x->dev, zview, NULL);
   vkDestroyImageView(x->dev, cview, NULL);
   vkFreeMemory(x->dev, zmem, NULL);
   vkFreeMemory(x->dev, cmem, NULL);
   vkDestroyImage(x->dev, zimg, NULL);
   vkDestroyImage(x->dev, cimg, NULL);

   printf("%-24s %s (%u wrong pixels)\n", p->name, bad ? "FAIL" : "ok", bad);
   return bad == 0;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "clipdepth.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "clipdepth.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-clipdepth",
      .apiVersion = VK_API_VERSION_1_1,
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
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!have.shaderClipDistance || !have.shaderCullDistance || !have.depthClamp) {
      fprintf(stderr, "FAILED: clip/cull distance or depth clamp not supported\n");
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
   VkPhysicalDeviceFeatures want = {
      .shaderClipDistance = VK_TRUE, .shaderCullDistance = VK_TRUE, .depthClamp = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
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

#define FULL { { 0, 0 }, { SIZE, SIZE } }
   const struct pass passes[] = {
      { "clip distance",  false, 0.0f, VK_COMPARE_OP_ALWAYS, 1.0f, 1.0f, 1, FULL,
        { { 0.5f, 0.5f }, 1 }, right_half, NULL },
      { "cull distance",  false, 0.0f, VK_COMPARE_OP_ALWAYS, 1.0f, 1.0f, 1, FULL,
        { { 0.5f, 0.5f }, 2 }, first_tri, off_diagonal },
      { "depth clip",     false, 0.0f, VK_COMPARE_OP_ALWAYS, 1.0f, 1.0f, 1, FULL,
        { { 0.5f, 1.5f }, 0 }, left_half, NULL },
      { "depth clamp",    true,  0.0f, VK_COMPARE_OP_LESS_OR_EQUAL, 0.5f, 0.5f, 1, FULL,
        { { 0.5f, 1.5f }, 0 }, all_px, NULL },
      { "depth clamp bound", true, 0.0f, VK_COMPARE_OP_GREATER_OR_EQUAL, 0.5f, 0.5f, 1, FULL,
        { { 0.5f, 1.5f }, 0 }, right_half, NULL },
      { "near clamp",     true,  0.0f, VK_COMPARE_OP_LESS_OR_EQUAL, 0.0f, 1.0f, 1, FULL,
        { { -0.5f, 0.5f }, 0 }, left_half, NULL },
      { "depth bias none", false, 0.0f, VK_COMPARE_OP_LESS, 0.5f, 1.0f, 1, FULL,
        { { 0.5f, 0.5f }, 0 }, no_px, NULL },
      { "depth bias -4",  false, -4.0f, VK_COMPARE_OP_LESS, 0.5f, 1.0f, 1, FULL,
        { { 0.5f, 0.5f }, 0 }, all_px, NULL },
      { "msaa scissor",   false, 0.0f, VK_COMPARE_OP_ALWAYS, 1.0f, 1.0f, 4,
        { { 4, 4 }, { 8, 8 } },
        { { 0.5f, 0.5f }, 0 }, inner_rect, NULL },
   };
   unsigned failed = 0;
   for (unsigned i = 0; i < sizeof(passes) / sizeof(passes[0]); i++)
      failed += !run_pass(&x, &passes[i]);

   vkDestroyShaderModule(x.dev, x.vs, NULL);
   vkDestroyShaderModule(x.dev, x.fs, NULL);
   vkDestroyDevice(x.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (clipdepth: %u pass(es) wrong)\n", failed);
      return 1;
   }
   printf("PASSED (clipdepth: clip/cull distances, depth clip/clamp and bias)\n");
   return 0;
}
