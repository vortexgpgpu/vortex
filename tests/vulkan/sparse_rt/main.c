/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Rendering into sparse attachments.
 *
 * Two render passes, each into a fully bound sparse RGBA8 colour attachment and
 * a sparse D32F depth attachment: 32 x 32, inside one 64 KiB tile, whose rows
 * linear addressing reaches; and 288 x 40, three tiles across, which it does
 * not. Each pass clears both attachments, draws its whole target at depth 0.5,
 * then the left half at 0.7, which the LESS test rejects, then the right half
 * at 0.3, which it keeps. A colour clear of a rectangle then follows; in the
 * wide pass it starts eight texels short of the first tile edge and spans it.
 * Every pixel's colour names the pixel and the pass that drew it, or the clear. Both attachments are then copied out and checked texel by
 * texel, and the colour attachment's memory is checked through a host mapping,
 * each texel at its place in its tile; lavapipe stores a tile's texels x, then
 * y. The wide pass is checked on vortexpipe only: llvmpipe's rasterizer writes
 * a sparse attachment linearly.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE 65536u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

struct ctx {
   VkPhysicalDevice pd;
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice        dev;
   VkQueue         queue;
   VkCommandBuffer cb;
   VkPipelineLayout pl;
   VkShaderModule  vs, fs;
};

struct params {
   uint32_t pass;
   float    depth;
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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
}

static bool
make_module(struct ctx *c, const char *path, VkShaderModule *sm)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return false;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkResult r = vkCreateShaderModule(c->dev, &smci, NULL, sm);
   free(spv);
   CHECK(r);
   return true;
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

static bool
alloc_mem(struct ctx *c, VkDeviceSize bytes, uint32_t bits, VkDeviceMemory *mem)
{
   uint32_t mt = find_mem(&c->mp, bits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return false; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bytes, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   return true;
}

/* A sparse image bound whole, at offset 0 of its own memory. */
static bool
make_sparse(struct ctx *c, VkFormat fmt, VkImageUsageFlags usage, bool residency,
            uint32_t w, uint32_t h, VkImage *img, VkDeviceMemory *mem, uint8_t **map)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
               (residency ? VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT : 0),
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = { w, h, 1 },
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, *img, &mr);
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, mem)) {
      return false;
   }
   CHECK(vkMapMemory(c->dev, *mem, 0, VK_WHOLE_SIZE, 0, (void **)map));
   memset(*map, 0x5c, mr.size);
   VkSparseMemoryBind b = { .resourceOffset = 0, .size = mr.size, .memory = *mem };
   VkSparseImageOpaqueMemoryBindInfo obi = { .image = *img, .bindCount = 1, .pBinds = &b };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageOpaqueBindCount = 1, .pImageOpaqueBinds = &obi,
   };
   CHECK(vkQueueBindSparse(c->queue, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

static bool
make_view(struct ctx *c, VkImage img, VkFormat fmt, VkImageAspectFlags aspect,
          VkImageView *view)
{
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { aspect, 0, 1, 0, 1 },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, view));
   return true;
}

static uint32_t
want_color(uint32_t x, uint32_t y, uint32_t w)
{
   const uint32_t pass = x >= w / 2 ? 2u : 0u;
   return (x & 255u) | ((y & 255u) << 8) | (((x >> 8) + (y >> 8) * 8u + pass * 64u) << 16) |
          0xff000000u;
}

/* The rectangle the colour clear covers, and its value as stored. */
struct rect { uint32_t x, y, w, h; };
#define CLEAR_RGBA8 0xffc08040u

static struct rect
clear_rect(uint32_t w, uint32_t h)
{
   return w > 128 ? (struct rect){ 120, 4, 50, h / 2 } : (struct rect){ 8, 4, 12, h / 2 };
}

static bool
run_pass(struct ctx *c, uint32_t w, uint32_t h)
{
   VkImage cimg, zimg;
   VkDeviceMemory cmem, zmem;
   uint8_t *cmap, *zmap;
   VkImageView cview, zview;
   if (!make_sparse(c, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true,
                    w, h, &cimg, &cmem, &cmap) ||
       !make_sparse(c, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false, w, h, &zimg, &zmem, &zmap) ||
       !make_view(c, cimg, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &cview) ||
       !make_view(c, zimg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, &zview)) {
      return false;
   }

   VkAttachmentDescription ad[2] = {
      { .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = VK_FORMAT_D32_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference car = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference zar = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sd = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &car, .pDepthStencilAttachment = &zar,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 2,
      .pAttachments = ad, .subpassCount = 1, .pSubpasses = &sd,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &rp));
   VkImageView views[2] = { cview, zview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 2, .pAttachments = views, .width = w, .height = h, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &fb));

   VkPipelineShaderStageCreateInfo st[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = c->vs, .pName = "main" },
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
   VkViewport vp = { 0, 0, (float)w, (float)h, 0, 1 };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkDynamicState dyn = VK_DYNAMIC_STATE_SCISSOR;
   VkPipelineDynamicStateCreateInfo dsi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1, .pDynamicStates = &dyn,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
      .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pDepthStencilState = &ds, .pColorBlendState = &cbs, .pDynamicState = &dsi,
      .layout = c->pl, .renderPass = rp,
   };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 8u,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, &rb));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, rb, &mr);
   VkDeviceMemory rbm;
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, &rbm)) {
      return false;
   }
   CHECK(vkBindBufferMemory(c->dev, rb, rbm, 0));
   uint32_t *rbp;
   CHECK(vkMapMemory(c->dev, rbm, 0, VK_WHOLE_SIZE, 0, (void **)&rbp));

   CHECK(vkResetCommandBuffer(c->cb, 0));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(c->cb, &cbbi));
   VkClearValue cv[2] = { { .color = { .float32 = { 0, 0, 0, 0 } } },
                          { .depthStencil = { 1.0f, 0 } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
      .framebuffer = fb, .renderArea = { { 0, 0 }, { w, h } },
      .clearValueCount = 2, .pClearValues = cv,
   };
   vkCmdBeginRenderPass(c->cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   const struct { uint32_t pass; float depth; int32_t x; uint32_t sw; } draws[3] = {
      { 0, 0.5f, 0, w }, { 1, 0.7f, 0, w / 2 }, { 2, 0.3f, (int32_t)(w / 2), w / 2 },
   };
   for (unsigned i = 0; i < 3; i++) {
      const struct params p = { draws[i].pass, draws[i].depth };
      VkRect2D sc = { { draws[i].x, 0 }, { draws[i].sw, h } };
      vkCmdSetScissor(c->cb, 0, 1, &sc);
      vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof p, &p);
      vkCmdDraw(c->cb, 3, 1, 0, 0);
   }
   const struct rect cr = clear_rect(w, h);
   const VkClearAttachment ca = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
      .clearValue = { .color = { .float32 = { 64 / 255.0f, 128 / 255.0f, 192 / 255.0f, 1 } } },
   };
   const VkClearRect crr = { { { (int32_t)cr.x, (int32_t)cr.y }, { cr.w, cr.h } }, 0, 1 };
   vkCmdClearAttachments(c->cb, 1, &ca, 1, &crr);
   vkCmdEndRenderPass(c->cb);
   VkBufferImageCopy r[2] = {
      { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } },
      { .bufferOffset = (VkDeviceSize)w * h * 4u,
        .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } },
   };
   vkCmdCopyImageToBuffer(c->cb, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &r[0]);
   vkCmdCopyImageToBuffer(c->cb, zimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &r[1]);
   CHECK(vkEndCommandBuffer(c->cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &c->cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   const uint32_t gw = 128, gh = 128, tx = (w + gw - 1) / gw;
   unsigned bad = 0;
   char first[160] = "";
   const float *zp = (const float *)(rbp + w * h);
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         const bool in_clear = x >= cr.x && x < cr.x + cr.w && y >= cr.y && y < cr.y + cr.h;
         const uint32_t want = in_clear ? CLEAR_RGBA8 : want_color(x, y, w);
         const float wz = x >= w / 2 ? 0.3f : 0.5f;
         uint32_t inmem;
         memcpy(&inmem, cmap + ((x / gw) + (y / gh) * tx) * TILE +
                        ((x % gw) + (y % gh) * gw) * 4u, 4);
         if (rbp[y * w + x] != want || inmem != want || zp[y * w + x] != wz) {
            if (!bad++) {
               snprintf(first, sizeof first, "(%u,%u): copied 0x%08x, in its tile 0x%08x, "
                        "want 0x%08x; depth %g, want %g", x, y, rbp[y * w + x], inmem, want,
                        zp[y * w + x], wz);
            }
         }
      }
   }

   vkUnmapMemory(c->dev, rbm);
   vkDestroyBuffer(c->dev, rb, NULL);
   vkFreeMemory(c->dev, rbm, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyFramebuffer(c->dev, fb, NULL);
   vkDestroyRenderPass(c->dev, rp, NULL);
   vkDestroyImageView(c->dev, cview, NULL);
   vkDestroyImageView(c->dev, zview, NULL);
   vkDestroyImage(c->dev, cimg, NULL);
   vkDestroyImage(c->dev, zimg, NULL);
   vkFreeMemory(c->dev, cmem, NULL);
   vkFreeMemory(c->dev, zmem, NULL);
   if (bad) {
      printf("FAILED (%ux%u: %u pixels wrong; first %s)\n", w, h, bad, first);
      return false;
   }
   printf("%ux%u: %u pixels right, colour in its tiles and depth, a %ux%u clear at "
          "(%u,%u) included\n", w, h, w * h, cr.w, cr.h, cr.x, cr.y);
   return true;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "FAILED: usage: %s sr.vert.spv sr.frag.spv\n", argv[0]);
      return 1;
   }
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_rt", .apiVersion = VK_API_VERSION_1_1,
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
   struct ctx c;
   memset(&c, 0, sizeof c);
   if (vkEnumeratePhysicalDevices(inst, &npd, &c.pd) < 0 || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(c.pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(c.pd, &have);
   if (!have.sparseBinding || !have.sparseResidencyImage2D) {
      printf("NotSupported: sparse binding or 2D sparse residency is not offered\n");
      return 0;
   }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs graphics and sparse binding\n");
      return 0;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures want = {
      .sparseBinding = VK_TRUE, .sparseResidencyImage2D = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &want,
   };
   if (vkCreateDevice(c.pd, &dci, NULL, &c.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);
   VkPushConstantRange pcr = { VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(struct params) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   if (vkCreatePipelineLayout(c.dev, &plci, NULL, &c.pl) != VK_SUCCESS ||
       vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: layout or command pool\n");
      return 1;
   }
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   if (vkAllocateCommandBuffers(c.dev, &cbai, &c.cb) != VK_SUCCESS ||
       !make_module(&c, argv[1], &c.vs) || !make_module(&c, argv[2], &c.fs)) {
      fprintf(stderr, "FAILED: command buffer or shaders\n");
      return 1;
   }
   const bool vortex = strncmp(props.deviceName, "vortexpipe", 10) == 0;
   const unsigned failed = !run_pass(&c, 32, 32) + (vortex && !run_pass(&c, 288, 40));
   if (failed) {
      printf("FAILED (sparse_rt: %u passes)\n", failed);
      return 1;
   }
   printf("PASSED (sparse_rt: rendering into sparse colour and depth attachments one tile "
           "and three tiles wide, each texel in its own tile)\n");
   return 0;
}
