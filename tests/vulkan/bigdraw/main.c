/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Draws with more primitives than one rasterizer pass numbers.
 *
 * The rasterizer carries a primitive's index in VX_RASTER_PID_BITS (16) bits,
 * so a draw that keeps more than 65,536 primitives must be split into passes,
 * and the passes must reach the target in draw order. Every primitive here is a
 * small triangle over exactly one pixel centre of a 256x256 target, in a colour
 * that names it; the first 65,536 cover every pixel once and the rest cover the
 * first pixels again, so the image is wrong wherever a primitive went missing,
 * aliased another, or was merged out of order.
 *
 *   vs     70,000 triangles from the vertex stage alone
 *   gs     9,000 points, each made into eight triangles by a geometry shader
 *          whose count is static: 72,000 primitives
 *   gsdyn  the same shader with each point's count (i * 7) % 9 decided at run
 *          time, so the primitives are ordered on the device and how many
 *          passes they take is known only once the shader has run
 *
 * Usage: bigdraw [-c vs|gs|gsdyn]. BIGDRAW_BITE=1 expects the draw to end at
 * 32,768 primitives, one front-end pass, which every case must then fail.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 256u
#define H 256u
#define NTRI 70000u
#define NPOINT 9000u

#define CHECK(x) do {                                                     \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      printf("FAILED (bigdraw: %s -> VkResult %d)\n", #x, (int)_r);        \
      exit(1);                                                            \
   }                                                                      \
} while (0)

static struct {
   VkDevice dev;
   VkQueue queue;
   VkPhysicalDeviceMemoryProperties mp;
   VkCommandPool pool;
   VkImage img;
   VkRenderPass rp;
   VkFramebuffer fb;
   VkBuffer rb;
   void *rb_map;
   VkPipelineLayout layout;
} vk;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk.mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (vk.mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   printf("FAILED (bigdraw: no memory type)\n");
   exit(1);
}

static VkShaderModule
module(const char *name)
{
   FILE *f = fopen(name, "rb");
   if (!f) {
      printf("FAILED (bigdraw: cannot open %s)\n", name);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc((size_t)sz);
   if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
      printf("FAILED (bigdraw: cannot read %s)\n", name);
      exit(1);
   }
   fclose(f);
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = (size_t)sz, .pCode = code };
   VkShaderModule m;
   CHECK(vkCreateShaderModule(vk.dev, &ci, NULL, &m));
   free(code);
   return m;
}

static void
init(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice phys;
   vkEnumeratePhysicalDevices(inst, &n, &phys);
   if (!n) {
      printf("FAILED (bigdraw: no device)\n");
      exit(1);
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(phys, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(phys, &vk.mp);
   uint32_t nq = 8, qf;
   VkQueueFamilyProperties qp[8];
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
   for (qf = 0; qf < nq && !(qp[qf].queueFlags & VK_QUEUE_GRAPHICS_BIT); qf++)
      ;
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = qf, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   VkPhysicalDeviceFeatures feat = { .geometryShader = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .pEnabledFeatures = &feat };
   CHECK(vkCreateDevice(phys, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, qf, 0, &vk.queue);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .queueFamilyIndex = qf };
   CHECK(vkCreateCommandPool(vk.dev, &pci, NULL, &vk.pool));

   VkImageCreateInfo imci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
   CHECK(vkCreateImage(vk.dev, &imci, NULL, &vk.img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk.dev, vk.img, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits, 0) };
   VkDeviceMemory img_mem;
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &img_mem));
   CHECK(vkBindImageMemory(vk.dev, vk.img, img_mem, 0));
   VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = vk.img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                 .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, &view));
   VkAttachmentDescription att = { .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                   .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                   .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                   .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                .colorAttachmentCount = 1, .pColorAttachments = &ref };
   VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 1, .pAttachments = &att,
                                   .subpassCount = 1, .pSubpasses = &sub };
   CHECK(vkCreateRenderPass(vk.dev, &rpci, NULL, &vk.rp));
   VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                   .renderPass = vk.rp, .attachmentCount = 1,
                                   .pAttachments = &view, .width = W, .height = H,
                                   .layers = 1 };
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &vk.fb));

   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = W * H * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   CHECK(vkCreateBuffer(vk.dev, &bci, NULL, &vk.rb));
   vkGetBufferMemoryRequirements(vk.dev, vk.rb, &mr);
   ai.allocationSize = mr.size;
   ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory rb_mem;
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &rb_mem));
   CHECK(vkBindBufferMemory(vk.dev, vk.rb, rb_mem, 0));
   CHECK(vkMapMemory(vk.dev, rb_mem, 0, W * H * 4, 0, &vk.rb_map));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 4 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .pushConstantRangeCount = 1,
                                       .pPushConstantRanges = &pcr };
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &vk.layout));
}

/* Draw `count` vertices -- triangles, or points through big.geom -- and read
 * the target back. */
static void
render(bool geom, bool dynamic, uint32_t count, uint32_t *img)
{
   VkShaderModule vs = module("big.vert.spv"), fs = module("big.frag.spv");
   VkShaderModule gs = geom ? module("big.geom.spv") : VK_NULL_HANDLE;
   const VkBool32 dyn = dynamic;
   VkSpecializationMapEntry me = { 0, 0, sizeof dyn };
   VkSpecializationInfo spec = { 1, &me, sizeof dyn, &dyn };
   VkPipelineShaderStageCreateInfo st[3] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_GEOMETRY_BIT, .module = gs, .pName = "main",
        .pSpecializationInfo = &spec },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = geom ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkViewport vp = { 0, 0, W, H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkGraphicsPipelineCreateInfo gp = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = geom ? 3u : 2u, .pStages = st, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = vk.layout,
      .renderPass = vk.rp };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));

   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = vk.pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkClearValue clear = { .color = { .float32 = { 0, 0, 0, 0 } } };
   VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                 .renderPass = vk.rp, .framebuffer = vk.fb,
                                 .renderArea = { { 0, 0 }, { W, H } },
                                 .clearValueCount = 1, .pClearValues = &clear };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   const uint32_t points = geom;
   vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &points);
   vkCmdDraw(cmd, count, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, vk.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.rb, 1, &reg);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   memcpy(img, vk.rb_map, W * H * 4);
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyShaderModule(vk.dev, vs, NULL);
   vkDestroyShaderModule(vk.dev, fs, NULL);
   if (gs)
      vkDestroyShaderModule(vk.dev, gs, NULL);
}

static uint32_t
rgba(unsigned r, unsigned g, unsigned b, unsigned a)
{
   return r | (g << 8) | (b << 16) | (a << 24);
}

static int
check(const char *name, const uint32_t *got, const uint32_t *want)
{
   unsigned bad = 0;
   for (unsigned k = 0; k < W * H; k++)
      if (got[k] != want[k] && bad++ < 4)
         printf("  %s: pixel (%u,%u) = %08x, expected %08x\n", name, k % W, k / W, got[k],
                want[k]);
   printf("%s: %s (%u pixels differ)\n", name, bad ? "FAILED" : "passed", bad);
   return bad != 0;
}

int
main(int argc, char **argv)
{
   const char *only = NULL;
   for (int a = 1; a + 1 < argc; a++)
      if (!strcmp(argv[a], "-c"))
         only = argv[a + 1];
   const bool bite = getenv("BIGDRAW_BITE") && atoi(getenv("BIGDRAW_BITE"));
   /* The draw as a wrong implementation would end it: after one pass. */
   const uint32_t limit = bite ? 32768u : UINT32_MAX;
   init();
   static uint32_t got[W * H], want[W * H];
   int failures = 0;

   if (!only || !strcmp(only, "vs")) {
      memset(want, 0, sizeof want);
      for (uint32_t t = 0; t < NTRI && t < limit; t++)
         want[t % 65536u] = rgba(t & 255u, (t >> 8) & 255u, t >> 16, 255);
      render(false, false, 3 * NTRI, got);
      failures += check("vs", got, want);
   }
   for (int dyn = 0; dyn < 2; dyn++) {
      if (only && strcmp(only, dyn ? "gsdyn" : "gs"))
         continue;
      memset(want, 0, sizeof want);
      uint32_t u = 0;
      for (uint32_t i = 0; i < NPOINT; i++) {
         const uint32_t n = dyn ? (i * 7u) % 9u : 8u;
         for (uint32_t j = 0; j < n && u < limit; j++, u++)
            want[(8u * i + j) % 65536u] = rgba(i & 255u, (i >> 8) & 255u, j * 32u, 255);
      }
      render(true, dyn, NPOINT, got);
      failures += check(dyn ? "gsdyn" : "gs", got, want);
   }

   if (failures) {
      printf("FAILED (bigdraw: %d case%s)\n", failures, failures == 1 ? "" : "s");
      return 1;
   }
   printf("PASSED\n");
   return 0;
}
