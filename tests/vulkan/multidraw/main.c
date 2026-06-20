/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Multi-draw depth-persistence test for the vortexpipe driver.
 *
 * Renders two overlapping triangles as TWO SEPARATE draw calls in one render
 * pass, each with its own pipeline (and self-contained vertex shader): the near
 * (blue, z=0.25) triangle first, then the far (red, z=0.75) triangle. With the
 * depth test (LESS) and a framebuffer whose depth attachment PERSISTS across
 * draws, the far triangle's fragments lose to the near triangle's already-
 * written depth — so the centre stays blue.
 *
 * This is the cross-draw-residency check the single-draw `depth` test cannot
 * make: a driver that re-clears (or otherwise loses) the depth buffer between
 * draws lets the second, farther triangle overwrite the first and the centre
 * comes out red. The two draws share one renderpass clear (depth=1.0), so the
 * depth value written by the first draw is the only thing that can reject the
 * second. Two pipelines (not gl_VertexIndex/firstVertex) select the geometry,
 * so the test does not depend on the driver honouring a vertex-index base.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe: vertex shading +
 * hardware rasterization + fragment shading + the OM depth test all run on the
 * Vortex device, and the depth attachment must remain device-resident between
 * the two draws.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH        64u
#define HEIGHT       64u
#define FORMAT       VK_FORMAT_R8G8B8A8_UNORM
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT

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

/* allocate + bind device-local memory for an image */
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

int
main(int argc, char **argv)
{
   const char *near_vs_path = (argc > 1) ? argv[1] : "near.vert.spv";
   const char *far_vs_path  = (argc > 2) ? argv[2] : "far.vert.spv";
   const char *fs_path      = (argc > 3) ? argv[3] : "depth.frag.spv";

   /* --- instance + device ----------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-multidraw", .apiVersion = VK_API_VERSION_1_1,
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

   /* --- colour + depth attachments -------------------------------- */
   VkImageCreateInfo cimci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage cimg;
   CHECK(vkCreateImage(dev, &cimci, NULL, &cimg));
   VkDeviceMemory cmem;
   CHECK(alloc_image(dev, &mp, cimg, &cmem));

   VkImageCreateInfo dimci = cimci;
   dimci.format = DEPTH_FORMAT;
   dimci.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage dimg;
   CHECK(vkCreateImage(dev, &dimci, NULL, &dimg));
   VkDeviceMemory dmem;
   CHECK(alloc_image(dev, &mp, dimg, &dmem));

   VkImageViewCreateInfo civci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = cimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView cview;
   CHECK(vkCreateImageView(dev, &civci, NULL, &cview));

   VkImageViewCreateInfo divci = civci;
   divci.image  = dimg;
   divci.format = DEPTH_FORMAT;
   divci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkImageView dview;
   CHECK(vkCreateImageView(dev, &divci, NULL, &dview));

   /* --- render pass (colour + depth) ------------------------------ */
   VkAttachmentDescription att[2] = {
      { .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = DEPTH_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference cref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &dref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fbviews[2] = { cview, dview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = fbviews,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* --- two pipelines (near / far), shared FS + depth test -------- */
   VkShaderModule near_vs = load_module(dev, near_vs_path);
   VkShaderModule far_vs  = load_module(dev, far_vs_path);
   VkShaderModule fs      = load_module(dev, fs_path);
   if (!near_vs || !far_vs || !fs) return 1;

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

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
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkPipelineShaderStageCreateInfo near_stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = near_vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkPipelineShaderStageCreateInfo far_stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = far_vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = pl, .renderPass = rp, .subpass = 0,
   };
   VkPipeline near_pipe, far_pipe;
   gpci.pStages = near_stages;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &near_pipe));
   gpci.pStages = far_stages;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &far_pipe));

   /* --- host-visible readback buffer ------------------------------ */
   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   uint32_t bmt = find_mem(&mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, rb, bmem, 0));

   /* --- command buffer -------------------------------------------- */
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

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   VkClearValue clears[2] = {
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
      { .depthStencil = { 1.0f, 0 } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 2, .pClearValues = clears,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   /* Two SEPARATE draws sharing one renderpass clear: near (blue) first, then
    * far (red). The depth value the near draw writes is the only thing that can
    * reject the far draw — far wins the centre iff the depth attachment was
    * lost between draws. */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, near_pipe);
   vkCmdDraw(cmd, 3, 1, 0, 0);   /* near (blue) */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, far_pipe);
   vkCmdDraw(cmd, 3, 1, 0, 0);   /* far  (red)  */
   vkCmdEndRenderPass(cmd);

   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify: the centre must remain the near (blue) triangle --- */
   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
   const uint8_t *centre = px + ((size_t)(HEIGHT / 2) * WIDTH + WIDTH / 2) * 4;
   uint8_t pr = centre[0], pg = centre[1], pb = centre[2];
   /* near (blue) drawn first must survive the later far (red) draw: the depth
    * attachment persisted across the two draws iff blue beats red. */
   bool depth_ok = (pb > pr) && (pb > 80);
   vkUnmapMemory(dev, bmem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, far_pipe, NULL);
   vkDestroyPipeline(dev, near_pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, far_vs, NULL);
   vkDestroyShaderModule(dev, near_vs, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImageView(dev, cview, NULL);
   vkFreeMemory(dev, dmem, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkDestroyImage(dev, cimg, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!depth_ok) {
      printf("FAILED (centre RGB = %u,%u,%u -- expected blue; the far triangle "
             "from draw 1 overwrote the near one from draw 0 -> the depth "
             "attachment was not preserved across draws)\n", pr, pg, pb);
      return 1;
   }
   printf("PASSED (multi-draw depth persistence: near triangle survived the "
          "later far draw, centre RGB = %u,%u,%u)\n", pr, pg, pb);
   return 0;
}
