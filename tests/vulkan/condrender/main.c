/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Draws inside VK_EXT_conditional_rendering blocks, with the predicate written
 * by the host and by an occlusion query.
 *
 * The predicate lives in a buffer and llvmpipe tests it at the top of its draw,
 * returning before rendering anything when it fails. A draw that takes the
 * device path never reaches that test, so the device path has to test it
 * itself or the geometry is drawn regardless -- with no error and no fallback,
 * because a render condition is context state that never passes through the
 * shader translator that refuses what it cannot do.
 *
 * Vulkan has no query-predicated render condition: the extension only reads a
 * buffer. The way an application predicates a draw on visibility is to copy an
 * occlusion query's result into that buffer with vkCmdCopyQueryPoolResults, so
 * the query cases do exactly that, and the predicate then depends on the
 * device's occlusion count reaching the copy before the conditional draw reads
 * it. Their predicate buffer starts out holding the value that gives the wrong
 * answer, so a copy that never landed is caught.
 *
 *   host 0 / host 1              predicate written by the host
 *   query visible / hidden       predicate copied from an occlusion query over
 *                                a full-screen draw / the same draw moved
 *                                wholly outside the viewport
 *
 * Each runs normal and inverted. A false predicate must leave the target at
 * the clear and a true one must render it, so a driver that ignores the
 * condition, one that drops every conditional draw and one that ignores the
 * inversion are each told apart from a correct one.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define NPX       ((long)WIDTH * (long)HEIGHT)
#define FORMAT    VK_FORMAT_R8G8B8A8_UNORM

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return -1;                                                   \
   }                                                               \
} while (0)

enum pred_source { PRED_HOST, PRED_QUERY };

struct cond_case {
   const char      *name;
   enum pred_source source;
   uint32_t         host_value;   /* PRED_HOST: the predicate word */
   bool             visible;      /* PRED_QUERY: whether the queried draw is */
   bool             inverted;
   long             want_px;
};

static const struct cond_case cases[] = {
   { "host 0",                  PRED_HOST,  0u, false, false, 0   },
   { "host 1",                  PRED_HOST,  1u, false, false, NPX },
   { "host 0 inverted",         PRED_HOST,  0u, false, true,  NPX },
   { "host 1 inverted",         PRED_HOST,  1u, false, true,  0   },
   { "query visible",           PRED_QUERY, 0u, true,  false, NPX },
   { "query hidden",            PRED_QUERY, 0u, false, false, 0   },
   { "query visible inverted",  PRED_QUERY, 0u, true,  true,  0   },
   { "query hidden inverted",   PRED_QUERY, 0u, false, true,  NPX },
};
#define NCASES (sizeof cases / sizeof cases[0])

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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
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

static int
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer *out_buf,
            VkDeviceMemory *out_mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, out_buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no host memory\n");
      return -1;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, out_mem));
   CHECK(vkBindBufferMemory(dev, *out_buf, *out_mem, 0));
   return 0;
}

/* Run one case: for a query case, an occlusion query over the triangle in a
 * first render pass, copied into the predicate word; then the triangle inside a
 * conditional-rendering block in a second, freshly cleared one. Reports how
 * many pixels the conditional draw covered and the predicate word it read.
 * Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, const struct cond_case *cc,
       long *out_covered, uint32_t *out_predicate)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   uint32_t imt = find_mem(mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no image memory\n");
      return -1;
   }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   /* Both passes clear, so the readback holds the conditional draw alone. */
   VkAttachmentDescription att = {
      .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0,
      .size = 2 * sizeof(float),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   /* No vertex input: condrender.vert builds its positions from
    * gl_VertexIndex. */
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
      .blendEnable = VK_FALSE,
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
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

   const VkDeviceSize bytes = (VkDeviceSize)NPX * 4;
   VkBuffer rb;
   VkDeviceMemory bmem;
   if (host_buffer(dev, mp, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   &rb, &bmem) < 0) {
      return -1;
   }

   /* The predicate word the block is conditioned on. A zero means the work
    * inside is discarded, anything else that it executes; the inverted bit
    * swaps the two. A query case starts it at the value that would give the
    * wrong answer, so only the copied count can make the case pass. */
   VkBuffer pbuf;
   VkDeviceMemory pmem;
   if (host_buffer(dev, mp, sizeof(uint32_t),
                   VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, &pbuf, &pmem) < 0) {
      return -1;
   }
   uint32_t *pval;
   CHECK(vkMapMemory(dev, pmem, 0, sizeof(uint32_t), 0, (void **)&pval));
   *pval = cc->source == PRED_HOST ? cc->host_value
                                   : (cc->visible ? 0u : 1u);
   vkUnmapMemory(dev, pmem);

   VkQueryPoolCreateInfo qpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = 1,
   };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));

   PFN_vkCmdBeginConditionalRenderingEXT p_begin =
      (PFN_vkCmdBeginConditionalRenderingEXT)
      vkGetDeviceProcAddr(dev, "vkCmdBeginConditionalRenderingEXT");
   PFN_vkCmdEndConditionalRenderingEXT p_end =
      (PFN_vkCmdEndConditionalRenderingEXT)
      vkGetDeviceProcAddr(dev, "vkCmdEndConditionalRenderingEXT");
   if (!p_begin || !p_end) {
      fprintf(stderr, "FAILED: conditional rendering entry points missing\n");
      return -1;
   }

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
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   const float inside[2] = { 0.0f, 0.0f };
   /* x in [3, 7] clip space: every vertex is beyond the right plane. */
   const float outside[2] = { 4.0f, 0.0f };

   if (cc->source == PRED_QUERY) {
      vkCmdResetQueryPool(cmd, qp, 0, 1);
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0,
                         sizeof inside, cc->visible ? inside : outside);
      vkCmdBeginQuery(cmd, qp, 0, 0);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndQuery(cmd, qp, 0);
      vkCmdEndRenderPass(cmd);
      /* A 32-bit result is the predicate word itself: non-zero exactly when
       * some sample passed. */
      vkCmdCopyQueryPoolResults(cmd, qp, 0, 1, pbuf, 0, sizeof(uint32_t),
                                VK_QUERY_RESULT_WAIT_BIT);
      VkMemoryBarrier to_pred = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT, 0,
                           1, &to_pred, 0, NULL, 0, NULL);
   }

   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inside,
                      inside);
   VkConditionalRenderingBeginInfoEXT crbi = {
      .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
      .buffer = pbuf, .offset = 0,
      .flags = cc->inverted ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0,
   };
   p_begin(cmd, &crbi);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   p_end(cmd);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   VkMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, NULL,
                        0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   CHECK(vkMapMemory(dev, pmem, 0, sizeof(uint32_t), 0, (void **)&pval));
   *out_predicate = *pval;
   vkUnmapMemory(dev, pmem);

   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
   *out_covered = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * 4;
         if (p[0] || p[1] || p[2]) {
            (*out_covered)++;
         }
      }
   }
   vkUnmapMemory(dev, bmem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyQueryPool(dev, qp, NULL);
   vkFreeMemory(dev, pmem, NULL);
   vkDestroyBuffer(dev, pbuf, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "condrender.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "condrender.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-condrender",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
      return 1;
   }

   uint32_t npd = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &npd, &pd) != VK_SUCCESS || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceConditionalRenderingFeaturesEXT crf_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &crf_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   if (!crf_have.conditionalRendering) {
      printf("FAILED (condrender: conditionalRendering not advertised; "
             "lavapipe always exposes it)\n");
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
   /* Conditional rendering is an extension and an opt-in feature. */
   VkPhysicalDeviceConditionalRenderingFeaturesEXT crf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
      .conditionalRendering = VK_TRUE,
   };
   const char *dev_exts[] = { VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &crf,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = dev_exts,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: cannot create device\n");
      return 1;
   }
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      return 1;
   }

   unsigned bad = 0;
   for (unsigned c = 0; c < NCASES; c++) {
      const struct cond_case *cc = &cases[c];
      long covered = -1;
      uint32_t predicate = 0;
      if (render(dev, queue, qf, &mp, vs, fs, cc, &covered, &predicate) < 0) {
         return 1;
      }
      printf("%-24s predicate=%u drew %ld px\n", cc->name, predicate, covered);
      /* The copied word is checked on its own, so a wrong count is reported
       * as the query's fault rather than the condition's. */
      if (cc->source == PRED_QUERY && (predicate != 0) != cc->visible) {
         printf("  %s: occlusion query copied %u into the predicate, want %s\n",
                cc->name, predicate, cc->visible ? "non-zero" : "0");
         bad++;
      }
      if (covered != cc->want_px) {
         printf("  %s: %ld px drawn, want %ld\n", cc->name, covered,
                cc->want_px);
         bad++;
      }
   }

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad) {
      printf("FAILED (condrender: %u checks wrong; a false predicate that "
             "still drew means the condition was never tested, which is what "
             "happens when the draw bypasses the path that tests it)\n", bad);
      return 1;
   }
   printf("PASSED (condrender: %u cases -- host and query-copied predicates, "
          "normal and inverted -- each drew all %ld px or nothing as "
          "predicated)\n", (unsigned)NCASES, NPX);
   return 0;
}
