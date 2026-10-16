/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A primitives-generated query (VK_EXT_primitives_generated_query) over draws
 * the device runs.
 *
 * The count is llvmpipe's, kept by the draw module as it assembles primitives,
 * so a draw that takes the device path has to produce it itself or the query
 * reads zero with no error. Every count is exact:
 *
 *   strip     a 10-vertex triangle strip, 8 triangles
 *   lines     a 6-vertex line list, 3 lines
 *   discard   the strip again under rasterizer discard: primitives are
 *             generated ahead of the rasterizer, so it still counts 8
 *             (primitivesGeneratedQueryWithRasterizerDiscard)
 *   stream1   the strip under a query on vertex stream 1: with no geometry
 *             shader every primitive goes to stream 0, so it counts 0
 *             (primitivesGeneratedQueryWithNonZeroStreams)
 *
 * A case whose feature is not advertised runs as an empty query, which must
 * still read 0. The strip covers the target and the lines are drawn over it,
 * so the footprint is checked first and a draw that went wrong is reported as
 * such rather than blamed on the query.
 *
 * Results are read with vkGetQueryPoolResults and with
 * vkCmdCopyQueryPoolResults into a buffer, and the two must agree.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define FORMAT    VK_FORMAT_R8G8B8A8_UNORM

#define STRIP_VERTS  10u
#define STRIP_PRIMS  8u
#define LINE_VERTS   6u
#define LINE_PRIMS   3u
/* Must match the rows primgen.vert places its lines on. */
static const uint32_t line_rows[LINE_PRIMS] = { 8u, 32u, 56u };

enum { Q_STRIP, Q_LINES, Q_DISCARD, Q_STREAM1, NQUERIES };

static const char *const query_names[NQUERIES] = {
   "strip", "lines", "discard", "stream1",
};

#define CHECK(x) do {                                                    \
   VkResult _r = (x);                                                    \
   if (_r != VK_SUCCESS) {                                               \
      printf("FAILED (primgen: %s -> VkResult %d)\n", #x, (int)_r);      \
      return 1;                                                          \
   }                                                                     \
} while (0)

struct push {
   uint32_t strip;
   uint32_t height;
};

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
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
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize bytes, VkBuffer *out_buf, VkDeviceMemory *out_mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkResult r = vkCreateBuffer(dev, &bci, NULL, out_buf);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   r = vkAllocateMemory(dev, &mai, NULL, out_mem);
   if (r != VK_SUCCESS) {
      return r;
   }
   return vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
}

static VkResult
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl,
              VkShaderModule vs, VkShaderModule fs,
              VkPrimitiveTopology topology, VkBool32 discard, VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
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
      .topology = topology,
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
      .rasterizerDiscardEnable = discard,
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
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static bool
has_device_extension(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   if (vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS) {
      return false;
   }
   VkExtensionProperties *exts = calloc(n, sizeof(*exts));
   bool found = false;
   if (exts && vkEnumerateDeviceExtensionProperties(pd, NULL, &n, exts) == VK_SUCCESS) {
      for (uint32_t i = 0; i < n; i++) {
         if (strcmp(exts[i].extensionName, name) == 0) {
            found = true;
            break;
         }
      }
   }
   free(exts);
   return found;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "primgen.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "primgen.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-primgen",
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
      printf("FAILED (primgen: no physical device)\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* lavapipe advertises the extension and its base feature unconditionally,
    * so a device built on it that lacks either has lost them, not opted out. */
   if (!has_device_extension(pd, VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME)) {
      printf("FAILED (primgen: %s not advertised; lavapipe always exposes it)\n",
             VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME);
      return 1;
   }
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &pgq_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   if (!pgq_have.primitivesGeneratedQuery) {
      printf("FAILED (primgen: primitivesGeneratedQuery feature not "
             "advertised; lavapipe always exposes it)\n");
      return 1;
   }
   const bool with_discard =
      pgq_have.primitivesGeneratedQueryWithRasterizerDiscard;
   /* A query on another stream is begun through the transform-feedback entry
    * point, so it also needs that extension and a second stream. */
   bool with_stream1 = pgq_have.primitivesGeneratedQueryWithNonZeroStreams &&
      has_device_extension(pd, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
   if (with_stream1) {
      VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
      };
      VkPhysicalDeviceProperties2 props2 = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &xfb,
      };
      vkGetPhysicalDeviceProperties2(pd, &props2);
      with_stream1 = xfb.maxTransformFeedbackStreams > 1;
   }
   if (!with_discard) {
      printf("note: primitivesGeneratedQueryWithRasterizerDiscard not "
             "advertised; the discard query is left empty\n");
   }
   if (!with_stream1) {
      printf("note: no query on a non-zero stream (needs "
             "primitivesGeneratedQueryWithNonZeroStreams and %s with 2+ "
             "streams); the stream1 query is left empty\n",
             VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
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
      printf("FAILED (primgen: no graphics queue)\n");
      return 1;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
      .primitivesGeneratedQuery = VK_TRUE,
      .primitivesGeneratedQueryWithRasterizerDiscard = with_discard,
      .primitivesGeneratedQueryWithNonZeroStreams = with_stream1,
   };
   const char *dev_exts[] = {
      VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME,
      VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &pgq,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = with_stream1 ? 2u : 1u,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   PFN_vkCmdBeginQueryIndexedEXT p_begin_indexed = NULL;
   PFN_vkCmdEndQueryIndexedEXT p_end_indexed = NULL;
   if (with_stream1) {
      p_begin_indexed = (PFN_vkCmdBeginQueryIndexedEXT)
         vkGetDeviceProcAddr(dev, "vkCmdBeginQueryIndexedEXT");
      p_end_indexed = (PFN_vkCmdEndQueryIndexedEXT)
         vkGetDeviceProcAddr(dev, "vkCmdEndQueryIndexedEXT");
      if (!p_begin_indexed || !p_end_indexed) {
         printf("FAILED (primgen: indexed query entry points missing)\n");
         return 1;
      }
   }

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      printf("FAILED (primgen: cannot load %s / %s)\n", vs_path, fs_path);
      return 1;
   }

   /* --- colour target ---------------------------------------------- */
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
   uint32_t imt = find_mem(&mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) {
      printf("FAILED (primgen: no image memory)\n");
      return 1;
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

   /* --- pipelines ---------------------------------------------------- */
   const VkShaderStageFlags pc_stages =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
   VkPushConstantRange pcr = {
      .stageFlags = pc_stages, .offset = 0, .size = sizeof(struct push),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkPipeline strip_pipe;
   VkPipeline line_pipe;
   VkPipeline discard_pipe;
   CHECK(make_pipeline(dev, rp, pl, vs, fs,
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_FALSE,
                       &strip_pipe));
   CHECK(make_pipeline(dev, rp, pl, vs, fs,
                       VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_FALSE, &line_pipe));
   CHECK(make_pipeline(dev, rp, pl, vs, fs,
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_TRUE,
                       &discard_pipe));

   /* --- readback buffers + query pool ------------------------------- */
   const VkDeviceSize px_bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBuffer pxbuf;
   VkDeviceMemory pxmem;
   CHECK(host_buffer(dev, &mp, px_bytes, &pxbuf, &pxmem));

   const VkDeviceSize stride = sizeof(uint64_t);
   VkBuffer qbuf;
   VkDeviceMemory qmem;
   CHECK(host_buffer(dev, &mp, NQUERIES * stride, &qbuf, &qmem));
   uint8_t *qmap;
   CHECK(vkMapMemory(dev, qmem, 0, NQUERIES * stride, 0, (void **)&qmap));
   memset(qmap, 0xff, (size_t)(NQUERIES * stride));
   vkUnmapMemory(dev, qmem);

   VkQueryPoolCreateInfo qpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
      .queryCount = NQUERIES,
   };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));

   /* --- record ------------------------------------------------------ */
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
   vkCmdResetQueryPool(cmd, qp, 0, NQUERIES);
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

   const struct push strip_pc = { .strip = 1u, .height = HEIGHT };
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, discard_pipe);
   vkCmdPushConstants(cmd, pl, pc_stages, 0, sizeof strip_pc, &strip_pc);
   vkCmdBeginQuery(cmd, qp, Q_DISCARD, 0);
   if (with_discard) {
      vkCmdDraw(cmd, STRIP_VERTS, 1, 0, 0);
   }
   vkCmdEndQuery(cmd, qp, Q_DISCARD);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, strip_pipe);
   vkCmdPushConstants(cmd, pl, pc_stages, 0, sizeof strip_pc, &strip_pc);
   if (with_stream1) {
      p_begin_indexed(cmd, qp, Q_STREAM1, 0, 1);
      vkCmdDraw(cmd, STRIP_VERTS, 1, 0, 0);
      p_end_indexed(cmd, qp, Q_STREAM1, 1);
   } else {
      vkCmdBeginQuery(cmd, qp, Q_STREAM1, 0);
      vkCmdEndQuery(cmd, qp, Q_STREAM1);
   }

   vkCmdBeginQuery(cmd, qp, Q_STRIP, 0);
   vkCmdDraw(cmd, STRIP_VERTS, 1, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_STRIP);

   const struct push line_pc = { .strip = 0u, .height = HEIGHT };
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipe);
   vkCmdPushConstants(cmd, pl, pc_stages, 0, sizeof line_pc, &line_pc);
   vkCmdBeginQuery(cmd, qp, Q_LINES, 0);
   vkCmdDraw(cmd, LINE_VERTS, 1, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_LINES);

   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          pxbuf, 1, &region);
   vkCmdCopyQueryPoolResults(cmd, qp, 0, NQUERIES, qbuf, 0, stride,
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
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

   /* --- read back ---------------------------------------------------- */
   uint64_t got[NQUERIES];
   CHECK(vkGetQueryPoolResults(dev, qp, 0, NQUERIES, sizeof got, got, stride,
                               VK_QUERY_RESULT_64_BIT |
                               VK_QUERY_RESULT_WAIT_BIT));
   uint64_t copied[NQUERIES];
   CHECK(vkMapMemory(dev, qmem, 0, NQUERIES * stride, 0, (void **)&qmap));
   memcpy(copied, qmap, sizeof copied);
   vkUnmapMemory(dev, qmem);

   /* The strip tiles the target, so every pixel is red or, on a line row,
    * green; each line spans the full width, so its row is mostly green. */
   uint8_t *px;
   CHECK(vkMapMemory(dev, pxmem, 0, px_bytes, 0, (void **)&px));
   long painted = 0;
   long green_on_rows = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      bool line_row = false;
      for (uint32_t l = 0; l < LINE_PRIMS; l++) {
         line_row |= (y == line_rows[l]);
      }
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * 4;
         const bool red = p[0] == 255 && p[1] == 0 && p[2] == 0;
         const bool green = p[0] == 0 && p[1] == 255 && p[2] == 0;
         if (red || (green && line_row)) {
            painted++;
         }
         if (green && line_row) {
            green_on_rows++;
         }
      }
   }
   vkUnmapMemory(dev, pxmem);

   for (unsigned q = 0; q < NQUERIES; q++) {
      printf("query %s: %llu\n", query_names[q], (unsigned long long)got[q]);
   }

   vkDestroyQueryPool(dev, qp, NULL);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyBuffer(dev, qbuf, NULL);
   vkFreeMemory(dev, qmem, NULL);
   vkDestroyBuffer(dev, pxbuf, NULL);
   vkFreeMemory(dev, pxmem, NULL);
   vkDestroyPipeline(dev, strip_pipe, NULL);
   vkDestroyPipeline(dev, line_pipe, NULL);
   vkDestroyPipeline(dev, discard_pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   const long npx = (long)WIDTH * HEIGHT;
   const long green_min = (long)LINE_PRIMS * (WIDTH - 2);
   if (painted != npx || green_on_rows < green_min) {
      printf("FAILED (primgen: %ld of %ld px painted, %ld line px (want >= "
             "%ld), so the query results cannot be judged)\n",
             painted, npx, green_on_rows, green_min);
      return 1;
   }
   for (unsigned q = 0; q < NQUERIES; q++) {
      if (got[q] != copied[q]) {
         printf("FAILED (primgen: query %s is %llu from vkGetQueryPoolResults "
                "but %llu from vkCmdCopyQueryPoolResults)\n", query_names[q],
                (unsigned long long)got[q], (unsigned long long)copied[q]);
         return 1;
      }
   }
   const uint64_t want[NQUERIES] = {
      [Q_STRIP]   = STRIP_PRIMS,
      [Q_LINES]   = LINE_PRIMS,
      [Q_DISCARD] = with_discard ? STRIP_PRIMS : 0u,
      [Q_STREAM1] = 0u,
   };
   unsigned bad = 0;
   for (unsigned q = 0; q < NQUERIES; q++) {
      if (got[q] != want[q]) {
         printf("  %s: counted %llu primitives, want %llu\n", query_names[q],
                (unsigned long long)got[q], (unsigned long long)want[q]);
         bad++;
      }
   }
   if (bad) {
      printf("FAILED (primgen: %u of %u queries wrong; zero means the counter "
             "was never updated, more means primitives were counted that the "
             "topology never assembled or that went to another stream)\n",
             bad, (unsigned)NQUERIES);
      return 1;
   }
   printf("PASSED (primgen: strip %llu, lines %llu, discard %llu%s, stream1 "
          "%llu%s; copy agrees)\n", (unsigned long long)got[Q_STRIP],
          (unsigned long long)got[Q_LINES], (unsigned long long)got[Q_DISCARD],
          with_discard ? "" : " (empty)", (unsigned long long)got[Q_STREAM1],
          with_stream1 ? "" : " (empty)");
   return 0;
}
