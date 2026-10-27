/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Transform feedback, vertex streams and queries from a geometry shader.
 *
 * Every captured word is an integer the host rebuilds from the primitive and
 * vertex it belongs to, in the order the specification gives: input primitive
 * by input primitive, each one's in emission order, strips split into
 * triangles whose odd ones are wound back with the provoking vertex first or
 * last. Every buffer starts filled with a sentinel and every byte is compared,
 * so a hole written, a primitive lost or one written past the room shows.
 *
 *   streams     points emitted on all four vertex streams, each stream into
 *               its own buffer, the last one a primitive short of room; the
 *               stream queries (written, needed) and primitives-generated
 *               queries of streams 1-3
 *   strip_*     strips from a static and from a run-time count, captured into
 *               two buffers of stream 0, the second one vertex short: a
 *               primitive is written only where every buffer of its stream has
 *               room, so both stop at the same primitive -- with a run-time
 *               count, in the middle of one invocation's strip. Both provoking
 *               modes; the stream-0 query
 *   resume      two draws in two Begin/End pairs, the second resuming from the
 *               counter buffer the first saved
 *   mixed_strip triangles drawn on stream 0, which nothing captures, and strips
 *               on stream 1, captured and never drawn, with a static and with
 *               a run-time count: the capture must be stream 1's strips, and
 *               only stream 0's triangles may reach the target. Strips on a
 *               stream other than 0 go past what the device advertises
 *               (transformFeedbackStreamsLinesTriangles is false); the case
 *               pins that the geometry stage still keeps each stream's
 *               primitives to its own list
 *   mixed_points points on stream 0, drawn and not captured, and a run-time
 *               count of points on stream 1, captured into a buffer one
 *               primitive short: stream 1 is committed from its own count
 *   restart     a triangle strip cut by a restart index into a static-count
 *               geometry shader, so only the device knows how many input
 *               triangles it gets: the capture, the stream query and the
 *               geometry statistics of the triangles the runs make
 *   stats       pipeline statistics around four-invocation geometry shaders:
 *               invocations (every instanced one), primitives, clipper
 *               invocations and primitives; and again under rasterizer
 *               discard, which changes none of them and draws nothing
 *
 * GSXFB_BITE=1 expects what a wrong implementation would produce (the capture
 * in invocation order, the overflow judged per buffer, statistics counting
 * input primitives); every case must then fail.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32u
#define H 32u
#define SENTINEL 0xDEADBEEFu
#define BUF 4096u

#define CHECK(x) do {                                                     \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      printf("FAILED (gsxfb: %s -> VkResult %d)\n", #x, (int)_r);          \
      exit(1);                                                            \
   }                                                                      \
} while (0)

static bool bite;
static int failures;

static PFN_vkCmdBindTransformFeedbackBuffersEXT BindXfb;
static PFN_vkCmdBeginTransformFeedbackEXT BeginXfb;
static PFN_vkCmdEndTransformFeedbackEXT EndXfb;
static PFN_vkCmdBeginQueryIndexedEXT BeginQueryIndexed;
static PFN_vkCmdEndQueryIndexedEXT EndQueryIndexed;

static struct {
   VkDevice dev;
   VkQueue queue;
   VkPhysicalDeviceMemoryProperties mp;
   VkCommandPool pool;
   VkImage img;
   VkRenderPass rp;
   VkFramebuffer fb;
   VkPipelineLayout layout;
   VkBuffer rb;
   void *rb_map;
   VkBuffer xb[4];
   uint32_t *xmap[4];
   VkBuffer ctr;
   uint32_t *ctr_map;
} vk;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk.mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (vk.mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   printf("FAILED (gsxfb: no memory type)\n");
   exit(1);
}

static void *
buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *b)
{
   VkBufferCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = usage };
   CHECK(vkCreateBuffer(vk.dev, &ci, NULL, b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(vk.dev, *b, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   VkDeviceMemory m;
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &m));
   CHECK(vkBindBufferMemory(vk.dev, *b, m, 0));
   void *map;
   CHECK(vkMapMemory(vk.dev, m, 0, size, 0, &map));
   return map;
}

static VkShaderModule
module(const char *name)
{
   FILE *f = fopen(name, "rb");
   if (!f) {
      printf("FAILED (gsxfb: cannot open %s)\n", name);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc((size_t)sz);
   if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
      printf("FAILED (gsxfb: cannot read %s)\n", name);
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
      printf("FAILED (gsxfb: no device)\n");
      exit(1);
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(phys, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(phys, &vk.mp);

   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT tf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
      .pNext = &pgq };
   VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                    .pNext = &tf };
   vkGetPhysicalDeviceFeatures2(phys, &f2);
   if (!f2.features.geometryShader || !f2.features.pipelineStatisticsQuery ||
       !tf.transformFeedback || !tf.geometryStreams || !pgq.primitivesGeneratedQuery ||
       !pgq.primitivesGeneratedQueryWithNonZeroStreams) {
      printf("FAILED (gsxfb: device lacks geometry streams, transform feedback or the "
             "queries)\n");
      exit(1);
   }
   VkPhysicalDeviceTransformFeedbackPropertiesEXT tfp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT };
   VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                      .pNext = &tfp };
   vkGetPhysicalDeviceProperties2(phys, &p2);
   if (tfp.maxTransformFeedbackStreams < 4) {
      printf("FAILED (gsxfb: %u transform feedback streams)\n",
             tfp.maxTransformFeedbackStreams);
      exit(1);
   }

   uint32_t nq = 8, qf;
   VkQueueFamilyProperties qp[8];
   vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
   for (qf = 0; qf < nq && !(qp[qf].queueFlags & VK_QUEUE_GRAPHICS_BIT); qf++)
      ;
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = qf, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
      .primitivesGeneratedQuery = VK_TRUE,
      .primitivesGeneratedQueryWithNonZeroStreams = VK_TRUE };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT tf_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
      .pNext = &pgq_on, .transformFeedback = VK_TRUE, .geometryStreams = VK_TRUE };
   VkPhysicalDeviceFeatures feat = { .geometryShader = VK_TRUE,
                                     .pipelineStatisticsQuery = VK_TRUE };
   const char *ext[] = { VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
                         VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &tf_on,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 2, .ppEnabledExtensionNames = ext,
                              .pEnabledFeatures = &feat };
   CHECK(vkCreateDevice(phys, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, qf, 0, &vk.queue);
   BindXfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdBindTransformFeedbackBuffersEXT");
   BeginXfb = (PFN_vkCmdBeginTransformFeedbackEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdBeginTransformFeedbackEXT");
   EndXfb = (PFN_vkCmdEndTransformFeedbackEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdEndTransformFeedbackEXT");
   BeginQueryIndexed = (PFN_vkCmdBeginQueryIndexedEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdBeginQueryIndexedEXT");
   EndQueryIndexed = (PFN_vkCmdEndQueryIndexedEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdEndQueryIndexedEXT");

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
   vk.rb_map = buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &vk.rb);
   for (unsigned b = 0; b < 4; b++)
      vk.xmap[b] = buffer(BUF, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, &vk.xb[b]);
   vk.ctr_map = buffer(64, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT, &vk.ctr);
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &vk.layout));
}

struct vtx { float x, y; uint32_t n; };

static VkPipeline
pipeline_restart(const char *geom, VkPrimitiveTopology topo, bool dynamic, bool last,
                 bool discard, bool restart)
{
   VkShaderModule vs = module("gsx.vert.spv"), fs = module("gsx.frag.spv"), gs = module(geom);
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
   VkVertexInputBindingDescription bind = { 0, sizeof(struct vtx), VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription attr[2] = {
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct vtx, x) },
      { 1, 0, VK_FORMAT_R32_UINT, offsetof(struct vtx, n) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind,
      .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attr };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = topo,
      .primitiveRestartEnable = restart };
   VkViewport vp = { 0, 0, W, H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
      .provokingVertexMode = last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                  : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = last ? &pv : NULL, .rasterizerDiscardEnable = discard,
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
      .stageCount = 3, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .layout = vk.layout, .renderPass = vk.rp };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
   vkDestroyShaderModule(vk.dev, vs, NULL);
   vkDestroyShaderModule(vk.dev, fs, NULL);
   vkDestroyShaderModule(vk.dev, gs, NULL);
   return p;
}

static VkPipeline
pipeline(const char *geom, VkPrimitiveTopology topo, bool dynamic, bool last, bool discard)
{
   return pipeline_restart(geom, topo, dynamic, last, discard, false);
}

static VkCommandBuffer
begin(void)
{
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = vk.pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   return cmd;
}

static void
pass_begin(VkCommandBuffer cmd)
{
   VkClearValue clear = { .color = { .float32 = { 0, 0, 0, 0 } } };
   VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                 .renderPass = vk.rp, .framebuffer = vk.fb,
                                 .renderArea = { { 0, 0 }, { W, H } },
                                 .clearValueCount = 1, .pClearValues = &clear };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
}

/* End the pass, read the target back, run and free the command buffer. */
static void
finish(VkCommandBuffer cmd)
{
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, vk.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.rb, 1, &reg);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
}

static VkBuffer
vertices(const struct vtx *v, unsigned n)
{
   VkBuffer b;
   void *map = buffer(sizeof(struct vtx) * n, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &b);
   memcpy(map, v, sizeof(struct vtx) * n);
   return b;
}

static void
sentinel(void)
{
   for (unsigned b = 0; b < 4; b++)
      for (unsigned w = 0; w < BUF / 4; w++)
         vk.xmap[b][w] = SENTINEL;
}

/* Compare a buffer with what the host expects of it, every word. */
static void
check_buffer(const char *name, unsigned b, const uint32_t *want)
{
   unsigned bad = 0;
   for (unsigned w = 0; w < BUF / 4; w++)
      if (vk.xmap[b][w] != want[w] && bad++ < 4)
         printf("  %s: buffer %u word %u = %08x, expected %08x\n", name, b, w, vk.xmap[b][w],
                want[w]);
   printf("%s_buffer%u: %s\n", name, b, bad ? "FAILED" : "passed");
   failures += bad != 0;
}

static void
check_value(const char *name, const char *what, uint64_t got, uint64_t want)
{
   const bool ok = got == want;
   if (!ok)
      printf("  %s: %s = %llu, expected %llu\n", name, what, (unsigned long long)got,
             (unsigned long long)want);
   printf("%s_%s: %s\n", name, what, ok ? "passed" : "FAILED");
   failures += !ok;
}

static VkQueryPool
query_pool(VkQueryType type, VkQueryPipelineStatisticFlags stats, uint32_t count)
{
   VkQueryPoolCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                .queryType = type, .queryCount = count,
                                .pipelineStatistics = stats };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(vk.dev, &ci, NULL, &qp));
   return qp;
}

#define NPOINTS 6u

/* Points on all four streams, each into its own buffer, buffer 3 one
 * primitive short. */
static void
case_streams(void)
{
   struct vtx v[NPOINTS];
   for (unsigned i = 0; i < NPOINTS; i++)
      v[i] = (struct vtx){ -0.8f + 0.25f * i, 0.0f, 0 };
   VkBuffer vb = vertices(v, NPOINTS);
   VkPipeline p = pipeline("streams.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, false,
                           false);
   VkQueryPool xq = query_pool(VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, 0, 4);
   VkQueryPool gq = query_pool(VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT, 0, 4);
   sentinel();
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, xq, 0, 4);
   vkCmdResetQueryPool(cmd, gq, 0, 4);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   const VkDeviceSize offs[4] = { 0, 0, 0, 0 };
   /* Room for every stream's primitives, but one fewer of stream 3's. */
   const VkDeviceSize sizes[4] = { 8u * NPOINTS, 16u * NPOINTS, 24u * NPOINTS,
                                   32u * NPOINTS - 8u };
   BindXfb(cmd, 0, 4, vk.xb, offs, sizes);
   for (unsigned s = 0; s < 4; s++) {
      BeginQueryIndexed(cmd, xq, s, 0, s);
      BeginQueryIndexed(cmd, gq, s, 0, s);
   }
   BeginXfb(cmd, 0, 0, NULL, NULL);
   vkCmdDraw(cmd, NPOINTS, 1, 0, 0);
   EndXfb(cmd, 0, 0, NULL, NULL);
   for (unsigned s = 0; s < 4; s++) {
      EndQueryIndexed(cmd, xq, s, s);
      EndQueryIndexed(cmd, gq, s, s);
   }
   finish(cmd);

   static uint32_t want[BUF / 4];
   uint64_t xr[8], gr[4];
   CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 4, sizeof xr, xr, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   CHECK(vkGetQueryPoolResults(vk.dev, gq, 0, 4, sizeof gr, gr, 8,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   for (unsigned s = 0; s < 4; s++) {
      for (unsigned w = 0; w < BUF / 4; w++)
         want[w] = SENTINEL;
      const unsigned per = s + 1, room = s == 3 ? NPOINTS * per - 1 : NPOINTS * per;
      for (unsigned i = 0; i < NPOINTS; i++) {
         for (unsigned j = 0; j < per; j++) {
            /* The wrong order runs the stream's emissions before the next
             * primitive's. */
            const unsigned at = bite ? j * NPOINTS + i : i * per + j;
            if (at < room) {
               want[2 * at] = i;
               want[2 * at + 1] = 100u * s + j;
            }
         }
      }
      check_buffer("streams", s, want);
      char what[32];
      snprintf(what, sizeof what, "written%u", s);
      check_value("streams", what, xr[2 * s], NPOINTS * per - (s == 3 ? 1u : 0u));
      snprintf(what, sizeof what, "needed%u", s);
      check_value("streams", what, xr[2 * s + 1], NPOINTS * per);
      snprintf(what, sizeof what, "generated%u", s);
      check_value("streams", what, gr[s], NPOINTS * per + (bite && s == 0 ? 1u : 0u));
   }
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, xq, NULL);
   vkDestroyQueryPool(vk.dev, gq, NULL);
}

/* What strip.geom captures for input triangles whose strips have n[i]
 * triangles, into words of buffer 0 (8-byte vertices) and buffer 1 (12-byte,
 * one word at 4), up to `room` primitives. */
static unsigned
expect_strip(const unsigned *n, unsigned prims, bool last, unsigned room, uint32_t *w0,
             uint32_t *w1, unsigned base)
{
   unsigned t = base;
   for (unsigned p = 0; p < prims; p++) {
      for (unsigned k = 0; k < n[p]; k++, t++) {
         unsigned v[3] = { k, k + 1, k + 2 };
         /* The wrong order keeps odd triangles as emitted. */
         if ((k & 1) && !bite) {
            if (last) {
               v[0] = k + 1; v[1] = k; v[2] = k + 2;
            } else {
               v[0] = k; v[1] = k + 2; v[2] = k + 1;
            }
         }
         if (t >= room)
            continue;
         for (unsigned c = 0; c < 3; c++) {
            w0[2 * (3 * t + c)] = p;
            w0[2 * (3 * t + c) + 1] = v[c];
            w1[3 * (3 * t + c) + 1] = 1000u * p + v[c];
         }
      }
   }
   return t;
}

#define NTRIS 5u

static void
case_strip(bool dynamic, bool last)
{
   static const unsigned dyn_n[NTRIS] = { 3, 0, 6, 1, 4 };
   unsigned n[NTRIS], total = 0;
   struct vtx v[3 * NTRIS];
   for (unsigned i = 0; i < NTRIS; i++) {
      n[i] = dynamic ? dyn_n[i] : 2u;
      total += n[i];
      for (unsigned k = 0; k < 3; k++)
         v[3 * i + k] = (struct vtx){ -0.9f + 0.3f * i, -0.5f + 0.1f * k, n[i] };
   }
   char name[32];
   snprintf(name, sizeof name, "strip_%s_%s", dynamic ? "dyn" : "static", last ? "last" : "first");
   VkBuffer vb = vertices(v, 3 * NTRIS);
   VkPipeline p = pipeline("strip.geom.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, dynamic, last,
                           false);
   VkQueryPool xq = query_pool(VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, 0, 1);
   sentinel();
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, xq, 0, 1);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   /* Buffer 0 holds every primitive; buffer 1 is one vertex short of that,
    * which stops both at the last primitive. */
   const VkDeviceSize offs[2] = { 0, 0 };
   const VkDeviceSize sizes[2] = { 24u * total, 36u * total - 12u };
   BindXfb(cmd, 0, 2, vk.xb, offs, sizes);
   BeginQueryIndexed(cmd, xq, 0, 0, 0);
   BeginXfb(cmd, 0, 0, NULL, NULL);
   vkCmdDraw(cmd, 3 * NTRIS, 1, 0, 0);
   EndXfb(cmd, 0, 0, NULL, NULL);
   EndQueryIndexed(cmd, xq, 0, 0);
   finish(cmd);

   static uint32_t w0[BUF / 4], w1[BUF / 4];
   for (unsigned w = 0; w < BUF / 4; w++)
      w0[w] = w1[w] = SENTINEL;
   /* The wrong rule would let buffer 0 take the last primitive too. */
   expect_strip(n, NTRIS, last, total - 1, w0, w1, 0);
   check_buffer(name, 0, w0);
   check_buffer(name, 1, w1);
   uint64_t xr[2];
   CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 1, sizeof xr, xr, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   check_value(name, "written", xr[0], total - 1);
   check_value(name, "needed", xr[1], total);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, xq, NULL);
}

/* Pixels drawn in the left and in the right half of the target. */
static void
coverage(unsigned *left, unsigned *right)
{
   *left = *right = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++)
         if (((const uint32_t *)vk.rb_map)[y * W + x])
            ++*(x < W / 2 ? left : right);
}

/* Stream 0 drawn and not captured, stream 1's strips captured and not drawn. */
static void
case_mixed_strip(bool dynamic)
{
   static const unsigned dyn_n[NTRIS] = { 3, 0, 5, 1, 4 };
   unsigned n[NTRIS], total = 0;
   struct vtx v[3 * NTRIS];
   for (unsigned i = 0; i < NTRIS; i++) {
      n[i] = dynamic ? dyn_n[i] : 3u;
      total += n[i];
      v[3 * i + 0] = (struct vtx){ -0.9f, -0.8f, n[i] };
      v[3 * i + 1] = (struct vtx){ -0.3f, -0.8f, n[i] };
      v[3 * i + 2] = (struct vtx){ -0.6f, -0.2f, n[i] };
   }
   const char *name = dynamic ? "mixed_strip_dyn" : "mixed_strip_static";
   VkBuffer vb = vertices(v, 3 * NTRIS);
   VkPipeline p = pipeline("mixed_strip.geom.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, dynamic,
                           false, false);
   VkQueryPool xq = query_pool(VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, 0, 1);
   sentinel();
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, xq, 0, 1);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   const VkDeviceSize offs[1] = { 0 };
   const VkDeviceSize sizes[1] = { 24u * total };
   BindXfb(cmd, 0, 1, vk.xb, offs, sizes);
   BeginQueryIndexed(cmd, xq, 0, 0, 1);
   BeginXfb(cmd, 0, 0, NULL, NULL);
   vkCmdDraw(cmd, 3 * NTRIS, 1, 0, 0);
   EndXfb(cmd, 0, 0, NULL, NULL);
   EndQueryIndexed(cmd, xq, 0, 1);
   finish(cmd);

   static uint32_t w0[BUF / 4], w1[3 * (BUF / 4)];
   for (unsigned w = 0; w < BUF / 4; w++)
      w0[w] = SENTINEL;
   expect_strip(n, NTRIS, false, total, w0, w1, 0);
   check_buffer(name, 0, w0);
   uint64_t xr[2];
   CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 1, sizeof xr, xr, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   check_value(name, "written", xr[0], total);
   check_value(name, "needed", xr[1], total);
   /* Stream 1's strips lie in the right half; drawing them is wrong. */
   unsigned left, right;
   coverage(&left, &right);
   check_value(name, "left_drawn", left != 0, 1);
   check_value(name, "right_clear", right == 0, !bite);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, xq, NULL);
}

/* Points on stream 0, drawn, and a run-time count on stream 1, captured into a
 * buffer one primitive short. */
static void
case_mixed_points(void)
{
   static const unsigned cnt[NPOINTS] = { 3, 0, 5, 1, 2, 4 };
   unsigned total = 0;
   struct vtx v[NPOINTS];
   for (unsigned i = 0; i < NPOINTS; i++) {
      v[i] = (struct vtx){ -0.8f + 0.25f * i, 0.0f, cnt[i] };
      total += cnt[i];
   }
   VkBuffer vb = vertices(v, NPOINTS);
   VkPipeline p = pipeline("mixed_points.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false,
                           false, false);
   VkQueryPool xq = query_pool(VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, 0, 1);
   sentinel();
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, xq, 0, 1);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   const VkDeviceSize offs[1] = { 0 };
   const VkDeviceSize sizes[1] = { 8u * (total - 1u) };
   BindXfb(cmd, 0, 1, vk.xb, offs, sizes);
   BeginQueryIndexed(cmd, xq, 0, 0, 1);
   BeginXfb(cmd, 0, 0, NULL, NULL);
   vkCmdDraw(cmd, NPOINTS, 1, 0, 0);
   const VkDeviceSize coff = 0;
   EndXfb(cmd, 0, 1, &vk.ctr, &coff);
   EndQueryIndexed(cmd, xq, 0, 1);
   finish(cmd);

   static uint32_t want[BUF / 4];
   for (unsigned w = 0; w < BUF / 4; w++)
      want[w] = SENTINEL;
   unsigned at = 0;
   for (unsigned i = 0; i < NPOINTS; i++)
      for (unsigned j = 0; j < cnt[i]; j++, at++)
         if (at < total - 1u) {
            want[2 * at] = i;
            want[2 * at + 1] = j;
         }
   check_buffer("mixed_points", 0, want);
   uint64_t xr[2];
   CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 1, sizeof xr, xr, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   /* Committing stream 1 from stream 0's count is wrong. */
   check_value("mixed_points", "written", xr[0], bite ? NPOINTS : total - 1u);
   check_value("mixed_points", "needed", xr[1], total);
   check_value("mixed_points", "counter", vk.ctr_map[0], 8u * (total - 1u));
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, xq, NULL);
}

/* An indexed triangle strip cut by a restart index into strip.geom with its
 * static count: two runs of five vertices make six input triangles where the
 * eleven positions would make nine. */
static void
case_restart(void)
{
   static const uint16_t idx[] = { 0, 1, 2, 3, 4, 0xffff, 5, 6, 7, 8, 9 };
   const unsigned nidx = sizeof idx / sizeof idx[0], tris = 6;
   struct vtx v[10];
   for (unsigned i = 0; i < 10; i++)
      v[i] = (struct vtx){ -0.9f + 0.15f * i, -0.5f + 0.3f * (i & 1u), 2 };
   VkBuffer vb = vertices(v, 10);
   VkBuffer ib;
   memcpy(buffer(sizeof idx, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib), idx, sizeof idx);
   VkPipeline p = pipeline_restart("strip.geom.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false,
                                   false, false, true);
   VkQueryPool xq = query_pool(VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, 0, 1);
   const VkQueryPipelineStatisticFlags flags =
      VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT;
   VkQueryPool sq = query_pool(VK_QUERY_TYPE_PIPELINE_STATISTICS, flags, 1);
   sentinel();
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, xq, 0, 1);
   vkCmdResetQueryPool(cmd, sq, 0, 1);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT16);
   const VkDeviceSize offs[2] = { 0, 0 };
   const VkDeviceSize sizes[2] = { BUF, BUF };
   BindXfb(cmd, 0, 2, vk.xb, offs, sizes);
   BeginQueryIndexed(cmd, xq, 0, 0, 0);
   vkCmdBeginQuery(cmd, sq, 0, 0);
   BeginXfb(cmd, 0, 0, NULL, NULL);
   vkCmdDrawIndexed(cmd, nidx, 1, 0, 0, 0);
   EndXfb(cmd, 0, 0, NULL, NULL);
   vkCmdEndQuery(cmd, sq, 0);
   EndQueryIndexed(cmd, xq, 0, 0);
   finish(cmd);

   unsigned n[6];
   for (unsigned i = 0; i < tris; i++)
      n[i] = 2;
   static uint32_t w0[BUF / 4], w1[BUF / 4];
   for (unsigned w = 0; w < BUF / 4; w++)
      w0[w] = w1[w] = SENTINEL;
   expect_strip(n, tris, false, ~0u, w0, w1, 0);
   check_buffer("restart", 0, w0);
   check_buffer("restart", 1, w1);
   uint64_t xr[2], sr[3];
   CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 1, sizeof xr, xr, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   CHECK(vkGetQueryPoolResults(vk.dev, sq, 0, 1, sizeof sr, sr, sizeof sr,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   check_value("restart", "written", xr[0], 2u * tris);
   check_value("restart", "needed", xr[1], 2u * tris);
   /* Counting the positions' closed form of nine triangles is wrong. */
   check_value("restart", "ia_primitives", sr[0], bite ? 9u : tris);
   check_value("restart", "gs_invocations", sr[1], bite ? 9u : tris);
   check_value("restart", "gs_primitives", sr[2], 2u * tris);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, xq, NULL);
   vkDestroyQueryPool(vk.dev, sq, NULL);
}

/* Two draws in two Begin/End pairs, the second resuming where the first's
 * counter buffer says it stopped. */
static void
case_resume(void)
{
   struct vtx v[3 * NTRIS];
   unsigned n[NTRIS];
   for (unsigned i = 0; i < NTRIS; i++) {
      n[i] = 2;
      for (unsigned k = 0; k < 3; k++)
         v[3 * i + k] = (struct vtx){ -0.9f + 0.3f * i, -0.5f + 0.1f * k, 2 };
   }
   VkBuffer vb = vertices(v, 3 * NTRIS);
   VkPipeline p = pipeline("strip.geom.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, false,
                           false);
   sentinel();
   for (unsigned seg = 0; seg < 2; seg++) {
      VkCommandBuffer cmd = begin();
      pass_begin(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
      VkDeviceSize zero = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
      const VkDeviceSize offs[2] = { 0, 0 };
      const VkDeviceSize sizes[2] = { BUF, BUF };
      BindXfb(cmd, 0, 2, vk.xb, offs, sizes);
      const VkBuffer ctrs[2] = { vk.ctr, vk.ctr };
      const VkDeviceSize coffs[2] = { 0, 4 };
      BeginXfb(cmd, 0, seg ? 2 : 0, seg ? ctrs : NULL, seg ? coffs : NULL);
      /* The second segment draws the input triangles from the third on. */
      vkCmdDraw(cmd, seg ? 3 * (NTRIS - 2) : 6, 1, seg ? 6 : 0, 0);
      EndXfb(cmd, 0, 2, ctrs, coffs);
      finish(cmd);
   }
   static uint32_t w0[BUF / 4], w1[BUF / 4];
   for (unsigned w = 0; w < BUF / 4; w++)
      w0[w] = w1[w] = SENTINEL;
   /* Each segment numbers its input primitives from 0; the second continues
    * the buffers -- or, wrongly, restarts them. */
   unsigned t = expect_strip(n, 2, false, ~0u, w0, w1, 0);
   expect_strip(n, NTRIS - 2, false, ~0u, w0, w1, bite ? 0 : t);
   check_buffer("resume", 0, w0);
   check_buffer("resume", 1, w1);
   check_value("resume", "counter0", vk.ctr_map[0], 24u * 2u * NTRIS);
   check_value("resume", "counter1", vk.ctr_map[1], 36u * 2u * NTRIS);
   vkDestroyPipeline(vk.dev, p, NULL);
}

/* Pipeline statistics around four-invocation geometry shaders, drawn and
 * discarded. */
static void
case_stats(bool discard)
{
   struct vtx v[NPOINTS];
   for (unsigned i = 0; i < NPOINTS; i++)
      v[i] = (struct vtx){ -0.8f + 0.1f * i, -0.8f + 0.25f * i, 0 };
   VkBuffer vb = vertices(v, NPOINTS);
   VkPipeline p = pipeline("inst.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, false,
                           discard);
   const VkQueryPipelineStatisticFlags flags =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
   VkQueryPool sq = query_pool(VK_QUERY_TYPE_PIPELINE_STATISTICS, flags, 1);
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, sq, 0, 1);
   pass_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
   vkCmdBeginQuery(cmd, sq, 0, 0);
   vkCmdDraw(cmd, NPOINTS, 1, 0, 0);
   vkCmdEndQuery(cmd, sq, 0);
   finish(cmd);
   uint64_t r[4];
   CHECK(vkGetQueryPoolResults(vk.dev, sq, 0, 1, sizeof r, r, sizeof r,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   const char *name = discard ? "stats_discard" : "stats";
   /* Every instanced invocation counts; counting input primitives is wrong. */
   check_value(name, "gs_invocations", r[0], bite ? NPOINTS : 4u * NPOINTS);
   check_value(name, "gs_primitives", r[1], 8u * NPOINTS);
   check_value(name, "clip_invocations", r[2], 8u * NPOINTS);
   check_value(name, "clip_primitives", r[3], 8u * NPOINTS);
   unsigned drawn = 0;
   for (unsigned k = 0; k < W * H; k++)
      drawn += ((const uint32_t *)vk.rb_map)[k] != 0;
   check_value(name, "drawn", drawn != 0, !discard);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, sq, NULL);
}

int
main(void)
{
   bite = getenv("GSXFB_BITE") && atoi(getenv("GSXFB_BITE"));
   init();
   case_streams();
   for (int dyn = 0; dyn < 2; dyn++)
      for (int last = 0; last < 2; last++)
         case_strip(dyn, last);
   case_resume();
   case_mixed_strip(false);
   case_mixed_strip(true);
   case_mixed_points();
   case_restart();
   case_stats(false);
   case_stats(true);
   if (failures) {
      printf("FAILED (gsxfb: %d check%s)\n", failures, failures == 1 ? "" : "s");
      return 1;
   }
   printf("PASSED\n");
   return 0;
}
