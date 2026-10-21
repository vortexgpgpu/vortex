/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Device / llvmpipe / device interleave against one storage buffer.
 *
 * Three dispatches of the same read-modify-write kernel (data[i] += 1), all
 * bound to the same buffer:
 *
 *   1. workgroup 4    -- fits the device cap, runs on Vortex
 *   2. workgroup 128  -- exceeds it, so vortexpipe falls back to llvmpipe
 *   3. workgroup 4    -- runs on Vortex again
 *
 * Correct result is i+3. The middle dispatch is the subject: llvmpipe executes
 * on the CPU and writes the host allocation directly, without any of the
 * gallium transfers a driver could interpose on. A driver that keeps a device
 * copy of the buffer and skips uploads it believes are up to date will miss
 * that write entirely -- dispatch 3 would resume from dispatch 1's output and
 * produce i+2, and the llvmpipe increment would vanish with no error anywhere.
 *
 * MESA_VORTEX_STRICT is 0 here, which is unusual for this suite and is the
 * point rather than a concession: strict mode turns the fallback into a no-op,
 * and a no-op cannot write the host buffer behind the driver's back. This test
 * needs the fallback to really run.
 *
 * The same holds for transform feedback, whose filled sizes the device keeps
 * between its draws. One capture segment draws three triangles into a layered
 * framebuffer, each vertex capturing its index:
 *
 *   1. xfbmix_a.vert  -- writes gl_Layer, runs on Vortex once per layer and
 *                        captures once
 *   2. xfbmix_b.vert  -- also counts its invocations in a storage buffer; a
 *                        layered draw with a side effect cannot replay per
 *                        layer, so it falls back to llvmpipe
 *   3. xfbmix_a.vert  -- on Vortex again
 *
 * The capture must be one continuous stream of vertices 0-8 with the counter
 * at 72 bytes: llvmpipe has to resume from the size the device reached, and
 * the device from the size llvmpipe left, without losing llvmpipe's bytes.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N           256u
/* Maximum local_size_x the test will request. The actual value is
 * specialized at pipeline creation to
 * min(LOCAL_SIZE_MAX, VkPhysicalDeviceProperties.limits.maxComputeWorkGroupSize[0])
 * — vortexpipe reports the Vortex hardware cap (NUM_THREADS × NUM_WARPS)
 * there, so the same shader runs on every Vortex config without baking
 * the cap into the .comp source. N must be a multiple of the chosen
 * value; the loop below rounds down if needed. */
#define LOCAL_SIZE_MAX  64u

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

#define XM_SIZE     16u
#define XM_VERTS    9u
#define XM_STRIDE   8u
#define XM_BYTES    256u
#define XM_SENTINEL 0xDEADBEEFu

static VkShaderModule
xm_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return VK_NULL_HANDLE;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
      sm = VK_NULL_HANDLE;
   }
   free(spv);
   return sm;
}

static uint32_t
xm_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
xm_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize bytes,
          VkBufferUsageFlags usage, VkBuffer *buf, VkDeviceMemory *mem, uint32_t **map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   VkResult r = vkCreateBuffer(dev, &bci, NULL, buf);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = xm_mem_type(mp, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   if ((r = vkAllocateMemory(dev, &mai, NULL, mem)) != VK_SUCCESS ||
       (r = vkBindBufferMemory(dev, *buf, *mem, 0)) != VK_SUCCESS) {
      return r;
   }
   return vkMapMemory(dev, *mem, 0, bytes, 0, (void **)map);
}

/* The capture segment described at the top; 0 on success. */
static int
xfb_mix(VkPhysicalDevice pd, const char *vs_a_path, const char *vs_b_path, const char *fs_path)
{
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDeviceVulkan12Features f12_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &xfb_have,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   if (!xfb_have.transformFeedback || !f12_have.shaderOutputLayer ||
       !have.features.vertexPipelineStoresAndAtomics) {
      printf("FAILED (capture: transformFeedback %u, shaderOutputLayer %u, "
             "vertexPipelineStoresAndAtomics %u)\n", xfb_have.transformFeedback,
             f12_have.shaderOutputLayer, have.features.vertexPipelineStoresAndAtomics);
      return 1;
   }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf && qf == UINT32_MAX; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qf = i;
      }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("FAILED (capture: no graphics queue)\n");
      return 1;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
      .transformFeedback = VK_TRUE,
   };
   VkPhysicalDeviceVulkan12Features f12_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &xfb_on,
      .shaderOutputLayer = VK_TRUE,
   };
   VkPhysicalDeviceFeatures feats = { .vertexPipelineStoresAndAtomics = VK_TRUE };
   const char *ext = VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f12_on,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &ext, .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)
      vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT");
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb = (PFN_vkCmdBeginTransformFeedbackEXT)
      vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT");
   PFN_vkCmdEndTransformFeedbackEXT end_xfb = (PFN_vkCmdEndTransformFeedbackEXT)
      vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT");
   VkShaderModule vs_a = xm_module(dev, vs_a_path);
   VkShaderModule vs_b = xm_module(dev, vs_b_path);
   VkShaderModule fs = xm_module(dev, fs_path);
   if (!bind_xfb || !begin_xfb || !end_xfb || !vs_a || !vs_b || !fs) {
      printf("FAILED (capture: entry points or shaders missing)\n");
      return 1;
   }

   /* A two-layer colour target. */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { XM_SIZE, XM_SIZE, 1 },
      .mipLevels = 1, .arrayLayers = 2, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = imr.size,
      .memoryTypeIndex = xm_mem_type(&mp, imr.memoryTypeBits, 0),
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 2 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkAttachmentDescription att = {
      .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference attref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &attref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 1, .pAttachments = &view, .width = XM_SIZE, .height = XM_SIZE,
      .layers = 2,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkBuffer xbuf, cbuf, sbuf;
   VkDeviceMemory xmem, cmem, smem;
   uint32_t *xmap, *cmap, *smap;
   CHECK(xm_buffer(dev, &mp, XM_BYTES, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
                   &xbuf, &xmem, &xmap));
   CHECK(xm_buffer(dev, &mp, 16, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
                   &cbuf, &cmem, &cmap));
   CHECK(xm_buffer(dev, &mp, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &sbuf, &smem, &smap));
   for (uint32_t w = 0; w < XM_BYTES / 4u; w++) {
      xmap[w] = XM_SENTINEL;
   }
   memset(cmap, 0, 16);
   memset(smap, 0, 16);

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi = { sbuf, 0, 16 };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs_a, .pName = "main" },
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
   VkViewport vpt = { 0, 0, (float)XM_SIZE, (float)XM_SIZE, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { XM_SIZE, XM_SIZE } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vpt, .scissorCount = 1, .pScissors = &sc,
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
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs, .layout = pl, .renderPass = rp,
   };
   VkPipeline pipe_a, pipe_b;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe_a));
   stages[0].module = vs_b;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe_b));

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { XM_SIZE, XM_SIZE } }, .clearValueCount = 1,
      .pClearValues = &clear,
   };
   const VkDeviceSize xoff = 0, xsize = XM_BYTES, coff = 4;
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
   bind_xfb(cb, 0, 1, &xbuf, &xoff, &xsize);
   begin_xfb(cb, 0, 0, NULL, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_a);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_b);
   vkCmdDraw(cb, 3, 1, 3, 0);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_a);
   vkCmdDraw(cb, 3, 1, 6, 0);
   end_xfb(cb, 0, 1, &cbuf, &coff);
   vkCmdEndRenderPass(cb);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned bad = 0;
   for (uint32_t w = 0; w < XM_BYTES / 4u; w++) {
      const uint32_t v = w / 2u;
      const uint32_t want = v >= XM_VERTS ? XM_SENTINEL : (w & 1u) ? v * 3u + 1u : v;
      if (xmap[w] != want) {
         if (bad < 5) {
            printf("  capture word %u = 0x%08x, want 0x%08x\n", w, xmap[w], want);
         }
         bad++;
      }
   }
   if (cmap[1] != XM_VERTS * XM_STRIDE) {
      printf("  counter %u, want %u\n", cmap[1], XM_VERTS * XM_STRIDE);
      bad++;
   }
   if (smap[0] != 3u) {
      printf("  the side-effect draw ran %u vertex invocations, want 3\n", smap[0]);
      bad++;
   }
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipe_a, NULL);
   vkDestroyPipeline(dev, pipe_b, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyBuffer(dev, xbuf, NULL);
   vkFreeMemory(dev, xmem, NULL);
   vkDestroyBuffer(dev, cbuf, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkDestroyBuffer(dev, sbuf, NULL);
   vkFreeMemory(dev, smem, NULL);
   vkDestroyShaderModule(dev, vs_a, NULL);
   vkDestroyShaderModule(dev, vs_b, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   if (bad) {
      printf("FAILED (capture: %u words wrong across device/llvmpipe/device; a gap "
             "or overwrite means one side resumed from a stale filled size)\n", bad);
      return 1;
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const char *spv_path  = (argc > 1) ? argv[1] : "add1.comp.spv";
   const char *spv_big   = (argc > 2) ? argv[2] : "add1_shared.comp.spv";
   const char *vs_a_path = (argc > 3) ? argv[3] : "xfbmix_a.vert.spv";
   const char *vs_b_path = (argc > 4) ? argv[4] : "xfbmix_b.vert.spv";
   const char *fs_path   = (argc > 5) ? argv[5] : "xfbmix.frag.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-compute-smoke",
      .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   /* --- physical device ------------------------------------------- */
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* compute-capable queue family */
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   /* --- logical device + queue ------------------------------------ */
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

   /* --- storage buffer + host-visible memory ---------------------- */
   const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(float);
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t mt = UINT32_MAX;
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));

   /* seed: data[i] = i */
   float *p;
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < N; i++) p[i] = (float)i;
   vkUnmapMemory(dev, mem);

   /* --- shader module --------------------------------------------- */
   size_t spv_size = 0;
   uint32_t *spv = read_spirv(spv_path, &spv_size);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(spv);

   /* The shared-memory variant, for the dispatch that must run on llvmpipe. */
   size_t big_size_bytes = 0;
   uint32_t *spv_b = read_spirv(spv_big, &big_size_bytes);
   if (!spv_b) return 1;
   VkShaderModuleCreateInfo smci_b = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = big_size_bytes, .pCode = spv_b,
   };
   VkShaderModule sm_big;
   CHECK(vkCreateShaderModule(dev, &smci_b, NULL, &sm_big));
   free(spv_b);

   /* --- descriptor + pipeline layout ------------------------------ */
   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- pick local_size_x from device limits ---------------------- *
    * Cap the requested size by what the device advertises, then round
    * down to the largest power of two that divides N — keeps dispatch
    * geometry simple (N / local_size workgroups, every thread in
    * range). `add1.comp` declares `layout(local_size_x_id = 0)` so the
    * value flows in as Vulkan specialization constant ID 0. */
   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX
                                                    : dev_max_x;
   while (local_size > 1 && (N % local_size) != 0)
      local_size >>= 1;
   if (local_size == 0)
      local_size = 1;
   /* add1_shared.comp's workgroup, which exceeds the device cap and uses shared
    * memory, so vortexpipe refuses to re-tile it and hands it to llvmpipe. */
   const uint32_t big_size = 128u;   /* matches add1_shared.comp */
   if (N % big_size != 0) {
      printf("FAILED (N=%u is not a multiple of the oversized workgroup %u)\n",
             N, big_size);
      return 1;
   }
   printf("local_size_x=%u big=%u (device max=%u)\n",
          local_size, big_size, dev_max_x);

   /* --- compute pipeline (→ vortexpipe create_compute_state) ------ */
   VkSpecializationMapEntry sme = {
      .constantID = 0, .offset = 0, .size = sizeof(uint32_t),
   };
   VkSpecializationInfo spec_info = {
      .mapEntryCount = 1, .pMapEntries = &sme,
      .dataSize = sizeof(uint32_t), .pData = &local_size,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec_info,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   /* No specialization: add1_shared.comp fixes its own local_size, which must
    * match its shared array. */
   cpci.stage.module = sm_big;
   cpci.stage.pSpecializationInfo = NULL;
   VkPipeline pipe_big;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe_big));

   /* --- descriptor set -------------------------------------------- */
   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));

   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

   VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = bytes };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   /* --- command buffer: bind + dispatch --------------------------- */
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
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, 1, &ds, 0, NULL);
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
   };
   /* device -> llvmpipe -> device, each reading what the previous one wrote. */
   vkCmdDispatch(cb, N / local_size, 1, 1);
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        1, &mb, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_big);
   vkCmdDispatch(cb, N / big_size, 1, 1);
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        1, &mb, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdDispatch(cb, N / local_size, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back + verify ---------------------------------------- */
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   unsigned fails = 0;
   for (uint32_t i = 0; i < N; i++) {
      float want = (float)i + 3.0f;   /* three dispatches of data[i] += 1 */
      if (p[i] != want) {
         if (fails < 5)
            fprintf(stderr, "  data[%u] = %f, want %f\n", i, p[i], want);
         fails++;
      }
   }
   vkUnmapMemory(dev, mem);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe_big, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm_big, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   const int capture_failed = xfb_mix(pd, vs_a_path, vs_b_path, fs_path);
   vkDestroyInstance(inst, NULL);
   if (capture_failed)
      return 1;

   if (fails) {
      printf("FAILED (%u/%u mismatches; i+2 everywhere means the llvmpipe "
             "dispatch's write was lost -- the device copy was reused without "
             "being re-uploaded after work ran on the CPU)\n", fails, N);
      return 1;
   }
   printf("PASSED (%u elements, device/llvmpipe/device: data[i] = i + 3 -- the "
          "fallback's write was visible to the dispatch after it; one capture "
          "segment across device/llvmpipe/device draws is continuous)\n", N);
   return 0;
}
