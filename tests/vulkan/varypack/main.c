/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Varyings laid out by location.
 *
 * A location holds four 32-bit components. Array elements, matrix columns and
 * struct members each start a location of their own, and several variables
 * may share a location by component (varypack.vert): two float arrays and a
 * vec2 array in components 0, 1 and 2-3 of locations 3-4, a vec3 and a float
 * in location 7, a struct whose vec3 member takes the location after its float.
 * A vertex record that packed array elements or members back to back would put
 * a[1] where b[0] lives, and the struct's vec3 inside its float's location.
 *
 * One triangle is drawn under transform feedback. Vertex v writes 100 * v + k
 * as the k-th of twenty captured floats, so the capture of its three vertices
 * is known exactly, and the fragment stage stores the provoking vertex's
 * twenty values -- 1..20 -- which must agree with what was captured for it.
 * Bytes past the three vertices must keep their sentinel.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE      16u
#define FORMAT    VK_FORMAT_R8G8B8A8_UNORM
#define NVALUES   20u
#define STRIDE    (NVALUES * 4u)
#define XFB_BYTES 512u
#define SENTINEL  0xDEADBEEFu

#define CHECK(x) do {                                                    \
   VkResult _r = (x);                                                    \
   if (_r != VK_SUCCESS) {                                               \
      printf("FAILED (varypack: %s -> VkResult %d)\n", #x, (int)_r);     \
      return 1;                                                          \
   }                                                                     \
} while (0)

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      return VK_NULL_HANDLE;
   }
   fseek(f, 0, SEEK_END);
   const long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *spv = malloc((size_t)sz);
   const bool ok = spv && fread(spv, 1, (size_t)sz, f) == (size_t)sz;
   fclose(f);
   VkShaderModule sm = VK_NULL_HANDLE;
   if (ok) {
      VkShaderModuleCreateInfo smci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = (size_t)sz, .pCode = spv,
      };
      if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
         sm = VK_NULL_HANDLE;
      }
   }
   free(spv);
   return sm;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize bytes,
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
      .memoryTypeIndex = find_mem(mp, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   if ((r = vkAllocateMemory(dev, &mai, NULL, mem)) != VK_SUCCESS) {
      return r;
   }
   if ((r = vkBindBufferMemory(dev, *buf, *mem, 0)) != VK_SUCCESS) {
      return r;
   }
   return vkMapMemory(dev, *mem, 0, bytes, 0, (void **)map);
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "varypack.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "varypack.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-varypack", .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) {
      printf("FAILED (varypack: no physical device)\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &xfb_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   if (!xfb_have.transformFeedback || !have.features.fragmentStoresAndAtomics) {
      printf("FAILED (varypack: transformFeedback %u, fragmentStoresAndAtomics %u)\n",
             xfb_have.transformFeedback, have.features.fragmentStoresAndAtomics);
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
      printf("FAILED (varypack: no graphics queue)\n");
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
   VkPhysicalDeviceFeatures feats = { .fragmentStoresAndAtomics = VK_TRUE };
   const char *ext = VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &xfb_on,
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
   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!bind_xfb || !begin_xfb || !end_xfb || !vs || !fs) {
      printf("FAILED (varypack: entry points or shaders missing)\n");
      return 1;
   }

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = FORMAT, .extent = { SIZE, SIZE, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = imr.size,
      .memoryTypeIndex = find_mem(&mp, imr.memoryTypeBits, 0),
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkAttachmentDescription att = {
      .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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
      .attachmentCount = 1, .pAttachments = &view, .width = SIZE, .height = SIZE, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkBuffer xbuf, obuf;
   VkDeviceMemory xmem, omem;
   uint32_t *xmap, *omap;
   CHECK(host_buffer(dev, &mp, XFB_BYTES, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
                     &xbuf, &xmem, &xmap));
   CHECK(host_buffer(dev, &mp, NVALUES * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     &obuf, &omem, &omap));
   for (uint32_t w = 0; w < XFB_BYTES / 4u; w++) {
      xmap[w] = SENTINEL;
   }
   memset(omap, 0, NVALUES * 4u);

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
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
   VkDescriptorBufferInfo dbi = { obuf, 0, NVALUES * 4u };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

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
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport vpt = { 0, 0, (float)SIZE, (float)SIZE, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { SIZE, SIZE } };
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
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

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
      .renderArea = { { 0, 0 }, { SIZE, SIZE } }, .clearValueCount = 1, .pClearValues = &clear,
   };
   const VkDeviceSize xoff = 0, xsize = XFB_BYTES;
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
   bind_xfb(cb, 0, 1, &xbuf, &xoff, &xsize);
   begin_xfb(cb, 0, 0, NULL, NULL);
   vkCmdDraw(cb, 3, 1, 0, 0);
   end_xfb(cb, 0, 0, NULL, NULL);
   vkCmdEndRenderPass(cb);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned bad_capture = 0, bad_fs = 0;
   for (uint32_t w = 0; w < XFB_BYTES / 4u; w++) {
      const uint32_t vtx = w / NVALUES, k = w % NVALUES;
      float want_f = 100.0f * (float)vtx + (float)(k + 1u);
      uint32_t want;
      memcpy(&want, &want_f, 4);
      if (vtx >= 3u) {
         want = SENTINEL;
      }
      if (xmap[w] != want) {
         if (bad_capture < 4) {
            printf("  capture: vertex %u value %u is 0x%08x, want 0x%08x\n", vtx, k + 1u,
                   xmap[w], want);
         }
         bad_capture++;
      }
   }
   for (uint32_t k = 0; k < NVALUES; k++) {
      float got;
      memcpy(&got, &omap[k], 4);
      if (got != (float)(k + 1u)) {
         if (bad_fs < 4) {
            printf("  fragment: value %u read %g, want %u\n", k + 1u, got, k + 1u);
         }
         bad_fs++;
      }
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
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
   vkDestroyBuffer(dev, obuf, NULL);
   vkFreeMemory(dev, omem, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (bad_capture || bad_fs) {
      printf("FAILED (varypack: %u captured and %u fragment-read values wrong; varyings "
             "sharing a location overwrote each other or were read from the wrong "
             "place)\n", bad_capture, bad_fs);
      return 1;
   }
   printf("PASSED (varypack: %u values per vertex captured and read back by location)\n",
          NVALUES);
   return 0;
}
