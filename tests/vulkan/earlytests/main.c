/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Early fragment tests against a shader with side effects. Each case draws a
 * full-screen quad over a cleared D24S8 attachment, and again over a D32F_S8
 * one, with a shader that counts
 * its invocations in a storage buffer:
 *
 *   late_fail     no early_fragment_tests, depth test failing everywhere: the
 *                 shader still runs for every fragment (its store is visible),
 *                 and no colour lands;
 *   early_fail    the same with early_fragment_tests: the test runs first, so
 *                 the shader runs for no fragment;
 *   early_pass    early tests passing everywhere: every fragment runs, colour
 *                 and depth land;
 *   early_discard early tests passing, then a shader that discards: the depth
 *                 and stencil the tests wrote stay written, though no colour
 *                 does;
 *   early_stencil a stencil test failing everywhere under early tests: no
 *                 fragment runs.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     32u
#define HEIGHT    32u
#define NPIX      (WIDTH * HEIGHT)
#define FB_FORMAT VK_FORMAT_R8G8B8A8_UNORM

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* Matches the push_constant block: a float, then a vec4 at offset 16. */
struct pc_t {
   float z;
   float pad[3];
   float color[4];
};

enum { FS_LATE, FS_EARLY, FS_EARLY_DISCARD, FS_COUNT };

struct ecase {
   const char *name;
   unsigned    fs;
   float       clear_depth;
   float       z;
   bool        stencil;          /* stencil test on, against reference 1 */
   VkCompareOp stencil_op;
   uint32_t    want_count;
   bool        want_green;       /* else the black clear */
   int         want_stencil;     /* -1: not checked */
   bool        want_depth_z;     /* the stored depth is the quad's */
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
   if (out_size) *out_size = (size_t)sz;
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

static VkDeviceMemory
bind_memory(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkBuffer buf, VkImage img, VkMemoryPropertyFlags want)
{
   VkMemoryRequirements mr;
   if (buf)
      vkGetBufferMemoryRequirements(dev, buf, &mr);
   else
      vkGetImageMemoryRequirements(dev, img, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits, want);
   if (mt == UINT32_MAX)
      return VK_NULL_HANDLE;
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   if (vkAllocateMemory(dev, &mai, NULL, &mem) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   VkResult r = buf ? vkBindBufferMemory(dev, buf, mem, 0)
                    : vkBindImageMemory(dev, img, mem, 0);
   if (r != VK_SUCCESS) {
      vkFreeMemory(dev, mem, NULL);
      return VK_NULL_HANDLE;
   }
   return mem;
}

static VkBuffer
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize size,
            VkBufferUsageFlags usage, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
   };
   VkBuffer b;
   if (vkCreateBuffer(dev, &bci, NULL, &b) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   *mem = bind_memory(dev, mp, b, VK_NULL_HANDLE,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   return *mem ? b : VK_NULL_HANDLE;
}

static int
run(const char *const paths[1 + FS_COUNT], VkFormat zs_format, const char *zs_name)
{
   const bool float_depth = zs_format == VK_FORMAT_D32_SFLOAT_S8_UINT;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-earlytests", .apiVersion = VK_API_VERSION_1_1,
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
   VkPhysicalDeviceFeatures feats = { .fragmentStoresAndAtomics = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* --- attachments ------------------------------------------------ */
   VkImageCreateInfo cimci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FB_FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage cimg, zimg;
   CHECK(vkCreateImage(dev, &cimci, NULL, &cimg));
   VkDeviceMemory cmem = bind_memory(dev, &mp, VK_NULL_HANDLE, cimg, 0);
   VkImageCreateInfo zimci = cimci;
   zimci.format = zs_format;
   zimci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   CHECK(vkCreateImage(dev, &zimci, NULL, &zimg));
   VkDeviceMemory zmem = bind_memory(dev, &mp, VK_NULL_HANDLE, zimg, 0);
   if (!cmem || !zmem) { fprintf(stderr, "FAILED: image memory\n"); return 1; }
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = cimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FB_FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView cview, zview;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &cview));
   ivci.image = zimg;
   ivci.format = zs_format;
   ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &zview));

   VkAttachmentDescription att[2] = {
      { .format = FB_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = zs_format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference zref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &zref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkImageView views[2] = { cview, zview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = views,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* --- counter and readback buffers ------------------------------- */
   VkDeviceMemory smem, rmem;
   VkBuffer sbuf = host_buffer(dev, &mp, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &smem);
   const VkDeviceSize cbytes = NPIX * 4, zbytes = NPIX * 4, sbytes = NPIX;
   VkBuffer rbuf = host_buffer(dev, &mp, cbytes + zbytes + sbytes,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rmem);
   if (!sbuf || !rbuf) { fprintf(stderr, "FAILED: buffers\n"); return 1; }

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
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorBufferInfo dbi = { sbuf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkPushConstantRange pcr = {
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct pc_t),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkShaderModule mods[1 + FS_COUNT];
   for (unsigned i = 0; i < 1 + FS_COUNT; i++) {
      mods[i] = load_module(dev, paths[i]);
      if (!mods[i]) return 1;
   }

   const struct ecase cases[] = {
      { "late_fail",     FS_LATE,          0.25f, 0.5f, false, VK_COMPARE_OP_ALWAYS,
        NPIX, false, -1, false },
      { "early_fail",    FS_EARLY,         0.25f, 0.5f, false, VK_COMPARE_OP_ALWAYS,
        0,    false, -1, false },
      { "early_pass",    FS_EARLY,         1.0f,  0.5f, false, VK_COMPARE_OP_ALWAYS,
        NPIX, true,  -1, true  },
      { "early_discard", FS_EARLY_DISCARD, 1.0f,  0.5f, true,  VK_COMPARE_OP_ALWAYS,
        NPIX, false, 1,  true  },
      { "early_stencil", FS_EARLY,         1.0f,  0.5f, true,  VK_COMPARE_OP_EQUAL,
        0,    false, 0,  false },
   };
   const unsigned ncases = sizeof(cases) / sizeof(cases[0]);

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   unsigned failed = 0;
   for (unsigned ci = 0; ci < ncases; ci++) {
      const struct ecase *c = &cases[ci];
      VkPipelineShaderStageCreateInfo stages[2] = {
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mods[0], .pName = "main" },
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mods[1 + c->fs], .pName = "main" },
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
         .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
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
      const VkStencilOpState sop = {
         .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_REPLACE,
         .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = c->stencil_op,
         .compareMask = 0xff, .writeMask = 0xff, .reference = 1,
      };
      VkPipelineDepthStencilStateCreateInfo dss = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
         .depthCompareOp = VK_COMPARE_OP_LESS,
         .stencilTestEnable = c->stencil, .front = sop, .back = sop,
      };
      VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
      VkPipelineColorBlendStateCreateInfo cb = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .attachmentCount = 1, .pAttachments = &cba,
      };
      VkGraphicsPipelineCreateInfo gpci = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = &ms, .pDepthStencilState = &dss,
         .pColorBlendState = &cb, .layout = pl, .renderPass = rp, .subpass = 0,
      };
      VkPipeline pipe;
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

      uint32_t *count;
      CHECK(vkMapMemory(dev, smem, 0, 4, 0, (void **)&count));
      *count = 0;
      vkUnmapMemory(dev, smem);

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      VkClearValue clears[2] = {
         { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
         { .depthStencil = { c->clear_depth, 0 } },
      };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 2, .pClearValues = clears,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &dset, 0, NULL);
      const struct pc_t pc = { c->z, { 0, 0, 0 }, { 0.0f, 1.0f, 0.0f, 1.0f } };
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof pc, &pc);
      vkCmdDraw(cmd, 6, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &region);
      region.bufferOffset = cbytes;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vkCmdCopyImageToBuffer(cmd, zimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &region);
      region.bufferOffset = cbytes + zbytes;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      vkCmdCopyImageToBuffer(cmd, zimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbuf, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      CHECK(vkMapMemory(dev, smem, 0, 4, 0, (void **)&count));
      const uint32_t n = *count;
      vkUnmapMemory(dev, smem);
      uint8_t *rb;
      CHECK(vkMapMemory(dev, rmem, 0, cbytes + zbytes + sbytes, 0, (void **)&rb));
      /* The depth left behind: as 24-bit unorm within one step of rounding,
       * or exactly the float. */
      const float want_zf = c->want_depth_z ? c->z : c->clear_depth;
      const uint32_t want_z = (uint32_t)(want_zf * 16777215.0f + 0.5f);
      unsigned bad_colour = 0, bad_depth = 0, bad_stencil = 0;
      for (uint32_t i = 0; i < NPIX; i++) {
         const uint8_t *px = rb + (size_t)i * 4;
         const bool green = px[1] > 200 && px[0] < 50 && px[2] < 50;
         const bool black = px[0] < 50 && px[1] < 50 && px[2] < 50;
         if (c->want_green ? !green : !black)
            bad_colour++;
         if (float_depth) {
            if (((const float *)(rb + cbytes))[i] != want_zf)
               bad_depth++;
         } else {
            const uint32_t z = ((const uint32_t *)(rb + cbytes))[i] & 0xffffffu;
            if (z + 1u < want_z || z > want_z + 1u)
               bad_depth++;
         }
         if (c->want_stencil >= 0 && rb[cbytes + zbytes + i] != (uint8_t)c->want_stencil)
            bad_stencil++;
      }
      vkUnmapMemory(dev, rmem);
      const bool ok = n == c->want_count && !bad_colour && !bad_depth && !bad_stencil;
      printf("%-6s %-14s %s  invocations=%u (want %u) colour_bad=%u depth_bad=%u "
             "stencil_bad=%u\n", zs_name, c->name, ok ? "pass" : "FAIL", n, c->want_count,
             bad_colour, bad_depth, bad_stencil);
      failed += !ok;
      vkDestroyPipeline(dev, pipe, NULL);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (failed) {
      printf("FAILED (%s: %u case(s))\n", zs_name, failed);
      return 1;
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const char *const paths[1 + FS_COUNT] = {
      argc > 1 ? argv[1] : "et.vert.spv",
      argc > 2 ? argv[2] : "et_late.frag.spv",
      argc > 3 ? argv[3] : "et_early.frag.spv",
      argc > 4 ? argv[4] : "et_early_discard.frag.spv",
   };
   if (run(paths, VK_FORMAT_D24_UNORM_S8_UINT, "D24S8") ||
       run(paths, VK_FORMAT_D32_SFLOAT_S8_UINT, "D32FS8"))
      return 1;
   printf("PASSED (early fragment tests run ahead of the shader and late ones after it)\n");
   return 0;
}
