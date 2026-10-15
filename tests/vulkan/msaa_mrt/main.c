/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Per-sample depth testing across two colour attachments of a 4x pass.
 *
 * A flat quad at depth 0.5 is drawn first, then a sloped triangle whose plane
 * crosses 0.5 on a diagonal through the 16x16 target, with a LESS test into a
 * 4x D32F attachment cleared to 1.0. Each primitive writes its own colour into
 * a 4x RGBA8 attachment and its own id into a 4x R32F attachment. Both are
 * then copied out per sample through texelFetch on a sampler2DMS.
 *
 * Every sample must hold the primitive that is nearer AT THAT SAMPLE's
 * position, computed here from the plane equations, and the two attachments
 * must agree on which one it was. Along the crossing the decision differs
 * between the samples of one pixel, which a test at the pixel centre, or one
 * depth per pixel, cannot reproduce.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH         16u
#define HEIGHT        16u
#define SAMPLES       4u
#define COLOR_FORMAT  VK_FORMAT_R8G8B8A8_UNORM
#define ID_FORMAT     VK_FORMAT_R32_SFLOAT
#define DEPTH_FORMAT  VK_FORMAT_D32_SFLOAT

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* Standard 4x positions in 1/8 pixel: sample k sits at (x, y) / 8. */
static const int k_pos8[SAMPLES][2] = { { 3, 1 }, { 7, 3 }, { 1, 5 }, { 5, 7 } };

/* The sloped plane is z = Z0 + x/64 + y/128 with Z0 = 0.5 - 192.5/1024, so in
 * units of 1/8 pixel (x = a/8, y = b/8) it is nearer than 0.5 exactly when
 * 2a + b <= 192 -- integer arithmetic, no rounding to decide. */
static bool
sloped_wins(uint32_t px, uint32_t py, uint32_t k)
{
   const int a = (int)px * 8 + k_pos8[k][0];
   const int b = (int)py * 8 + k_pos8[k][1];
   return 2 * a + b <= 192;
}

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
   *out_size = (size_t)sz;
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
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
bind_memory(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            const VkMemoryRequirements *mr, VkMemoryPropertyFlags want,
            VkDeviceMemory *out)
{
   uint32_t mt = find_mem(mp, mr->memoryTypeBits, want);
   if (mt == UINT32_MAX) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr->size, .memoryTypeIndex = mt,
   };
   return vkAllocateMemory(dev, &mai, NULL, out);
}

static VkResult
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
           VkFormat fmt, VkImageAspectFlags aspect, VkSampleCountFlagBits samples,
           VkImageUsageFlags usage, VkImage *img, VkDeviceMemory *mem,
           VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkResult r = vkCreateImage(dev, &ici, NULL, img);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, *img, &mr);
   r = bind_memory(dev, mp, &mr, 0, mem);
   if (r == VK_SUCCESS) {
      r = vkBindImageMemory(dev, *img, *mem, 0);
   }
   if (r != VK_SUCCESS) {
      return r;
   }
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = { .aspectMask = aspect, .levelCount = 1, .layerCount = 1 },
   };
   return vkCreateImageView(dev, &vci, NULL, view);
}

static VkResult
make_pipeline(VkDevice dev, VkShaderModule vs, VkShaderModule fs,
              VkRenderPass rp, VkPipelineLayout layout,
              VkSampleCountFlagBits samples, uint32_t ncolor, bool depth,
              VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo st[2] = {
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
      .rasterizationSamples = samples,
   };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth, .depthWriteEnable = depth,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState cba[2] = {
      { .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT },
      { .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT },
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = ncolor, .pAttachments = cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = depth ? &ds : NULL,
      .pColorBlendState = &cb,
      .layout = layout, .renderPass = rp, .subpass = 0,
   };
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static void
memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage,
               VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
               VkAccessFlags dst_access)
{
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = src_access, .dstAccessMask = dst_access,
   };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* Which primitive a stored sample shows: 0 the quad, 1 the triangle, -1 the
 * clear, -2 anything else. */
static int
color_prim(const float *c)
{
   const bool r = c[0] == 1.0f, g = c[1] == 1.0f;
   const bool r0 = c[0] == 0.0f, g0 = c[1] == 0.0f, b0 = c[2] == 0.0f;
   if (c[3] != 1.0f) {
      return -2;
   }
   if (r && g0 && b0) {
      return 0;
   }
   if (r0 && g && b0) {
      return 1;
   }
   if (r0 && g0 && c[2] == 1.0f) {
      return -1;
   }
   return -2;
}

static int
id_prim(float v)
{
   return v == 1.0f ? 0 : v == 2.0f ? 1 : v == 0.0f ? -1 : -2;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_mrt",
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
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(pd, &have);
   if (!(props.limits.framebufferColorSampleCounts & SAMPLES) ||
       !(props.limits.framebufferDepthSampleCounts & SAMPLES) ||
       !(props.limits.sampledImageColorSampleCounts & SAMPLES) ||
       !have.fragmentStoresAndAtomics) {
      printf("FAILED (device lacks 4x colour/depth/sampled counts or fragment "
             "stores: color=0x%x depth=0x%x sampled=0x%x stores=%u)\n",
             (unsigned)props.limits.framebufferColorSampleCounts,
             (unsigned)props.limits.framebufferDepthSampleCounts,
             (unsigned)props.limits.sampledImageColorSampleCounts,
             (unsigned)have.fragmentStoresAndAtomics);
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
   VkPhysicalDeviceFeatures want = { .fragmentStoresAndAtomics = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   const VkSampleCountFlagBits S = (VkSampleCountFlagBits)SAMPLES;
   VkImage cimg, iimg, dimg, ssimg;
   VkDeviceMemory cmem, imem, dmem, ssmem;
   VkImageView cview, iview, dview, ssview;
   CHECK(make_image(dev, &mp, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT, S,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    &cimg, &cmem, &cview));
   CHECK(make_image(dev, &mp, ID_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT, S,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    &iimg, &imem, &iview));
   CHECK(make_image(dev, &mp, DEPTH_FORMAT, VK_IMAGE_ASPECT_DEPTH_BIT, S,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &dimg, &dmem, &dview));
   CHECK(make_image(dev, &mp, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    &ssimg, &ssmem, &ssview));

   /* --- the 4x pass: two colour attachments and depth ---------------- */
   VkAttachmentDescription atts[3] = {
      { .format = COLOR_FORMAT, .samples = S,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { .format = ID_FORMAT, .samples = S,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { .format = DEPTH_FORMAT, .samples = S,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference crefs[2] = {
      { .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
      { .attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference dref = {
      .attachment = 2, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 2, .pColorAttachments = crefs,
      .pDepthStencilAttachment = &dref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 3, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp_draw;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp_draw));
   VkImageView draw_views[3] = { cview, iview, dview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp_draw, .attachmentCount = 3, .pAttachments = draw_views,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb_draw;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_draw));

   /* --- the single-sample read pass --------------------------------- */
   VkAttachmentDescription ratt = {
      .format = COLOR_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription rsub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &crefs[0],
   };
   VkRenderPassCreateInfo rrpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ratt,
      .subpassCount = 1, .pSubpasses = &rsub,
   };
   VkRenderPass rp_read;
   CHECK(vkCreateRenderPass(dev, &rrpci, NULL, &rp_read));
   fbci.renderPass = rp_read;
   fbci.attachmentCount = 1;
   fbci.pAttachments = &ssview;
   VkFramebuffer fb_read;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb_read));

   /* Two vec4 per sample: the colour, then the id. */
   const VkDeviceSize sbytes = (VkDeviceSize)WIDTH * HEIGHT * SAMPLES * 2 * 16;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sbytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer sbuf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &sbuf));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, sbuf, &bmr);
   VkDeviceMemory sbmem;
   CHECK(bind_memory(dev, &mp, &bmr,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sbmem));
   CHECK(vkBindBufferMemory(dev, sbuf, sbmem, 0));
   void *smap;
   CHECK(vkMapMemory(dev, sbmem, 0, sbytes, 0, &smap));
   const float *samples = smap;

   VkShaderModule vs = load_module(dev, "mrt.vert.spv");
   VkShaderModule fs = load_module(dev, "mrt.frag.spv");
   VkShaderModule read_vs = load_module(dev, "mrt_quad.vert.spv");
   VkShaderModule read_fs = load_module(dev, "mrt_read.frag.spv");
   if (!vs || !fs || !read_vs || !read_fs) {
      return 1;
   }

   VkDescriptorSetLayoutBinding dslb[3] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl_draw, pl_read;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl_draw));
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 4,
   };
   plci.setLayoutCount = 1;
   plci.pSetLayouts = &dsl;
   plci.pushConstantRangeCount = 1;
   plci.pPushConstantRanges = &pcr;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl_read));

   VkPipeline pipe_draw, pipe_read;
   CHECK(make_pipeline(dev, vs, fs, rp_draw, pl_draw, S, 2, true, &pipe_draw));
   CHECK(make_pipeline(dev, read_vs, read_fs, rp_read, pl_read,
                       VK_SAMPLE_COUNT_1_BIT, 1, false, &pipe_read));

   /* A multisample image is fetched by integer coordinate, never filtered. */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 2 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorImageInfo dii[2] = {
      { .sampler = sampler, .imageView = cview,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { .sampler = sampler, .imageView = iview,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
   };
   VkDescriptorBufferInfo dbi = { .buffer = sbuf, .offset = 0, .range = sbytes };
   VkWriteDescriptorSet wds[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
        .dstBinding = 2, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

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

   VkClearValue clears[3] = {
      { .color = { .float32 = { 0.0f, 0.0f, 1.0f, 1.0f } } },
      { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
      { .depthStencil = { 1.0f, 0 } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp_draw, .framebuffer = fb_draw,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 3, .pClearValues = clears,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_draw);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdDraw(cmd, 3, 1, 6, 0);
   vkCmdEndRenderPass(cmd);

   memory_barrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

   rpbi.renderPass = rp_read;
   rpbi.framebuffer = fb_read;
   rpbi.clearValueCount = 1;
   rpbi.pClearValues = &clears[1];
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_read);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_read, 0, 1,
                           &dset, 0, NULL);
   int32_t width = (int32_t)WIDTH;
   vkCmdPushConstants(cmd, pl_read, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(width),
                      &width);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   memory_barrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned wrong = 0, disagree = 0, mixed = 0, sloped = 0;
   int printed = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         unsigned wins = 0;
         bool pixel_ok = true;
         for (uint32_t k = 0; k < SAMPLES; k++) {
            const float *s = samples + ((size_t)(y * WIDTH + x) * SAMPLES + k) * 8;
            const int want = sloped_wins(x, y, k) ? 1 : 0;
            const int got_c = color_prim(s);
            const int got_i = id_prim(s[4]);
            wins += (unsigned)want;
            if (got_c != got_i) {
               disagree++;
               pixel_ok = false;
            }
            if (got_c != want || got_i != want) {
               wrong++;
               pixel_ok = false;
            }
         }
         sloped += wins;
         if (wins != 0 && wins != SAMPLES) {
            mixed++;
         }
         if (!pixel_ok && printed < 6) {
            printed++;
            printf("  (%u,%u):", x, y);
            for (uint32_t k = 0; k < SAMPLES; k++) {
               const float *s = samples + ((size_t)(y * WIDTH + x) * SAMPLES + k) * 8;
               printf(" s%u want=%d color=(%.2f,%.2f,%.2f,%.2f) id=%.2f", k,
                      sloped_wins(x, y, k) ? 1 : 0, s[0], s[1], s[2], s[3], s[4]);
            }
            printf("\n");
         }
      }
   }
   const unsigned total = WIDTH * HEIGHT * SAMPLES;
   /* The expectation itself must exercise the per-sample decision: a crossing
    * that missed every pixel's interior would pass a per-pixel depth test. */
   const bool scene_ok = mixed > 0 && sloped > 0 && sloped < total;
   printf("msaa_mrt: %s  wrong=%u disagree=%u of %u samples; mixed pixels=%u, "
          "triangle samples=%u\n", (wrong || disagree || !scene_ok) ? "FAIL" : "pass",
          wrong, disagree, total, mixed, sloped);

   vkUnmapMemory(dev, sbmem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipeline(dev, pipe_read, NULL);
   vkDestroyPipeline(dev, pipe_draw, NULL);
   vkDestroyPipelineLayout(dev, pl_read, NULL);
   vkDestroyPipelineLayout(dev, pl_draw, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, read_fs, NULL);
   vkDestroyShaderModule(dev, read_vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkFreeMemory(dev, sbmem, NULL);
   vkDestroyBuffer(dev, sbuf, NULL);
   vkDestroyFramebuffer(dev, fb_read, NULL);
   vkDestroyFramebuffer(dev, fb_draw, NULL);
   vkDestroyRenderPass(dev, rp_read, NULL);
   vkDestroyRenderPass(dev, rp_draw, NULL);
   vkDestroyImageView(dev, ssview, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImageView(dev, iview, NULL);
   vkDestroyImageView(dev, cview, NULL);
   vkFreeMemory(dev, ssmem, NULL);
   vkFreeMemory(dev, dmem, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkDestroyImage(dev, ssimg, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkDestroyImage(dev, iimg, NULL);
   vkDestroyImage(dev, cimg, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (wrong || disagree || !scene_ok) {
      printf("FAILED (%u samples show the wrong primitive, %u where the "
             "attachments disagree)\n", wrong, disagree);
      return 1;
   }
   printf("PASSED (per-sample depth decides both attachments, %u pixels split "
          "between the primitives)\n", mixed);
   return 0;
}
