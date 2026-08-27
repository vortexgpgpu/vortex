/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A multisample attachment read directly rather than resolved.
 *
 * The device renders a multisample pass into a resident plane holding one
 * colour per sample, and that plane reaches the application through the
 * pass-end resolve. A pass that ends without one has nowhere to put its colour:
 * writing the samples back would need a transfer that carries a sample index,
 * which the read path does not have. The driver says so and leaves the
 * attachment at its clear.
 *
 * This is the only way an application can observe that. A multisample image
 * cannot be the source of a copy, so the attachment is read by sampling it as a
 * sampler2DMS in a second pass and writing sample 0 into a single-sample target
 * that can be copied out.
 *
 * WHAT IS ASSERTED: the fill. An application that renders red and reads the
 * attachment it rendered into must get red. Keeping the clear is not a lesser
 * outcome to be tolerated -- it is a wrong image handed to the application, and
 * a test that accepted it would be recording the defect as the specification.
 *
 * THIS CASE FAILS TODAY, and that is what it is for. The device serves a
 * multisample pass only through the pass-end resolve, and by the time the read
 * is noticed -- at sync-out, after the pass -- there is nowhere left to put the
 * colour. The driver says so and leaves the clear.
 *
 * Standing aside up front is NOT available as the fix, though it reads like the
 * obvious one. It would need the pass to know its attachment is one the
 * application will sample, and the driver cannot tell: lavapipe gives every
 * multisample colour attachment PIPE_BIND_SAMPLER_VIEW whether or not the
 * application asked for VK_IMAGE_USAGE_SAMPLED_BIT, because it needs a sampler
 * view for its own resolve blits. Refusing on that bind flag refuses every
 * multisample pass -- it was tried, and it took the msaa case's device path away
 * with it. The property that separates this case from msaa is the Vulkan usage,
 * and Gallium has already folded it away by the time the driver sees the
 * resource. So the fix is either that usage plumbed through to the driver, or
 * the samples written back to the attachment when a pass ends without a
 * resolve -- the transfer carrying a sample index that the read path lacks.
 *
 * The uniform scene is what makes the failure precise rather than approximate:
 * every sample of every pixel is covered, so a correct read is one colour over
 * the whole target, and a mixture would mean a half-written plane read as
 * though it were single-sample -- a plausible-looking wrong image, reported
 * separately from the clear so the two are never confused.
 *
 * STRICT := 0 because standing aside to llvmpipe is a correct way to serve this
 * pass, and the assertion is the pixel rather than where it was produced.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   32u
#define HEIGHT  32u
#define FORMAT  VK_FORMAT_R8G8B8A8_UNORM
#define SAMPLES VK_SAMPLE_COUNT_4_BIT

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
   if (r != VK_SUCCESS) return r;
   return vkBindImageMemory(dev, img, *out, 0);
}

int
main(int argc, char **argv)
{
   const char *vs_path    = (argc > 1) ? argv[1] : "msaa_read.vert.spv";
   const char *fill_path  = (argc > 2) ? argv[2] : "msaa_fill.frag.spv";
   const char *fetch_path = (argc > 3) ? argv[3] : "msaa_fetch.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa_read",
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
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   if (!(props.limits.framebufferColorSampleCounts &
         (VkSampleCountFlags)SAMPLES)) {
      printf("FAILED (device does not advertise 4x colour samples: 0x%x)\n",
             (unsigned)props.limits.framebufferColorSampleCounts);
      return 1;
   }
   /* Sampling a multisample image is what the read is performed through, so a
    * device that cannot do it cannot run this case at all. */
   if (!(props.limits.sampledImageColorSampleCounts &
         (VkSampleCountFlags)SAMPLES)) {
      printf("FAILED (device cannot sample 4x colour images: 0x%x)\n",
             (unsigned)props.limits.sampledImageColorSampleCounts);
      return 1;
   }

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

   /* --- the multisample attachment, kept rather than resolved ------ */
   VkImageCreateInfo msci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = SAMPLES, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage msimg;
   CHECK(vkCreateImage(dev, &msci, NULL, &msimg));
   VkDeviceMemory msmem;
   CHECK(alloc_image(dev, &mp, msimg, &msmem));

   VkImageViewCreateInfo msvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = msimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView msview;
   CHECK(vkCreateImageView(dev, &msvci, NULL, &msview));

   /* --- the single-sample target the fetch pass writes -------------- */
   VkImageCreateInfo ssci = msci;
   ssci.samples = VK_SAMPLE_COUNT_1_BIT;
   ssci.usage   = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   VkImage ssimg;
   CHECK(vkCreateImage(dev, &ssci, NULL, &ssimg));
   VkDeviceMemory ssmem;
   CHECK(alloc_image(dev, &mp, ssimg, &ssmem));

   VkImageViewCreateInfo ssvci = msvci;
   ssvci.image = ssimg;
   VkImageView ssview;
   CHECK(vkCreateImageView(dev, &ssvci, NULL, &ssview));

   /* --- pass 1: render into the multisample attachment, no resolve -- */
   VkAttachmentDescription att1 = {
      .format = FORMAT, .samples = SAMPLES,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkAttachmentReference ref1 = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub1 = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref1,
   };
   VkRenderPassCreateInfo rpci1 = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att1,
      .subpassCount = 1, .pSubpasses = &sub1,
   };
   VkRenderPass rp1;
   CHECK(vkCreateRenderPass(dev, &rpci1, NULL, &rp1));
   VkFramebufferCreateInfo fbci1 = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp1, .attachmentCount = 1, .pAttachments = &msview,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb1;
   CHECK(vkCreateFramebuffer(dev, &fbci1, NULL, &fb1));

   /* --- pass 2: fetch sample 0 into the single-sample target ------- */
   VkAttachmentDescription att2 = {
      .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference ref2 = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub2 = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref2,
   };
   VkRenderPassCreateInfo rpci2 = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att2,
      .subpassCount = 1, .pSubpasses = &sub2,
   };
   VkRenderPass rp2;
   CHECK(vkCreateRenderPass(dev, &rpci2, NULL, &rp2));
   VkFramebufferCreateInfo fbci2 = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp2, .attachmentCount = 1, .pAttachments = &ssview,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb2;
   CHECK(vkCreateFramebuffer(dev, &fbci2, NULL, &fb2));

   const VkDeviceSize rb_bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = rb_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

   /* --- shaders, layouts, pipelines -------------------------------- */
   VkShaderModule vs    = load_module(dev, vs_path);
   VkShaderModule fill  = load_module(dev, fill_path);
   VkShaderModule fetch = load_module(dev, fetch_path);
   if (!vs || !fill || !fetch) return 1;

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci0 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl_fill;
   CHECK(vkCreatePipelineLayout(dev, &plci0, NULL, &pl_fill));

   VkPipelineLayoutCreateInfo plci1 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl_fetch;
   CHECK(vkCreatePipelineLayout(dev, &plci1, NULL, &pl_fetch));

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
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };

   VkPipelineMultisampleStateCreateInfo ms_multi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = SAMPLES,
   };
   VkPipelineMultisampleStateCreateInfo ms_single = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };

   VkPipelineShaderStageCreateInfo st_fill[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fill, .pName = "main" },
   };
   VkGraphicsPipelineCreateInfo gp_fill = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st_fill,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms_multi, .pColorBlendState = &cb,
      .layout = pl_fill, .renderPass = rp1, .subpass = 0,
   };
   VkPipeline pipe_fill;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_fill, NULL,
                                   &pipe_fill));

   VkPipelineShaderStageCreateInfo st_fetch[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fetch, .pName = "main" },
   };
   VkGraphicsPipelineCreateInfo gp_fetch = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st_fetch,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms_single, .pColorBlendState = &cb,
      .layout = pl_fetch, .renderPass = rp2, .subpass = 0,
   };
   VkPipeline pipe_fetch;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_fetch, NULL,
                                   &pipe_fetch));

   /* A multisample image is fetched by integer coordinate, never filtered, so
    * the sampler's filter and address modes are not consulted. */
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

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
   };
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
   VkDescriptorImageInfo dii = {
      .sampler = sampler, .imageView = msview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
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

   /* --- both passes, then read the single-sample target ------------ */
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   /* Green clear, red fill: the two legal answers, two channels apart. */
   VkClearValue clear1 = { .color = { .float32 = { 0.0f, 1.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi1 = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp1, .framebuffer = fb1,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear1,
   };
   vkCmdBeginRenderPass(cmd, &rpbi1, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_fill);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   /* Blue clear on the fetch target: a colour neither legal answer can be, so
    * a target the fetch pass never wrote is a failure and not a silent pass. */
   VkClearValue clear2 = { .color = { .float32 = { 0.0f, 0.0f, 1.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi2 = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp2, .framebuffer = fb2,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear2,
   };
   vkCmdBeginRenderPass(cmd, &rpbi2, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_fetch);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_fetch,
                           0, 1, &dset, 0, NULL);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, ssimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, rb_bytes, 0, (void **)&px));
   unsigned as_clear = 0, as_fill = 0, other = 0;
   uint8_t sample_other[3] = { 0, 0, 0 };
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      const uint8_t *p = px + (size_t)i * 4;
      if (p[0] == 0 && p[1] == 255 && p[2] == 0) {
         as_clear++;
      } else if (p[0] == 255 && p[1] == 0 && p[2] == 0) {
         as_fill++;
      } else {
         if (other == 0) {
            sample_other[0] = p[0]; sample_other[1] = p[1]; sample_other[2] = p[2];
         }
         other++;
      }
   }
   vkUnmapMemory(dev, bmem);

   const uint32_t total = WIDTH * HEIGHT;
   const bool ok = (as_fill == total);
   printf("msaa_read: %s  fill=%u clear=%u other=%u of %u",
          ok ? "pass" : "FAIL", as_fill, as_clear, other, total);
   if (other) {
      printf("  first other=(%u,%u,%u)",
             sample_other[0], sample_other[1], sample_other[2]);
   }
   printf("\n");
   if (!ok) {
      printf("outcome: %s\n",
             as_clear == total
                ? "the attachment kept its clear -- the render never reached it"
                : "neither the fill nor the clear -- see the counts above");
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipeline(dev, pipe_fetch, NULL);
   vkDestroyPipeline(dev, pipe_fill, NULL);
   vkDestroyPipelineLayout(dev, pl_fetch, NULL);
   vkDestroyPipelineLayout(dev, pl_fill, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, fetch, NULL);
   vkDestroyShaderModule(dev, fill, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyFramebuffer(dev, fb2, NULL);
   vkDestroyFramebuffer(dev, fb1, NULL);
   vkDestroyRenderPass(dev, rp2, NULL);
   vkDestroyRenderPass(dev, rp1, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyImageView(dev, ssview, NULL);
   vkFreeMemory(dev, ssmem, NULL);
   vkDestroyImage(dev, ssimg, NULL);
   vkDestroyImageView(dev, msview, NULL);
   vkFreeMemory(dev, msmem, NULL);
   vkDestroyImage(dev, msimg, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!ok) {
      printf("FAILED\n");
      return 1;
   }
   printf("PASSED (one colour over the whole target, and a legal one)\n");
   return 0;
}
