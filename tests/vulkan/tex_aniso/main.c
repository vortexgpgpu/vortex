/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Anisotropic filtering for the vortexpipe driver.
 *
 * A 64x64 texture whose level 0 alternates black and white columns and whose
 * every level above is solid red, drawn with texture coordinates that stretch
 * each pixel's footprint over 4 texels along u and 1 along v. The answer does
 * not depend on how an implementation places its probes:
 *
 *   isotropic (maxAnisotropy 1)  -- the level of the longer axis, 2: red
 *   maxAnisotropy 16             -- 4 probes one texel apart at level 0, whose
 *                                   blends of the alternating columns average to
 *                                   grey at any phase
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   32u
#define HEIGHT  32u
#define FB_FORMAT VK_FORMAT_R8G8B8A8_UNORM


#define TEX_N    64u
#define LEVELS   7u

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


/* The mip chain, tightly packed level after level. */
static uint8_t chain[(TEX_N * TEX_N * 4 * 4) / 3 + 64];

static VkDeviceSize
fill_chain(VkDeviceSize offs[LEVELS])
{
   VkDeviceSize off = 0;
   for (uint32_t l = 0; l < LEVELS; l++) {
      const uint32_t n = TEX_N >> l;
      offs[l] = off;
      for (uint32_t i = 0; i < n * n; i++) {
         uint8_t *t = chain + off + (VkDeviceSize)i * 4;
         const uint8_t c = (l == 0 && ((i % n) & 1u)) ? 255 : 0;
         t[0] = l ? 255 : c;
         t[1] = l ? 0 : c;
         t[2] = l ? 0 : c;
         t[3] = 255;
      }
      off += (VkDeviceSize)n * n * 4;
   }
   return off;
}

int
main(int argc, char **argv)
{
   VkDeviceSize level_off[LEVELS];
   const VkDeviceSize chain_bytes = fill_chain(level_off);

   const char *vs_path = (argc > 1) ? argv[1] : "tex_aniso.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "tex_aniso.frag.spv";

   /* --- instance / device / queue --------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-tex_aniso",
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
   VkPhysicalDeviceFeatures feats = { .samplerAnisotropy = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* --- render target / pass / readback --------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FB_FORMAT,
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
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return 1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FB_FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = FB_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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

   /* --- the mipmapped texture ------------------------------------- */
   VkImageCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { TEX_N, TEX_N, 1 }, .mipLevels = LEVELS, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage tex;
   CHECK(vkCreateImage(dev, &tci, NULL, &tex));
   VkMemoryRequirements tmr;
   vkGetImageMemoryRequirements(dev, tex, &tmr);
   uint32_t tmt = find_mem(&mp, tmr.memoryTypeBits, 0);
   if (tmt == UINT32_MAX) { fprintf(stderr, "FAILED: no texture memory\n"); return 1; }
   VkMemoryAllocateInfo tmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = tmr.size, .memoryTypeIndex = tmt,
   };
   VkDeviceMemory tmem;
   CHECK(vkAllocateMemory(dev, &tmai, NULL, &tmem));
   CHECK(vkBindImageMemory(dev, tex, tmem, 0));
   VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = tex, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 },
   };
   VkImageView texview;
   CHECK(vkCreateImageView(dev, &tvci, NULL, &texview));

   VkBufferCreateInfo sbci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = chain_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
   };
   VkBuffer sbuf;
   CHECK(vkCreateBuffer(dev, &sbci, NULL, &sbuf));
   VkMemoryRequirements smr;
   vkGetBufferMemoryRequirements(dev, sbuf, &smr);
   uint32_t smt = find_mem(&mp, smr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (smt == UINT32_MAX) { fprintf(stderr, "FAILED: no staging memory\n"); return 1; }
   VkMemoryAllocateInfo smai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = smr.size, .memoryTypeIndex = smt,
   };
   VkDeviceMemory smem;
   CHECK(vkAllocateMemory(dev, &smai, NULL, &smem));
   CHECK(vkBindBufferMemory(dev, sbuf, smem, 0));
   void *sp;
   CHECK(vkMapMemory(dev, smem, 0, chain_bytes, 0, &sp));
   memcpy(sp, chain, chain_bytes);
   vkUnmapMemory(dev, smem);

   /* --- pipeline --------------------------------------------------- */
   VkShaderModule vs = load_module(dev, vs_path);
   if (!vs) return 1;
   VkShaderModule fs = load_module(dev, fs_path);
   if (!fs) return 1;

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

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

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

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 2,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));

   const char *case_name[2] = { "isotropic", "anisotropic x16" };
   const float max_aniso[2] = { 1.0f, 16.0f };
   unsigned failed = 0;
   for (uint32_t c = 0; c < 2; c++) {
      VkSamplerCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
         .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
         .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
         .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
         .anisotropyEnable = max_aniso[c] > 1.0f,
         .maxAnisotropy = max_aniso[c],
         .maxLod = (float)LEVELS,
      };
      VkSampler sampler;
      CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
      };
      VkDescriptorSet dset;
      CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
      VkDescriptorImageInfo dii = {
         .sampler = sampler, .imageView = texview,
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      VkWriteDescriptorSet wds = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &dii,
      };
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      if (c == 0) {
         VkImageMemoryBarrier to_dst = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = tex,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 },
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         };
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_dst);
         VkBufferImageCopy copies[LEVELS];
         for (uint32_t l = 0; l < LEVELS; l++) {
            copies[l] = (VkBufferImageCopy){
               .bufferOffset = level_off[l],
               .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1 },
               .imageExtent = { TEX_N >> l, TEX_N >> l, 1 },
            };
         }
         vkCmdCopyBufferToImage(cmd, sbuf, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                LEVELS, copies);
         VkImageMemoryBarrier to_read = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = tex,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 },
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         };
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                              &to_read);
      }
      VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 1, .pClearValues = &clear,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &dset, 0, NULL);
      vkCmdDraw(cmd, 6, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      /* Count the pixels that read as the expected answer: red for the
       * isotropic sampler, grey (all channels near 128) for the anisotropic one. */
      uint8_t *px;
      CHECK(vkMapMemory(dev, bmem, 0, rb_bytes, 0, (void **)&px));
      uint32_t good = 0;
      for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
         const uint8_t *p = px + (size_t)i * 4;
         const bool red = p[0] > 200 && p[1] < 50 && p[2] < 50;
         const bool grey = abs((int)p[0] - 128) < 20 && abs((int)p[1] - 128) < 20 &&
                           abs((int)p[2] - 128) < 20;
         good += (c == 0) ? red : grey;
      }
      vkUnmapMemory(dev, bmem);
      const bool ok = good >= WIDTH * HEIGHT * 95u / 100u;
      printf("%s: %s (%u of %u pixels read as %s)\n", case_name[c], ok ? "pass" : "FAIL",
             good, WIDTH * HEIGHT, c == 0 ? "the red upper levels" : "grey");
      if (!ok)
         failed++;
      vkDestroySampler(dev, sampler, NULL);
   }

   if (failed) {
      printf("FAILED (%u case(s))\n", failed);
      return 1;
   }
   printf("PASSED (anisotropic footprint walked at the minor axis's level)\n");
   return 0;
}
