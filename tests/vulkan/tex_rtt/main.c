/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A texture written and sampled in turn. Each round writes a new pattern into
 * texture T -- with a render pass in even rounds, with a compute shader's image
 * stores in odd ones -- and then renders target O by sampling T with a plain
 * texture() lookup, which the TEX unit serves. O is copied out and every texel
 * checked against the round's pattern.
 *
 * The TEX unit reads its own copy of a texture, built on the device from the
 * texture's memory. A write the copy does not notice leaves O showing an
 * earlier round's pattern, so every round after the first fails unless both
 * kinds of write make the next sample rebuild it. T is 70x45, so its rows carry
 * padding the copy must step over.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FMT    VK_FORMAT_R8G8B8A8_UNORM
#define W      70u
#define H      45u
#define ROUNDS 6u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* rtt_pattern.glsl */
static uint32_t
pattern(uint32_t x, uint32_t y, uint32_t k)
{
   uint32_t v = (x * 2654435761u) ^ (y * 40503u) ^ ((k + 1u) * 0x9e3779b9u);
   v ^= v >> 13;
   return v * 0x85ebca6bu;
}

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "FAILED: cannot open %s\n", path); return VK_NULL_HANDLE; }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *buf = malloc((size_t)sz);
   VkShaderModule sm = VK_NULL_HANDLE;
   if (buf && fread(buf, 1, (size_t)sz, f) == (size_t)sz) {
      VkShaderModuleCreateInfo smci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = (size_t)sz, .pCode = buf,
      };
      if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS)
         sm = VK_NULL_HANDLE;
   }
   free(buf);
   fclose(f);
   return sm;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static void
full_barrier(VkCommandBuffer cmd)
{
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

struct target {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   VkFramebuffer fb;
};

static VkResult
make_target(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkRenderPass rp,
            VkImageUsageFlags usage, struct target *t)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = FMT, .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
   };
   VkResult r = vkCreateImage(dev, &ici, NULL, &t->img);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, t->img, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(mp, mr.memoryTypeBits, 0),
   };
   if ((r = vkAllocateMemory(dev, &mai, NULL, &t->mem)) != VK_SUCCESS ||
       (r = vkBindImageMemory(dev, t->img, t->mem, 0)) != VK_SUCCESS)
      return r;
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FMT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   if ((r = vkCreateImageView(dev, &vci, NULL, &t->view)) != VK_SUCCESS)
      return r;
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 1, .pAttachments = &t->view, .width = W, .height = H, .layers = 1,
   };
   return vkCreateFramebuffer(dev, &fbci, NULL, &t->fb);
}

static VkPipeline
make_graphics(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl, VkShaderModule vs,
              VkShaderModule fs)
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
   VkViewport vp = { 0, 0, (float)W, (float)H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
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
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .layout = pl, .renderPass = rp,
   };
   VkPipeline p = VK_NULL_HANDLE;
   if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return p;
}

static void
draw(VkCommandBuffer cmd, VkRenderPass rp, const struct target *t, VkPipeline p)
{
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
      .framebuffer = t->fb, .renderArea = { { 0, 0 }, { W, H } },
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
}

int
main(int argc, char **argv)
{
   const char *vs_path = argc > 1 ? argv[1] : "rtt.vert.spv";
   const char *wfs_path = argc > 2 ? argv[2] : "rtt_write.frag.spv";
   const char *wcs_path = argc > 3 ? argv[3] : "rtt_write.comp.spv";
   const char *sfs_path = argc > 4 ? argv[4] : "rtt_sample.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-tex-rtt", .apiVersion = VK_API_VERSION_1_1,
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
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if ((qfp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qf = i; break; }
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

   VkShaderModule vs = load_module(dev, vs_path), wfs = load_module(dev, wfs_path);
   VkShaderModule wcs = load_module(dev, wcs_path), sfs = load_module(dev, sfs_path);
   if (!vs || !wfs || !wcs || !sfs)
      return 1;

   /* Both images stay in GENERAL: attachment, sampled, stored and copied. */
   VkAttachmentDescription att = {
      .format = FMT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_GENERAL, .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_GENERAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   struct target tex, out;
   CHECK(make_target(dev, &mp, rp, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                     &tex));
   CHECK(make_target(dev, &mp, rp, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &out));

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = W * H * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, rb, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = find_mem(&mp, mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory rbm;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &rbm));
   CHECK(vkBindBufferMemory(dev, rb, rbm, 0));
   uint32_t *px;
   CHECK(vkMapMemory(dev, rbm, 0, W * H * 4, 0, (void **)&px));

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler smp;
   CHECK(vkCreateSampler(dev, &sci, NULL, &smp));

   /* Layouts: the sampling pass reads T, the compute writer stores to it. */
   VkDescriptorSetLayoutBinding sb = {
      0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL,
   };
   VkDescriptorSetLayoutBinding cb = {
      0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
   };
   VkDescriptorSetLayout sdsl, cdsl;
   dslci.pBindings = &sb;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &sdsl));
   dslci.pBindings = &cb;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &cdsl));
   VkPushConstantRange wpc = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 };
   VkPushConstantRange spc = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8 };
   VkPushConstantRange cpc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
   };
   VkPipelineLayout wpl, spl, cpl;
   plci.pPushConstantRanges = &wpc;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &wpl));
   plci.setLayoutCount = 1;
   plci.pSetLayouts = &sdsl;
   plci.pPushConstantRanges = &spc;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &spl));
   plci.pSetLayouts = &cdsl;
   plci.pPushConstantRanges = &cpc;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &cpl));

   VkPipeline wpipe = make_graphics(dev, rp, wpl, vs, wfs);
   VkPipeline spipe = make_graphics(dev, rp, spl, vs, sfs);
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = wcs, .pName = "main" },
      .layout = cpl,
   };
   VkPipeline cpipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe));
   if (!wpipe || !spipe) { fprintf(stderr, "FAILED: graphics pipeline\n"); return 1; }

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetLayout layouts[2] = { sdsl, cdsl };
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 2, .pSetLayouts = layouts,
   };
   VkDescriptorSet sets[2];
   CHECK(vkAllocateDescriptorSets(dev, &dsai, sets));
   VkDescriptorImageInfo sii = { smp, tex.view, VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorImageInfo cii = { VK_NULL_HANDLE, tex.view, VK_IMAGE_LAYOUT_GENERAL };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[0],
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &sii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[1],
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &cii },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

   VkCommandPoolCreateInfo cpoolci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf,
   };
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpoolci, NULL, &cpool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   for (uint32_t k = 0; k < ROUNDS; k++) {
      const bool by_compute = (k & 1u) != 0;
      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      full_barrier(cmd);
      if (by_compute) {
         const uint32_t pc[3] = { k, W, H };
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &sets[1],
                                 0, NULL);
         vkCmdPushConstants(cmd, cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
         vkCmdDispatch(cmd, (W + 3) / 4, (H + 3) / 4, 1);
      } else {
         vkCmdPushConstants(cmd, wpl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &k);
         draw(cmd, rp, &tex, wpipe);
      }
      full_barrier(cmd);
      const float inv[2] = { 1.0f / (float)W, 1.0f / (float)H };
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, spl, 0, 1, &sets[0],
                              0, NULL);
      vkCmdPushConstants(cmd, spl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof inv, inv);
      draw(cmd, rp, &out, spipe);
      full_barrier(cmd);
      VkBufferImageCopy bic = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { W, H, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, out.img, VK_IMAGE_LAYOUT_GENERAL, rb, 1, &bic);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      unsigned bad = 0, stale = 0;
      for (uint32_t y = 0; y < H; y++) {
         for (uint32_t x = 0; x < W; x++) {
            const uint32_t want = pattern(x, y, k), got = px[y * W + x];
            if (got == want)
               continue;
            if (k && got == pattern(x, y, k - 1))
               stale++;
            if (bad++ < 4)
               fprintf(stderr, "  round %u (%u,%u): 0x%08x, want 0x%08x\n", k, x, y, got, want);
         }
      }
      if (bad) {
         printf("FAILED (round %u, written by %s: %u of %u texels differ, %u of them "
                "still the previous round's)\n", k, by_compute ? "image stores" : "a render pass",
                bad, W * H, stale);
         return 1;
      }
      printf("round %u (%s): %u texels sampled as written\n", k,
             by_compute ? "image stores" : "render pass", W * H);
   }

   vkDestroyCommandPool(dev, cpool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (tex_rtt: %u rounds of write then sample, render passes and image "
          "stores in turn)\n", ROUNDS);
   return 0;
}
