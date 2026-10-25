/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Message passing under the Vulkan memory model, after the CTS
 * memory_model.message_passing cases. Every invocation writes a payload word,
 * releases its guard word with an atomic exchange, then acquires its partner's
 * guard with another; when the partner's guard reads as released, the
 * partner's payload must read back as the partner's value. A failure flags the
 * invocation in a buffer the host checks.
 *
 *   image     device scope, payload in a storage image, partner in another
 *             workgroup
 *   buffer    the same with the payload in a storage buffer
 *   subgroup  subgroup scope, payload in shared memory, partner in the same
 *             subgroup; lanes whose partner is inactive return first, so the
 *             subgroup barrier runs with a subgroup that has already left
 *   vertex    the image variant in a vertex shader, one vertex per
 *             invocation of a point draw with rasterization discarded
 *
 * The guard is a buffer reached through its device address. Each variant runs
 * ITERS dispatches (or draws) in one submission, clearing payload and guard
 * before each.
 *
 * usage: msgpass <spv>... [-v variant] [-d dim] [-n wgs_per_dim] [-i iters]
 * Each .spv selects its variant from its file name (mp_<variant>.<stage>.spv).
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

enum variant { V_IMAGE, V_BUFFER, V_SUBGROUP, V_VERTEX, V_COUNT };
static const char *const vname[V_COUNT] = { "image", "buffer", "subgroup", "vertex" };

struct ctx {
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
};

struct buf {
   VkBuffer b;
   VkDeviceMemory m;
   VkDeviceSize size;
};

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
   if (buf)
      *out_size = (size_t)sz;
   return buf;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   return UINT32_MAX;
}

static int
make_buf(struct ctx *c, VkDeviceSize size, VkBufferUsageFlags usage,
         VkMemoryPropertyFlags want, struct buf *out)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, &out->b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, out->b, &mr);
   uint32_t mt = find_mem(&c->mp, mr.memoryTypeBits, want);
   if (mt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no memory type for a %llu-byte buffer\n",
              (unsigned long long)size);
      return 1;
   }
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? &mafi : NULL,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, &out->m));
   CHECK(vkBindBufferMemory(c->dev, out->b, out->m, 0));
   out->size = size;
   return 0;
}

static void
free_buf(struct ctx *c, struct buf *b)
{
   vkDestroyBuffer(c->dev, b->b, NULL);
   vkFreeMemory(c->dev, b->m, NULL);
}

/* One variant: ITERS dispatches of NUM_WG x NUM_WG workgroups of DIM x DIM.
 * Each invocation ORs 1 into its result word when it saw its partner released
 * without the partner's payload, and 2 when it saw its partner released at
 * all. Returns the number of failed invocations, or -1 on a setup error;
 * *out_checked counts the invocations that checked a payload. */
static long
run_variant(struct ctx *c, enum variant v, const char *spv_path,
            uint32_t dim, uint32_t nwg, uint32_t iters, uint32_t *out_checked)
{
   const uint32_t side = dim * nwg;
   const uint32_t count = side * side;
   const VkDeviceSize words = (VkDeviceSize)count * sizeof(uint32_t);
   long result = -1;

   struct buf guard, res, payload = { 0 }, readback;
   if (make_buf(c, words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &guard) ||
       make_buf(c, words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &res) ||
       make_buf(c, words, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback))
      return -1;
   if (v == V_BUFFER &&
       make_buf(c, words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &payload))
      return -1;

   VkImage img = VK_NULL_HANDLE;
   VkDeviceMemory imem = VK_NULL_HANDLE;
   VkImageView view = VK_NULL_HANDLE;
   const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   const bool image_payload = (v == V_IMAGE || v == V_VERTEX);
   const bool graphics = (v == V_VERTEX);
   const VkPipelineBindPoint bind_point = graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS
                                                   : VK_PIPELINE_BIND_POINT_COMPUTE;
   if (image_payload) {
      VkImageCreateInfo imci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
         .extent = { side, side, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      if (vkCreateImage(c->dev, &imci, NULL, &img) != VK_SUCCESS)
         return -1;
      VkMemoryRequirements imr;
      vkGetImageMemoryRequirements(c->dev, img, &imr);
      uint32_t imt = find_mem(&c->mp, imr.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VkMemoryAllocateInfo imai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = imr.size, .memoryTypeIndex = imt,
      };
      if (imt == UINT32_MAX ||
          vkAllocateMemory(c->dev, &imai, NULL, &imem) != VK_SUCCESS ||
          vkBindImageMemory(c->dev, img, imem, 0) != VK_SUCCESS)
         return -1;
      VkImageViewCreateInfo ivci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_R32_UINT, .subresourceRange = range,
      };
      if (vkCreateImageView(c->dev, &ivci, NULL, &view) != VK_SUCCESS)
         return -1;
   }

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0,
        .descriptorType = image_payload ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
   };
   const uint32_t nbind = (v == V_SUBGROUP) ? 1u : 2u;
   const VkDescriptorSetLayoutBinding *binds = (v == V_SUBGROUP) ? &dslb[1] : dslb;
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nbind, .pBindings = binds,
   };
   VkDescriptorSetLayout dsl;
   if (vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, &dsl) != VK_SUCCESS)
      return -1;
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_ALL, .size = sizeof(VkDeviceAddress),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   if (vkCreatePipelineLayout(c->dev, &plci, NULL, &pl) != VK_SUCCESS)
      return -1;

   size_t sz = 0;
   uint32_t *spv = read_spirv(spv_path, &sz);
   if (!spv)
      return -1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   if (vkCreateShaderModule(c->dev, &smci, NULL, &sm) != VK_SUCCESS)
      return -1;
   free(spv);
   const uint32_t spec[2] = { dim, nwg };
   const VkSpecializationMapEntry sme[2] = {
      { 0, 0, sizeof(uint32_t) }, { 1, sizeof(uint32_t), sizeof(uint32_t) },
   };
   VkSpecializationInfo si = {
      .mapEntryCount = 2, .pMapEntries = sme,
      .dataSize = sizeof spec, .pData = spec,
   };
   const VkPipelineShaderStageCreateInfo stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = graphics ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_COMPUTE_BIT,
      .module = sm, .pName = "main", .pSpecializationInfo = &si,
   };
   VkPipeline pipe;
   VkRenderPass rp = VK_NULL_HANDLE;
   VkFramebuffer fb = VK_NULL_HANDLE;
   if (graphics) {
      /* Points with rasterization discarded: the vertex shader is all that
       * runs, once per vertex. */
      VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS };
      VkRenderPassCreateInfo rpci = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .subpassCount = 1, .pSubpasses = &sub,
      };
      if (vkCreateRenderPass(c->dev, &rpci, NULL, &rp) != VK_SUCCESS)
         return -1;
      VkFramebufferCreateInfo fbci = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
         .width = side, .height = side, .layers = 1,
      };
      if (vkCreateFramebuffer(c->dev, &fbci, NULL, &fb) != VK_SUCCESS)
         return -1;
      VkPipelineVertexInputStateCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      };
      VkPipelineInputAssemblyStateCreateInfo ia = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
      };
      VkViewport vp = { 0, 0, (float)side, (float)side, 0, 1 };
      VkRect2D sc = { { 0, 0 }, { side, side } };
      VkPipelineViewportStateCreateInfo vs = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
      };
      VkPipelineRasterizationStateCreateInfo rs = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .rasterizerDiscardEnable = VK_TRUE, .polygonMode = VK_POLYGON_MODE_FILL,
         .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f,
      };
      VkPipelineMultisampleStateCreateInfo ms = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      };
      VkGraphicsPipelineCreateInfo gpci = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 1, .pStages = &stage,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms,
         .layout = pl, .renderPass = rp,
      };
      if (vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe) != VK_SUCCESS)
         return -1;
   } else {
      VkComputePipelineCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = stage, .layout = pl,
      };
      if (vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe) != VK_SUCCESS)
         return -1;
   }
   vkDestroyShaderModule(c->dev, sm, NULL);

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   if (vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp) != VK_SUCCESS)
      return -1;
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   if (vkAllocateDescriptorSets(c->dev, &dsai, &ds) != VK_SUCCESS)
      return -1;
   VkDescriptorBufferInfo fbi = { res.b, 0, VK_WHOLE_SIZE };
   VkDescriptorBufferInfo pbi = { payload.b, 0, VK_WHOLE_SIZE };
   VkDescriptorImageInfo pii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &fbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = image_payload ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &pbi, .pImageInfo = &pii },
   };
   vkUpdateDescriptorSets(c->dev, (v == V_SUBGROUP) ? 1u : 2u, wds, 0, NULL);

   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = guard.b,
   };
   const VkDeviceAddress gaddr = vkGetBufferDeviceAddress(c->dev, &bdai);

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   if (vkCreateCommandPool(c->dev, &cpi, NULL, &cp) != VK_SUCCESS)
      return -1;
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   if (vkAllocateCommandBuffers(c->dev, &cbai, &cb) != VK_SUCCESS)
      return -1;
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (vkBeginCommandBuffer(cb, &cbbi) != VK_SUCCESS)
      return -1;

   vkCmdFillBuffer(cb, res.b, 0, VK_WHOLE_SIZE, 0);
   if (img) {
      VkImageMemoryBarrier ib = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = img, .subresourceRange = range,
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
   }
   vkCmdBindDescriptorSets(cb, bind_point, pl, 0, 1, &ds, 0, NULL);
   vkCmdBindPipeline(cb, bind_point, pipe);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof gaddr, &gaddr);
   VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
   for (uint32_t it = 0; it < iters; it++) {
      vkCmdFillBuffer(cb, guard.b, 0, VK_WHOLE_SIZE, 0);
      if (payload.b)
         vkCmdFillBuffer(cb, payload.b, 0, VK_WHOLE_SIZE, 0);
      if (img) {
         VkClearColorValue zero = { .uint32 = { 0, 0, 0, 0 } };
         vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
      }
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      if (graphics) {
         VkRenderPassBeginInfo rpbi = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = rp, .framebuffer = fb, .renderArea = { { 0, 0 }, { side, side } },
         };
         vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
         vkCmdDraw(cb, count, 1, 0, 0);
         vkCmdEndRenderPass(cb);
      } else {
         vkCmdDispatch(cb, nwg, nwg, 1);
      }
      mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   }
   VkBufferCopy bc = { 0, 0, words };
   vkCmdCopyBuffer(cb, res.b, readback.b, 1, &bc);
   mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
   if (vkEndCommandBuffer(cb) != VK_SUCCESS)
      return -1;

   VkSubmitInfo sinfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   if (vkQueueSubmit(c->queue, 1, &sinfo, VK_NULL_HANDLE) != VK_SUCCESS ||
       vkQueueWaitIdle(c->queue) != VK_SUCCESS)
      return -1;

   uint32_t *p;
   if (vkMapMemory(c->dev, readback.m, 0, words, 0, (void **)&p) != VK_SUCCESS)
      return -1;
   result = 0;
   uint32_t checked = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (p[i] & 2u)
         checked++;
      if (p[i] & 1u) {
         if (result < 8)
            printf("  %s: invocation %u failed\n", vname[v], i);
         result++;
      }
   }
   vkUnmapMemory(c->dev, readback.m);
   *out_checked = checked;

   vkDestroyCommandPool(c->dev, cp, NULL);
   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   if (graphics) {
      vkDestroyFramebuffer(c->dev, fb, NULL);
      vkDestroyRenderPass(c->dev, rp, NULL);
   }
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   if (img) {
      vkDestroyImageView(c->dev, view, NULL);
      vkDestroyImage(c->dev, img, NULL);
      vkFreeMemory(c->dev, imem, NULL);
   }
   if (payload.b)
      free_buf(c, &payload);
   free_buf(c, &readback);
   free_buf(c, &res);
   free_buf(c, &guard);
   return result;
}

int
main(int argc, char **argv)
{
   const char *spv[V_COUNT] = { NULL };
   int only = -1;
   uint32_t dim = 0, nwg = 0, iters = 0;
   for (int i = 1; i < argc; i++) {
      const char *a = argv[i];
      if (!strcmp(a, "-v") && i + 1 < argc) {
         const char *n = argv[++i];
         for (int k = 0; k < V_COUNT; k++) {
            if (!strcmp(n, vname[k]))
               only = k;
         }
         if (only < 0) {
            fprintf(stderr, "FAILED: unknown variant %s\n", n);
            return 1;
         }
      } else if (!strcmp(a, "-d") && i + 1 < argc) {
         dim = (uint32_t)atoi(argv[++i]);
      } else if (!strcmp(a, "-n") && i + 1 < argc) {
         nwg = (uint32_t)atoi(argv[++i]);
      } else if (!strcmp(a, "-i") && i + 1 < argc) {
         iters = (uint32_t)atoi(argv[++i]);
      } else {
         for (int k = 0; k < V_COUNT; k++) {
            char key[32];
            snprintf(key, sizeof key, "mp_%s.", vname[k]);
            if (strstr(a, key))
               spv[k] = a;
         }
      }
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msgpass", .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   struct ctx c = { 0 };
   uint32_t npd = 1;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &c.pd));
   if (npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceSubgroupProperties sgp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp,
   };
   vkGetPhysicalDeviceProperties2(c.pd, &props);
   printf("device: %s\n", props.properties.deviceName);
   const uint32_t max_inv = props.properties.limits.maxComputeWorkGroupInvocations;

   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(c.pd, &f2);
   if (!v12.vulkanMemoryModel || !v12.vulkanMemoryModelDeviceScope ||
       !v12.bufferDeviceAddress) {
      fprintf(stderr, "FAILED: device lacks the Vulkan memory model or buffer device "
              "addresses\n");
      return 1;
   }
   const VkSubgroupFeatureFlags sg_need = VK_SUBGROUP_FEATURE_BASIC_BIT |
      VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
   if ((sgp.supportedOperations & sg_need) != sg_need ||
       !(sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
      fprintf(stderr, "FAILED: device lacks compute subgroup ballot/shuffle\n");
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   c.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
         c.qf = i;
         break;
      }
   }
   free(qfp);
   if (c.qf == UINT32_MAX) {
      fprintf(stderr, "FAILED: no compute queue\n");
      return 1;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan12Features v12_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .vulkanMemoryModel = VK_TRUE, .vulkanMemoryModelDeviceScope = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
   };
   VkPhysicalDeviceFeatures f_en = {
      .vertexPipelineStoresAndAtomics = f2.features.vertexPipelineStoresAndAtomics,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &v12_en,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &f_en,
   };
   CHECK(vkCreateDevice(c.pd, &dci, NULL, &c.dev));
   vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);

   long total = 0;
   int ran = 0;
   for (int k = 0; k < V_COUNT; k++) {
      if (!spv[k] || (only >= 0 && k != only))
         continue;
      if (k == V_VERTEX && !f2.features.vertexPipelineStoresAndAtomics) {
         fprintf(stderr, "FAILED: device lacks vertexPipelineStoresAndAtomics\n");
         return 1;
      }
      /* The CTS shape: the widest square workgroup the device allows, so a
       * subgroup-mapped workgroup ends in a partial subgroup. */
      uint32_t d = dim;
      if (!d) {
         d = 31;
         while (d * d > max_inv)
            d--;
      }
      uint32_t n = nwg ? nwg : 2u;
      uint32_t it = iters ? iters : 4u;
      uint32_t checked = 0;
      long f = run_variant(&c, (enum variant)k, spv[k], d, n, it, &checked);
      if (f < 0) {
         fprintf(stderr, "FAILED: %s: setup error\n", vname[k]);
         return 1;
      }
      printf("%s: %ux%u workgroups of %ux%u, subgroup %u, %u dispatches: %u invocations "
             "checked a released partner, %ld failed\n", vname[k], n, n, d, d,
             sgp.subgroupSize, it, checked, f);
      if (checked == 0) {
         printf("FAILED (%s: no invocation saw its partner released, so nothing was "
                "checked)\n", vname[k]);
         return 1;
      }
      total += f;
      ran++;
   }

   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (!ran) {
      fprintf(stderr, "FAILED: no variant ran\n");
      return 1;
   }
   if (total) {
      printf("FAILED (%ld invocations saw a released guard without its payload)\n",
             total);
      return 1;
   }
   printf("PASSED\n");
   return 0;
}
