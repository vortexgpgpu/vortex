/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * discard and demote in the fragment stage.
 *
 * Both are core Vulkan -- discard needs no feature bit and
 * shaderDemoteToHelperInvocation is advertised -- and the device implements
 * them together: a per-lane live flag that the fragment wrapper ANDs into
 * coverage, plus a sink address that a suppressed lane's stores are steered
 * into. Neither had any coverage in this suite, so the whole mechanism was
 * carried by the fact that nothing exercised it.
 *
 * Five phases, each isolating one claim the implementation makes:
 *
 *   kill    the discarded fragment does not reach the attachment
 *   demote  the demoted lane keeps executing, so a live neighbour in its quad
 *           can still shuffle from it
 *   sink    the demoted lane's buffer store does not commit
 *   vswrite a vertex shader's buffer store reaches the host at all
 *   atomic  the demoted lane's atomic does not commit either
 *
 * Odd columns are suppressed in every phase, which leaves each pixel quad half
 * alive. That is what makes the demote claim testable at all: with a coarser
 * pattern the quads would be wholly on one side of the split and a demote
 * implemented as an early exit would look identical to a correct one.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u

/* Sample columns are even (kept) with their odd (suppressed) partner beside
 * them, so each pair straddles one quad. Three rows rather than one: the
 * suppression predicate is a function of x alone, so a result that varied with
 * y would mean the quad grid, not the shader, decided it. */
static const struct { const char *where; uint32_t x, y; } SAMPLES[] = {
   { "lower-left",  24, 16 },
   { "centre",      32, 32 },
   { "upper-right", 40, 48 },
};
#define NUM_SAMPLES (sizeof(SAMPLES) / sizeof(SAMPLES[0]))

/* Half the columns survive. */
#define EXPECT_PX  ((long)(WIDTH / 2) * HEIGHT)

/* The kept lanes write dFdx(v_uv.x) * 64 = 1.0 in red and a literal 1.0 in
 * green; the suppressed ones must leave the clear colour. Both live at the ends
 * of the range, so a failure moves a channel by the whole 255. */
#define EXPECT_R  255
#define EXPECT_G  255
#define EXPECT_B  0

/* One count of slack for the UNORM8 encoding, and no more. Every defect this
 * test can see moves a channel the full 255 or moves a pixel count by
 * thousands, so a tolerance loose enough to matter would have to be loose
 * enough to accept one of them. */
#define TOL 1

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return -1;                                                   \
   }                                                               \
} while (0)

/* What one phase measured. */
struct phase_result {
   uint8_t kept[NUM_SAMPLES][3];        /* the even column of each pair */
   uint8_t killed[NUM_SAMPLES][3];      /* its odd partner */
   long    covered;                     /* pixels that are not the clear colour */
   long    stores;                      /* words the shader committed */
   long    stores_odd;                  /* of those, ones a suppressed lane wrote */
   uint32_t head[3];                    /* the first words, for the per-vertex case */
   long     extra_idx;                  /* first word written outside head, or -1 */
   uint32_t extra_val;
};

/* What discard_vswrite.vert stores into word `v`. */
#define VS_MARK(v) (0xA5u + (unsigned)(v))

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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
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

/* Draw the triangle with one fragment shader and report what landed. When
 * `with_hits` is set the shader also writes one word per pixel into a storage
 * buffer, and the words it committed are counted. Returns 0, or -1 on a Vulkan
 * error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, bool with_hits,
       struct phase_result *out)
{
   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
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
   uint32_t imt = find_mem(mp, imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return -1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
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

   /* The per-pixel store target, and the descriptor plumbing that reaches it.
    * Only the sink phase asks for one; the other two run with an empty pipeline
    * layout so that a descriptor problem cannot be mistaken for a discard
    * problem. */
   const VkDeviceSize hit_bytes = (VkDeviceSize)WIDTH * HEIGHT * sizeof(uint32_t);
   VkBuffer              hb    = VK_NULL_HANDLE;
   VkDeviceMemory        hmem  = VK_NULL_HANDLE;
   VkDescriptorSetLayout dsl   = VK_NULL_HANDLE;
   VkDescriptorPool      dpool = VK_NULL_HANDLE;
   VkDescriptorSet       dset  = VK_NULL_HANDLE;
   if (with_hits) {
      VkBufferCreateInfo hbci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = hit_bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      };
      CHECK(vkCreateBuffer(dev, &hbci, NULL, &hb));
      VkMemoryRequirements hmr;
      vkGetBufferMemoryRequirements(dev, hb, &hmr);
      uint32_t hmt = find_mem(mp, hmr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (hmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return -1; }
      VkMemoryAllocateInfo hmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = hmr.size, .memoryTypeIndex = hmt,
      };
      CHECK(vkAllocateMemory(dev, &hmai, NULL, &hmem));
      CHECK(vkBindBufferMemory(dev, hb, hmem, 0));

      /* Cleared before the draw: every word the test counts afterwards was put
       * there by a fragment. */
      void *hp;
      CHECK(vkMapMemory(dev, hmem, 0, hit_bytes, 0, &hp));
      memset(hp, 0, (size_t)hit_bytes);
      vkUnmapMemory(dev, hmem);

      VkDescriptorSetLayoutBinding dslb = {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         /* Both stages: the sink phase writes from the fragment shader and the
          * vswrite phase from the vertex shader, against the same binding. */
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
      };
      VkDescriptorSetLayoutCreateInfo dslci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1, .pBindings = &dslb,
      };
      CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

      VkDescriptorPoolSize dps = {
         .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
      };
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
      };
      CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));

      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
      };
      CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

      VkDescriptorBufferInfo dbi = {
         .buffer = hb, .offset = 0, .range = hit_bytes,
      };
      VkWriteDescriptorSet wds = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi,
      };
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);
   }

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = with_hits ? 1u : 0u,
      .pSetLayouts = with_hits ? &dsl : NULL,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   /* No vertex input: the vertex shader builds its position and its varying
    * from gl_VertexIndex, so nothing but the fragment stage is under test. */
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

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   uint32_t bmt = find_mem(mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return -1; }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, rb, bmem, 0));

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
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   if (with_hits) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1,
                              &dset, 0, NULL);
   }
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
   out->covered = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * 4;
         if (p[0] || p[1] || p[2]) { out->covered++; }
      }
   }
   for (unsigned s = 0; s < NUM_SAMPLES; s++) {
      const size_t base = ((size_t)SAMPLES[s].y * WIDTH + SAMPLES[s].x) * 4;
      memcpy(out->kept[s],   px + base,     3);
      memcpy(out->killed[s], px + base + 4, 3);
   }
   vkUnmapMemory(dev, bmem);

   out->stores = 0;
   out->stores_odd = 0;
   if (with_hits) {
      uint32_t *hp;
      CHECK(vkMapMemory(dev, hmem, 0, hit_bytes, 0, (void **)&hp));
      for (uint32_t y = 0; y < HEIGHT; y++) {
         for (uint32_t x = 0; x < WIDTH; x++) {
            if (hp[(size_t)y * WIDTH + x]) {
               out->stores++;
               if (x & 1u) { out->stores_odd++; }
            }
         }
      }
      memcpy(out->head, hp, sizeof out->head);
      out->extra_idx = -1;
      for (size_t w = 3; w < (size_t)WIDTH * HEIGHT; w++) {
         if (hp[w]) {
            out->extra_idx = (long)w;
            out->extra_val = hp[w];
            break;
         }
      }
      vkUnmapMemory(dev, hmem);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   if (with_hits) {
      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkDestroyDescriptorSetLayout(dev, dsl, NULL);
      vkFreeMemory(dev, hmem, NULL);
      vkDestroyBuffer(dev, hb, NULL);
   }
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   return 0;
}

/* Report one phase. Returns true when it holds. */
static bool
judge(const char *name, const struct phase_result *r, bool want_deriv,
      bool want_stores)
{
   bool ok = (r->covered == EXPECT_PX);
   bool killed_lit = false;
   bool kept_dark = false;

   for (unsigned s = 0; s < NUM_SAMPLES; s++) {
      const int dr = (int)r->kept[s][0] - EXPECT_R;
      const int dg = (int)r->kept[s][1] - EXPECT_G;
      const int db = (int)r->kept[s][2] - EXPECT_B;
      if (dr < -TOL || dr > TOL || dg < -TOL || dg > TOL ||
          db < -TOL || db > TOL) {
         ok = false;
         if (r->kept[s][0] == 0 && r->kept[s][1] == 0) { kept_dark = true; }
      }
      if (r->killed[s][0] || r->killed[s][1] || r->killed[s][2]) {
         ok = false;
         killed_lit = true;
      }
   }

   if (want_stores) {
      /* Every kept lane must have committed its word, and no suppressed one
       * may have. */
      if (r->stores != EXPECT_PX || r->stores_odd != 0) { ok = false; }
   }

   if (ok) {
      if (want_stores) {
         printf("PASSED (discard %s: %ld px kept, %ld stores committed, none "
                "from a suppressed lane)\n", name, r->covered, r->stores);
      } else {
         printf("PASSED (discard %s: %ld px kept, suppressed columns clear, "
                "dFdx=%u across the half-suppressed quad)\n",
                name, r->covered, r->kept[0][0]);
      }
      return true;
   }

   for (unsigned s = 0; s < NUM_SAMPLES; s++) {
      printf("   %s: kept(x=%u)=%u,%u,%u  suppressed(x=%u)=%u,%u,%u\n",
             SAMPLES[s].where, SAMPLES[s].x,
             r->kept[s][0], r->kept[s][1], r->kept[s][2], SAMPLES[s].x + 1,
             r->killed[s][0], r->killed[s][1], r->killed[s][2]);
   }
   printf("FAILED (discard %s: %ld px kept of %ld expected", name, r->covered,
          EXPECT_PX);
   if (killed_lit) {
      printf("; a suppressed column reached the attachment, so the live flag "
             "is not reaching coverage");
   }
   if (kept_dark && want_deriv) {
      printf("; a kept lane's derivative is zero, so the lane beside it stopped "
             "executing when it was demoted instead of becoming a helper");
   }
   if (want_stores) {
      printf("; %ld stores committed of %ld expected, %ld of them from a "
             "suppressed lane", r->stores, EXPECT_PX, r->stores_odd);
      if (r->stores_odd > 0) {
         printf(" -- a demoted lane's write reached memory");
      }
   }
   printf(")\n");
   return false;
}

/* Report the vertex-stage write. Returns true when it holds. */
static bool
judge_vswrite(const struct phase_result *r)
{
   bool ok = (r->stores == 3);
   for (unsigned v = 0; v < 3; v++) {
      if (r->head[v] != VS_MARK(v)) { ok = false; }
   }
   if (ok) {
      printf("PASSED (discard vswrite: one word per vertex reached the host, "
             "%u %u %u)\n", r->head[0], r->head[1], r->head[2]);
      return true;
   }
   printf("FAILED (discard vswrite: got %u %u %u, expected %u %u %u, over %ld "
          "words written%s", r->head[0], r->head[1], r->head[2],
          VS_MARK(0), VS_MARK(1), VS_MARK(2), r->stores,
          (r->head[0] == 0 && r->head[1] == 0 && r->head[2] == 0)
             ? " -- nothing arrived, so the vertex stage's write stayed in the "
               "device copy of the buffer" : "");
   if (r->extra_idx >= 0) {
      printf("; word %ld holds %u, which no vertex of a 3-vertex draw should "
             "have written", r->extra_idx, r->extra_val);
   }
   printf(")\n");
   return false;
}

/* Report the contended-counter phase. Returns true when it holds. */
static bool
judge_atomic(const struct phase_result *r)
{
   /* One word touched, holding one increment per lane that was allowed to run.
    * The word count matters as much as the value: an atomic that escaped
    * suppression and also missed its address would light a second word. */
   if (r->head[0] == (uint32_t)EXPECT_PX && r->stores == 1) {
      printf("PASSED (discard atomic: counter reached %u, one increment per "
             "kept lane and none from a suppressed one)\n", r->head[0]);
      return true;
   }
   printf("FAILED (discard atomic: counter reached %u, expected %ld, over %ld "
          "words touched", r->head[0], EXPECT_PX, r->stores);
   if (r->head[0] == (uint32_t)(2 * EXPECT_PX)) {
      printf(" -- every lane incremented it, so a demoted lane's atomic reached "
             "memory while its plain stores did not");
   }
   printf(")\n");
   return false;
}

int
main(int argc, char **argv)
{
   const char *vs_path     = (argc > 1) ? argv[1] : "discard.vert.spv";
   const char *kill_path   = (argc > 2) ? argv[2] : "discard_kill.frag.spv";
   const char *demote_path = (argc > 3) ? argv[3] : "discard_demote.frag.spv";
   const char *sink_path   = (argc > 4) ? argv[4] : "discard_sink.frag.spv";
   const char *vsw_path    = (argc > 5) ? argv[5] : "discard_vswrite.vert.spv";
   const char *atom_path   = (argc > 6) ? argv[6] : "discard_atomic.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-discard",
      .apiVersion = VK_API_VERSION_1_1,
   };
   const char *inst_exts[] = {
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = inst_exts,
   };
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
      return 1;
   }

   uint32_t npd = 1;
   VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(inst, &npd, &pd) != VK_SUCCESS || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n"); return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* The device must say it supports demote before the test may hold it to it.
    * Reporting the feature off is a legitimate answer; silently passing a test
    * of a feature the driver disclaims is not. */
   VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures demote = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
   };
   VkPhysicalDeviceFeatures2 feats2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &demote,
   };
   vkGetPhysicalDeviceFeatures2(pd, &feats2);
   if (!demote.shaderDemoteToHelperInvocation) {
      printf("FAILED (discard: the device does not advertise "
             "shaderDemoteToHelperInvocation)\n");
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures demote_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
      .shaderDemoteToHelperInvocation = VK_TRUE,
   };
   const char *dev_exts[] = {
      VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &demote_on,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice dev;
   if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: cannot create device\n");
      return 1;
   }
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   VkShaderModule vs   = load_module(dev, vs_path);
   VkShaderModule kill = load_module(dev, kill_path);
   VkShaderModule dem  = load_module(dev, demote_path);
   VkShaderModule sink = load_module(dev, sink_path);
   VkShaderModule vsw  = load_module(dev, vsw_path);
   VkShaderModule atom = load_module(dev, atom_path);
   if (!vs || !kill || !dem || !sink || !vsw || !atom) {
      return 1;
   }

   bool ok = true;
   struct phase_result r;

   memset(&r, 0, sizeof r);
   if (render(dev, queue, qf, &mp, vs, kill, false, &r) < 0) { return 1; }
   ok &= judge("kill", &r, false, false);

   memset(&r, 0, sizeof r);
   if (render(dev, queue, qf, &mp, vs, dem, false, &r) < 0) { return 1; }
   ok &= judge("demote", &r, true, false);

   memset(&r, 0, sizeof r);
   if (render(dev, queue, qf, &mp, vs, sink, true, &r) < 0) { return 1; }
   ok &= judge("sink", &r, false, true);

   /* Paired with the discarding fragment shader, which writes no memory, so the
    * only words in the buffer afterwards are the vertex stage's. */
   memset(&r, 0, sizeof r);
   if (render(dev, queue, qf, &mp, vsw, kill, true, &r) < 0) { return 1; }
   ok &= judge_vswrite(&r);

   memset(&r, 0, sizeof r);
   if (render(dev, queue, qf, &mp, vs, atom, true, &r) < 0) { return 1; }
   ok &= judge_atomic(&r);

   vkDestroyShaderModule(dev, atom, NULL);
   vkDestroyShaderModule(dev, vsw, NULL);
   vkDestroyShaderModule(dev, sink, NULL);
   vkDestroyShaderModule(dev, dem, NULL);
   vkDestroyShaderModule(dev, kill, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return ok ? 0 : 1;
}
