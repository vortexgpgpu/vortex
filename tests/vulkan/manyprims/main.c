/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One draw with more primitives than a RASTER pass addresses.
 *
 * The RASTER unit carries a primitive's index in VX_RASTER_PID_BITS (16) bits,
 * from the bin's pid list through the record fetch to the fragment's payload,
 * so a pass holds at most 65536 primitive records and the driver splits a larger
 * draw into passes. This draw is 33792 triangles in one vkCmdDraw, each clipped
 * by the near plane into two records: 67584 in all. One pass would carry pids
 * past 65535, which the unit either truncates (the fragment shades with another
 * primitive's varyings) or drops (a bin's count wraps), depending on how the
 * primitives fall into bins. Split by the driver's worst case, the first pass
 * is exactly full.
 *
 * The triangles are drawn as two instances of 16896, a count the pass size does
 * not divide, so a pass boundary falls inside an instance.
 *
 * Every triangle lies inside one pixel of a 64x64 target, so the whole draw is
 * one bin. Every 32nd covers its pixel's centre, at a pixel of its own, and
 * writes its own triangle index; the rest cover no sample. The covering ones
 * span every pass boundary, and every pixel of the image is checked exactly:
 *
 *   covered pixel p   triangle 32*p+31, encoded one byte per channel
 *   any other pixel   the clear colour
 *
 * Two stippled line strips then cross a pass boundary (32768 segments): one
 * run whose stipple counter carries from its segment before the boundary to
 * its segment after it, and a run restarted exactly at the boundary, whose
 * counter starts again at 0. Each row of the second image is checked against
 * the stipple pattern.
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

/* Matches manyprims.vert: two instances of TRIS_PER_INST triangles. */
#define NUM_TRIS      33792u
#define TRIS_PER_INST (NUM_TRIS / 2u)
#define COVER_STRIDE  32u

/* Matches manyprims_line.vert: rows and lengths of the stippled segments. */
#define LINE_X0       4u
#define LINE_A        5u      /* the first run's visible segment */
#define LINE_B        40u     /* the visible segment after the pass boundary */
#define ROW_CARRY     10u
#define ROW_HEAD_RUN1 20u
#define ROW_HEAD_RUN2 30u
#define PASS_SEGS     32768u  /* the driver's pass: GFX_RASTER_MAX_PRIMS / 2 */
#define CARRY_ZEROS   32789u  /* zero-length segments between the carry draw's two */
#define STIPPLE_PATTERN 0x00ffu
#define STIPPLE_FACTOR  1u
#define RESTART       0xffffffffu

/* One render: the pipeline's topology, and the draws it records. Lines are
 * stippled rectangular lines with primitive restart, drawn from `indices`. */
struct draw_desc {
   VkPrimitiveTopology topology;
   bool lines;
   const uint32_t *indices;
   uint32_t index_count;
   uint32_t ndraws;
   struct { uint32_t count, instances, first_index; } draws[2];
};

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return -1;                                                   \
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

/* Record d's draws into a cleared target and read the image back into out_px
 * (RGBA8, row-major). Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, const struct draw_desc *d, uint8_t *out_px)
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

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   /* No vertex input: the vertex shader builds every triangle from
    * gl_VertexIndex. */
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = d->topology, .primitiveRestartEnable = d->lines,
   };
   VkViewport vp = { 0, 0, (float)WIDTH, (float)HEIGHT, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { WIDTH, HEIGHT } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationLineStateCreateInfoEXT ls = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
      .lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
      .stippledLineEnable = VK_TRUE,
      .lineStippleFactor = STIPPLE_FACTOR, .lineStipplePattern = STIPPLE_PATTERN,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = d->lines ? &ls : NULL,
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

   VkBuffer ib = VK_NULL_HANDLE;
   VkDeviceMemory imem_idx = VK_NULL_HANDLE;
   if (d->indices) {
      VkBufferCreateInfo ibci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = (VkDeviceSize)d->index_count * 4, .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      };
      CHECK(vkCreateBuffer(dev, &ibci, NULL, &ib));
      VkMemoryRequirements ibr;
      vkGetBufferMemoryRequirements(dev, ib, &ibr);
      VkMemoryAllocateInfo ibai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = ibr.size,
         .memoryTypeIndex = find_mem(mp, ibr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      if (ibai.memoryTypeIndex == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return -1; }
      CHECK(vkAllocateMemory(dev, &ibai, NULL, &imem_idx));
      CHECK(vkBindBufferMemory(dev, ib, imem_idx, 0));
      void *map;
      CHECK(vkMapMemory(dev, imem_idx, 0, VK_WHOLE_SIZE, 0, &map));
      memcpy(map, d->indices, (size_t)d->index_count * 4);
      vkUnmapMemory(dev, imem_idx);
   }

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
   if (ib) {
      vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
   }
   for (uint32_t i = 0; i < d->ndraws; i++) {
      if (ib) {
         vkCmdDrawIndexed(cmd, d->draws[i].count, d->draws[i].instances,
                          d->draws[i].first_index, 0, 0);
      } else {
         vkCmdDraw(cmd, d->draws[i].count, d->draws[i].instances, 0, 0);
      }
   }
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
   memcpy(out_px, px, (size_t)bytes);
   vkUnmapMemory(dev, bmem);

   vkDestroyCommandPool(dev, cp, NULL);
   if (ib) {
      vkDestroyBuffer(dev, ib, NULL);
      vkFreeMemory(dev, imem_idx, NULL);
   }
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-manyprims",
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
   VkPhysicalDeviceLineRasterizationFeaturesEXT lf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &lf,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!lf.rectangularLines || !lf.stippledRectangularLines) {
      fprintf(stderr, "FAILED: stippled rectangular lines missing\n");
      return 1;
   }
   VkPhysicalDeviceLineRasterizationFeaturesEXT lfe = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
      .rectangularLines = VK_TRUE, .stippledRectangularLines = VK_TRUE,
   };
   const char *dev_exts[] = { VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &lfe,
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

   VkShaderModule vs = load_module(dev, "manyprims.vert.spv");
   VkShaderModule fs = load_module(dev, "manyprims.frag.spv");
   VkShaderModule lvs = load_module(dev, "manyprims_line.vert.spv");
   VkShaderModule lfs = load_module(dev, "manyprims_line.frag.spv");
   if (!vs || !fs || !lvs || !lfs) {
      return 1;
   }
   uint8_t *px = malloc((size_t)WIDTH * HEIGHT * 4);
   if (!px) {
      return 1;
   }

   /* Triangles: two instances, each a count the pass size does not divide. */
   const struct draw_desc tris = {
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .ndraws = 1, .draws = { { 3 * TRIS_PER_INST, 2, 0 } },
   };
   if (render(dev, queue, qf, &mp, vs, fs, &tris, px) < 0) {
      return 1;
   }
   uint32_t covered = 0, errors = 0;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      const uint32_t tri = i * COVER_STRIDE + COVER_STRIDE - 1;
      uint8_t want[3] = { 0, 0, 0 };
      if (tri < NUM_TRIS) {
         want[0] = tri & 0xff;
         want[1] = (tri >> 8) & 0xff;
         want[2] = (tri >> 16) & 0xff;
         covered++;
      }
      const uint8_t *got = px + (size_t)i * 4;
      if (memcmp(got, want, 3) != 0) {
         if (errors < 10) {
            printf("   pixel (%u, %u): triangle %u, got %02x%02x%02x, expected %02x%02x%02x\n",
                   i % WIDTH, i / WIDTH, (got[2] << 16) | (got[1] << 8) | got[0],
                   got[2], got[1], got[0], want[2], want[1], want[0]);
         }
         errors++;
      }
   }

   /* Stippled line strips, two indexed draws:
    *  - one run whose visible segments sit either side of the pass boundary,
    *    joined by zero-length segments: the second continues the first's
    *    stipple counter, which crosses into the next pass;
    *  - a run of exactly one pass, then a restarted run whose first segment is
    *    the first of the next pass: it starts its counter at 0. */
   const uint32_t carry_count = 1 + CARRY_ZEROS + 2;
   const uint32_t head_count = 1 + PASS_SEGS + 1 + 2;
   uint32_t *idx = malloc((size_t)(carry_count + head_count) * 4);
   if (!idx) {
      return 1;
   }
   uint32_t n = 0;
   idx[n++] = 0;
   for (uint32_t i = 0; i <= CARRY_ZEROS; i++) {
      idx[n++] = 1;
   }
   idx[n++] = 2;
   idx[n++] = 3;
   for (uint32_t i = 0; i < PASS_SEGS; i++) {
      idx[n++] = 4;
   }
   idx[n++] = RESTART;
   idx[n++] = 5;
   idx[n++] = 6;
   const struct draw_desc lines = {
      .topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, .lines = true,
      .indices = idx, .index_count = n,
      .ndraws = 2, .draws = { { carry_count, 1, 0 }, { head_count, 1, carry_count } },
   };
   if (render(dev, queue, qf, &mp, lvs, lfs, &lines, px) < 0) {
      return 1;
   }
   free(idx);
   uint32_t lit = 0, line_errors = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         /* The counter along each row runs from 0 at LINE_X0: carried through
          * the first draw's boundary, restarted by the second draw's head. */
         uint32_t len = 0;
         if (y == ROW_CARRY) {
            len = LINE_A + LINE_B;
         } else if (y == ROW_HEAD_RUN1) {
            len = LINE_A;
         } else if (y == ROW_HEAD_RUN2) {
            len = LINE_B;
         }
         bool want = false;
         if (x >= LINE_X0 && x < LINE_X0 + len) {
            const uint32_t bit = ((x - LINE_X0) / STIPPLE_FACTOR) & 15u;
            want = (STIPPLE_PATTERN >> bit) & 1u;
         }
         const uint8_t *got = px + ((size_t)y * WIDTH + x) * 4;
         const bool on = got[0] || got[1] || got[2];
         lit += on;
         if (on != want) {
            if (line_errors < 10) {
               printf("   line pixel (%u, %u): %s, expected %s\n", x, y,
                      on ? "lit" : "clear", want ? "lit" : "clear");
            }
            line_errors++;
         }
      }
   }
   free(px);

   const bool ok = errors == 0 && line_errors == 0;
   if (ok) {
      printf("PASSED (manyprims: %u clipped triangles in 2 instances, %u covered pixels exact; "
             "stipple carry and restart across the pass boundary, %u pixels exact)\n",
             NUM_TRIS, covered, lit);
   } else {
      printf("FAILED (manyprims: %u of %u triangle pixels wrong, %u of %u line pixels wrong)\n",
             errors, WIDTH * HEIGHT, line_errors, WIDTH * HEIGHT);
   }

   vkDestroyShaderModule(dev, lvs, NULL);
   vkDestroyShaderModule(dev, lfs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return ok ? 0 : 1;
}
