/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Colour attachment formats the output merger has to encode, rather than store
 * verbatim.
 *
 * Every other test in the suite renders to a 32-bit UNORM attachment, which the
 * merger writes through untouched. These four do not:
 *
 *   r8, rg8    a texel is one or two bytes, so every buffer size, row pitch and
 *              readback along the path has to be measured in the attachment's
 *              own width instead of four.
 *   srgb       the merger applies the transfer function on write (and its
 *              inverse on the read that feeds a blend), so the stored bytes are
 *              not the shader's values.
 *   srgb_bgra  the same, on the other channel order, because the encode and the
 *              pack order are independent and each can be right alone.
 *
 * All four take the same fragment colour (1.0, 0.5, 0.25) over a zero clear.
 * Three distinct channel values make a permutation visible, and the two sRGB
 * phases expect the same three numbers in opposite orders -- so a phase that
 * passed by ignoring the attachment's channel order would fail the other.
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

/* Expected bytes are stated as memory bytes, low byte first, `nbytes` of them.
 * `wrong` is what the phase produces if the merger is left in its pass-through
 * format, which is the defect each phase exists to catch. */
struct phase {
   const char *name;
   VkFormat    fmt;
   unsigned    bpp;
   unsigned    nbytes;
   uint8_t     expect[3];
   uint8_t     wrong[3];
   const char *wrong_why;
};

/* The fragment colour is (1.0, 0.5, 0.25). The device quantises each channel to
 * UNORM8 by truncation -- 255, 127, 63 -- and the merger then encodes:
 *
 *   R8    stores red only                        -> 255
 *   RG8   stores red then green                  -> 255, 127
 *   SRGB  applies the sRGB transfer function to
 *         each colour channel, leaving alpha     -> 255, 187, 136
 *
 * The sRGB numbers are the standard encode of the already-quantised linear
 * values (127/255 -> 187.2, 63/255 -> 136.2). A merger left in its pass-through
 * format stores the linear values instead, which is 60 and 73 away -- nowhere
 * near the tolerance below. */
static const struct phase PHASES[] = {
   { "r8",        VK_FORMAT_R8_UNORM,       1, 1,
     { 255, 0, 0 }, { 63, 0, 0 },
     "the byte stored is the fragment's blue, so the merger took its red from "
     "the wrong lane" },
   { "rg8",       VK_FORMAT_R8G8_UNORM,     2, 2,
     { 255, 127, 0 }, { 63, 127, 0 },
     "red and blue are exchanged, so the merger took its channels from the "
     "wrong lanes" },
   { "srgb",      VK_FORMAT_R8G8B8A8_SRGB,  4, 3,
     { 255, 187, 136 }, { 255, 127, 63 },
     "the stored bytes are the shader's linear values, so the sRGB transfer "
     "function was never applied" },
   { "srgb_bgra", VK_FORMAT_B8G8R8A8_SRGB,  4, 3,
     { 136, 187, 255 }, { 63, 127, 255 },
     "the stored bytes are the shader's linear values, so the sRGB transfer "
     "function was never applied" },
};
#define NUM_PHASES (sizeof(PHASES) / sizeof(PHASES[0]))

/* The device applies the transfer function to a linear value already truncated
 * to 8 bits, where llvmpipe converts in float; on the steep part of the sRGB
 * curve that is worth a few LSBs. Every defect signature above is at least 60
 * away, so this cannot absorb one. */
#define TOL 5

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

/* Draw the triangle into an attachment of this phase's format and report its
 * centre texel and its footprint. Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, const struct phase *ph,
       uint8_t centre[3], long *out_covered)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = ph->fmt,
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
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ph->fmt,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = ph->fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
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

   /* No vertex input: the vertex shader builds its positions from
    * gl_VertexIndex, so nothing but the fragment colour is under test. */
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
   /* Blending is off: the transfer function on the read side is only reachable
    * through a blend, and this test covers the write side. */
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

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * ph->bpp;
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
   *out_covered = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * ph->bpp;
         for (unsigned c = 0; c < ph->nbytes; c++) {
            if (p[c]) { (*out_covered)++; break; }
         }
      }
   }
   /* The triangle's lower-left corner is at (-0.9,-0.9) and it spans 1.8 NDC on
    * each axis, so a quarter of the way in on both axes is comfortably inside
    * it and clear of every edge. */
   memset(centre, 0, 3);
   memcpy(centre, px + ((size_t)(HEIGHT / 4) * WIDTH + WIDTH / 4) * ph->bpp,
          ph->nbytes);
   vkUnmapMemory(dev, bmem);

   vkDestroyCommandPool(dev, cp, NULL);
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

static void
fmt_bytes(char *out, size_t n, const uint8_t *b, unsigned count)
{
   int k = 0;
   for (unsigned i = 0; i < count && k >= 0 && (size_t)k < n; i++) {
      k += snprintf(out + k, n - (size_t)k, i ? ",%u" : "%u", b[i]);
   }
}

/* Report one phase. Returns true when it passed. */
static bool
check_phase(const struct phase *ph, const uint8_t centre[3], long covered)
{
   /* Footprint sanity: the triangle spans 1.8 NDC on both axes, ~1659 px on a
    * 64px framebuffer. If the draw itself went wrong the bytes mean nothing. */
   const long EXPECT_PX = 1659;
   const bool area_ok = covered >= EXPECT_PX - 200 && covered <= EXPECT_PX + 200;
   bool color_ok = true, is_wrong = true;
   for (unsigned i = 0; i < ph->nbytes; i++) {
      const int d = (int)centre[i] - (int)ph->expect[i];
      const int w = (int)centre[i] - (int)ph->wrong[i];
      if (d < -TOL || d > TOL) {
         color_ok = false;
      }
      if (w < -TOL || w > TOL) {
         is_wrong = false;
      }
   }
   char got[32], want[32];
   fmt_bytes(got, sizeof got, centre, ph->nbytes);
   fmt_bytes(want, sizeof want, ph->expect, ph->nbytes);
   if (area_ok && color_ok) {
      printf("PASSED (omfmt %s: %u-byte texel, memory bytes = %s over %ld px)\n",
             ph->name, ph->bpp, got, covered);
      return true;
   }
   printf("FAILED (omfmt %s: %u-byte texel, %ld px covered, memory bytes = %s "
          "-- expected ~%ld px at %s%s%s)\n",
          ph->name, ph->bpp, covered, got, EXPECT_PX, want,
          is_wrong ? "; " : "", is_wrong ? ph->wrong_why : "");
   return false;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "omfmt.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "omfmt.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-omfmt",
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
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
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

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      return 1;
   }

   bool ok = true;
   for (unsigned i = 0; i < NUM_PHASES; i++) {
      /* An attachment format the driver does not advertise is a failure here,
       * not a skip: the whole point is that these four are renderable. */
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, PHASES[i].fmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
         printf("FAILED (omfmt %s: not advertised as a colour attachment)\n",
                PHASES[i].name);
         ok = false;
         continue;
      }
      uint8_t centre[3] = { 0, 0, 0 };
      long covered = 0;
      if (render(dev, queue, qf, &mp, vs, fs, &PHASES[i], centre, &covered) < 0) {
         return 1;
      }
      if (!check_phase(&PHASES[i], centre, covered)) {
         ok = false;
      }
   }

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return ok ? 0 : 1;
}
