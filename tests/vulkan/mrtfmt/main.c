/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two colour attachments of different formats in one draw.
 *
 * `omfmt` establishes that the merger can encode R8, RG8 and sRGB; this asks
 * whether it can encode two of them at once. Every render target carries its
 * own merger state, so the storage format and the texel width belong to the
 * attachment rather than to the framebuffer, and each of the two phases below
 * pins one half of that:
 *
 *   rgba_srgb  two formats of the same width. Only the encode differs, so a
 *              driver that carried one format for the whole set stores RT1's
 *              linear values where its sRGB encode belongs.
 *   r8_rg8     two widths. Sizes, row pitches and readbacks are per attachment
 *              here, so a driver that measured RT1 in RT0's texel width writes
 *              it at the wrong stride -- visible as a smear in the footprint,
 *              not just as wrong bytes at the centre.
 *   rgba_r8    two channel orders, which is the G-buffer case. The merger reads
 *              an R8 texel's channel from a different lane than an RGBA8 one,
 *              so the fragment kernel has to pack the two targets differently
 *              in the same shader.
 *
 * The fragment shader writes the same three values to both targets in opposite
 * orders, so an attachment left at the clear and an attachment written with its
 * neighbour's colour both fail.
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
#define NUM_RT  2u

/* Expected bytes are stated as memory bytes, low byte first, `nbytes` of them. */
struct rtdesc {
   VkFormat fmt;
   unsigned bpp;
   unsigned nbytes;
   uint8_t  expect[3];
};

struct phase {
   const char   *name;
   struct rtdesc rt[NUM_RT];
   const char   *why;
};

/* RT0 takes the fragment colour (1.0, 0.5, 0.25) and RT1 its reverse. The
 * device quantises each channel to UNORM8 by truncation -- 255, 127, 63 -- and
 * then each attachment's own format encodes it:
 *
 *   R8G8B8A8  stores the truncated values verbatim
 *   SRGB      applies the sRGB transfer function to the colour channels, so
 *             63, 127, 255 is stored as 136, 187, 255
 *   R8, RG8   store the first one or two channels at one or two bytes a texel
 */
static const struct phase PHASES[] = {
   { "rgba_srgb",
     { { VK_FORMAT_R8G8B8A8_UNORM, 4, 3, { 255, 127,  63 } },
       { VK_FORMAT_R8G8B8A8_SRGB,  4, 3, { 136, 187, 255 } } },
     "RT1 holds its linear values, so the merger encoded it in RT0's format" },
   { "r8_rg8",
     { { VK_FORMAT_R8_UNORM,       1, 1, { 255,   0,   0 } },
       { VK_FORMAT_R8G8_UNORM,     2, 2, {  63, 127,   0 } } },
     "RT1 is wrong at the centre or covers the wrong area, so the merger "
     "measured it in RT0's texel width" },
   { "rgba_r8",
     { { VK_FORMAT_R8G8B8A8_UNORM, 4, 3, { 255, 127,  63 } },
       { VK_FORMAT_R8_UNORM,       1, 1, {  63,   0,   0 } } },
     "RT1 holds the fragment's blue, so the shader packed it in RT0's "
     "channel order" },
};
#define NUM_PHASES (sizeof(PHASES) / sizeof(PHASES[0]))

/* The device applies the sRGB transfer function to a linear value already
 * truncated to 8 bits, where llvmpipe converts in float; on the steep part of
 * the curve that is worth a few LSBs. Every defect signature here is at least
 * 60 away, so this cannot absorb one. */
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

/* Draw the triangle into this phase's two attachments and report each one's
 * centre texel and footprint. Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs, const struct phase *ph,
       uint8_t centre[NUM_RT][3], long covered[NUM_RT])
{
   VkImage        img[NUM_RT];
   VkDeviceMemory imem[NUM_RT];
   VkImageView    view[NUM_RT];
   VkBuffer       rb[NUM_RT];
   VkDeviceMemory bmem[NUM_RT];

   for (unsigned k = 0; k < NUM_RT; k++) {
      VkImageCreateInfo imci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = ph->rt[k].fmt,
         .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      CHECK(vkCreateImage(dev, &imci, NULL, &img[k]));
      VkMemoryRequirements imr;
      vkGetImageMemoryRequirements(dev, img[k], &imr);
      uint32_t imt = find_mem(mp, imr.memoryTypeBits, 0);
      if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return -1; }
      VkMemoryAllocateInfo imai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = imr.size, .memoryTypeIndex = imt,
      };
      CHECK(vkAllocateMemory(dev, &imai, NULL, &imem[k]));
      CHECK(vkBindImageMemory(dev, img[k], imem[k], 0));

      VkImageViewCreateInfo ivci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = img[k], .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = ph->rt[k].fmt,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1, .layerCount = 1,
         },
      };
      CHECK(vkCreateImageView(dev, &ivci, NULL, &view[k]));
   }

   VkAttachmentDescription att[NUM_RT];
   VkAttachmentReference   attref[NUM_RT];
   for (unsigned k = 0; k < NUM_RT; k++) {
      att[k] = (VkAttachmentDescription){
         .format = ph->rt[k].fmt, .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      };
      attref[k] = (VkAttachmentReference){
         .attachment = k, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
   }
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = NUM_RT, .pColorAttachments = attref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = NUM_RT, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = NUM_RT, .pAttachments = view,
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
    * gl_VertexIndex, so nothing but the fragment colours is under test. */
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
   VkPipelineColorBlendAttachmentState cba[NUM_RT];
   for (unsigned k = 0; k < NUM_RT; k++) {
      cba[k] = (VkPipelineColorBlendAttachmentState){
         .blendEnable = VK_FALSE,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
   }
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = NUM_RT, .pAttachments = cba,
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

   for (unsigned k = 0; k < NUM_RT; k++) {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = (VkDeviceSize)WIDTH * HEIGHT * ph->rt[k].bpp,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &rb[k]));
      VkMemoryRequirements bmr;
      vkGetBufferMemoryRequirements(dev, rb[k], &bmr);
      uint32_t bmt = find_mem(mp, bmr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return -1; }
      VkMemoryAllocateInfo bmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = bmr.size, .memoryTypeIndex = bmt,
      };
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem[k]));
      CHECK(vkBindBufferMemory(dev, rb[k], bmem[k], 0));
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
   VkClearValue clear[NUM_RT];
   for (unsigned k = 0; k < NUM_RT; k++) {
      clear[k] = (VkClearValue){ .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   }
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = NUM_RT, .pClearValues = clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   for (unsigned k = 0; k < NUM_RT; k++) {
      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb[k], 1, &region);
   }
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   for (unsigned k = 0; k < NUM_RT; k++) {
      const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * ph->rt[k].bpp;
      uint8_t *px;
      CHECK(vkMapMemory(dev, bmem[k], 0, bytes, 0, (void **)&px));
      covered[k] = 0;
      for (uint32_t y = 0; y < HEIGHT; y++) {
         for (uint32_t x = 0; x < WIDTH; x++) {
            const uint8_t *p = px + ((size_t)y * WIDTH + x) * ph->rt[k].bpp;
            for (unsigned c = 0; c < ph->rt[k].nbytes; c++) {
               if (p[c]) { covered[k]++; break; }
            }
         }
      }
      /* The triangle's lower-left corner is at (-0.9,-0.9) and it spans 1.8 NDC
       * on each axis, so a quarter of the way in on both axes is comfortably
       * inside it and clear of every edge. */
      memset(centre[k], 0, 3);
      memcpy(centre[k],
             px + ((size_t)(HEIGHT / 4) * WIDTH + WIDTH / 4) * ph->rt[k].bpp,
             ph->rt[k].nbytes);
      vkUnmapMemory(dev, bmem[k]);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   for (unsigned k = 0; k < NUM_RT; k++) {
      vkFreeMemory(dev, bmem[k], NULL);
      vkDestroyBuffer(dev, rb[k], NULL);
      vkDestroyImageView(dev, view[k], NULL);
      vkFreeMemory(dev, imem[k], NULL);
      vkDestroyImage(dev, img[k], NULL);
   }
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

/* Report one phase. Both attachments have to be right: a check that looked at
 * RT0 alone would pass with RT1 never written. Returns true when it passed. */
static bool
check_phase(const struct phase *ph, uint8_t centre[NUM_RT][3],
            const long covered[NUM_RT])
{
   /* Footprint sanity: the triangle spans 1.8 NDC on both axes, ~1659 px on a
    * 64px framebuffer. A target written at the wrong row stride lands outside
    * that band even when its centre texel happens to look right. */
   const long EXPECT_PX = 1659;
   bool ok = true;
   for (unsigned k = 0; k < NUM_RT; k++) {
      const struct rtdesc *r = &ph->rt[k];
      const bool area_ok = covered[k] >= EXPECT_PX - 200 &&
                           covered[k] <= EXPECT_PX + 200;
      bool color_ok = true;
      for (unsigned i = 0; i < r->nbytes; i++) {
         const int d = (int)centre[k][i] - (int)r->expect[i];
         if (d < -TOL || d > TOL) {
            color_ok = false;
         }
      }
      char got[32], want[32];
      fmt_bytes(got, sizeof got, centre[k], r->nbytes);
      fmt_bytes(want, sizeof want, r->expect, r->nbytes);
      if (area_ok && color_ok) {
         printf("PASSED (mrtfmt %s rt%u: %u-byte texel, memory bytes = %s "
                "over %ld px)\n", ph->name, k, r->bpp, got, covered[k]);
         continue;
      }
      printf("FAILED (mrtfmt %s rt%u: %u-byte texel, %ld px covered, memory "
             "bytes = %s -- expected ~%ld px at %s; %s)\n",
             ph->name, k, r->bpp, covered[k], got, EXPECT_PX, want, ph->why);
      ok = false;
   }
   return ok;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "mrtfmt.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "mrtfmt.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-mrtfmt",
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

   /* One draw writes both targets, so the device has to bind at least two. */
   if (props.limits.maxColorAttachments < NUM_RT) {
      printf("FAILED (mrtfmt: device binds %u colour attachments, needs %u)\n",
             props.limits.maxColorAttachments, NUM_RT);
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
       * not a skip: omfmt already renders each of these alone, so the only
       * thing under test is carrying two of them at once. */
      bool advertised = true;
      for (unsigned k = 0; k < NUM_RT; k++) {
         VkFormatProperties fp;
         vkGetPhysicalDeviceFormatProperties(pd, PHASES[i].rt[k].fmt, &fp);
         if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
            printf("FAILED (mrtfmt %s rt%u: not advertised as a colour "
                   "attachment)\n", PHASES[i].name, k);
            advertised = false;
         }
      }
      if (!advertised) {
         ok = false;
         continue;
      }
      uint8_t centre[NUM_RT][3];
      long covered[NUM_RT] = { 0 };
      memset(centre, 0, sizeof centre);
      if (render(dev, queue, qf, &mp, vs, fs, &PHASES[i], centre, covered) < 0) {
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
