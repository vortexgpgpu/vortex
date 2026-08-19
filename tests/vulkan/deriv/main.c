/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Screen-space derivatives in the fragment stage.
 *
 * dFdx and dFdy are core fragment features -- nothing gates them behind a
 * feature bit, so they are advertised unconditionally -- and the device
 * computes them by shuffling between the four lanes of a pixel quad. Nothing in
 * the suite used them, so the quad lane packing they depend on had no direct
 * coverage; it was reachable only indirectly, through texture LOD.
 *
 * The varying spans 0..1 over the 64-pixel target, so the expected derivative
 * is 1/64 per pixel. The shader scales by 64, making the answer exactly 1.0 and
 * a dead derivative exactly 0 -- a full-range separation rather than a fraction
 * of a quantisation step.
 *
 * The cross terms are checked too: v_uv.x must not vary along y. A quad shuffle
 * that picked the wrong neighbour would leave the axis derivatives plausible
 * while making the cross terms non-zero.
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

/* Three interior points, well away from the target edge. The derivative is
 * constant across a linear varying, so all three must agree; a disagreement
 * would mean it depends on where in the quad grid the pixel lands. */
static const struct { const char *where; float fx, fy; } SAMPLES[] = {
   { "lower-left",  0.25f, 0.25f },
   { "centre",      0.50f, 0.50f },
   { "upper-right", 0.75f, 0.75f },
};
#define NUM_SAMPLES (sizeof(SAMPLES) / sizeof(SAMPLES[0]))

/* dFdx(u) and dFdy(v) are both 1/64 per pixel, scaled to 1.0 -> 255. The cross
 * terms are zero. A dead derivative reads 0 where 255 belongs, which is the
 * whole dynamic range away. */
#define EXPECT_R  255
#define EXPECT_G  255
#define EXPECT_B  0

/* The expected values sit at the ends of the range, so one count of slack
 * covers the UNORM8 encoding without admitting anything else. It must stay this
 * tight: every failure this test exists to catch moves a channel by the full
 * 255, and a tolerance chosen loosely enough to matter would have to be large
 * enough to accept one of them. */
#define TOL 1

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

/* Draw the triangle and report the three sample texels and the footprint.
 * Returns 0, or -1 on a Vulkan error. */
static int
render(VkDevice dev, VkQueue queue, uint32_t qf,
       const VkPhysicalDeviceMemoryProperties *mp,
       VkShaderModule vs, VkShaderModule fs,
       uint8_t got[NUM_SAMPLES][3], long *out_covered)
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
   /* No vertex input: the vertex shader builds its positions and its varying
    * from gl_VertexIndex, so nothing but the derivative is under test. */
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
         const uint8_t *p = px + ((size_t)y * WIDTH + x) * 4;
         if (p[0] || p[1] || p[2]) { (*out_covered)++; }
      }
   }
   for (unsigned s = 0; s < NUM_SAMPLES; s++) {
      const uint32_t sx = (uint32_t)(SAMPLES[s].fx * (float)WIDTH);
      const uint32_t sy = (uint32_t)(SAMPLES[s].fy * (float)HEIGHT);
      memcpy(got[s], px + ((size_t)sy * WIDTH + sx) * 4, 3);
   }
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

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "deriv.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "deriv.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-deriv",
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

   uint8_t got[NUM_SAMPLES][3];
   long covered = 0;
   memset(got, 0, sizeof got);
   if (render(dev, queue, qf, &mp, vs, fs, got, &covered) < 0) {
      return 1;
   }

   /* Footprint sanity: this triangle is deliberately oversized, so it covers the
    * whole target. Anything less means the draw itself went wrong and the
    * derivatives below would be read from pixels the primitive never shaded. */
   const long EXPECT_PX = (long)WIDTH * HEIGHT;
   bool ok = covered == EXPECT_PX;
   if (!ok) {
      printf("FAILED (deriv: %ld px covered, expected the full %ld -- the draw "
             "itself is wrong, so the shaded values mean nothing)\n",
             covered, EXPECT_PX);
   }

   bool r_varies = false;
   for (unsigned k = 0; k < NUM_SAMPLES; k++) {
      const int dr = (int)got[k][0] - EXPECT_R;
      const int dg = (int)got[k][1] - EXPECT_G;
      const int db = (int)got[k][2] - EXPECT_B;
      if (dr < -TOL || dr > TOL || dg < -TOL || dg > TOL ||
          db < -TOL || db > TOL) {
         ok = false;
      }
      if (got[k][0] != got[0][0]) {
         r_varies = true;
      }
   }
   if (ok) {
      printf("PASSED (deriv: dFdx=%u dFdy=%u cross=%u at three points over "
             "%ld px)\n", got[0][0], got[0][1], got[0][2], covered);
   } else {
      for (unsigned k = 0; k < NUM_SAMPLES; k++) {
         printf("   %s: dFdx=%u dFdy=%u cross=%u (expected %u, %u, %u)\n",
                SAMPLES[k].where, got[k][0], got[k][1], got[k][2],
                EXPECT_R, EXPECT_G, EXPECT_B);
      }
      printf("FAILED (deriv: %ld px covered%s%s%s)\n", covered,
             (got[0][0] == 0 && got[0][1] == 0)
                ? "; both derivatives are zero, so the quad shuffle returned "
                  "the lane's own value instead of its neighbour's" : "",
             got[0][2] > TOL
                ? "; a cross term is non-zero, so the shuffle picked a "
                  "neighbour on the wrong axis" : "",
             r_varies ? "; dFdx differs between samples, though the varying is "
                        "linear across the whole primitive" : "");
   }

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return ok ? 0 : 1;
}
