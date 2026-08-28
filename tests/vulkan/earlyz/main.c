/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Depth-test invariants that an early-Z stage must not break.
 *
 * Nothing here measures a cull -- a cull is invisible in the image by
 * construction, and no configuration currently performs one. What the cases fix
 * is the two properties a cull is allowed to preserve and nothing else:
 *
 *   overdraw  several full-screen quads at different depths, drawn far-to-near
 *             and then near-to-far. Both orders must produce the same image. An
 *             over-eager cull drops a fragment that should have won and the two
 *             orders diverge.
 *
 *   fsdepth   the fragment shader writes a depth contradicting its plane. The
 *             test must follow what the shader wrote. A stage that tested the
 *             plane instead keeps the other quad, so the colour says which
 *             depth was used rather than only that something was wrong.
 *
 * Full-screen quads throughout: coverage is then never the variable, so a
 * failure is a depth decision and cannot be an edge rule.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   32u
#define HEIGHT  32u
#define FORMAT       VK_FORMAT_R8G8B8A8_UNORM
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* Matches the push_constant block in the shaders: two floats, then a vec4 which
 * std430 places at offset 16. */
struct pc_t {
   float z;
   float zwrite;
   float pad[2];
   float color[4];
};

/* One draw. */
struct draw {
   float z;         /* plane depth */
   float zwrite;    /* depth the shader writes, when the shader writes one */
   float color[4];
};

struct zcase {
   const char        *name;
   bool               fs_writes_depth;
   uint32_t           ndraws;
   struct draw        draws[4];
   uint8_t            expect[3];   /* the colour every pixel must end up */
};

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
   /* Three depths well apart, one colour each, so the surviving colour names
    * the surviving depth. */
   struct zcase cases[] = {
      /* Far to near: every quad passes LESS as it is drawn, so all three run
       * and the nearest is left. */
      { "overdraw_far_to_near", false, 3,
        { { 0.75f, 0.0f, { 1, 0, 0, 1 } },
          { 0.50f, 0.0f, { 0, 1, 0, 1 } },
          { 0.25f, 0.0f, { 0, 0, 1, 1 } } },
        { 0, 0, 255 } },

      /* Near to far: the first quad wins and the other two fail the test. Same
       * image as above -- that equality is the assertion. */
      { "overdraw_near_to_far", false, 3,
        { { 0.25f, 0.0f, { 0, 0, 1, 1 } },
          { 0.50f, 0.0f, { 0, 1, 0, 1 } },
          { 0.75f, 0.0f, { 1, 0, 0, 1 } } },
        { 0, 0, 255 } },

      /* The red quad's plane is nearest (0.25) but it writes 0.75, so green at
       * 0.50 must beat it. Testing the plane instead keeps red, and red is not
       * a colour any correct ordering can produce here. */
      { "fsdepth", true, 2,
        { { 0.25f, 0.75f, { 1, 0, 0, 1 } },
          { 0.50f, 0.50f, { 0, 1, 0, 1 } } },
        { 0, 255, 0 } },
   };
   const uint32_t ncases = (uint32_t)(sizeof(cases) / sizeof(cases[0]));

   const char *vs_path  = (argc > 1) ? argv[1] : "earlyz.vert.spv";
   const char *fs_path  = (argc > 2) ? argv[2] : "earlyz.frag.spv";
   const char *fsd_path = (argc > 3) ? argv[3] : "earlyz_depth.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-earlyz",
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

   /* --- colour + depth attachments -------------------------------- */
   VkImageCreateInfo cimci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage cimg;
   CHECK(vkCreateImage(dev, &cimci, NULL, &cimg));
   VkDeviceMemory cmem;
   CHECK(alloc_image(dev, &mp, cimg, &cmem));

   VkImageCreateInfo dimci = cimci;
   dimci.format = DEPTH_FORMAT;
   dimci.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage dimg;
   CHECK(vkCreateImage(dev, &dimci, NULL, &dimg));
   VkDeviceMemory dmem;
   CHECK(alloc_image(dev, &mp, dimg, &dmem));

   VkImageViewCreateInfo civci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = cimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView cview;
   CHECK(vkCreateImageView(dev, &civci, NULL, &cview));

   VkImageViewCreateInfo divci = civci;
   divci.image  = dimg;
   divci.format = DEPTH_FORMAT;
   divci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkImageView dview;
   CHECK(vkCreateImageView(dev, &divci, NULL, &dview));

   VkAttachmentDescription att[2] = {
      { .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = DEPTH_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference cref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &dref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fbviews[2] = { cview, dview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = fbviews,
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

   /* --- two pipelines: with and without a shader-written depth ---- */
   VkShaderModule vs  = load_module(dev, vs_path);
   VkShaderModule fs  = load_module(dev, fs_path);
   VkShaderModule fsd = load_module(dev, fsd_path);
   if (!vs || !fs || !fsd) return 1;

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0, .size = sizeof(struct pc_t),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

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
   /* LESS with depth writes on: monotone, which is the condition the driver
    * requires before it arms early-Z at all. */
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
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

   VkPipeline pipes[2];
   for (uint32_t k = 0; k < 2; k++) {
      VkPipelineShaderStageCreateInfo stages[2] = {
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
           .module = k ? fsd : fs, .pName = "main" },
      };
      VkGraphicsPipelineCreateInfo gpci = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = &ms, .pDepthStencilState = &ds,
         .pColorBlendState = &cb,
         .layout = pl, .renderPass = rp, .subpass = 0,
      };
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL,
                                      &pipes[k]));
   }

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

   /* --- run every case -------------------------------------------- */
   unsigned failed = 0;
   for (uint32_t ci = 0; ci < ncases; ci++) {
      const struct zcase *c = &cases[ci];

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));

      /* Clear depth to 1.0 so every quad is in front of it, and colour to
       * black -- a colour no case expects, so an undrawn target is a failure
       * rather than an accidental pass. */
      VkClearValue clears[2] = {
         { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
         { .depthStencil = { 1.0f, 0 } },
      };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 2, .pClearValues = clears,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipes[c->fs_writes_depth ? 1 : 0]);
      for (uint32_t d = 0; d < c->ndraws; d++) {
         struct pc_t pc = { .z = c->draws[d].z, .zwrite = c->draws[d].zwrite };
         memcpy(pc.color, c->draws[d].color, sizeof(pc.color));
         vkCmdPushConstants(cmd, pl,
                            VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, sizeof(pc), &pc);
         vkCmdDraw(cmd, 6, 1, 0, 0);
      }
      vkCmdEndRenderPass(cmd);

      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));

      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      /* Every pixel is covered by every quad, so a single disagreeing pixel is
       * as much a failure as all of them. */
      uint8_t *px;
      CHECK(vkMapMemory(dev, bmem, 0, rb_bytes, 0, (void **)&px));
      uint32_t bad = 0;
      uint8_t got[3] = { px[0], px[1], px[2] };
      for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
         const uint8_t *p = px + (size_t)i * 4;
         if (p[0] != c->expect[0] || p[1] != c->expect[1] || p[2] != c->expect[2]) {
            if (bad == 0) { got[0] = p[0]; got[1] = p[1]; got[2] = p[2]; }
            bad++;
         }
      }
      vkUnmapMemory(dev, bmem);

      bool ok = (bad == 0);
      printf("%s: %s  got=(%u,%u,%u) want=(%u,%u,%u) wrong=%u of %u\n",
             c->name, ok ? "pass" : "FAIL",
             got[0], got[1], got[2],
             c->expect[0], c->expect[1], c->expect[2], bad, WIDTH * HEIGHT);
      if (!ok) failed++;
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipes[0], NULL);
   vkDestroyPipeline(dev, pipes[1], NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyShaderModule(dev, fsd, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImageView(dev, cview, NULL);
   vkFreeMemory(dev, dmem, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkDestroyImage(dev, cimg, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u case(s))\n", failed);
      return 1;
   }
   printf("PASSED (draw order does not change the image; a written depth wins)\n");
   return 0;
}
