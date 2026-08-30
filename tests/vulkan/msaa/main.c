/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * 4x multisample + resolve test for the vortexpipe driver.
 *
 * The output merger keeps one colour per sample and the pass-end resolve
 * averages a pixel's four samples into one. Two cases separate the two halves
 * of that:
 *
 *   msaa_solid -- an oversized quad covers every sample of every pixel, so the
 *                 resolve averages four identical values. Every resolved pixel
 *                 must be exactly the fill. A pixel that comes back partial
 *                 here means samples were lost, not that an edge crossed it.
 *   msaa_edge  -- one triangle with no axis-aligned and no 45-degree edge. Its
 *                 interior must be exactly the fill, its exterior exactly the
 *                 clear, and the band between them must be non-empty and lie
 *                 strictly between the two. That band is the whole point of
 *                 multisampling: a single-sample render produces none of it,
 *                 so the case cannot pass by rasterizing at 1x.
 *
 * The shaded colour is a constant, so a resolved value carries coverage and
 * nothing else. Clear is black and fill is opaque red, which leaves green and
 * blue at zero in both operands -- any pixel with green or blue set is
 * corruption rather than a coverage fraction, and is counted separately.
 *
 * The exact partial values are deliberately not asserted. A box filter over
 * four samples admits more than one rounding, and pinning one of them would
 * test an implementation rather than the feature.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u
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
   if (buf) *out_size = (size_t)sz;
   return buf;
}

/* first memory type satisfying `want`; UINT32_MAX if none. */
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

/* Allocate + bind device-local memory for `img`, returning its memory. */
static bool
bind_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
           VkImage img, VkDeviceMemory *out_mem)
{
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits, 0);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return false; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) return false;
   return vkBindImageMemory(dev, img, *out_mem, 0) == VK_SUCCESS;
}

struct msaacase {
   const char *name;
   const char *vs_path;   /* filled from argv */
   uint32_t    verts;
   bool        want_partial;  /* the edge case must produce a partial band */
};

int
main(int argc, char **argv)
{
   const char *quad_vs = (argc > 1) ? argv[1] : "msaa.vert.spv";
   const char *tri_vs  = (argc > 2) ? argv[2] : "msaa_tri.vert.spv";
   const char *fs_path = (argc > 3) ? argv[3] : "msaa.frag.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   /* --- physical device + graphics queue family ------------------- */
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* 4x has to be advertised for both the colour attachment and the
    * rasterizer. A device that does not offer it cannot run this case, and
    * saying so is a failure rather than a skip: the driver claims 4x. */
   const VkSampleCountFlags need = (VkSampleCountFlags)SAMPLES;
   if (!(props.limits.framebufferColorSampleCounts & need)) {
      printf("FAILED (device does not advertise 4x colour samples: 0x%x)\n",
             (unsigned)props.limits.framebufferColorSampleCounts);
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

   /* --- logical device + queue ------------------------------------ */
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

   /* --- multisample colour attachment + single-sample resolve target -- */
   VkImageCreateInfo msci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = SAMPLES, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage msimg;
   CHECK(vkCreateImage(dev, &msci, NULL, &msimg));
   VkDeviceMemory msmem;
   if (!bind_image(dev, &mp, msimg, &msmem)) return 1;

   VkImageCreateInfo rsci = msci;
   rsci.samples = VK_SAMPLE_COUNT_1_BIT;
   rsci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   VkImage rsimg;
   CHECK(vkCreateImage(dev, &rsci, NULL, &rsimg));
   VkDeviceMemory rsmem;
   if (!bind_image(dev, &mp, rsimg, &rsmem)) return 1;

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView msview, rsview;
   ivci.image = msimg;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &msview));
   ivci.image = rsimg;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &rsview));

   /* --- render pass: attachment 0 multisampled, 1 the resolve target -- *
    * The multisample attachment is never stored -- the resolve is what the
    * pass produces -- so only the resolve target carries STORE and a layout
    * the copy below can read. */
   VkAttachmentDescription atts[2] = {
      { .format = FORMAT, .samples = SAMPLES,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
      { .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
   };
   VkAttachmentReference msref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference rsref = {
      .attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &msref,
      .pResolveAttachments = &rsref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fbviews[2] = { msview, rsview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = fbviews,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* --- shared fragment module, layout, readback buffer, pool ------ */
   VkShaderModule fs = load_module(dev, fs_path);
   if (!fs) return 1;

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
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

   struct msaacase cases[] = {
      { "msaa_solid", quad_vs, 6, false },
      { "msaa_edge",  tri_vs,  3, true  },
   };

   unsigned failed = 0;
   for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
      const struct msaacase *c = &cases[ci];

      VkShaderModule vs = load_module(dev, c->vs_path);
      if (!vs) return 1;

      VkPipelineShaderStageCreateInfo stages[2] = {
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
      };
      /* gl_VertexIndex-driven: no vertex buffers, no attributes. */
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
         .rasterizationSamples = SAMPLES,
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

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));

      /* pClearValues is indexed by attachment, so the resolve target needs an
       * entry even though it is never cleared. */
      VkClearValue clears[2] = {
         { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
         { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
      };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 2, .pClearValues = clears,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdDraw(cmd, c->verts, 1, 0, 0);
      vkCmdEndRenderPass(cmd);

      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, rsimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));

      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      /* Clear and fill agree on green and blue, so a resolved pixel is a
       * point on the red axis and its red value is the covered fraction.
       * Anything off that axis is corruption, never coverage. */
      uint8_t *px;
      CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
      unsigned fill = 0, clear = 0, partial = 0, offaxis = 0;
      uint8_t lo = 255, hi = 0;
      for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
         const uint8_t *p = px + (size_t)i * 4;
         if (p[1] != 0 || p[2] != 0) { offaxis++; continue; }
         if (p[0] == 255)     fill++;
         else if (p[0] == 0)  clear++;
         else {
            partial++;
            if (p[0] < lo) lo = p[0];
            if (p[0] > hi) hi = p[0];
         }
      }
      vkUnmapMemory(dev, bmem);

      bool ok;
      if (c->want_partial) {
         ok = offaxis == 0 && fill > 0 && clear > 0 && partial > 0;
      } else {
         ok = offaxis == 0 && partial == 0 && fill == WIDTH * HEIGHT;
      }
      if (partial > 0) {
         printf("%s: %s (fill=%u clear=%u partial=%u offaxis=%u, "
                "partial red in [%u,%u])\n", c->name, ok ? "pass" : "FAIL",
                fill, clear, partial, offaxis, lo, hi);
      } else {
         printf("%s: %s (fill=%u clear=%u partial=%u offaxis=%u)\n",
                c->name, ok ? "pass" : "FAIL", fill, clear, partial, offaxis);
      }
      if (!ok) failed++;

      vkDestroyPipeline(dev, pipe, NULL);
      vkDestroyShaderModule(dev, vs, NULL);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u case(s))\n", failed);
      return 1;
   }
   printf("PASSED (4x multisample + resolve: solid fill exact, edge band present)\n");
   return 0;
}
