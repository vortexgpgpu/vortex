/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Hello-triangle Vulkan test for the vortexpipe driver.
 *
 * Renders one triangle off-screen into a 64x64 RGBA8 image, copies
 * it to a host-visible buffer, and checks the result: the centre
 * pixel is inside the triangle (non-black), a corner is the clear
 * colour (black), and the coloured-pixel count is within the
 * triangle's expected coverage.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe: it exercises
 * vkCreateGraphicsPipelines (vertex stage -> vortexpipe) and
 * vkCmdDraw. The fragment stage stays on llvmpipe's CPU path.
 *
 * It is also the suite's only test that binds both a compute and a graphics
 * pipeline in one context, which is its own driver case: compute kernels and
 * fragment shaders are linked at the same fixed device address, so only one can
 * be resident and whichever holds it must release it when the other runs. A
 * compute dispatch either side of the draw makes both handovers happen, and the
 * buffer must come back incremented twice.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CN      64u    /* compute elements, one dispatch's worth */
#define CLOCAL  16u    /* compute local_size_x upper bound */
#define WIDTH   64u
#define HEIGHT  64u
#define FORMAT  VK_FORMAT_R8G8B8A8_UNORM

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

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "triangle.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "triangle.frag.spv";
   const char *cs_path = (argc > 3) ? argv[3] : "triangle.comp.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-triangle",
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

   /* --- colour attachment image ----------------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT,
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
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   /* --- render pass + framebuffer --------------------------------- */
   VkAttachmentDescription att = {
      .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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

   /* --- shader modules -------------------------------------------- */
   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) return 1;

   /* --- graphics pipeline (vertex stage -> vortexpipe) ------------ */
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- the compute half ------------------------------------------- */
   VkShaderModule cs = load_module(dev, cs_path);
   if (cs == VK_NULL_HANDLE) {
      fprintf(stderr, "FAILED: could not load %s\n", cs_path);
      return 1;
   }
   const VkDeviceSize cbytes = (VkDeviceSize)CN * sizeof(uint32_t);
   VkBufferCreateInfo cbci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = cbytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer cbuf;
   CHECK(vkCreateBuffer(dev, &cbci, NULL, &cbuf));
   VkMemoryRequirements cmr;
   vkGetBufferMemoryRequirements(dev, cbuf, &cmr);
   VkPhysicalDeviceMemoryProperties cmp;
   vkGetPhysicalDeviceMemoryProperties(pd, &cmp);
   uint32_t cmt = find_mem(&cmp, cmr.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (cmt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no host-visible memory for the compute buffer\n");
      return 1;
   }
   VkMemoryAllocateInfo cmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = cmr.size, .memoryTypeIndex = cmt,
   };
   VkDeviceMemory cmem;
   CHECK(vkAllocateMemory(dev, &cmai, NULL, &cmem));
   CHECK(vkBindBufferMemory(dev, cbuf, cmem, 0));
   uint32_t *cp_map;
   CHECK(vkMapMemory(dev, cmem, 0, cbytes, 0, (void **)&cp_map));
   for (uint32_t i = 0; i < CN; i++)
      cp_map[i] = i;
   vkUnmapMemory(dev, cmem);

   VkDescriptorSetLayoutBinding cdslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo cdslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &cdslb,
   };
   VkDescriptorSetLayout cdsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &cdslci, NULL, &cdsl));
   VkPipelineLayoutCreateInfo cplci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &cdsl,
   };
   VkPipelineLayout cpl;
   CHECK(vkCreatePipelineLayout(dev, &cplci, NULL, &cpl));

   uint32_t clocal = CLOCAL < props.limits.maxComputeWorkGroupSize[0]
                   ? CLOCAL : props.limits.maxComputeWorkGroupSize[0];
   while (clocal > 1 && (CN % clocal) != 0)
      clocal >>= 1;
   if (clocal == 0)
      clocal = 1;
   VkSpecializationMapEntry csme = {
      .constantID = 0, .offset = 0, .size = sizeof(uint32_t),
   };
   VkSpecializationInfo cspec = {
      .mapEntryCount = 1, .pMapEntries = &csme,
      .dataSize = sizeof(uint32_t), .pData = &clocal,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main",
         .pSpecializationInfo = &cspec,
      },
      .layout = cpl,
   };
   VkPipeline cpipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe));

   VkDescriptorPoolSize cdps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
   };
   VkDescriptorPoolCreateInfo cdpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &cdps,
   };
   VkDescriptorPool cdp;
   CHECK(vkCreateDescriptorPool(dev, &cdpci, NULL, &cdp));
   VkDescriptorSetAllocateInfo cdsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = cdp, .descriptorSetCount = 1, .pSetLayouts = &cdsl,
   };
   VkDescriptorSet cds;
   CHECK(vkAllocateDescriptorSets(dev, &cdsai, &cds));
   VkDescriptorBufferInfo cdbi = {
      .buffer = cbuf, .offset = 0, .range = cbytes,
   };
   VkWriteDescriptorSet cwds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = cds, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &cdbi,
   };
   vkUpdateDescriptorSets(dev, 1, &cwds, 0, NULL);

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

   /* --- host-visible readback buffer ------------------------------ */
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

   /* --- command buffer: render pass + copy-to-buffer -------------- */
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

   /* Compute first: the kernel takes the shared device address. */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpl,
                           0, 1, &cds, 0, NULL);
   vkCmdDispatch(cmd, CN / clocal, 1, 1);

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

   /* Compute again, now that a fragment shader has taken the address the
    * kernel was using. Without the handover the kernel cannot load and the
    * dispatch fails outright. */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpl,
                           0, 1, &cds, 0, NULL);
   vkCmdDispatch(cmd, CN / clocal, 1, 1);

   /* render pass finalLayout left the image in TRANSFER_SRC_OPTIMAL */
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

   /* --- read back + verify ---------------------------------------- */
   uint8_t *px;
   CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));

   unsigned colored = 0;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      const uint8_t *p = px + (size_t)i * 4;
      if (p[0] || p[1] || p[2]) colored++;
   }
   const uint8_t *centre = px + ((size_t)(HEIGHT / 2) * WIDTH + WIDTH / 2) * 4;
   const uint8_t *corner = px + ((size_t)2 * WIDTH + 2) * 4;
   bool centre_lit = centre[0] || centre[1] || centre[2];
   bool corner_clear = !(corner[0] || corner[1] || corner[2]);
   vkUnmapMemory(dev, bmem);

   /* the triangle covers roughly an eighth of the frame */
   bool coverage_ok = colored > 100u && colored < 3500u;

   /* Both dispatches ran, and the second one survived the draw taking the
    * device address in between: data[i] was seeded i and incremented twice. */
   unsigned cfails = 0;
   CHECK(vkMapMemory(dev, cmem, 0, cbytes, 0, (void **)&cp_map));
   for (uint32_t i = 0; i < CN; i++) {
      if (cp_map[i] != i + 2u) {
         if (cfails < 5)
            fprintf(stderr, "  compute data[%u] = %u, want %u\n",
                    i, cp_map[i], i + 2u);
         cfails++;
      }
   }
   vkUnmapMemory(dev, cmem);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!centre_lit || !corner_clear || !coverage_ok || cfails) {
      printf("FAILED (centre_lit=%d corner_clear=%d colored=%u compute_fails=%u)\n",
             centre_lit, corner_clear, colored, cfails);
      return 1;
   }
   printf("PASSED (triangle rendered, %u/%u pixels covered; compute dispatched"
          " either side of the draw)\n", colored, WIDTH * HEIGHT);
   return 0;
}
