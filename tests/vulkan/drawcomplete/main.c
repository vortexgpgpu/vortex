/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three draw forms on the device path, each checked by an outcome a host
 * fallback could not produce under MESA_VORTEX_STRICT:
 *
 *   restart   primitive restart on an indexed triangle list
 *             (VK_EXT_primitive_topology_list_restart). The restart index
 *             drops the triangle it leaves incomplete and starts counting
 *             again, so the draw must match the same list with the restart
 *             and the orphaned index taken out. A restart index taken as an
 *             ordinary index, or skipped, shifts the grouping by one: the
 *             right triangle is lost either way;
 *   no FS     a pipeline without a fragment shader still tests and writes
 *             depth, and writes no colour. It covers the left half at
 *             z = 0.25, then a green quad at z = 0.5 under LESS must land on
 *             the right half only, the left half keeping the clear colour;
 *   no        a render pass with no attachments still runs the fragment
 *   attach-   stage. A shader that only counts its invocations in a storage
 *   ments     buffer, over a full-target quad, must count every pixel once.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define FB_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define ZS_FORMAT VK_FORMAT_D32_SFLOAT

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

/* Memory of the wanted kind bound to a buffer or image. */
static VkDeviceMemory
bind_memory(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkBuffer buf, VkImage img, VkMemoryPropertyFlags want)
{
   VkMemoryRequirements mr;
   if (buf)
      vkGetBufferMemoryRequirements(dev, buf, &mr);
   else
      vkGetImageMemoryRequirements(dev, img, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits, want);
   if (mt == UINT32_MAX)
      return VK_NULL_HANDLE;
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   if (vkAllocateMemory(dev, &mai, NULL, &mem) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   VkResult r = buf ? vkBindBufferMemory(dev, buf, mem, 0)
                    : vkBindImageMemory(dev, img, mem, 0);
   if (r != VK_SUCCESS) {
      vkFreeMemory(dev, mem, NULL);
      return VK_NULL_HANDLE;
   }
   return mem;
}

struct pipe_desc {
   VkRenderPass rp;
   bool         restart;
   VkShaderModule fs;        /* VK_NULL_HANDLE: no fragment stage */
   bool         depth_test, depth_write;
   VkCompareOp  depth_op;
   uint32_t     num_color;   /* colour attachments in the subpass */
};

static VkPipeline
make_pipeline(VkDevice dev, VkPipelineLayout pl, VkShaderModule vs,
              const struct pipe_desc *d)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = d->fs, .pName = "main" },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = d->restart,
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
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = d->depth_test, .depthWriteEnable = d->depth_write,
      .depthCompareOp = d->depth_op,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = d->num_color, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = d->fs ? 2 : 1, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb,
      .layout = pl, .renderPass = d->rp, .subpass = 0,
   };
   VkPipeline p = VK_NULL_HANDLE;
   if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return p;
}

static bool
is_green(const uint8_t *px, uint32_t x, uint32_t y)
{
   const uint8_t *p = px + ((size_t)y * WIDTH + x) * 4;
   return p[0] < 64 && p[1] > 192 && p[2] < 64;
}

static uint32_t
count_green(const uint8_t *px)
{
   uint32_t n = 0;
   for (uint32_t y = 0; y < HEIGHT; y++)
      for (uint32_t x = 0; x < WIDTH; x++)
         n += is_green(px, x, y);
   return n;
}

static bool
has_extension(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
   VkExtensionProperties *ep = calloc(n, sizeof(*ep));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, ep);
   bool found = false;
   for (uint32_t i = 0; i < n && !found; i++)
      found = strcmp(ep[i].extensionName, name) == 0;
   free(ep);
   return found;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "drawcomplete.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "drawcomplete.frag.spv";
   const char *cs_path = (argc > 3) ? argv[3] : "drawcomplete_count.frag.spv";

   /* --- instance / device / queue --------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-drawcomplete",
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

   const char *restart_ext = VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME;
   VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT lr = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &lr,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!has_extension(pd, restart_ext) || !lr.primitiveTopologyListRestart) {
      fprintf(stderr, "FAILED: primitiveTopologyListRestart not advertised\n");
      return 1;
   }
   if (!f2.features.fragmentStoresAndAtomics) {
      fprintf(stderr, "FAILED: fragmentStoresAndAtomics not advertised\n");
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

   VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT want_lr = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
      .primitiveTopologyListRestart = VK_TRUE,
   };
   VkPhysicalDeviceFeatures want = { .fragmentStoresAndAtomics = VK_TRUE };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want_lr,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &restart_ext,
      .pEnabledFeatures = &want,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   const VkMemoryPropertyFlags host =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   /* --- attachments ------------------------------------------------ */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FB_FORMAT,
      .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkDeviceMemory imem = bind_memory(dev, &mp, VK_NULL_HANDLE, img, 0);
   if (!imem) { fprintf(stderr, "FAILED: no image memory\n"); return 1; }
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FB_FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkImageCreateInfo zci = imci;
   zci.format = ZS_FORMAT;
   zci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage zimg;
   CHECK(vkCreateImage(dev, &zci, NULL, &zimg));
   VkDeviceMemory zmem = bind_memory(dev, &mp, VK_NULL_HANDLE, zimg, 0);
   if (!zmem) { fprintf(stderr, "FAILED: no depth memory\n"); return 1; }
   VkImageViewCreateInfo zvci = ivci;
   zvci.image = zimg;
   zvci.format = ZS_FORMAT;
   zvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkImageView zview;
   CHECK(vkCreateImageView(dev, &zvci, NULL, &zview));

   /* --- render passes ---------------------------------------------- */
   VkAttachmentDescription atts[2] = {
      { .format = FB_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = ZS_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference zref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &zref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = atts,
      .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkImageView fbviews[2] = { view, zview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 2, .pAttachments = fbviews,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkSubpassDescription sub0 = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
   };
   VkRenderPassCreateInfo rp0ci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1, .pSubpasses = &sub0,
   };
   VkRenderPass rp0;
   CHECK(vkCreateRenderPass(dev, &rp0ci, NULL, &rp0));
   VkFramebufferCreateInfo fb0ci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp0, .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb0;
   CHECK(vkCreateFramebuffer(dev, &fb0ci, NULL, &fb0));

   /* --- buffers ---------------------------------------------------- */
   const VkDeviceSize rb_bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = rb_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkDeviceMemory rbmem = bind_memory(dev, &mp, rb, VK_NULL_HANDLE, host);
   if (!rbmem) { fprintf(stderr, "FAILED: no readback memory\n"); return 1; }

   /* The restart draw's indices, then the same list without the restart and
    * the two indices it orphans: the reference. */
   static const uint16_t indices[14] = {
      0, 1, 2, 3, 0xffff, 4, 5, 6,
      0, 1, 2, 4, 5, 6,
   };
   const uint32_t restart_count = 8, ref_first = 8, ref_count = 6;
   VkBufferCreateInfo xci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(indices), .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
   };
   VkBuffer xbuf;
   CHECK(vkCreateBuffer(dev, &xci, NULL, &xbuf));
   VkDeviceMemory xmem = bind_memory(dev, &mp, xbuf, VK_NULL_HANDLE, host);
   if (!xmem) { fprintf(stderr, "FAILED: no index memory\n"); return 1; }
   void *xp;
   CHECK(vkMapMemory(dev, xmem, 0, sizeof(indices), 0, &xp));
   memcpy(xp, indices, sizeof(indices));
   vkUnmapMemory(dev, xmem);

   VkBufferCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(uint32_t), .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer sbuf;
   CHECK(vkCreateBuffer(dev, &sci, NULL, &sbuf));
   VkDeviceMemory smem = bind_memory(dev, &mp, sbuf, VK_NULL_HANDLE, host);
   if (!smem) { fprintf(stderr, "FAILED: no storage memory\n"); return 1; }
   uint32_t *counter;
   CHECK(vkMapMemory(dev, smem, 0, sizeof(uint32_t), 0, (void **)&counter));
   *counter = 0;

   /* --- descriptors and layout ------------------------------------- */
   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorBufferInfo dbi = { sbuf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- pipelines -------------------------------------------------- */
   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   VkShaderModule cfs = load_module(dev, cs_path);
   if (!vs || !fs || !cfs) return 1;

   const struct pipe_desc d_restart = { rp, true, fs, false, false, VK_COMPARE_OP_ALWAYS, 1 };
   const struct pipe_desc d_list    = { rp, false, fs, false, false, VK_COMPARE_OP_ALWAYS, 1 };
   const struct pipe_desc d_nofs    = { rp, false, VK_NULL_HANDLE, true, true,
                                        VK_COMPARE_OP_ALWAYS, 1 };
   const struct pipe_desc d_less    = { rp, false, fs, true, false, VK_COMPARE_OP_LESS, 1 };
   const struct pipe_desc d_count   = { rp0, false, cfs, false, false, VK_COMPARE_OP_ALWAYS, 0 };
   VkPipeline p_restart = make_pipeline(dev, pl, vs, &d_restart);
   VkPipeline p_list    = make_pipeline(dev, pl, vs, &d_list);
   VkPipeline p_nofs    = make_pipeline(dev, pl, vs, &d_nofs);
   VkPipeline p_less    = make_pipeline(dev, pl, vs, &d_less);
   VkPipeline p_count   = make_pipeline(dev, pl, vs, &d_count);
   if (!p_restart || !p_list || !p_nofs || !p_less || !p_count) {
      fprintf(stderr, "FAILED: pipeline creation\n");
      return 1;
   }

   /* --- command buffer --------------------------------------------- */
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
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkClearValue clears[2] = {
      { .color = { .float32 = { 0.0f, 0.0f, 1.0f, 1.0f } } },
      { .depthStencil = { 1.0f, 0 } },
   };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 2, .pClearValues = clears,
   };
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };

   /* Passes 0-2 render into the colour + depth pass and read the colour back:
    * the restart draw, its reference, and the depth-only occluder followed by
    * the green quad. Pass 3 is the attachment-less count. */
   static uint8_t shot[3][WIDTH * HEIGHT * 4];
   for (uint32_t pass = 0; pass < 4; pass++) {
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      if (pass < 3) {
         vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      } else {
         VkRenderPassBeginInfo rp0bi = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = rp0, .framebuffer = fb0,
            .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         };
         vkCmdBeginRenderPass(cmd, &rp0bi, VK_SUBPASS_CONTENTS_INLINE);
      }
      float z = 0.5f;
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(z), &z);
      switch (pass) {
      case 0:
      case 1:
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pass == 0 ? p_restart : p_list);
         vkCmdBindIndexBuffer(cmd, xbuf, 0, VK_INDEX_TYPE_UINT16);
         vkCmdDrawIndexed(cmd, pass == 0 ? restart_count : ref_count, 1,
                          pass == 0 ? 0 : ref_first, 0, 0);
         break;
      case 2: {
         const float zn = 0.25f;
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_nofs);
         vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(zn), &zn);
         vkCmdDraw(cmd, 6, 1, 7, 0);
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_less);
         vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(z), &z);
         vkCmdDraw(cmd, 6, 1, 13, 0);
         break;
      }
      default:
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p_count);
         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1,
                                 &dset, 0, NULL);
         vkCmdDraw(cmd, 6, 1, 13, 0);
         break;
      }
      vkCmdEndRenderPass(cmd);
      if (pass < 3)
         vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));
      if (pass < 3) {
         uint8_t *px;
         CHECK(vkMapMemory(dev, rbmem, 0, rb_bytes, 0, (void **)&px));
         memcpy(shot[pass], px, sizeof(shot[pass]));
         vkUnmapMemory(dev, rbmem);
      }
   }

   unsigned failed = 0;

   /* restart: the reference must have drawn, and the two must agree. */
   const uint32_t ref_px = count_green(shot[1]);
   const bool restart_ok = ref_px > 0 &&
                           memcmp(shot[0], shot[1], sizeof(shot[0])) == 0;
   printf("restart on a triangle list: %s  %u px, reference %u px\n",
          restart_ok ? "pass" : "FAIL", count_green(shot[0]), ref_px);
   failed += !restart_ok;

   /* no FS: green exactly where the depth-only quad did not reach, and the
    * clear colour (blue) left untouched where it did. */
   uint32_t wrong = 0;
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint8_t *p = shot[2] + ((size_t)y * WIDTH + x) * 4;
         const bool blue = p[0] == 0 && p[1] == 0 && p[2] == 255 && p[3] == 255;
         wrong += (x >= WIDTH / 2) ? !is_green(shot[2], x, y) : !blue;
      }
   }
   printf("depth-only draw without a fragment shader: %s  %u px wrong, %u green\n",
          wrong ? "FAIL" : "pass", wrong, count_green(shot[2]));
   failed += wrong != 0;

   /* no attachments: one invocation per pixel. */
   const uint32_t n = *counter;
   printf("render pass without attachments: %s  %u invocations, want %u\n",
          n == WIDTH * HEIGHT ? "pass" : "FAIL", n, WIDTH * HEIGHT);
   failed += n != WIDTH * HEIGHT;

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkUnmapMemory(dev, smem);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, p_count, NULL);
   vkDestroyPipeline(dev, p_less, NULL);
   vkDestroyPipeline(dev, p_nofs, NULL);
   vkDestroyPipeline(dev, p_list, NULL);
   vkDestroyPipeline(dev, p_restart, NULL);
   vkDestroyShaderModule(dev, cfs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkFreeMemory(dev, smem, NULL);
   vkDestroyBuffer(dev, sbuf, NULL);
   vkFreeMemory(dev, xmem, NULL);
   vkDestroyBuffer(dev, xbuf, NULL);
   vkFreeMemory(dev, rbmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyFramebuffer(dev, fb0, NULL);
   vkDestroyRenderPass(dev, rp0, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, zview, NULL);
   vkFreeMemory(dev, zmem, NULL);
   vkDestroyImage(dev, zimg, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u check(s))\n", failed);
      return 1;
   }
   printf("PASSED (list restart, depth-only draw without a fragment shader, "
          "attachment-less render pass)\n");
   return 0;
}
