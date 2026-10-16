/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A pipeline-statistics query over graphics draws and a dispatch the device
 * runs.
 *
 * The counters are llvmpipe's, bumped as its draw module and rasterizer walk
 * the geometry, so a draw that takes the device path has to produce them
 * itself or the query reads zero with no error. Every draw here has one right
 * answer for each counter, so each is asserted exactly: a counter that is
 * merely non-zero can still be counting the wrong draws, instances or rows.
 *
 * The geometry is two triangles that tile the 64x64 target exactly once, with
 * no depth test, so the fragment shader runs once per pixel per instance.
 *
 *   instanced    3 instances of both triangles
 *   offscreen    both triangles moved wholly outside the viewport
 *   discard      rasterizer discard: everything up to clipping, no fragments
 *   cull_all     cull mode FRONT_AND_BACK: likewise, culled after clipping
 *   two_draws    one triangle per vkCmdDraw, summed in one query
 *   multi_draw   vkCmdDrawMultiEXT, two draws of 2 instances each
 *   dispatch     a 2-row compute grid, no graphics work at all
 *
 * Results are read twice, with vkGetQueryPoolResults and with
 * vkCmdCopyQueryPoolResults into a buffer, and the two must agree.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define FORMAT    VK_FORMAT_R8G8B8A8_UNORM
#define NPX       ((uint64_t)WIDTH * HEIGHT)

#define VERTS      6u
#define TRIS       2u
#define INSTANCES  3u
#define MD_INSTANCES 2u

#define CS_COLS          256u
#define CS_ROWS          2u
#define CS_LOCAL_MAX     64u

#define STATS (VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |     \
               VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |   \
               VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |   \
               VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |        \
               VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |         \
               VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT | \
               VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT)

/* Result slots, in ascending bit order as Vulkan writes them. */
enum { IA_VERTS, IA_PRIMS, VS_INV, CLIP_INV, CLIP_PRIMS, FS_INV, CS_INV, NSTATS };

static const char *const stat_names[NSTATS] = {
   "ia_vertices", "ia_primitives", "vs_invocations",
   "clip_invocations", "clip_primitives", "fs_invocations", "cs_invocations",
};

enum {
   Q_INSTANCED, Q_OFFSCREEN, Q_DISCARD, Q_CULL_ALL, Q_TWO_DRAWS, Q_MULTI_DRAW,
   Q_DISPATCH, NQUERIES
};

static const char *const query_names[NQUERIES] = {
   "instanced", "offscreen", "discard", "cull_all", "two_draws", "multi_draw",
   "dispatch",
};

/* A counter Vulkan leaves implementation-dependent for that draw. */
#define ANY UINT64_MAX

#define CHECK(x) do {                                                    \
   VkResult _r = (x);                                                    \
   if (_r != VK_SUCCESS) {                                               \
      printf("FAILED (gfxstats: %s -> VkResult %d)\n", #x, (int)_r);     \
      return 1;                                                          \
   }                                                                     \
} while (0)

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
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

static VkResult
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer *out_buf,
            VkDeviceMemory *out_mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = usage,
   };
   VkResult r = vkCreateBuffer(dev, &bci, NULL, out_buf);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   r = vkAllocateMemory(dev, &mai, NULL, out_mem);
   if (r != VK_SUCCESS) {
      return r;
   }
   return vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
}

static bool
has_device_extension(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   if (vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS) {
      return false;
   }
   VkExtensionProperties *exts = calloc(n, sizeof(*exts));
   bool found = false;
   if (exts && vkEnumerateDeviceExtensionProperties(pd, NULL, &n, exts) == VK_SUCCESS) {
      for (uint32_t i = 0; i < n; i++) {
         if (strcmp(exts[i].extensionName, name) == 0) {
            found = true;
            break;
         }
      }
   }
   free(exts);
   return found;
}

static VkResult
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl,
              VkShaderModule vs, VkShaderModule fs, VkBool32 discard,
              VkCullModeFlags cull, VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
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
      .rasterizerDiscardEnable = discard,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = cull,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   /* No depth/stencil state: instances overlap exactly, and each must still
    * reach the fragment shader. */
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
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "gfxstats.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "gfxstats.frag.spv";
   const char *cs_path = (argc > 3) ? argv[3] : "gfxstats.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-gfxstats",
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
   if (npd == 0) {
      printf("FAILED (gfxstats: no physical device)\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   VkPhysicalDeviceMultiDrawFeaturesEXT md_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &md_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   if (!have.features.pipelineStatisticsQuery) {
      printf("FAILED (gfxstats: pipelineStatisticsQuery not advertised)\n");
      return 1;
   }
   const bool multi_draw =
      has_device_extension(pd, VK_EXT_MULTI_DRAW_EXTENSION_NAME) &&
      md_have.multiDraw;
   if (!multi_draw) {
      printf("note: %s not advertised; the multi_draw query is left empty\n",
             VK_EXT_MULTI_DRAW_EXTENSION_NAME);
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
         qf = i;
         break;
      }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("FAILED (gfxstats: no graphics+compute queue)\n");
      return 1;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceMultiDrawFeaturesEXT md_feat = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
      .multiDraw = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 feats = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = multi_draw ? &md_feat : NULL,
      .features = { .pipelineStatisticsQuery = VK_TRUE },
   };
   const char *dev_exts[] = { VK_EXT_MULTI_DRAW_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &feats,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = multi_draw ? 1u : 0u,
      .ppEnabledExtensionNames = dev_exts,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   PFN_vkCmdDrawMultiEXT p_draw_multi = NULL;
   if (multi_draw) {
      p_draw_multi = (PFN_vkCmdDrawMultiEXT)
         vkGetDeviceProcAddr(dev, "vkCmdDrawMultiEXT");
      if (!p_draw_multi) {
         printf("FAILED (gfxstats: vkCmdDrawMultiEXT missing)\n");
         return 1;
      }
   }

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   VkShaderModule cs = load_module(dev, cs_path);
   if (!vs || !fs || !cs) {
      printf("FAILED (gfxstats: cannot load %s / %s / %s)\n", vs_path, fs_path,
             cs_path);
      return 1;
   }

   /* --- colour target ---------------------------------------------- */
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
   if (imt == UINT32_MAX) {
      printf("FAILED (gfxstats: no image memory)\n");
      return 1;
   }
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

   /* --- graphics pipelines ------------------------------------------ */
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0,
      .size = 2 * sizeof(float),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkPipeline pipe, discard_pipe, cull_pipe;
   CHECK(make_pipeline(dev, rp, pl, vs, fs, VK_FALSE, VK_CULL_MODE_NONE, &pipe));
   CHECK(make_pipeline(dev, rp, pl, vs, fs, VK_TRUE, VK_CULL_MODE_NONE,
                       &discard_pipe));
   CHECK(make_pipeline(dev, rp, pl, vs, fs, VK_FALSE,
                       VK_CULL_MODE_FRONT_AND_BACK, &cull_pipe));

   /* --- compute pipeline -------------------------------------------- */
   uint32_t local_size = CS_LOCAL_MAX < props.limits.maxComputeWorkGroupSize[0]
                       ? CS_LOCAL_MAX : props.limits.maxComputeWorkGroupSize[0];
   while (local_size > 1 && (CS_COLS % local_size) != 0) {
      local_size >>= 1;
   }
   const uint32_t cs_groups_x = CS_COLS / local_size;
   const uint32_t cs_n = CS_COLS * CS_ROWS;

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo cplci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout cpl;
   CHECK(vkCreatePipelineLayout(dev, &cplci, NULL, &cpl));
   VkSpecializationMapEntry sme = {
      .constantID = 0, .offset = 0, .size = sizeof(uint32_t),
   };
   VkSpecializationInfo spec = {
      .mapEntryCount = 1, .pMapEntries = &sme,
      .dataSize = sizeof(uint32_t), .pData = &local_size,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = cpl,
   };
   VkPipeline cpipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe));

   const VkDeviceSize cs_bytes = (VkDeviceSize)cs_n * sizeof(uint32_t);
   VkBuffer csbuf;
   VkDeviceMemory csmem;
   CHECK(host_buffer(dev, &mp, cs_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     &csbuf, &csmem));
   uint32_t *csmap;
   CHECK(vkMapMemory(dev, csmem, 0, cs_bytes, 0, (void **)&csmap));
   memset(csmap, 0, (size_t)cs_bytes);
   vkUnmapMemory(dev, csmem);

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi = { .buffer = csbuf, .offset = 0, .range = cs_bytes };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   /* --- readback buffers + query pool ------------------------------- */
   const VkDeviceSize px_bytes = NPX * 4;
   VkBuffer pxbuf;
   VkDeviceMemory pxmem;
   CHECK(host_buffer(dev, &mp, px_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     &pxbuf, &pxmem));

   const VkDeviceSize stride = NSTATS * sizeof(uint64_t);
   const VkDeviceSize qbytes = NQUERIES * stride;
   VkBuffer qbuf;
   VkDeviceMemory qmem;
   CHECK(host_buffer(dev, &mp, qbytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     &qbuf, &qmem));
   uint8_t *qmap;
   CHECK(vkMapMemory(dev, qmem, 0, qbytes, 0, (void **)&qmap));
   memset(qmap, 0xff, (size_t)qbytes);
   vkUnmapMemory(dev, qmem);

   VkQueryPoolCreateInfo qpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS, .queryCount = NQUERIES,
      .pipelineStatistics = STATS,
   };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));

   /* --- record ------------------------------------------------------ */
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
   vkCmdResetQueryPool(cmd, qp, 0, NQUERIES);
   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

   const float inside[2] = { 0.0f, 0.0f };
   /* x in [3, 5] clip space: every vertex is beyond the right plane. */
   const float outside[2] = { 4.0f, 0.0f };

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, discard_pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inside,
                      inside);
   vkCmdBeginQuery(cmd, qp, Q_DISCARD, 0);
   vkCmdDraw(cmd, VERTS, 1, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_DISCARD);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cull_pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inside,
                      inside);
   vkCmdBeginQuery(cmd, qp, Q_CULL_ALL, 0);
   vkCmdDraw(cmd, VERTS, 1, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_CULL_ALL);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof outside,
                      outside);
   vkCmdBeginQuery(cmd, qp, Q_OFFSCREEN, 0);
   vkCmdDraw(cmd, VERTS, 1, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_OFFSCREEN);

   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inside,
                      inside);
   vkCmdBeginQuery(cmd, qp, Q_TWO_DRAWS, 0);
   vkCmdDraw(cmd, VERTS / 2, 1, 0, 0);
   vkCmdDraw(cmd, VERTS / 2, 1, VERTS / 2, 0);
   vkCmdEndQuery(cmd, qp, Q_TWO_DRAWS);

   vkCmdBeginQuery(cmd, qp, Q_INSTANCED, 0);
   vkCmdDraw(cmd, VERTS, INSTANCES, 0, 0);
   vkCmdEndQuery(cmd, qp, Q_INSTANCED);

   vkCmdBeginQuery(cmd, qp, Q_MULTI_DRAW, 0);
   if (multi_draw) {
      const VkMultiDrawInfoEXT md[2] = {
         { .firstVertex = 0, .vertexCount = VERTS / 2 },
         { .firstVertex = VERTS / 2, .vertexCount = VERTS / 2 },
      };
      p_draw_multi(cmd, 2, md, MD_INSTANCES, 0, sizeof md[0]);
   }
   vkCmdEndQuery(cmd, qp, Q_MULTI_DRAW);

   vkCmdEndRenderPass(cmd);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &ds,
                           0, NULL);
   vkCmdBeginQuery(cmd, qp, Q_DISPATCH, 0);
   vkCmdDispatch(cmd, cs_groups_x, CS_ROWS, 1);
   vkCmdEndQuery(cmd, qp, Q_DISPATCH);

   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          pxbuf, 1, &region);
   vkCmdCopyQueryPoolResults(cmd, qp, 0, NQUERIES, qbuf, 0, stride,
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   VkMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, NULL,
                        0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back ---------------------------------------------------- */
   uint64_t got[NQUERIES][NSTATS];
   CHECK(vkGetQueryPoolResults(dev, qp, 0, NQUERIES, sizeof got, got, stride,
                               VK_QUERY_RESULT_64_BIT |
                               VK_QUERY_RESULT_WAIT_BIT));
   uint64_t copied[NQUERIES][NSTATS];
   CHECK(vkMapMemory(dev, qmem, 0, qbytes, 0, (void **)&qmap));
   memcpy(copied, qmap, sizeof copied);
   vkUnmapMemory(dev, qmem);

   uint8_t *px;
   CHECK(vkMapMemory(dev, pxmem, 0, px_bytes, 0, (void **)&px));
   uint64_t covered = 0;
   for (uint32_t i = 0; i < NPX; i++) {
      const uint8_t *p = px + (size_t)i * 4;
      if (p[0] == 255 && p[1] == 0 && p[2] == 0) {
         covered++;
      }
   }
   vkUnmapMemory(dev, pxmem);

   CHECK(vkMapMemory(dev, csmem, 0, cs_bytes, 0, (void **)&csmap));
   uint32_t cs_bad = 0;
   for (uint32_t i = 0; i < cs_n; i++) {
      if (csmap[i] != i * 3u + 1u) {
         cs_bad++;
      }
   }
   vkUnmapMemory(dev, csmem);

   for (unsigned q = 0; q < NQUERIES; q++) {
      printf("query %-10s:", query_names[q]);
      for (unsigned s = 0; s < NSTATS; s++) {
         printf(" %s=%llu", stat_names[s], (unsigned long long)got[q][s]);
      }
      printf("\n");
   }

   vkDestroyQueryPool(dev, qp, NULL);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyBuffer(dev, qbuf, NULL);
   vkFreeMemory(dev, qmem, NULL);
   vkDestroyBuffer(dev, pxbuf, NULL);
   vkFreeMemory(dev, pxmem, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyBuffer(dev, csbuf, NULL);
   vkFreeMemory(dev, csmem, NULL);
   vkDestroyPipeline(dev, cpipe, NULL);
   vkDestroyPipelineLayout(dev, cpl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipeline(dev, discard_pipe, NULL);
   vkDestroyPipeline(dev, cull_pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, cs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (covered != NPX) {
      printf("FAILED (gfxstats: the draws covered %llu of %llu px, so the "
             "query results cannot be judged)\n",
             (unsigned long long)covered, (unsigned long long)NPX);
      return 1;
   }
   if (cs_bad) {
      printf("FAILED (gfxstats: the dispatch wrote %u of %u words wrong, so "
             "its query result cannot be judged)\n", cs_bad, cs_n);
      return 1;
   }
   for (unsigned q = 0; q < NQUERIES; q++) {
      for (unsigned s = 0; s < NSTATS; s++) {
         if (got[q][s] != copied[q][s]) {
            printf("FAILED (gfxstats: query %s %s is %llu from "
                   "vkGetQueryPoolResults but %llu from "
                   "vkCmdCopyQueryPoolResults)\n", query_names[q],
                   stat_names[s], (unsigned long long)got[q][s],
                   (unsigned long long)copied[q][s]);
            return 1;
         }
      }
   }

   /* A non-indexed draw has no vertex to reuse, so the vertex shader runs once
    * per assembled vertex. Nothing is clipped in a draw that lies inside the
    * viewport, so clipping passes every primitive through unchanged. Two
    * triangles tiling the target cover each pixel once per instance, and the
    * shader writes no helper-sensitive output, so fragment invocations are the
    * covered pixels exactly. Rasterizer discard and FRONT_AND_BACK culling
    * both drop primitives after clipping, so everything before it still
    * counts. A primitive wholly outside the clip volume may be counted by the
    * clipper's output as zero or more, so that one counter is left open. */
   const uint64_t md_draws = multi_draw ? 1u : 0u;
   const uint64_t want[NQUERIES][NSTATS] = {
      [Q_INSTANCED]  = { VERTS * INSTANCES, TRIS * INSTANCES, VERTS * INSTANCES,
                         TRIS * INSTANCES, TRIS * INSTANCES, NPX * INSTANCES, 0 },
      [Q_OFFSCREEN]  = { VERTS, TRIS, VERTS, TRIS, ANY, 0, 0 },
      [Q_DISCARD]    = { VERTS, TRIS, VERTS, TRIS, TRIS, 0, 0 },
      [Q_CULL_ALL]   = { VERTS, TRIS, VERTS, TRIS, TRIS, 0, 0 },
      [Q_TWO_DRAWS]  = { VERTS, TRIS, VERTS, TRIS, TRIS, NPX, 0 },
      [Q_MULTI_DRAW] = { md_draws * VERTS * MD_INSTANCES,
                         md_draws * TRIS * MD_INSTANCES,
                         md_draws * VERTS * MD_INSTANCES,
                         md_draws * TRIS * MD_INSTANCES,
                         md_draws * TRIS * MD_INSTANCES,
                         md_draws * NPX * MD_INSTANCES, 0 },
      [Q_DISPATCH]   = { 0, 0, 0, 0, 0, 0, cs_n },
   };
   unsigned bad = 0;
   for (unsigned q = 0; q < NQUERIES; q++) {
      for (unsigned s = 0; s < NSTATS; s++) {
         if (want[q][s] != ANY && got[q][s] != want[q][s]) {
            printf("  %s %s = %llu, want %llu\n", query_names[q], stat_names[s],
                   (unsigned long long)got[q][s],
                   (unsigned long long)want[q][s]);
            bad++;
         }
      }
   }
   if (bad) {
      printf("FAILED (gfxstats: %u counters wrong; zero means the counter was "
             "never updated, which is what happens when the work bypasses the "
             "pipeline that owns it, and a multiple of the right value means "
             "instances, draws or rows were counted wrongly)\n", bad);
      return 1;
   }

   printf("PASSED (gfxstats: %u queries exact -- instanced fs %llu, two draws "
          "fs %llu, multi-draw fs %llu, discard/cull_all/offscreen fs 0, "
          "dispatch cs %u; copy agrees)\n", (unsigned)NQUERIES,
          (unsigned long long)got[Q_INSTANCED][FS_INV],
          (unsigned long long)got[Q_TWO_DRAWS][FS_INV],
          (unsigned long long)got[Q_MULTI_DRAW][FS_INV], cs_n);
   return 0;
}
