/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * An acceleration structure bound to a draw.
 *
 * A shader reaches an acceleration structure through a descriptor, and the
 * driver must repoint that descriptor at the device copy of the BVH before the
 * shader runs -- a BVH holds internal links to its bottom-level structures, so
 * a host pointer left in place, or a verbatim byte copy, gives the device a
 * tree it cannot walk.
 *
 * The dispatch path does this. The draw path relocates buffer and image
 * descriptors and leaves acceleration structures alone, and nothing refuses the
 * device path for a draw that binds one. Every ray-query test in this suite --
 * rtquery, rtquery_id, rtquery_anyhit -- is a compute dispatch, so the draw
 * path has never been asked the question.
 *
 * Same scene and same camera as rtquery: one opaque triangle in the z=0 plane,
 * one orthographic ray per pixel down -Z. Here the ray query runs in the
 * fragment shader and the answer lands in the colour attachment.
 *
 * The check is a count band rather than a single pixel, so a shader that
 * answers with a constant fails whichever constant it picks: an all-miss frame
 * and an all-hit frame are both outside the band, and the centre/corner pair
 * pins the triangle where it belongs.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u

/* The triangle covers an eighth of the frame; the band is wide enough to
 * survive edge-rule differences and far too narrow for a constant answer. */
#define HITS_MIN   200u
#define HITS_MAX  1500u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static VkDevice dev = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties memprops;

/* VK_KHR_acceleration_structure entry points are extension functions, not in
 * the loader's static exports -- resolve them at run time. Buffer device
 * address is core Vulkan 1.2, so it links directly. */
static PFN_vkCreateAccelerationStructureKHR           p_CreateAccelStruct;
static PFN_vkDestroyAccelerationStructureKHR          p_DestroyAccelStruct;
static PFN_vkGetAccelerationStructureBuildSizesKHR    p_GetAccelBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR        p_CmdBuildAccelStructs;
static PFN_vkGetAccelerationStructureDeviceAddressKHR p_GetAccelAddress;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

/* a buffer + host-coherent, device-addressable memory; filled from src */
static bool
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void *src,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
   };
   if (vkCreateBuffer(dev, &bci, NULL, buf) != VK_SUCCESS) {
      return false;
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mr.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) {
      return false;
   }
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &mafi, .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS) {
      return false;
   }
   vkBindBufferMemory(dev, *buf, *mem, 0);
   if (src) {
      void *p;
      if (vkMapMemory(dev, *mem, 0, size, 0, &p) != VK_SUCCESS) {
         return false;
      }
      memcpy(p, src, size);
      vkUnmapMemory(dev, *mem);
   }
   return true;
}

static VkDeviceAddress
buffer_addr(VkBuffer buf)
{
   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf,
   };
   return vkGetBufferDeviceAddress(dev, &bdai);
}

/* Build one acceleration structure. `geom` + `prim_count` describe the
 * geometry; `type` is bottom- or top-level. Returns the AS handle and records
 * the build into `cmd`; the backing + scratch buffers leak (a smoke test
 * exits). */
static VkAccelerationStructureKHR
build_as(VkCommandBuffer cmd, VkAccelerationStructureTypeKHR type,
         const VkAccelerationStructureGeometryKHR *geom, uint32_t prim_count)
{
   VkAccelerationStructureBuildGeometryInfoKHR bgi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = type,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1, .pGeometries = geom,
   };
   VkAccelerationStructureBuildSizesInfoKHR sizes = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
   };
   p_GetAccelBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                        &bgi, &prim_count, &sizes);

   VkBuffer asbuf; VkDeviceMemory asmem;
   if (!make_buffer(sizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                    NULL, &asbuf, &asmem)) {
      return VK_NULL_HANDLE;
   }

   VkAccelerationStructureCreateInfoKHR aci = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = asbuf, .size = sizes.accelerationStructureSize, .type = type,
   };
   VkAccelerationStructureKHR as;
   if (p_CreateAccelStruct(dev, &aci, NULL, &as) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
   }

   VkBuffer scratch; VkDeviceMemory scratchmem;
   if (!make_buffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    NULL, &scratch, &scratchmem)) {
      return VK_NULL_HANDLE;
   }

   bgi.dstAccelerationStructure  = as;
   bgi.scratchData.deviceAddress = buffer_addr(scratch);

   VkAccelerationStructureBuildRangeInfoKHR range = {
      .primitiveCount = prim_count,
   };
   const VkAccelerationStructureBuildRangeInfoKHR *pranges = &range;
   p_CmdBuildAccelStructs(cmd, 1, &bgi, &pranges);

   /* The next build, and then the fragment shader, read this AS. */
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
   };
   vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0, 1, &mb, 0, NULL, 0, NULL);
   return as;
}

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

static VkShaderModule
load_module(const char *path)
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

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "rtqdraw.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "rtqdraw.frag.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-rtqdraw",
      .apiVersion = VK_API_VERSION_1_2,
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
   vkGetPhysicalDeviceMemoryProperties(pd, &memprops);

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

   /* --- device: ray-query + acceleration-structure + BDA ----------- */
   const char *exts[] = {
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
   };
   VkPhysicalDeviceRayQueryFeaturesKHR rqf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
      .rayQuery = VK_TRUE,
   };
   VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rqf, .accelerationStructure = VK_TRUE,
   };
   VkPhysicalDeviceBufferDeviceAddressFeatures bdaf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &asf, .bufferDeviceAddress = VK_TRUE,
   };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &bdaf,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   p_CreateAccelStruct = (PFN_vkCreateAccelerationStructureKHR)
      vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
   p_DestroyAccelStruct = (PFN_vkDestroyAccelerationStructureKHR)
      vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");
   p_GetAccelBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)
      vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
   p_CmdBuildAccelStructs = (PFN_vkCmdBuildAccelerationStructuresKHR)
      vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");
   p_GetAccelAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
      vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR");
   if (!p_CreateAccelStruct || !p_DestroyAccelStruct || !p_GetAccelBuildSizes ||
       !p_CmdBuildAccelStructs || !p_GetAccelAddress) {
      fprintf(stderr, "FAILED: acceleration-structure entry points "
              "unavailable (create=%p sizes=%p build=%p addr=%p)\n",
              (void *)p_CreateAccelStruct, (void *)p_GetAccelBuildSizes,
              (void *)p_CmdBuildAccelStructs, (void *)p_GetAccelAddress);
      return 1;
   }

   /* --- triangle geometry (one triangle in the z=0 plane) ---------- */
   const float verts[3][3] = {
      { -0.5f, -0.5f, 0.0f },
      {  0.5f, -0.5f, 0.0f },
      {  0.0f,  0.5f, 0.0f },
   };
   const uint32_t indices[3] = { 0, 1, 2 };
   VkBuffer vbuf, ibuf; VkDeviceMemory vmem, imem;
   if (!make_buffer(sizeof(verts),
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
          verts, &vbuf, &vmem) ||
       !make_buffer(sizeof(indices),
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
          indices, &ibuf, &imem)) {
      fprintf(stderr, "FAILED: geometry buffers\n"); return 1;
   }

   /* --- command buffer --------------------------------------------- */
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
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

   /* --- bottom-level AS (the triangle) ----------------------------- */
   VkAccelerationStructureGeometryKHR tri_geom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
      .geometry.triangles = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
         .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
         .vertexData.deviceAddress = buffer_addr(vbuf),
         .vertexStride = 3 * sizeof(float),
         .maxVertex = 2,
         .indexType = VK_INDEX_TYPE_UINT32,
         .indexData.deviceAddress = buffer_addr(ibuf),
      },
   };
   VkAccelerationStructureKHR blas =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
               &tri_geom, 1);
   if (!blas) { fprintf(stderr, "FAILED: BLAS\n"); return 1; }

   /* --- top-level AS (one instance of the BLAS) -------------------- */
   VkAccelerationStructureDeviceAddressInfoKHR adai = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
      .accelerationStructure = blas,
   };
   VkAccelerationStructureInstanceKHR instance = {
      .transform = {{ {1,0,0,0}, {0,1,0,0}, {0,0,1,0} }},
      .mask = 0xFF,
      .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
      .accelerationStructureReference = p_GetAccelAddress(dev, &adai),
   };
   VkBuffer instbuf; VkDeviceMemory instmem;
   if (!make_buffer(sizeof(instance),
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
          &instance, &instbuf, &instmem)) {
      fprintf(stderr, "FAILED: instance buffer\n"); return 1;
   }
   VkAccelerationStructureGeometryKHR inst_geom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
         .arrayOfPointers = VK_FALSE,
         .data.deviceAddress = buffer_addr(instbuf),
      },
   };
   VkAccelerationStructureKHR tlas =
      build_as(cmd, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
               &inst_geom, 1);
   if (!tlas) { fprintf(stderr, "FAILED: TLAS\n"); return 1; }

   /* --- colour attachment ------------------------------------------ */
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
   VkImage cimg;
   CHECK(vkCreateImage(dev, &imci, NULL, &cimg));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, cimg, &imr);
   uint32_t imt = find_mem(imr.memoryTypeBits, 0);
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return 1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory cmem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &cmem));
   CHECK(vkBindImageMemory(dev, cimg, cmem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = cimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView cview;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &cview));

   /* --- render pass + framebuffer ---------------------------------- */
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
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &cview,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* --- descriptor set (the acceleration structure, in the FS) ----- */
   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = 1,
   };
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
   VkWriteDescriptorSetAccelerationStructureKHR was = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 1, .pAccelerationStructures = &tlas,
   };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &was,
      .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
   };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   /* --- graphics pipeline ------------------------------------------ */
   VkShaderModule vs = load_module(vs_path);
   VkShaderModule fs = load_module(fs_path);
   if (!vs || !fs) { return 1; }

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   /* No vertex input: the vertex shader builds its position from
    * gl_VertexIndex, so nothing but the ray query is under test. */
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

   /* --- readback buffer -------------------------------------------- */
   const VkDeviceSize obytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBuffer rb; VkDeviceMemory rbmem;
   if (!make_buffer(obytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL,
                    &rb, &rbmem)) {
      fprintf(stderr, "FAILED: readback buffer\n"); return 1;
   }

   /* --- draw (after the AS builds, same command buffer) ------------ */
   VkClearValue clear = { .color = { .float32 = { 0.0f, 1.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1,
                           &dset, 0, NULL);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   VkBufferImageCopy creg = {
      .bufferOffset = 0,
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &creg);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify ------------------------------------------------------ */
   uint8_t *px;
   CHECK(vkMapMemory(dev, rbmem, 0, obytes, 0, (void **)&px));
   unsigned hits = 0, misses = 0, bad = 0, uncovered = 0;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      const uint8_t *p = px + (size_t)i * 4;
      if      (p[0] == 255 && p[1] == 0 && p[2] == 0 && p[3] == 255) { hits++; }
      else if (p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 255)   { misses++; }
      /* The clear is green, so a pixel the draw never reached is told apart
       * from a ray that answered wrongly. */
      else if (p[0] == 0 && p[1] == 255 && p[2] == 0 && p[3] == 255) { uncovered++; }
      else                                                           { bad++; }
   }
   const uint8_t *centre = px + (((size_t)(HEIGHT / 2) * WIDTH) + WIDTH / 2) * 4;
   const uint8_t *corner = px + (((size_t)1 * WIDTH) + 1) * 4;
   const bool centre_hit = centre[0] == 255 && centre[1] == 0 && centre[2] == 0;
   const bool corner_miss = corner[0] == 0 && corner[1] == 0 && corner[2] == 0;
   uint8_t cpx[4], kpx[4];
   memcpy(cpx, centre, 4);
   memcpy(kpx, corner, 4);
   vkUnmapMemory(dev, rbmem);

   p_DestroyAccelStruct(dev, tlas, NULL);
   p_DestroyAccelStruct(dev, blas, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   /* The triangle covers about an eighth of the frame; the centre is inside
    * it, a corner is outside. */
   const bool ok = bad == 0 && uncovered == 0 && centre_hit && corner_miss &&
                   hits > HITS_MIN && hits < HITS_MAX;
   if (!ok) {
      printf("FAILED (rtqdraw: hits=%u misses=%u uncovered=%u bad=%u "
             "centre=%u,%u,%u,%u corner=%u,%u,%u,%u",
             hits, misses, uncovered, bad,
             cpx[0], cpx[1], cpx[2], cpx[3], kpx[0], kpx[1], kpx[2], kpx[3]);
      if (uncovered) {
         printf("; the draw did not cover the target, so the ray answers say "
                "nothing");
      } else if (hits == 0) {
         printf("; every ray missed -- the fragment stage's acceleration "
                "structure was never pointed at the device copy of the BVH");
      }
      printf(")\n");
      return 1;
   }
   printf("PASSED (rtqdraw: %u of %u rays hit the triangle from the fragment "
          "stage)\n", hits, WIDTH * HEIGHT);
   return 0;
}
