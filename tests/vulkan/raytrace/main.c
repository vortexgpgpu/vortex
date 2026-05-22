/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Ray-query smoke test for the vortexpipe driver (Phase 7, step 1).
 *
 * Builds a one-triangle acceleration structure, then dispatches a
 * compute shader that casts one orthographic primary ray per pixel
 * via VK_KHR_ray_query and writes red on a hit, black on a miss.
 *
 * This is the Phase 7 correctness oracle: vortexpipe cannot yet
 * translate rayQueryEXT, so the compute kernel falls back to
 * llvmpipe and the ray tracing runs on lavapipe's software path.
 * Later phases (BVH build + traversal on the Vortex SIMT cores) are
 * validated pixel-for-pixel against the image this test produces.
 * See docs/proposals/vulkan_support_proposal.md, Phase 7.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

#define HIT_COLOR   0xff0000ffu
#define MISS_COLOR  0xff000000u

static VkDevice dev = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties memprops;

/* VK_KHR_acceleration_structure entry points are extension functions,
 * not in the loader's static exports -- resolve them at run time.
 * Buffer device address is core Vulkan 1.2, so it links directly. */
static PFN_vkCreateAccelerationStructureKHR         p_CreateAccelStruct;
static PFN_vkDestroyAccelerationStructureKHR        p_DestroyAccelStruct;
static PFN_vkGetAccelerationStructureBuildSizesKHR  p_GetAccelBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR      p_CmdBuildAccelStructs;
static PFN_vkGetAccelerationStructureDeviceAddressKHR p_GetAccelAddress;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
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
   if (vkCreateBuffer(dev, &bci, NULL, buf) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mr.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX)
      return false;
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &mafi, .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS)
      return false;
   vkBindBufferMemory(dev, *buf, *mem, 0);
   if (src) {
      void *p;
      if (vkMapMemory(dev, *mem, 0, size, 0, &p) != VK_SUCCESS)
         return false;
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
 * geometry; `type` is bottom- or top-level. Returns the AS handle and
 * records the build into `cmd`; the backing + scratch buffers leak
 * (a smoke test exits). */
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
                    NULL, &asbuf, &asmem))
      return VK_NULL_HANDLE;

   VkAccelerationStructureCreateInfoKHR aci = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = asbuf, .size = sizes.accelerationStructureSize, .type = type,
   };
   VkAccelerationStructureKHR as;
   if (p_CreateAccelStruct(dev, &aci, NULL, &as) != VK_SUCCESS)
      return VK_NULL_HANDLE;

   VkBuffer scratch; VkDeviceMemory scratchmem;
   if (!make_buffer(sizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    NULL, &scratch, &scratchmem))
      return VK_NULL_HANDLE;

   bgi.dstAccelerationStructure  = as;
   bgi.scratchData.deviceAddress = buffer_addr(scratch);

   VkAccelerationStructureBuildRangeInfoKHR range = {
      .primitiveCount = prim_count,
   };
   const VkAccelerationStructureBuildRangeInfoKHR *pranges = &range;
   p_CmdBuildAccelStructs(cmd, 1, &bgi, &pranges);

   /* the next build / the shader reads this AS */
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
   };
   vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
   if (buf) *out_size = (size_t)sz;
   return buf;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "raytrace.comp.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-raytrace",
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
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   /* --- device: ray-query + acceleration-structure + BDA ---------- */
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

   /* --- triangle geometry (one triangle in the z=0 plane) --------- */
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

   /* --- command buffer -------------------------------------------- */
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

   /* --- bottom-level AS (the triangle) ---------------------------- */
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

   /* --- top-level AS (one instance of the BLAS) ------------------- */
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

   /* --- output storage buffer ------------------------------------- */
   const VkDeviceSize obytes = (VkDeviceSize)WIDTH * HEIGHT * sizeof(uint32_t);
   VkBuffer obuf; VkDeviceMemory omem;
   if (!make_buffer(obytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    NULL, &obuf, &omem)) {
      fprintf(stderr, "FAILED: output buffer\n"); return 1;
   }

   /* --- compute pipeline ------------------------------------------ */
   size_t spv_size = 0;
   uint32_t *spv = read_spirv(spv_path, &spv_size);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(spv);

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   const uint32_t dims[2] = { WIDTH, HEIGHT };
   VkSpecializationMapEntry sme[2] = {
      { 0, 0, sizeof(uint32_t) }, { 1, sizeof(uint32_t), sizeof(uint32_t) },
   };
   VkSpecializationInfo spec = {
      .mapEntryCount = 2, .pMapEntries = sme,
      .dataSize = sizeof(dims), .pData = dims,
   };
   VkComputePipelineCreateInfo cpc = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpc, NULL, &pipe));

   /* --- descriptor set (acceleration structure + output) ---------- */
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

   VkWriteDescriptorSetAccelerationStructureKHR was = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 1, .pAccelerationStructures = &tlas,
   };
   VkDescriptorBufferInfo dbi = { .buffer = obuf, .offset = 0, .range = obytes };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &was,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

   /* --- dispatch (after the AS builds, same command buffer) ------- */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, 1, &ds, 0, NULL);
   vkCmdDispatch(cmd, WIDTH / 8, HEIGHT / 8, 1);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- read back + verify ---------------------------------------- */
   uint32_t *px;
   CHECK(vkMapMemory(dev, omem, 0, obytes, 0, (void **)&px));
   unsigned hits = 0, misses = 0, bad = 0;
   for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
      if      (px[i] == HIT_COLOR)  hits++;
      else if (px[i] == MISS_COLOR) misses++;
      else                          bad++;
   }
   uint32_t centre = px[(HEIGHT / 2) * WIDTH + WIDTH / 2];
   uint32_t corner = px[1 * WIDTH + 1];
   vkUnmapMemory(dev, omem);

   p_DestroyAccelStruct(dev, tlas, NULL);
   p_DestroyAccelStruct(dev, blas, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   /* the triangle covers ~1/8 of the frame; the centre is inside it,
    * a corner is outside. */
   bool ok = bad == 0 && centre == HIT_COLOR && corner == MISS_COLOR &&
             hits > 200u && hits < 1500u;
   if (!ok) {
      printf("FAILED (hits=%u misses=%u bad=%u centre=0x%08x corner=0x%08x)\n",
             hits, misses, bad, centre, corner);
      return 1;
   }
   printf("PASSED (ray query: %u/%u rays hit the triangle)\n",
          hits, WIDTH * HEIGHT);
   return 0;
}
