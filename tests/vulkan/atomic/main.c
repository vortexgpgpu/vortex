/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Atomic coverage test for the vortexpipe driver.
 *
 * Dispatches atomic.comp, which contends the integer atomics the device
 * implements -- add, and, or, xor, signed and unsigned min/max, and exchange
 * -- on an SSBO from every invocation at once, so each AMO goes to the dcache.
 *
 * Every expectation here is independent of the order the device schedules its
 * warps, because a test that depended on that order would be flaky rather than
 * strict:
 *   - add/and/or/xor/min/max are commutative and associative, so the final
 *     memory value is fixed;
 *   - exchange is order-dependent, so it is checked with a multiset invariant:
 *     the old values handed back to the invocations, together with the value
 *     left in memory, must be exactly the initial value together with all the
 *     values written. A dropped or duplicated exchange breaks that equality
 *     without the test ever needing to know who went first.
 *
 * The buffers are seeded with sentinels and with identities chosen so that a
 * no-op dispatch cannot be mistaken for a pass: every expected result differs
 * from its seed.
 *
 * float and 64-bit atomics are deliberately not exercised -- RV32A has neither
 * -- but the test asserts that the device does not advertise them, so the
 * disclaim cannot silently rot into a false claim.
 *
 * The same op set on `shared` variables is covered by tests/vulkan/atomic_shared,
 * which is a separate test because the device implements only one of the two
 * address spaces today. compare-and-swap is absent from both: it lowers to an
 * LR/SC pair whose forward progress is a separate open defect.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_DIM   4u
#define IMG_SEED  0xA5A5A5A5u
#define INVOCS         64u
#define LOCAL_SIZE_MAX 64u
#define SENTINEL       0xDEADBEEFu

/* Slots in the unsigned target buffer; must match atomic.comp. */
#define G_ADD    0u
#define G_AND    1u
#define G_OR     2u
#define G_XOR    3u
#define G_UMIN   4u
#define G_UMAX   5u
#define G_XCHG   6u
#define G_SLOTS  7u

#define GI_MIN   0u
#define GI_MAX   1u
#define GI_SLOTS 2u

/* Seed for the exchange slot: distinct from every value the shader writes
 * (1..INVOCS), so it can be told apart in the multiset check. */
#define XCHG_SEED 0xA5A5A5A5u

/* Seed for invocation i's own AMO target. Distinct per lane and disjoint from
 * the values the shader writes (1..INVOCS), so a value landing on the wrong
 * lane is always visible. */
#define LANE_SEED(i) (0x5A5A0000u + (i))

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static const char *const g_name[G_SLOTS] = {
   "atomicAdd", "atomicAnd", "atomicOr", "atomicXor",
   "atomicMin.u", "atomicMax.u", "atomicExchange",
};

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "FAILED: cannot open %s\n", path);
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

static int
make_buffer(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize bytes,
            VkBuffer *out_buf, VkDeviceMemory *out_mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (vkCreateBuffer(dev, &bci, NULL, out_buf) != VK_SUCCESS) {
      return 1;
   }

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t mt = UINT32_MAX;
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want) {
         mt = i;
         break;
      }
   }
   if (mt == UINT32_MAX) {
      fprintf(stderr, "FAILED: no host-visible memory\n");
      return 1;
   }

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) {
      return 1;
   }
   if (vkBindBufferMemory(dev, *out_buf, *out_mem, 0) != VK_SUCCESS) {
      return 1;
   }
   return 0;
}

static int
cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a;
   uint32_t y = *(const uint32_t *)b;
   if (x < y) {
      return -1;
   }
   if (x > y) {
      return 1;
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "atomic.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-atomic",
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
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* RV32A has only 32-bit integer AMOs, so float and 64-bit atomics are
    * disclaimed rather than implemented. Assert the disclaim instead of
    * assuming it: if a feature ever flips back to true, this test does not
    * cover the ops behind it, and silence would let an unbacked claim ship.
    *
    * The check is on the FEATURES, not on the extension strings. Advertising
    * VK_EXT_shader_atomic_float while reporting every one of its features
    * unsupported is exactly how a driver is meant to disclaim it, so keying
    * this on the extension list would fail an honest device.
    *
    * Both structs are zero-initialized, so a driver that ignores an unknown
    * pNext leaves them reading "unsupported" -- the safe direction. */
   VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
   };
   VkPhysicalDeviceShaderAtomicInt64Features ai = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
      .pNext = &af,
   };
   VkPhysicalDeviceFeatures2 feats2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &ai,
   };
   vkGetPhysicalDeviceFeatures2(pd, &feats2);

   const struct { const char *name; VkBool32 got; const char *why; } disclaimed[] = {
      { "shaderBufferInt64Atomics",   ai.shaderBufferInt64Atomics,   "no 64-bit AMO" },
      { "shaderSharedInt64Atomics",   ai.shaderSharedInt64Atomics,   "no 64-bit AMO" },
      { "shaderBufferFloat32Atomics", af.shaderBufferFloat32Atomics, "no float AMO" },
      { "shaderBufferFloat32AtomicAdd", af.shaderBufferFloat32AtomicAdd, "no float AMO" },
      { "shaderSharedFloat32Atomics", af.shaderSharedFloat32Atomics, "no float AMO" },
      { "shaderSharedFloat32AtomicAdd", af.shaderSharedFloat32AtomicAdd, "no float AMO" },
   };
   int unbacked_found = 0;
   for (unsigned i = 0; i < sizeof(disclaimed) / sizeof(disclaimed[0]); i++) {
      if (disclaimed[i].got) {
         fprintf(stderr, "  %s is advertised, but this device has %s and this "
                 "test does not cover it\n", disclaimed[i].name,
                 disclaimed[i].why);
         unbacked_found++;
      }
   }
   if (unbacked_found) {
      printf("FAILED (%d unbacked atomic feature(s) advertised)\n",
             unbacked_found);
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
         qf = i;
         break;
      }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      fprintf(stderr, "FAILED: no compute queue\n");
      return 1;
   }

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

   /* --- three storage buffers: unsigned targets, signed targets, results --- */
   const VkDeviceSize gu_bytes = (VkDeviceSize)G_SLOTS * sizeof(uint32_t);
   const VkDeviceSize gi_bytes = (VkDeviceSize)GI_SLOTS * sizeof(int32_t);
   /* Three rows: contended-exchange old values, per-lane AMO targets, and the
    * old values those returned. */
   const uint32_t     n_res    = 3u * INVOCS;
   const VkDeviceSize r_bytes  = (VkDeviceSize)n_res * sizeof(uint32_t);

   VkBuffer gu_buf, gi_buf, r_buf;
   VkDeviceMemory gu_mem, gi_mem, r_mem;
   if (make_buffer(dev, pd, gu_bytes, &gu_buf, &gu_mem) ||
       make_buffer(dev, pd, gi_bytes, &gi_buf, &gi_mem) ||
       make_buffer(dev, pd, r_bytes,  &r_buf,  &r_mem)) {
      return 1;
   }

   /* --- storage image, the kernel's atomic-only descriptor ------------ */
   /* Linear tiling on host-visible memory so the result can be read by mapping
    * it, with no staging copy to get wrong. IMG_DIM > 1 keeps the row stride
    * off the degenerate single-texel case even though only (0,0) is written. */
   VkImageCreateInfo img_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .extent = { IMG_DIM, IMG_DIM, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &img_ci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   VkPhysicalDeviceMemoryProperties imp;
   vkGetPhysicalDeviceMemoryProperties(pd, &imp);
   uint32_t imt = UINT32_MAX;
   const VkMemoryPropertyFlags iwant =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < imp.memoryTypeCount; i++)
      if ((imr.memoryTypeBits & (1u << i)) &&
          (imp.memoryTypes[i].propertyFlags & iwant) == iwant) { imt = i; break; }
   if (imt == UINT32_MAX) {
      fprintf(stderr, "no host-visible memory type for the storage image\n");
      return 1;
   }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory img_mem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &img_mem));
   CHECK(vkBindImageMemory(dev, img, img_mem, 0));

   /* Seed every texel non-zero: an unexecuted dispatch must not look like a
    * pass, and (0,0) must end at exactly the invocation count, not seed+count. */
   VkImageSubresource isr = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT };
   VkSubresourceLayout ilay;
   vkGetImageSubresourceLayout(dev, img, &isr, &ilay);
   uint8_t *ibase;
   CHECK(vkMapMemory(dev, img_mem, 0, VK_WHOLE_SIZE, 0, (void **)&ibase));
   for (uint32_t y = 0; y < IMG_DIM; y++)
      for (uint32_t x = 0; x < IMG_DIM; x++)
         *(uint32_t *)(ibase + ilay.offset + y * ilay.rowPitch
                       + x * sizeof(uint32_t)) = IMG_SEED;
   *(uint32_t *)(ibase + ilay.offset) = 0u;   /* (0,0) accumulates from zero */
   vkUnmapMemory(dev, img_mem);

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView img_view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &img_view));

   /* Seeds: each is the identity for its op, and none equals its expected
    * result, so an unexecuted dispatch cannot look like a pass. */
   uint32_t *gu;
   CHECK(vkMapMemory(dev, gu_mem, 0, gu_bytes, 0, (void **)&gu));
   gu[G_ADD]  = 0u;
   gu[G_AND]  = 0xFFFFFFFFu;
   gu[G_OR]   = 0u;
   gu[G_XOR]  = 0u;
   gu[G_UMIN] = 0xFFFFFFFFu;
   gu[G_UMAX] = 0u;
   gu[G_XCHG] = XCHG_SEED;
   vkUnmapMemory(dev, gu_mem);

   int32_t *gi;
   CHECK(vkMapMemory(dev, gi_mem, 0, gi_bytes, 0, (void **)&gi));
   gi[GI_MIN] = INT32_MAX;
   gi[GI_MAX] = INT32_MIN;
   vkUnmapMemory(dev, gi_mem);

   uint32_t *rp;
   CHECK(vkMapMemory(dev, r_mem, 0, r_bytes, 0, (void **)&rp));
   for (uint32_t i = 0; i < n_res; i++) {
      rp[i] = SENTINEL;
   }
   for (uint32_t i = 0; i < INVOCS; i++) {
      rp[INVOCS + i] = LANE_SEED(i);
   }
   vkUnmapMemory(dev, r_mem);

   /* --- shader + pipeline ------------------------------------------- */
   size_t spv_size = 0;
   uint32_t *spv = read_spirv(spv_path, &spv_size);
   if (!spv) {
      return 1;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(spv);

   VkDescriptorSetLayoutBinding dslb[4];
   for (uint32_t i = 0; i < 3; i++) {
      dslb[i] = (VkDescriptorSetLayoutBinding){
         .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   dslb[3] = (VkDescriptorSetLayoutBinding){
      .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX : dev_max_x;
   while (local_size > 1 && (INVOCS % local_size) != 0) {
      local_size >>= 1;
   }
   if (local_size == 0) {
      local_size = 1;
   }
   printf("local_size_x=%u (device max=%u)\n", local_size, dev_max_x);

   /* The kernel indexes its own results row at INVOCS, so the invocation count
    * has to reach it as well as the workgroup size. */
   const uint32_t spec_data[2] = { local_size, INVOCS };
   VkSpecializationMapEntry sme[2] = {
      { .constantID = 0, .offset = 0,                .size = sizeof(uint32_t) },
      { .constantID = 1, .offset = sizeof(uint32_t), .size = sizeof(uint32_t) },
   };
   VkSpecializationInfo spec_info = {
      .mapEntryCount = 2, .pMapEntries = sme,
      .dataSize = sizeof(spec_data), .pData = spec_data,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec_info,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  .descriptorCount = 1 },
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

   VkDescriptorBufferInfo dbi[3] = {
      { .buffer = gu_buf, .offset = 0, .range = gu_bytes },
      { .buffer = gi_buf, .offset = 0, .range = gi_bytes },
      { .buffer = r_buf,  .offset = 0, .range = r_bytes  },
   };
   VkWriteDescriptorSet wds[4];
   for (uint32_t i = 0; i < 3; i++) {
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[i],
      };
   }
   VkDescriptorImageInfo dii = {
      .imageView = img_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   wds[3] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = 3, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &dii,
   };
   vkUpdateDescriptorSets(dev, 4, wds, 0, NULL);

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
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   /* A storage image must be in GENERAL to be written by a shader. */
   VkImageMemoryBarrier ibar = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        0, NULL, 0, NULL, 1, &ibar);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, INVOCS / local_size, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify ------------------------------------------------------- */
   unsigned fails = 0;

   /* The storage image: (0,0) took one atomic add per invocation, and no other
    * texel was touched. A descriptor the scan failed to find would leave the
    * device writing a host address, and (0,0) would still read 0. */
   uint8_t *ires;
   CHECK(vkMapMemory(dev, img_mem, 0, VK_WHOLE_SIZE, 0, (void **)&ires));
   uint32_t img_got = *(uint32_t *)(ires + ilay.offset);
   unsigned img_spill = 0;
   for (uint32_t y = 0; y < IMG_DIM; y++)
      for (uint32_t x = 0; x < IMG_DIM; x++) {
         if (x == 0 && y == 0)
            continue;
         if (*(uint32_t *)(ires + ilay.offset + y * ilay.rowPitch
                           + x * sizeof(uint32_t)) != IMG_SEED)
            img_spill++;
      }
   vkUnmapMemory(dev, img_mem);
   if (img_got != INVOCS) {
      fprintf(stderr, "  image atomic: (0,0) = %u, want %u\n", img_got, INVOCS);
      fails++;
   }
   if (img_spill) {
      fprintf(stderr, "  image atomic: %u texels outside (0,0) modified\n",
              img_spill);
      fails++;
   }

   uint32_t want_u[G_SLOTS];
   want_u[G_ADD]  = 0u;
   want_u[G_AND]  = 0xFFFFFFFFu;
   want_u[G_OR]   = 0u;
   want_u[G_XOR]  = 0u;
   want_u[G_UMIN] = 0xFFFFFFFFu;
   want_u[G_UMAX] = 0u;
   for (uint32_t i = 0; i < INVOCS; i++) {
      uint32_t v  = i + 1u;
      uint32_t mv = ((i & 1u) != 0u) ? (0x80000000u + v) : v;
      want_u[G_ADD] += v;
      want_u[G_AND] &= ~(1u << (i & 31u));
      want_u[G_OR]  |=  (1u << (i & 31u));
      want_u[G_XOR] ^= v;
      if (mv < want_u[G_UMIN]) {
         want_u[G_UMIN] = mv;
      }
      if (mv > want_u[G_UMAX]) {
         want_u[G_UMAX] = mv;
      }
   }
   want_u[G_XCHG] = 0u;                 /* checked by multiset, not by value */

   CHECK(vkMapMemory(dev, gu_mem, 0, gu_bytes, 0, (void **)&gu));
   for (uint32_t k = 0; k < G_SLOTS; k++) {
      if (k == G_XCHG) {
         continue;
      }
      if (gu[k] != want_u[k]) {
         fprintf(stderr, "  %s: got 0x%08x, want 0x%08x\n",
                 g_name[k], gu[k], want_u[k]);
         fails++;
      }
   }
   uint32_t xchg_final = gu[G_XCHG];
   vkUnmapMemory(dev, gu_mem);

   int32_t want_imin = INT32_MAX, want_imax = INT32_MIN;
   for (uint32_t i = 0; i < INVOCS; i++) {
      int32_t sv = (int32_t)i - 32;
      if (sv < want_imin) {
         want_imin = sv;
      }
      if (sv > want_imax) {
         want_imax = sv;
      }
   }
   CHECK(vkMapMemory(dev, gi_mem, 0, gi_bytes, 0, (void **)&gi));
   if (gi[GI_MIN] != want_imin) {
      fprintf(stderr, "  atomicMin.i: got %d, want %d\n", gi[GI_MIN], want_imin);
      fails++;
   }
   if (gi[GI_MAX] != want_imax) {
      fprintf(stderr, "  atomicMax.i: got %d, want %d\n", gi[GI_MAX], want_imax);
      fails++;
   }
   vkUnmapMemory(dev, gi_mem);

   CHECK(vkMapMemory(dev, r_mem, 0, r_bytes, 0, (void **)&rp));
   /* Only the two result rows are sentinel-seeded; the middle row holds the
    * per-lane AMO targets and is checked by value below. */
   unsigned sentinels = 0;
   for (uint32_t i = 0; i < INVOCS; i++) {
      if (rp[i] == SENTINEL) {
         sentinels++;
      }
      if (rp[2u * INVOCS + i] == SENTINEL) {
         sentinels++;
      }
   }
   if (sentinels) {
      fprintf(stderr,
              "  %u/%u result slots still hold the sentinel -- the dispatch did "
              "not write them. Under STRICT this means atomic.comp bailed to "
              "llvmpipe: an atomic op has no lowering in the translator.\n",
              sentinels, 2u * INVOCS);
      fails++;
   }

   /* Per-lane response routing. The multiset check below proves nothing was
    * dropped or duplicated; it is satisfied by ANY permutation of the returned
    * values across lanes. These two checks are what pin each result to the
    * invocation that asked for it. */
   unsigned routing_bad = 0, target_bad = 0;
   for (uint32_t i = 0; i < INVOCS; i++) {
      if (rp[2u * INVOCS + i] != LANE_SEED(i)) {
         if (routing_bad < 4) {
            fprintf(stderr,
                    "  lane %u got old value 0x%08x from its own target, want "
                    "0x%08x -- an AMO response reached the wrong lane\n",
                    i, rp[2u * INVOCS + i], LANE_SEED(i));
         }
         routing_bad++;
      }
      if (rp[INVOCS + i] != i + 1u) {
         if (target_bad < 4) {
            fprintf(stderr,
                    "  lane %u's target holds 0x%08x, want 0x%08x -- an AMO was "
                    "applied to the wrong address\n",
                    i, rp[INVOCS + i], i + 1u);
         }
         target_bad++;
      }
   }
   if (routing_bad || target_bad) {
      fprintf(stderr, "  %u misrouted response(s), %u misapplied address(es)\n",
              routing_bad, target_bad);
      fails++;
   }

   /* Exchange: {old values} + {final} must equal {seed} + {written values}. */
   uint32_t *saw = malloc((size_t)(INVOCS + 1u) * sizeof(uint32_t));
   uint32_t *exp = malloc((size_t)(INVOCS + 1u) * sizeof(uint32_t));
   if (!saw || !exp) {
      fprintf(stderr, "FAILED: out of memory\n");
      return 1;
   }
   for (uint32_t i = 0; i < INVOCS; i++) {
      saw[i] = rp[i];
      exp[i] = i + 1u;
   }
   saw[INVOCS] = xchg_final;
   exp[INVOCS] = XCHG_SEED;
   qsort(saw, INVOCS + 1u, sizeof(uint32_t), cmp_u32);
   qsort(exp, INVOCS + 1u, sizeof(uint32_t), cmp_u32);
   if (memcmp(saw, exp, (size_t)(INVOCS + 1u) * sizeof(uint32_t)) != 0) {
      fprintf(stderr,
              "  atomicExchange: the old values returned plus the final memory "
              "value are not a permutation of the seed plus the written "
              "values -- an exchange was dropped or duplicated\n");
      fails++;
   }
   free(saw);
   free(exp);

   vkUnmapMemory(dev, r_mem);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkFreeMemory(dev, gu_mem, NULL);
   vkFreeMemory(dev, gi_mem, NULL);
   vkFreeMemory(dev, r_mem, NULL);
   vkDestroyBuffer(dev, gu_buf, NULL);
   vkDestroyBuffer(dev, gi_buf, NULL);
   vkDestroyBuffer(dev, r_buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails) {
      printf("FAILED (%u mismatches)\n", fails);
      return 1;
   }
   printf("PASSED (%u invocations, %u unsigned + 2 signed SSBO atomic ops, "
          "per-lane response routing verified, image atomic = %u)\n",
          INVOCS, G_SLOTS, img_got);
   return 0;
}
