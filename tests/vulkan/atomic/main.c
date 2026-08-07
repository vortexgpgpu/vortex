/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Atomic coverage test for the vortexpipe driver.
 *
 * Dispatches atomic.comp, which contends the integer atomics the device
 * implements -- add, and, or, xor, signed and unsigned min/max, and exchange
 * -- on an SSBO from every invocation at once.
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
 * compare-and-swap and shared-memory atomics are also absent: the hardware
 * asserts on both, for the reasons atomic.comp records.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
   const uint32_t     n_res    = INVOCS;
   const VkDeviceSize r_bytes  = (VkDeviceSize)n_res * sizeof(uint32_t);

   VkBuffer gu_buf, gi_buf, r_buf;
   VkDeviceMemory gu_mem, gi_mem, r_mem;
   if (make_buffer(dev, pd, gu_bytes, &gu_buf, &gu_mem) ||
       make_buffer(dev, pd, gi_bytes, &gi_buf, &gi_mem) ||
       make_buffer(dev, pd, r_bytes,  &r_buf,  &r_mem)) {
      return 1;
   }

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

   VkDescriptorSetLayoutBinding dslb[3];
   for (uint32_t i = 0; i < 3; i++) {
      dslb[i] = (VkDescriptorSetLayoutBinding){
         .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = dslb,
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

   VkSpecializationMapEntry sme = {
      .constantID = 0, .offset = 0, .size = sizeof(uint32_t),
   };
   VkSpecializationInfo spec_info = {
      .mapEntryCount = 1, .pMapEntries = &sme,
      .dataSize = sizeof(uint32_t), .pData = &local_size,
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

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3,
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

   VkDescriptorBufferInfo dbi[3] = {
      { .buffer = gu_buf, .offset = 0, .range = gu_bytes },
      { .buffer = gi_buf, .offset = 0, .range = gi_bytes },
      { .buffer = r_buf,  .offset = 0, .range = r_bytes  },
   };
   VkWriteDescriptorSet wds[3];
   for (uint32_t i = 0; i < 3; i++) {
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[i],
      };
   }
   vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);

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

   uint32_t want_u[G_SLOTS];
   want_u[G_ADD]  = 0u;
   want_u[G_AND]  = 0xFFFFFFFFu;
   want_u[G_OR]   = 0u;
   want_u[G_XOR]  = 0u;
   want_u[G_UMIN] = 0xFFFFFFFFu;
   want_u[G_UMAX] = 0u;
   for (uint32_t i = 0; i < INVOCS; i++) {
      uint32_t v = i + 1u;
      want_u[G_ADD] += v;
      want_u[G_AND] &= ~(1u << (i & 31u));
      want_u[G_OR]  |=  (1u << (i & 31u));
      want_u[G_XOR] ^= v;
      if (v < want_u[G_UMIN]) {
         want_u[G_UMIN] = v;
      }
      if (v > want_u[G_UMAX]) {
         want_u[G_UMAX] = v;
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
   unsigned sentinels = 0;
   for (uint32_t i = 0; i < n_res; i++) {
      if (rp[i] == SENTINEL) {
         sentinels++;
      }
   }
   if (sentinels) {
      fprintf(stderr,
              "  %u/%u result slots still hold the sentinel -- the dispatch did "
              "not write them. Under STRICT this means atomic.comp bailed to "
              "llvmpipe: an atomic op has no lowering in the translator.\n",
              sentinels, n_res);
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
   printf("PASSED (%u invocations, %u unsigned + 2 signed SSBO atomic ops)\n",
          INVOCS, G_SLOTS);
   return 0;
}
