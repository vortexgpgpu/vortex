/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Shared-memory atomic coverage test for the vortexpipe driver.
 *
 * Dispatches atomic_shared.comp, which contends the same op set as
 * tests/vulkan/atomic -- add, and, or, xor, signed and unsigned min/max, and
 * exchange -- on `shared` variables rather than on an SSBO, so each AMO's
 * address lands in the local-memory aperture instead of the dcache. It is a
 * separate test from `atomic` because the device implements only one of the two
 * address spaces today, and the suite excludes it per backend on that basis.
 *
 * Several workgroups run, and each checks its own shared block against its own
 * expected row. The seeds derive from the global invocation id, so no two rows
 * expect the same answer -- a device whose workgroups shared one allocation
 * produces equal rows and fails, rather than passing on a coincidence.
 *
 * Every expectation is independent of the order the device schedules its warps:
 *   - add/and/or/xor/min/max are commutative and associative, so the final
 *     value is fixed;
 *   - exchange is order-dependent, so it is checked with a multiset invariant:
 *     the old values handed back, together with the value left in memory, must
 *     be exactly the seed together with all the values written.
 *
 * The unsigned min/max operands straddle 0x7fffffff on purpose. With only small
 * positive operands a signed compare and an unsigned compare agree, so those two
 * checks cannot distinguish AMOMINU from AMOMIN at all, and a device executing
 * the signed variant passes them.
 *
 * compare-and-swap is absent: it lowers to an LR/SC pair whose forward progress
 * is a separate open defect.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Must not exceed LSIZE_MAX in atomic_shared.comp, which sizes the per-lane
 * shared array. The kernel indexes that array by local invocation id, so a
 * larger workgroup would run off the end of it silently. */
#define LOCAL_SIZE_MAX 64u
#define SENTINEL       0xDEADBEEFu

/* Workgroups dispatched. More than one, so that shared blocks belonging to
 * different groups are required to be distinct storage. */
#define GROUPS         4u

/* Slots in a workgroup's output row; must match atomic_shared.comp. */
#define S_ADD    0u
#define S_AND    1u
#define S_OR     2u
#define S_XOR    3u
#define S_UMIN   4u
#define S_UMAX   5u
#define S_XCHG   6u
#define S_USLOTS 7u
#define S_IMIN   7u
#define S_IMAX   8u
#define S_SLOTS  9u

/* Seed for the exchange slot: distinct from every value the shader writes, so
 * it can be told apart in the multiset check. */
#define XCHG_SEED 0xA5A5A5A5u

/* Base of the per-invocation seeds; must match LANE_SEED in the kernel. */
#define LANE_SEED 0x5A5A0000u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static const char *const s_name[S_USLOTS] = {
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

/* {old values returned} + {value left in memory} must be a permutation of
 * {seed} + {values written}. A dropped or duplicated exchange breaks the
 * equality without the check needing to know who went first. */
static int
xchg_multiset_ok(const uint32_t *old_vals, uint32_t n, uint32_t final_val,
                 uint32_t first_written)
{
   uint32_t *saw = malloc((size_t)(n + 1u) * sizeof(uint32_t));
   uint32_t *exp = malloc((size_t)(n + 1u) * sizeof(uint32_t));
   if (!saw || !exp) {
      free(saw);
      free(exp);
      return 0;
   }
   for (uint32_t i = 0; i < n; i++) {
      saw[i] = old_vals[i];
      exp[i] = first_written + i;
   }
   saw[n] = final_val;
   exp[n] = XCHG_SEED;
   qsort(saw, n + 1u, sizeof(uint32_t), cmp_u32);
   qsort(exp, n + 1u, sizeof(uint32_t), cmp_u32);
   int ok = memcmp(saw, exp, (size_t)(n + 1u) * sizeof(uint32_t)) == 0;
   free(saw);
   free(exp);
   return ok;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "atomic_shared.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-atomic-shared",
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

   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX : dev_max_x;
   if (local_size == 0) {
      local_size = 1;
   }
   if (local_size > LOCAL_SIZE_MAX) {
      fprintf(stderr,
              "FAILED: workgroup size %u exceeds the kernel's per-lane shared "
              "array (%u)\n", local_size, LOCAL_SIZE_MAX);
      return 1;
   }
   const uint32_t invocs = GROUPS * local_size;
   printf("local_size_x=%u (device max=%u), %u workgroups\n",
          local_size, dev_max_x, GROUPS);

   /* --- two storage buffers: one result row per group, one old value per
    *     invocation ------------------------------------------------------ */
   const VkDeviceSize o_bytes  = (VkDeviceSize)GROUPS * S_SLOTS * sizeof(uint32_t);
   /* Two rows: the contended-exchange old values, then the old value each
    * invocation got from its own shared slot. */
   const VkDeviceSize xr_bytes = (VkDeviceSize)2u * invocs * sizeof(uint32_t);

   VkBuffer o_buf, xr_buf;
   VkDeviceMemory o_mem, xr_mem;
   if (make_buffer(dev, pd, o_bytes,  &o_buf,  &o_mem) ||
       make_buffer(dev, pd, xr_bytes, &xr_buf, &xr_mem)) {
      return 1;
   }

   /* Sentinels everywhere: the shader overwrites every slot, so a dispatch
    * that did not run cannot look like a pass. */
   uint32_t *op;
   CHECK(vkMapMemory(dev, o_mem, 0, o_bytes, 0, (void **)&op));
   for (uint32_t i = 0; i < GROUPS * S_SLOTS; i++) {
      op[i] = SENTINEL;
   }
   vkUnmapMemory(dev, o_mem);

   uint32_t *xrp;
   CHECK(vkMapMemory(dev, xr_mem, 0, xr_bytes, 0, (void **)&xrp));
   for (uint32_t i = 0; i < 2u * invocs; i++) {
      xrp[i] = SENTINEL;
   }
   vkUnmapMemory(dev, xr_mem);

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

   VkDescriptorSetLayoutBinding dslb[2];
   for (uint32_t i = 0; i < 2; i++) {
      dslb[i] = (VkDescriptorSetLayoutBinding){
         .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
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

   /* The kernel indexes its second results row at the invocation count, so
    * that has to reach it as well as the workgroup size. */
   const uint32_t spec_data[2] = { local_size, invocs };
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

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2,
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

   VkDescriptorBufferInfo dbi[2] = {
      { .buffer = o_buf,  .offset = 0, .range = o_bytes  },
      { .buffer = xr_buf, .offset = 0, .range = xr_bytes },
   };
   VkWriteDescriptorSet wds[2];
   for (uint32_t i = 0; i < 2; i++) {
      wds[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[i],
      };
   }
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

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
   vkCmdDispatch(cb, GROUPS, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify ------------------------------------------------------- */
   unsigned fails = 0;

   CHECK(vkMapMemory(dev, o_mem,  0, o_bytes,  0, (void **)&op));
   CHECK(vkMapMemory(dev, xr_mem, 0, xr_bytes, 0, (void **)&xrp));

   unsigned sentinels = 0;
   for (uint32_t i = 0; i < 2u * invocs; i++) {
      if (xrp[i] == SENTINEL) {
         sentinels++;
      }
   }
   if (sentinels) {
      fprintf(stderr,
              "  %u/%u result slots still hold the sentinel -- the dispatch did "
              "not write them. Under STRICT this means atomic_shared.comp bailed "
              "to llvmpipe: a shared atomic has no lowering in the translator.\n",
              sentinels, 2u * invocs);
      fails++;
   }

   /* Per-lane response routing. The multiset check below is satisfied by any
    * permutation of the returned values across lanes, so it cannot show that
    * each invocation was answered. Every invocation owns a distinct shared
    * slot with a distinct seed, which makes the expected value per-lane.
    *
    * Scope: this covers routing, not per-lane hart identity -- the LMEM path
    * can drop its per-lane tids and still pass here, because hart_id decides
    * LR/SC reservations rather than response delivery. */
   unsigned routing_bad = 0;
   for (uint32_t i = 0; i < invocs; i++) {
      if (xrp[invocs + i] != LANE_SEED + i) {
         if (routing_bad < 4) {
            fprintf(stderr,
                    "  lane %u got old value 0x%08x from its own shared slot, "
                    "want 0x%08x -- a shared AMO response reached the wrong "
                    "lane\n", i, xrp[invocs + i], LANE_SEED + i);
         }
         routing_bad++;
      }
   }
   if (routing_bad) {
      fprintf(stderr, "  %u misrouted shared AMO response(s)\n", routing_bad);
      fails++;
   }

   for (uint32_t g = 0; g < GROUPS; g++) {
      uint32_t want[S_USLOTS];
      want[S_ADD]  = 0u;
      want[S_AND]  = 0xFFFFFFFFu;
      want[S_OR]   = 0u;
      want[S_XOR]  = 0u;
      want[S_UMIN] = 0xFFFFFFFFu;
      want[S_UMAX] = 0u;
      int32_t w_imin = INT32_MAX, w_imax = INT32_MIN;
      for (uint32_t lid = 0; lid < local_size; lid++) {
         uint32_t gid = g * local_size + lid;
         uint32_t v   = gid + 1u;
         uint32_t mv  = ((lid & 1u) != 0u) ? (0x80000000u + v) : v;
         int32_t  sv  = (int32_t)gid - 32;
         want[S_ADD] += v;
         want[S_AND] &= ~(1u << (lid & 31u));
         want[S_OR]  |=  (1u << (lid & 31u));
         want[S_XOR] ^= v;
         if (mv < want[S_UMIN]) {
            want[S_UMIN] = mv;
         }
         if (mv > want[S_UMAX]) {
            want[S_UMAX] = mv;
         }
         if (sv < w_imin) {
            w_imin = sv;
         }
         if (sv > w_imax) {
            w_imax = sv;
         }
      }

      const uint32_t *row = op + (size_t)g * S_SLOTS;
      for (uint32_t k = 0; k < S_USLOTS; k++) {
         if (k == S_XCHG) {
            continue;                   /* checked by multiset, not by value */
         }
         if (row[k] != want[k]) {
            fprintf(stderr, "  wg%u %s: got 0x%08x, want 0x%08x\n",
                    g, s_name[k], row[k], want[k]);
            fails++;
         }
      }
      if ((int32_t)row[S_IMIN] != w_imin) {
         fprintf(stderr, "  wg%u atomicMin.i: got %d, want %d\n",
                 g, (int32_t)row[S_IMIN], w_imin);
         fails++;
      }
      if ((int32_t)row[S_IMAX] != w_imax) {
         fprintf(stderr, "  wg%u atomicMax.i: got %d, want %d\n",
                 g, (int32_t)row[S_IMAX], w_imax);
         fails++;
      }
      if (!xchg_multiset_ok(xrp + (size_t)g * local_size, local_size,
                            row[S_XCHG], g * local_size + 1u)) {
         fprintf(stderr,
                 "  wg%u atomicExchange: the old values returned plus the final "
                 "shared value are not a permutation of the seed plus the "
                 "written values -- an exchange was dropped or duplicated\n", g);
         fails++;
      }
   }

   vkUnmapMemory(dev, xr_mem);
   vkUnmapMemory(dev, o_mem);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkFreeMemory(dev, o_mem, NULL);
   vkFreeMemory(dev, xr_mem, NULL);
   vkDestroyBuffer(dev, o_buf, NULL);
   vkDestroyBuffer(dev, xr_buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (fails) {
      printf("FAILED (%u mismatches)\n", fails);
      return 1;
   }
   printf("PASSED (%u invocations in %u workgroups, %u unsigned + 2 signed "
          "shared-memory atomic ops, per-lane response routing verified)\n",
          invocs, GROUPS, S_USLOTS);
   return 0;
}
