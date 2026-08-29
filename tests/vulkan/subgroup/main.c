/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Subgroup coverage test for the vortexpipe driver.
 *
 * Dispatches two compute modules and reproduces every cross-lane result here
 * on the host:
 *
 *   subgroup.comp      ballot, elect, broadcastFirst, shuffle, vote (all/any/
 *                      allEqual), add/max reduce, inclusive/exclusive scan,
 *                      and a divergent even-lanes-only reduce.
 *   subgroup_ext.comp  shuffleUp/Down/Xor and a clustered add.
 *
 * They are separate modules on purpose: under MESA_VORTEX_STRICT=1 an
 * intrinsic the translator cannot lower bails the entire shader containing it,
 * so folding both sets into one kernel would let a single missing op erase the
 * evidence for all the others.
 *
 * The buffer is seeded with a sentinel rather than left undefined. A bail
 * turns the dispatch into a no-op, so the sentinel survives and is reported as
 * such -- that is what makes a pass evidence the *device* executed these ops
 * rather than the CPU.
 *
 * Nothing here hardcodes the lane mapping. Each invocation reports the
 * gl_SubgroupID and gl_SubgroupInvocationID it actually observed, and the host
 * groups invocations by those reported values before computing any
 * expectation. A subgroup is a Vortex warp, but which invocations land in
 * which warp is a runtime property, and a test that assumed it would fail for
 * the wrong reason if it ever changed.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_OPS      16u
#define NUM_EXT_OPS   6u
#define INVOCS       64u
#define SENTINEL     0xDEADBEEFu
/* Maximum local_size_x the test will request; the actual value is specialized
 * at pipeline creation to min(this, maxComputeWorkGroupSize[0]). vortexpipe
 * reports the Vortex hardware cap (NUM_THREADS x NUM_WARPS) there. */
#define LOCAL_SIZE_MAX 64u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* Slot names, so a mismatch names the op that broke rather than an index. */
static const char *const op_name[NUM_OPS] = {
   "gl_SubgroupID",      "gl_SubgroupInvocationID", "gl_SubgroupSize",
   "subgroupBallot",     "subgroupElect",           "subgroupBroadcastFirst",
   "subgroupShuffle.rev", "subgroupAll",            "subgroupAny",
   "subgroupAllEqual.uniform", "subgroupAllEqual.varying",
   "subgroupAdd",        "subgroupMax",             "subgroupInclusiveAdd",
   "subgroupExclusiveAdd", "subgroupAdd.divergent",
};

static const char *const ext_op_name[NUM_EXT_OPS] = {
   "gl_SubgroupID", "gl_SubgroupInvocationID",
   "subgroupShuffleUp", "subgroupShuffleDown", "subgroupShuffleXor",
   "subgroupClusteredAdd.2",
};

/* The value subgroup.comp and subgroup_ext.comp both derive per invocation. */
static uint32_t
lane_value(uint32_t gid)
{
   return gid + 1u;
}

struct vk_ctx {
   VkPhysicalDevice pd;
   VkDevice         dev;
   VkQueue          queue;
   uint32_t         qf;
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

/* Build one compute pipeline around `spv_path`, dispatch it over `buf`, and
 * wait for it. Every object created here is destroyed before returning, so the
 * two dispatches in main() share nothing but the buffer. */
static int
run_shader(const struct vk_ctx *c, const char *spv_path, VkBuffer buf,
           VkDeviceSize bytes, uint32_t groups, uint32_t local_size)
{
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
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   free(spv);

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, &pl));

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
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp));

   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, &ds));

   VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = bytes };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
   };
   vkUpdateDescriptorSets(c->dev, 1, &wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(c->dev, &cmpci, NULL, &cp));

   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cb));

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, groups, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));

   vkDestroyCommandPool(c->dev, cp, NULL);
   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyPipelineLayout(c->dev, pl, NULL);
   vkDestroyDescriptorSetLayout(c->dev, dsl, NULL);
   vkDestroyShaderModule(c->dev, sm, NULL);
   return 0;
}

/* ---- subgroup reconstruction ----------------------------------------- *
 * The lane identity each invocation reported. `peer[]` lists, for one
 * invocation, every invocation the device placed in the same subgroup —
 * derived from the reported (workgroup, gl_SubgroupID) pair, never assumed. */
struct lanes {
   uint32_t sgid[INVOCS];
   uint32_t lane[INVOCS];
   uint32_t size;                    /* gl_SubgroupSize, must be uniform */
};

/* Collect the invocations sharing gid's subgroup into `peer`, returning the
 * count. Two invocations share a subgroup only when they are in the same
 * workgroup AND report the same gl_SubgroupID: the id is scoped to the
 * workgroup, so comparing it alone would merge warp 0 of every group. */
static uint32_t
collect_peers(const struct lanes *l, uint32_t gid, uint32_t local_size,
              uint32_t *peer)
{
   uint32_t n = 0;
   uint32_t wg = gid / local_size;
   for (uint32_t g = 0; g < INVOCS; g++) {
      if (g / local_size == wg && l->sgid[g] == l->sgid[gid]) {
         peer[n++] = g;
      }
   }
   return n;
}

/* Read the lane identity a dispatch reported into `l`. Each dispatch is
 * reconstructed from its OWN output: gl_SubgroupID is the physical warp id, and
 * which warp a workgroup's threads land on is a scheduling decision that can
 * differ between two dispatches of the same geometry. Vulkan promises nothing
 * about that id across dispatches, so reusing one dispatch's labels for another
 * is wrong -- the partition is stable, the numbering is not. */
static void
build_lanes(const uint32_t *data, uint32_t stride, uint32_t sgid_off,
            uint32_t lane_off, uint32_t size, struct lanes *l)
{
   l->size = size;
   for (uint32_t g = 0; g < INVOCS; g++) {
      l->sgid[g] = data[g * stride + sgid_off];
      l->lane[g] = data[g * stride + lane_off];
   }
}

/* Find the peer occupying lane `want`; INVOCS if the subgroup has no such
 * lane (which makes any shuffle reading it undefined, so the caller skips). */
static uint32_t
peer_at_lane(const struct lanes *l, const uint32_t *peer, uint32_t n,
             uint32_t want)
{
   for (uint32_t k = 0; k < n; k++) {
      if (l->lane[peer[k]] == want) {
         return peer[k];
      }
   }
   return INVOCS;
}

int
main(int argc, char **argv)
{
   const char *spv_main = (argc > 1) ? argv[1] : "subgroup.comp.spv";
   const char *spv_ext  = (argc > 2) ? argv[2] : "subgroup_ext.comp.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-subgroup",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   /* --- physical device -------------------------------------------- */
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

   /* Subgroup properties drive both the expectations and the skip decision.
    * A subgroup is a Vortex warp, so subgroupSize is NUM_THREADS (4 in the
    * default config) -- not the 32 or 64 subgroup tests usually assume. */
   VkPhysicalDeviceSubgroupProperties sgp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &sgp,
   };
   vkGetPhysicalDeviceProperties2(pd, &props2);
   printf("subgroupSize=%u supportedOps=0x%x supportedStages=0x%x\n",
          sgp.subgroupSize, sgp.supportedOperations, sgp.supportedStages);

   const VkSubgroupFeatureFlags need_main =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
   const VkSubgroupFeatureFlags need_ext =
      VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
      VK_SUBGROUP_FEATURE_CLUSTERED_BIT;

   if (!(sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
       (sgp.supportedOperations & need_main) != need_main) {
      printf("NotSupported: device does not advertise the core subgroup "
             "operations in compute (ops=0x%x)\n", sgp.supportedOperations);
      return 0;
   }
   /* The relative-shuffle and clustered families are advertised by this
    * device, so they are exercised rather than skipped. If a future device
    * honestly withdraws them, this test drops that phase instead of failing —
    * but it never lets an advertised bit go untested. */
   const int run_ext = (sgp.supportedOperations & need_ext) == need_ext;
   if (!run_ext) {
      printf("skip: shuffle-relative/clustered not advertised (ops=0x%x)\n",
             sgp.supportedOperations);
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

   /* --- logical device + queue ------------------------------------- */
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
   struct vk_ctx ctx = { .pd = pd, .dev = dev, .queue = queue, .qf = qf };

   /* --- local_size: a whole number of full subgroups ----------------- *
    * Cross-lane expectations are only well-defined when every subgroup is
    * fully populated: a shuffle reading a lane no invocation occupies is
    * undefined by the spec. Requiring local_size to be a multiple of
    * subgroupSize (and INVOCS a multiple of local_size) guarantees that. */
   uint32_t dev_max_x = props.limits.maxComputeWorkGroupSize[0];
   uint32_t local_size = LOCAL_SIZE_MAX < dev_max_x ? LOCAL_SIZE_MAX : dev_max_x;
   while (local_size > 1 &&
          ((INVOCS % local_size) != 0 || (local_size % sgp.subgroupSize) != 0)) {
      local_size >>= 1;
   }
   if (local_size == 0) {
      local_size = 1;
   }
   printf("local_size_x=%u (device max=%u)\n", local_size, dev_max_x);
   if (local_size < sgp.subgroupSize) {
      printf("NotSupported: cannot form a full subgroup of %u within a "
             "workgroup of %u\n", sgp.subgroupSize, local_size);
      return 0;
   }

   /* --- buffer sized for the larger of the two dispatches ------------ */
   const uint32_t n_main = INVOCS * NUM_OPS;
   const uint32_t n_ext  = INVOCS * NUM_EXT_OPS;
   const VkDeviceSize bytes = (VkDeviceSize)n_main * sizeof(uint32_t);

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);
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
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));

   uint32_t *p = NULL;
   const uint32_t groups = INVOCS / local_size;
   unsigned fails = 0;

   /* ================= phase A/B: subgroup.comp ====================== */
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   for (uint32_t i = 0; i < n_main; i++) {
      p[i] = SENTINEL;
   }
   vkUnmapMemory(dev, mem);

   if (run_shader(&ctx, spv_main, buf, bytes, groups, local_size) != 0) {
      return 1;
   }

   uint32_t *got = malloc((size_t)n_main * sizeof(uint32_t));
   if (!got) {
      fprintf(stderr, "FAILED: out of memory\n");
      return 1;
   }
   CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
   memcpy(got, p, (size_t)n_main * sizeof(uint32_t));
   vkUnmapMemory(dev, mem);

   unsigned sentinels = 0;
   for (uint32_t i = 0; i < n_main; i++) {
      if (got[i] == SENTINEL) {
         sentinels++;
      }
   }
   if (sentinels) {
      fprintf(stderr,
              "  %u/%u slots still hold the sentinel -- the dispatch did not "
              "write them. Under STRICT this means subgroup.comp bailed to "
              "llvmpipe: a cross-lane intrinsic has no lowering in the "
              "translator.\n", sentinels, n_main);
      free(got);
      printf("FAILED (subgroup.comp did not execute on the device)\n");
      return 1;
   }

   /* Reconstruct the lane layout the device reported, then check it is
    * self-consistent before trusting it as the basis for every expectation. */
   struct lanes l;
   build_lanes(got, NUM_OPS, 0u, 1u, got[2], &l);
   for (uint32_t g = 0; g < INVOCS; g++) {
      if (got[g * NUM_OPS + 2u] != l.size) {
         fprintf(stderr, "  invoc %u gl_SubgroupSize: got %u, want %u "
                 "(must be uniform)\n", g, got[g * NUM_OPS + 2u], l.size);
         fails++;
      }
      if (l.lane[g] >= l.size) {
         fprintf(stderr, "  invoc %u gl_SubgroupInvocationID %u is outside "
                 "subgroup size %u\n", g, l.lane[g], l.size);
         fails++;
      }
   }
   if (l.size != sgp.subgroupSize) {
      fprintf(stderr, "  gl_SubgroupSize %u disagrees with the reported "
              "VkPhysicalDeviceSubgroupProperties.subgroupSize %u\n",
              l.size, sgp.subgroupSize);
      fails++;
   }
   if (fails) {
      free(got);
      printf("FAILED (%u lane-identity mismatches; cross-lane checks skipped "
             "because the reference cannot be built)\n", fails);
      return 1;
   }

   for (uint32_t gid = 0; gid < INVOCS; gid++) {
      uint32_t peer[INVOCS];
      uint32_t n = collect_peers(&l, gid, local_size, peer);
      uint32_t lane = l.lane[gid];

      /* Every subgroup must be full: the shuffles below read fixed lanes. */
      if (n != l.size) {
         fprintf(stderr, "  invoc %u: subgroup %u holds %u lanes, expected a "
                 "full %u\n", gid, l.sgid[gid], n, l.size);
         fails++;
         continue;
      }

      uint32_t ballot = 0, sum = 0, mx = 0, incl = 0, excl = 0;
      uint32_t even_sum = 0, min_lane = l.size, first_v = 0;
      int has_lane0 = 0;
      for (uint32_t k = 0; k < n; k++) {
         uint32_t m = peer[k];
         uint32_t ml = l.lane[m];
         uint32_t mv = lane_value(m);
         ballot |= 1u << ml;
         sum += mv;
         if (mv > mx) {
            mx = mv;
         }
         if (ml <= lane) {
            incl += mv;
         }
         if (ml < lane) {
            excl += mv;
         }
         if ((ml & 1u) == 0u) {
            even_sum += mv;
         }
         if (ml < min_lane) {
            min_lane = ml;
            first_v = mv;
         }
         if (ml == 0u) {
            has_lane0 = 1;
         }
      }

      uint32_t rev = peer_at_lane(&l, peer, n, l.size - 1u - lane);
      uint32_t want[NUM_OPS];
      want[0]  = l.sgid[gid];                      /* echoed, checked above */
      want[1]  = lane;
      want[2]  = l.size;
      want[3]  = ballot;
      want[4]  = (lane == min_lane) ? 1u : 0u;
      want[5]  = first_v;
      want[6]  = lane_value(rev);
      want[7]  = 1u;                               /* all v > 0 */
      want[8]  = has_lane0 ? 1u : 0u;
      want[9]  = 1u;                               /* size is uniform */
      want[10] = (n == 1u) ? 1u : 0u;              /* v is distinct per lane */
      want[11] = sum;
      want[12] = mx;
      want[13] = incl;
      want[14] = excl;
      want[15] = ((lane & 1u) == 0u) ? even_sum : 0u;

      for (uint32_t k = 3; k < NUM_OPS; k++) {
         if (got[gid * NUM_OPS + k] != want[k]) {
            if (fails < 12) {
               fprintf(stderr, "  invoc %u (sg %u lane %u) %s: got %u, want %u\n",
                       gid, l.sgid[gid], lane, op_name[k],
                       got[gid * NUM_OPS + k], want[k]);
            }
            fails++;
         }
      }
   }

   if (fails) {
      free(got);
      printf("FAILED (%u mismatches in subgroup.comp)\n", fails);
      return 1;
   }
   printf("phase A/B PASSED (%u invocations x %u subgroup ops, subgroupSize=%u)\n",
          INVOCS, NUM_OPS, l.size);

   /* ================= phase C: subgroup_ext.comp ==================== */
   if (run_ext) {
      CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
      for (uint32_t i = 0; i < n_main; i++) {
         p[i] = SENTINEL;
      }
      vkUnmapMemory(dev, mem);

      if (run_shader(&ctx, spv_ext, buf, bytes, groups, local_size) != 0) {
         free(got);
         return 1;
      }

      uint32_t *ge = malloc((size_t)n_ext * sizeof(uint32_t));
      if (!ge) {
         fprintf(stderr, "FAILED: out of memory\n");
         free(got);
         return 1;
      }
      CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **)&p));
      memcpy(ge, p, (size_t)n_ext * sizeof(uint32_t));
      vkUnmapMemory(dev, mem);

      sentinels = 0;
      for (uint32_t i = 0; i < n_ext; i++) {
         if (ge[i] == SENTINEL) {
            sentinels++;
         }
      }
      if (sentinels) {
         fprintf(stderr,
                 "  %u/%u slots still hold the sentinel -- subgroup_ext.comp "
                 "bailed. shuffleUp/Down/Xor and clustered reduce are "
                 "advertised by this device (VK_SUBGROUP_FEATURE_"
                 "SHUFFLE_RELATIVE_BIT | _CLUSTERED_BIT) but at least one of "
                 "them has no lowering in the translator.\n",
                 sentinels, n_ext);
         free(ge);
         free(got);
         printf("FAILED (subgroup_ext.comp did not execute on the device)\n");
         return 1;
      }

      /* Rebuild from this dispatch's own report rather than reusing phase A/B's
       * -- see build_lanes on why the warp numbering may differ. */
      struct lanes le;
      build_lanes(ge, NUM_EXT_OPS, 0u, 1u, l.size, &le);

      for (uint32_t gid = 0; gid < INVOCS; gid++) {
         uint32_t peer[INVOCS];
         uint32_t n = collect_peers(&le, gid, local_size, peer);
         uint32_t lane = le.lane[gid];
         uint32_t base = gid * NUM_EXT_OPS;

         if (lane >= le.size || n != le.size) {
            fprintf(stderr, "  invoc %u: ext dispatch reports lane %u of a "
                    "subgroup holding %u lanes, expected a full %u\n",
                    gid, lane, n, le.size);
            fails++;
            continue;
         }
         /* The warp *labels* may differ between dispatches, but the partition
          * must not: the same invocations have to end up together, or the two
          * dispatches disagree about the machine and neither reference is
          * trustworthy. */
         uint32_t mpeer[INVOCS];
         uint32_t mn = collect_peers(&l, gid, local_size, mpeer);
         int same = (mn == n);
         for (uint32_t k = 0; same && k < n; k++) {
            int found = 0;
            for (uint32_t j = 0; j < mn; j++) {
               if (mpeer[j] == peer[k]) {
                  found = 1;
                  break;
               }
            }
            same = found;
         }
         if (!same) {
            fprintf(stderr, "  invoc %u: the two dispatches grouped this "
                    "invocation with different peers -- the subgroup partition "
                    "is not stable\n", gid);
            fails++;
            continue;
         }

         /* shuffleUp/Down: only the lanes whose source is in range are
          * defined; the edge lanes are skipped rather than guessed. */
         if (lane >= 1u) {
            uint32_t src = peer_at_lane(&le, peer, n, lane - 1u);
            if (src != INVOCS && ge[base + 2u] != lane_value(src)) {
               if (fails < 12) {
                  fprintf(stderr, "  invoc %u (sg %u lane %u) %s: got %u, want %u\n",
                          gid, le.sgid[gid], lane, ext_op_name[2],
                          ge[base + 2u], lane_value(src));
               }
               fails++;
            }
         }
         if (lane + 1u < le.size) {
            uint32_t src = peer_at_lane(&le, peer, n, lane + 1u);
            if (src != INVOCS && ge[base + 3u] != lane_value(src)) {
               if (fails < 12) {
                  fprintf(stderr, "  invoc %u (sg %u lane %u) %s: got %u, want %u\n",
                          gid, le.sgid[gid], lane, ext_op_name[3],
                          ge[base + 3u], lane_value(src));
               }
               fails++;
            }
         }
         if (le.size >= 2u) {
            uint32_t src = peer_at_lane(&le, peer, n, lane ^ 1u);
            if (src != INVOCS && ge[base + 4u] != lane_value(src)) {
               if (fails < 12) {
                  fprintf(stderr, "  invoc %u (sg %u lane %u) %s: got %u, want %u\n",
                          gid, le.sgid[gid], lane, ext_op_name[4],
                          ge[base + 4u], lane_value(src));
               }
               fails++;
            }
            /* Cluster of 2 = lanes {lane & ~1, lane | 1}. A driver that drops
             * the cluster size returns the whole-subgroup sum instead. */
            uint32_t csum = 0;
            for (uint32_t k = 0; k < n; k++) {
               if ((le.lane[peer[k]] & ~1u) == (lane & ~1u)) {
                  csum += lane_value(peer[k]);
               }
            }
            if (ge[base + 5u] != csum) {
               if (fails < 12) {
                  fprintf(stderr, "  invoc %u (sg %u lane %u) %s: got %u, want %u\n",
                          gid, le.sgid[gid], lane, ext_op_name[5],
                          ge[base + 5u], csum);
               }
               fails++;
            }
         }
      }
      free(ge);

      if (fails) {
         free(got);
         printf("FAILED (%u mismatches in subgroup_ext.comp)\n", fails);
         return 1;
      }
      printf("phase C PASSED (%u invocations x %u relative-shuffle/clustered ops)\n",
             INVOCS, NUM_EXT_OPS);
   }

   free(got);

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("PASSED (%u invocations, subgroupSize=%u%s)\n",
          INVOCS, l.size, run_ext ? ", incl. relative-shuffle + clustered" : "");
   return 0;
}
