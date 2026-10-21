/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Float atomics (VK_EXT_shader_atomic_float, _float2) on the Vortex device:
 * add, min, max and exchange on a storage buffer, workgroup shared memory, an
 * R32_SFLOAT storage image and a buffer device address, all contended.
 *
 * The device has no float AMO, so these are built from integer AMOs. The
 * checks are what any serialisation of the atomics must satisfy, and they are
 * exact:
 *   - add: every target gets n equal exact addends, so the returned values are
 *     exactly init + j * addend for j = 0..n-1, each once, and the final value
 *     is init + n * addend;
 *   - min/max: the final value is the extreme of the non-NaN operands and the
 *     initial value (a NaN initial value does not count), and every returned
 *     value is the initial value or one of the operands and lies on the right
 *     side of the final value;
 *   - exchange: the returned values plus the final one are a permutation of
 *     the initial value and all operands.
 *
 * Under MESA_VORTEX_STRICT=1 a shader the device cannot run fails the dispatch
 * instead of running on llvmpipe, and the sentinel in every result slot shows it.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOTAL     256u
#define NSUM      4u
#define NMM       4u
#define NGRP_MAX  256u
#define LOCAL_MAX 64u
#define SENTINEL  0x7fbadbadu

/* Mirrors the shader's Acc block (std430: all floats, no padding). */
struct acc {
   float sum[NSUM];
   float mn[NMM];
   float mx[NMM];
   float xchg;
   float zmin;
   float zmax;
   float ret_sum[TOTAL];
   float ret_min[TOTAL];
   float ret_max[TOTAL];
   float ret_xchg[TOTAL];
   float ret_ssum[TOTAL];
   float ret_isum[TOTAL];
   float ret_imin[TOTAL];
   float ret_imax[TOTAL];
   float ret_bsum[TOTAL];
   float ret_bmin[TOTAL];
   float ret_bmax[TOTAL];
   float grp_sum[NGRP_MAX];
   float grp_min[NGRP_MAX];
   float grp_max[NGRP_MAX];
   float grp_nmin[NGRP_MAX];
   float grp_nmax[NGRP_MAX];
   float mm_op[TOTAL];
};

static const float addend[NSUM] = { 1.0f, 0.5f, 2.0f, -0.25f };
static const float sum_init[NSUM] = { 1000.0f, 0.0f, -64.0f, 3.5f };
#define SH_SUM_INIT   8.0f
#define SH_ADDEND     0.25f
#define BDA_INIT      -2.0f
#define BDA_ADDEND    0.125f
#define XCHG_INIT     -1.0f
/* NaNs in memory that the integer order of the bit patterns keeps, so every
 * min/max on them goes through the compare-and-swap fix-up. */
#define NAN_FOR_MIN   0xffc00000u
#define NAN_FOR_MAX   0x7fc00000u

static float
fbits(uint32_t u)
{
   float f;
   memcpy(&f, &u, 4);
   return f;
}

static uint32_t
ubits(float f)
{
   uint32_t u;
   memcpy(&u, &f, 4);
   return u;
}

/* The operand pool of the min/max atomics: both signed zeros and infinities,
 * denormals, a quiet and a signalling NaN of each sign, and ordinary values. */
static float
mm_operand(uint32_t i)
{
   static const uint32_t pool[] = {
      0x80000000u, 0x00000000u, 0x3fc00000u, 0xc0100000u, 0x7f800000u,
      0xff800000u, 0x7fc00000u, 0xffc00000u, 0x0001b4f8u, 0x8001b4f8u,
      0x7f61b1e6u, 0xc0e00000u, 0x42280000u, 0x3a83126fu, 0xba83126fu,
      0x40b00000u, 0x7f800001u, 0xff800001u, 0x3f800000u, 0xbf000000u,
   };
   const uint32_t n = sizeof(pool) / sizeof(pool[0]);
   /* A stride coprime to n spreads every pool value over every target. */
   return fbits(pool[(i * 7u + i / 13u) % n]);
}

/* min/max targets' initial values: a quiet NaN, a negative NaN, an infinity and
 * a zero, so the device meets a NaN already in memory on both signs. */
static const uint32_t mn_init_bits[NMM] = { 0x7fc00000u, 0xffc00000u, 0x7f800000u, 0x00000000u };
static const uint32_t mx_init_bits[NMM] = { 0xffc00000u, 0x7fc00000u, 0xff800000u, 0x80000000u };

static bool
same_value(float a, float b)
{
   if (isnan(a) || isnan(b))
      return isnan(a) && isnan(b);
   return a == b;          /* +0 and -0 are both acceptable for each other */
}

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

static unsigned fails;

static void
fail(const char *what, uint32_t i, float got, float want)
{
   if (fails < 12)
      fprintf(stderr, "  %s [%u]: got %a (0x%08x), want %a\n",
              what, i, got, ubits(got), want);
   fails++;
}

/* Every add to one target returns init + j * a for a distinct j in [0, n). */
static void
check_add_chain(const char *what, const float *ret, uint32_t stride,
                uint32_t first, uint32_t n, float init, float a, float final)
{
   bool *seen = calloc(n, 1);
   for (uint32_t j = 0; j < n; j++) {
      const uint32_t idx = first + j * stride;
      const float r = ret[idx];
      const float q = (r - init) / a;
      const int64_t s = (int64_t)q;
      if (isnan(r) || q != (float)s || s < 0 || s >= (int64_t)n ||
          init + (float)s * a != r || seen[s]) {
         fail(what, idx, r, init);
         continue;
      }
      seen[s] = true;
   }
   free(seen);
   if (final != init + (float)n * a)
      fail(what, UINT32_MAX, final, init + (float)n * a);
}

/* A min (is_max = false) or max target: its final value, and the values its
 * atomics returned. */
static void
check_minmax(const char *what, bool is_max, float init, const float *ops,
             const float *ret, uint32_t stride, uint32_t first, uint32_t n,
             float final)
{
   float want = init;
   for (uint32_t j = 0; j < n; j++) {
      const float v = ops[first + j * stride];
      if (isnan(v))
         continue;
      if (isnan(want) || (is_max ? v > want : v < want))
         want = v;
   }
   if (!same_value(final, want))
      fail(what, UINT32_MAX, final, want);
   if (!ret)
      return;
   for (uint32_t j = 0; j < n; j++) {
      const uint32_t idx = first + j * stride;
      const float r = ret[idx];
      bool known = ubits(r) == ubits(init);
      for (uint32_t l = 0; l < n && !known; l++)
         known = ubits(r) == ubits(ops[first + l * stride]) && !isnan(r);
      const bool beyond = !isnan(r) && (is_max ? r > final : r < final);
      if (!known || beyond || (isnan(r) && !isnan(init)))
         fail(what, idx, r, init);
   }
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

static int
make_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize size, VkBufferUsageFlags usage, bool bda,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateFlagsInfo mafi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = bda ? &mafi : NULL,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
   return 0;
}

int
main(int argc, char **argv)
{
   const char *spv_path = (argc > 1) ? argv[1] : "atomic_float.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-atomic-float",
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

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
   };
   VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .pNext = &af2,
   };
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &af,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   /* Float add and min/max need a device that runs a compare-and-swap as one
    * AMO; without one they are not offered, and that is an honest skip. */
   if (!af.shaderBufferFloat32Atomics || !af.shaderBufferFloat32AtomicAdd ||
       !af.shaderSharedFloat32Atomics || !af.shaderSharedFloat32AtomicAdd ||
       !af.shaderImageFloat32Atomics || !af.shaderImageFloat32AtomicAdd ||
       !af2.shaderBufferFloat32AtomicMinMax || !af2.shaderSharedFloat32AtomicMinMax ||
       !af2.shaderImageFloat32AtomicMinMax || !v12.bufferDeviceAddress) {
      printf("NotSupported: a float-atomic or buffer-device-address feature is not "
             "offered by this device\n");
      return 0;
   }
   VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
      .shaderBufferFloat32AtomicMinMax = VK_TRUE,
      .shaderSharedFloat32AtomicMinMax = VK_TRUE,
      .shaderImageFloat32AtomicMinMax = VK_TRUE,
   };
   VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
      .pNext = &af2_en,
      .shaderBufferFloat32Atomics = VK_TRUE, .shaderBufferFloat32AtomicAdd = VK_TRUE,
      .shaderSharedFloat32Atomics = VK_TRUE, .shaderSharedFloat32AtomicAdd = VK_TRUE,
      .shaderImageFloat32Atomics = VK_TRUE, .shaderImageFloat32AtomicAdd = VK_TRUE,
   };
   VkPhysicalDeviceVulkan12Features v12_en = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &af_en, .bufferDeviceAddress = VK_TRUE,
   };
   const char *exts[] = { "VK_EXT_shader_atomic_float", "VK_EXT_shader_atomic_float2" };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &v12_en,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* --- buffers: the accumulators, the BDA target, the image staging ---- */
   VkBuffer abuf, rbuf, sbuf;
   VkDeviceMemory amem, rmem, smem;
   const VkDeviceSize ibytes = 3u * NSUM * sizeof(float);
   if (make_buffer(dev, &mp, sizeof(struct acc), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   false, &abuf, &amem) ||
       make_buffer(dev, &mp, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true, &rbuf, &rmem) ||
       make_buffer(dev, &mp, ibytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &sbuf, &smem))
      return 1;

   struct acc *a;
   CHECK(vkMapMemory(dev, amem, 0, sizeof(struct acc), 0, (void **)&a));
   for (uint32_t w = 0; w < sizeof(struct acc) / 4u; w++)
      ((uint32_t *)a)[w] = SENTINEL;
   for (uint32_t k = 0; k < NSUM; k++)
      a->sum[k] = sum_init[k];
   for (uint32_t m = 0; m < NMM; m++) {
      a->mn[m] = fbits(mn_init_bits[m]);
      a->mx[m] = fbits(mx_init_bits[m]);
   }
   a->xchg = XCHG_INIT;
   a->zmin = 0.0f;
   a->zmax = -0.0f;
   for (uint32_t i = 0; i < TOTAL; i++)
      a->mm_op[i] = mm_operand(i);
   vkUnmapMemory(dev, amem);

   float *rp;
   CHECK(vkMapMemory(dev, rmem, 0, 64, 0, (void **)&rp));
   rp[0] = BDA_INIT;
   rp[1] = fbits(NAN_FOR_MIN);
   rp[2] = fbits(NAN_FOR_MAX);
   vkUnmapMemory(dev, rmem);

   float *sp;
   CHECK(vkMapMemory(dev, smem, 0, ibytes, 0, (void **)&sp));
   for (uint32_t k = 0; k < NSUM; k++) {
      sp[k] = sum_init[k];
      sp[NSUM + k] = fbits(mn_init_bits[k]);
      sp[2 * NSUM + k] = fbits(mx_init_bits[k]);
   }
   vkUnmapMemory(dev, smem);

   VkBufferDeviceAddressInfo bdai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = rbuf,
   };
   const VkDeviceAddress raddr = vkGetBufferDeviceAddress(dev, &bdai);

   /* --- the storage image ------------------------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_SFLOAT,
      .extent = { NSUM, 3, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   uint32_t imt = find_mem(&mp, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (imt == UINT32_MAX)
      imt = find_mem(&mp, imr.memoryTypeBits, 0);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_SFLOAT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   /* --- pipeline ---------------------------------------------------- */
   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0,
      .size = sizeof(VkDeviceAddress),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* One workgroup per CTA: the largest power of two the device allows. */
   uint32_t local = LOCAL_MAX;
   while (local > props.limits.maxComputeWorkGroupSize[0] ||
          local > props.limits.maxComputeWorkGroupInvocations)
      local >>= 1;
   const uint32_t groups = TOTAL / local;
   printf("local_size_x=%u, %u workgroups\n", local, groups);

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
   VkSpecializationMapEntry sme = { .constantID = 0, .offset = 0, .size = 4 };
   VkSpecializationInfo spec = {
      .mapEntryCount = 1, .pMapEntries = &sme, .dataSize = 4, .pData = &local,
   };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1 },
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
   VkDescriptorBufferInfo dbi = { .buffer = abuf, .offset = 0, .range = VK_WHOLE_SIZE };
   VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii },
   };
   vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);

   /* --- seed the image, dispatch, read the image back --------------- */
   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
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
   VkImageMemoryBarrier tob = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tob);
   VkBufferImageCopy region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { NSUM, 3, 1 },
   };
   vkCmdCopyBufferToImage(cb, sbuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
   VkImageMemoryBarrier togen = tob;
   togen.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   togen.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
   togen.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   togen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &togen);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(raddr), &raddr);
   vkCmdDispatch(cb, groups, 1, 1);
   VkImageMemoryBarrier tosrc = tob;
   tosrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   tosrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   tosrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
   tosrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tosrc);
   vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sbuf, 1, &region);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- verify -------------------------------------------------------- */
   CHECK(vkMapMemory(dev, amem, 0, sizeof(struct acc), 0, (void **)&a));
   CHECK(vkMapMemory(dev, rmem, 0, 64, 0, (void **)&rp));
   CHECK(vkMapMemory(dev, smem, 0, ibytes, 0, (void **)&sp));

   unsigned sentinels = 0;
   for (uint32_t w = 0; w < offsetof(struct acc, grp_sum) / 4u; w++)
      sentinels += ((uint32_t *)a)[w] == SENTINEL;

   const uint32_t per = TOTAL / NSUM;
   for (uint32_t k = 0; k < NSUM; k++) {
      check_add_chain("ssbo add", a->ret_sum, NSUM, k, per, sum_init[k], addend[k], a->sum[k]);
      check_add_chain("image add", a->ret_isum, NSUM, k, per, sum_init[k], addend[k], sp[k]);
   }
   check_add_chain("bda add", a->ret_bsum, 1, 0, TOTAL, BDA_INIT, BDA_ADDEND, rp[0]);
   for (uint32_t g = 0; g < groups; g++)
      check_add_chain("shared add", a->ret_ssum, 1, g * local, local,
                      SH_SUM_INIT, SH_ADDEND, a->grp_sum[g]);

   const uint32_t perm = TOTAL / NMM;
   for (uint32_t m = 0; m < NMM; m++) {
      check_minmax("ssbo min", false, fbits(mn_init_bits[m]), a->mm_op, a->ret_min,
                   NMM, m, perm, a->mn[m]);
      check_minmax("ssbo max", true, fbits(mx_init_bits[m]), a->mm_op, a->ret_max,
                   NMM, m, perm, a->mx[m]);
      check_minmax("image min", false, fbits(mn_init_bits[m]), a->mm_op, a->ret_imin,
                   NMM, m, perm, sp[NSUM + m]);
      check_minmax("image max", true, fbits(mx_init_bits[m]), a->mm_op, a->ret_imax,
                   NMM, m, perm, sp[2 * NSUM + m]);
   }
   check_minmax("bda min", false, fbits(NAN_FOR_MIN), a->mm_op, a->ret_bmin,
                1, 0, TOTAL, rp[1]);
   check_minmax("bda max", true, fbits(NAN_FOR_MAX), a->mm_op, a->ret_bmax,
                1, 0, TOTAL, rp[2]);
   for (uint32_t g = 0; g < groups; g++) {
      check_minmax("shared min", false, 1.0e30f, a->mm_op, NULL, 1, g * local, local,
                   a->grp_min[g]);
      check_minmax("shared max", true, -1.0e30f, a->mm_op, NULL, 1, g * local, local,
                   a->grp_max[g]);
      check_minmax("shared NaN min", false, fbits(NAN_FOR_MIN), a->mm_op, NULL, 1,
                   g * local, local, a->grp_nmin[g]);
      check_minmax("shared NaN max", true, fbits(NAN_FOR_MAX), a->mm_op, NULL, 1,
                   g * local, local, a->grp_nmax[g]);
   }
   /* Signed zeros, which check_minmax treats as equal: the device orders
    * -0 < +0, so these two finals are exact. */
   if (ubits(a->zmax) != 0x00000000u)
      fail("ssbo max(-0, +0)", 0, a->zmax, 0.0f);
   if (ubits(a->zmin) != 0x80000000u)
      fail("ssbo min(+0, -0)", 0, a->zmin, -0.0f);

   /* exchange: {returned} + {final} == {init} + {operands}, each once */
   {
      bool *seen = calloc(TOTAL + 1, 1);
      for (uint32_t i = 0; i <= TOTAL; i++) {
         const float r = i < TOTAL ? a->ret_xchg[i] : a->xchg;
         int64_t s = r == XCHG_INIT ? (int64_t)TOTAL : (int64_t)(r - 0.5f);
         if (s < 0 || s > (int64_t)TOTAL || (s < (int64_t)TOTAL && (float)s + 0.5f != r) ||
             seen[s]) {
            fail("ssbo exchange", i, r, XCHG_INIT);
            continue;
         }
         seen[s] = true;
      }
      free(seen);
   }

   vkUnmapMemory(dev, smem);
   vkUnmapMemory(dev, rmem);
   vkUnmapMemory(dev, amem);

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyBuffer(dev, abuf, NULL);
   vkFreeMemory(dev, amem, NULL);
   vkDestroyBuffer(dev, rbuf, NULL);
   vkFreeMemory(dev, rmem, NULL);
   vkDestroyBuffer(dev, sbuf, NULL);
   vkFreeMemory(dev, smem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (sentinels)
      fprintf(stderr, "  %u result slots still hold the sentinel: the dispatch "
              "did not run on the device\n", sentinels);
   if (fails || sentinels) {
      printf("FAILED (%u mismatches)\n", fails);
      return 1;
   }
   printf("PASSED (%u invocations: float add/min/max/exchange on buffer, shared, "
          "image and device address)\n", TOTAL);
   return 0;
}
