/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Storage images written without a format (shaderStorageImageWriteWithoutFormat).
 *
 * The shader's image has no format qualifier, so the texel encode cannot be
 * compiled in: the store has to take it from the view at run time. That is the
 * only way to write a format GLSL has no qualifier for -- B8G8R8A8 and A8 among
 * them -- and the same path serves every other format written without one.
 *
 * Each format gets the constant (1.0, -0.6, 0.3, 0.8) in every texel of a 16x16
 * image. Every field of every texel is checked: normalized fields within one
 * step of the exact value, half floats within their precision, the 10- and
 * 11-bit unsigned floats within theirs (the negative channel clamping to 0),
 * 32-bit floats exactly. The field layouts restate the Vulkan definitions.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W      16u
#define H      16u
#define LOCAL  4u

enum kind { UNORM, SNORM, F16, F32, UF };

struct phase {
   const char *name;
   VkFormat    fmt;
   unsigned    bpp;
   enum kind   kind;
   uint8_t     off[4];
   uint8_t     width[4];   /* 0: no such channel */
};

static const struct phase PHASES[] = {
   { "bgra8_unorm",   VK_FORMAT_B8G8R8A8_UNORM,      4, UNORM, { 16, 8, 0, 24 },  { 8, 8, 8, 8 } },
   { "a8_unorm",      VK_FORMAT_A8_UNORM_KHR,        1, UNORM, { 0, 0, 0, 0 },    { 0, 0, 0, 8 } },
   { "rgba8_snorm",   VK_FORMAT_R8G8B8A8_SNORM,      4, SNORM, { 0, 8, 16, 24 },  { 8, 8, 8, 8 } },
   { "rg16_snorm",    VK_FORMAT_R16G16_SNORM,        4, SNORM, { 0, 16 },         { 16, 16 } },
   { "a2b10g10r10",   VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, UNORM, { 0, 10, 20, 30 }, { 10, 10, 10, 2 } },
   { "rgba16_sfloat", VK_FORMAT_R16G16B16A16_SFLOAT, 8, F16,   { 0, 16, 32, 48 }, { 16, 16, 16, 16 } },
   { "b10g11r11",     VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, UF, { 0, 11, 22 },    { 11, 11, 10 } },
   { "rg32_sfloat",   VK_FORMAT_R32G32_SFLOAT,       8, F32,   { 0, 32 },         { 32, 32 } },
};
#define NUM_PHASES (sizeof(PHASES) / sizeof(PHASES[0]))

static const float COLOR[4] = { 1.0f, -0.6f, 0.3f, 0.8f };

static double
half_to_double(uint32_t h)
{
   const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
   const double v = e ? ldexp((double)(m | 0x400), (int)e - 25) : ldexp((double)m, -24);
   return (h & 0x8000) ? -v : v;
}

/* An unsigned float of `w` bits: 5 exponent bits, the rest mantissa. */
static double
ufloat_to_double(uint32_t v, unsigned w)
{
   const unsigned mb = w - 5;
   const uint32_t e = v >> mb, m = v & ((1u << mb) - 1);
   return e ? ldexp((double)(m | (1u << mb)), (int)e - 15 - (int)mb)
            : ldexp((double)m, -14 - (int)mb);
}

/* One texel's fields against the constant; on a wrong one, says which in
 * `why`. */
static bool
check_texel(const struct phase *ph, const uint8_t *p, char *why, size_t n)
{
   uint64_t lo = 0;
   memcpy(&lo, p, ph->bpp);
   for (unsigned c = 0; c < 4; c++) {
      const unsigned w = ph->width[c];
      if (!w)
         continue;
      const uint64_t raw = (lo >> ph->off[c]) & ((w >= 64) ? ~0ull : ((1ull << w) - 1));
      double got, want, tol;
      switch (ph->kind) {
      case UNORM:
         got = (double)raw;
         want = fmin(fmax(COLOR[c], 0.0), 1.0) * (double)((1u << w) - 1);
         tol = 1.0;
         break;
      case SNORM:
         got = (double)(((int64_t)(raw << (64 - w))) >> (64 - w));
         want = fmin(fmax(COLOR[c], -1.0), 1.0) * (double)((1u << (w - 1)) - 1);
         tol = 1.0;
         break;
      case F16:
         got = half_to_double((uint32_t)raw);
         want = COLOR[c];
         tol = fabs(want) / 1024.0;
         break;
      case UF:
         got = ufloat_to_double((uint32_t)raw, w);
         want = fmax(COLOR[c], 0.0);
         tol = want / ((w == 11) ? 64.0 : 32.0);
         break;
      default: {
         float f;
         const uint32_t bits = (uint32_t)raw;
         memcpy(&f, &bits, 4);
         got = f;
         want = COLOR[c];
         tol = 0.0;
         break;
      }
      }
      if (fabs(got - want) > tol) {
         snprintf(why, n, "%c = %g, expected %g", "RGBA"[c], got, want);
         return false;
      }
   }
   return true;
}

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

int
main(int argc, char **argv)
{
   const char *cs_path = (argc > 1) ? argv[1] : "imgfmt.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-imgfmt",
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
   VkPhysicalDeviceFeatures avail;
   vkGetPhysicalDeviceFeatures(pd, &avail);
   if (!avail.shaderStorageImageWriteWithoutFormat) {
      printf("FAILED: shaderStorageImageWriteWithoutFormat is not supported\n");
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures feats = { .shaderStorageImageWriteWithoutFormat = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   size_t sz = 0;
   uint32_t *spv = read_spirv(cs_path, &sz);
   if (!spv) return 1;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule cs;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &cs));
   free(spv);

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs, .pName = "main" },
      .layout = pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

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

   unsigned failed = 0;
   for (unsigned i = 0; i < NUM_PHASES; i++) {
      const struct phase *ph = &PHASES[i];
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, ph->fmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
         printf("%s: FAIL (not advertised as a storage image format)\n", ph->name);
         failed++;
         continue;
      }

      VkImageCreateInfo imci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = ph->fmt,
         .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      VkImage img;
      CHECK(vkCreateImage(dev, &imci, NULL, &img));
      VkMemoryRequirements imr;
      vkGetImageMemoryRequirements(dev, img, &imr);
      VkMemoryAllocateInfo imai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = imr.size, .memoryTypeIndex = find_mem(&mp, imr.memoryTypeBits, 0),
      };
      VkDeviceMemory imem;
      CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
      CHECK(vkBindImageMemory(dev, img, imem, 0));
      VkImageViewCreateInfo ivci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ph->fmt,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

      const VkDeviceSize bytes = (VkDeviceSize)W * H * ph->bpp;
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      };
      VkBuffer rb;
      CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
      VkMemoryRequirements bmr;
      vkGetBufferMemoryRequirements(dev, rb, &bmr);
      VkMemoryAllocateInfo bmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = bmr.size,
         .memoryTypeIndex = find_mem(&mp, bmr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      VkDeviceMemory bmem;
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
      CHECK(vkBindBufferMemory(dev, rb, bmem, 0));

      VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
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
      VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
      VkWriteDescriptorSet wds = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dii,
      };
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      VkImageMemoryBarrier to_gen = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &to_gen);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
      vkCmdDispatch(cmd, W / LOCAL, H / LOCAL, 1);
      VkImageMemoryBarrier to_src = to_gen;
      to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      to_src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_src);
      VkBufferImageCopy region = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { W, H, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      uint8_t *px;
      CHECK(vkMapMemory(dev, bmem, 0, bytes, 0, (void **)&px));
      unsigned bad = 0;
      char why[96] = "";
      for (uint32_t t = 0; t < W * H; t++) {
         char w[96];
         if (!check_texel(ph, px + (size_t)t * ph->bpp, w, sizeof w)) {
            if (!bad)
               snprintf(why, sizeof why, "texel (%u,%u): %s", t % W, t / W, w);
            bad++;
         }
      }
      vkUnmapMemory(dev, bmem);
      if (bad) {
         printf("%s: FAIL (%u of %u texels wrong; first %s)\n", ph->name, bad, W * H, why);
         failed++;
      } else {
         printf("%s: pass (%u texels)\n", ph->name, W * H);
      }

      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkFreeMemory(dev, bmem, NULL);
      vkDestroyBuffer(dev, rb, NULL);
      vkDestroyImageView(dev, view, NULL);
      vkFreeMemory(dev, imem, NULL);
      vkDestroyImage(dev, img, NULL);
   }

   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, cs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u format(s))\n", failed);
      return 1;
   }
   printf("PASSED (format-less image stores: every field of every format)\n");
   return 0;
}
