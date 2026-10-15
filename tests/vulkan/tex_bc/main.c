/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Block-compressed textures for the vortexpipe driver.
 *
 * Each case uploads an 8x4 texture of one BC format -- two 4x4 blocks side by
 * side -- and fetches every texel into an RGBA32F target. The texture stays
 * compressed in device memory and the sampler decodes the texel a fetch lands
 * on, so each texel checks both the decode and which block (and which texel of
 * it) the address reaches. The two blocks of a case differ, typically in the
 * order of their endpoints, so a fetch that read the neighbouring block or the
 * mirrored texel lands on a different value.
 *
 *   bc1, bc1a        RGB565 endpoints, the four-colour ramp, and (bc1a's
 *                    second block) the three-colour mode's transparent black
 *   bc3              BC1's colour with the eight- and six-step alpha ramps
 *   bc4, bc4sn, bc5  the red (and green) ramps, unsigned and snorm8
 *   bc6uf            mode 10: one region, 10-bit endpoints, half-float out
 *   bc7, bc7_srgb    mode 6: 7-bit endpoints with a p-bit, 4-bit indices
 *
 * The expected texels come from the format definitions directly, not from the
 * driver's decoder: each block here uses only the few features it names.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   8u
#define HEIGHT  4u
#define FB_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT

/* Normalized values are exact quotients; float rounding and the sRGB curve
 * are worth a few ULPs. */
#define TOL 1e-4f

enum bckind { BC1, BC1A, BC3, BC4, BC4SN, BC5, BC6UF, BC7, BC7_SRGB };

struct bccase {
   const char *name;
   VkFormat    fmt;
   enum bckind kind;
   uint32_t    block_bytes;
};

static const struct bccase CASES[] = {
   { "bc1",      VK_FORMAT_BC1_RGB_UNORM_BLOCK,  BC1,       8 },
   { "bc1a",     VK_FORMAT_BC1_RGBA_UNORM_BLOCK, BC1A,      8 },
   { "bc3",      VK_FORMAT_BC3_UNORM_BLOCK,      BC3,      16 },
   { "bc4",      VK_FORMAT_BC4_UNORM_BLOCK,      BC4,       8 },
   { "bc4sn",    VK_FORMAT_BC4_SNORM_BLOCK,      BC4SN,     8 },
   { "bc5",      VK_FORMAT_BC5_UNORM_BLOCK,      BC5,      16 },
   { "bc6uf",    VK_FORMAT_BC6H_UFLOAT_BLOCK,    BC6UF,    16 },
   { "bc7",      VK_FORMAT_BC7_UNORM_BLOCK,      BC7,      16 },
   { "bc7_srgb", VK_FORMAT_BC7_SRGB_BLOCK,       BC7_SRGB, 16 },
};
#define NUM_CASES (sizeof(CASES) / sizeof(CASES[0]))

/* A little-endian bit writer over one block. */
static void
put_bits(uint8_t *blk, uint32_t pos, uint32_t n, uint32_t v)
{
   for (uint32_t i = 0; i < n; i++, pos++) {
      if ((v >> i) & 1u)
         blk[pos >> 3] |= (uint8_t)(1u << (pos & 7u));
   }
}

/* The index of texel t (row-major in its block) in block b. Texel 0 stays below
 * half the range: as the first texel of its subset its index has one bit fewer. */
static uint32_t
index_of(uint32_t b, uint32_t t, uint32_t bits)
{
   const uint32_t n = 1u << bits;
   return b ? (t * 7u) % n : t % n;
}

/* ---- BC1/BC3 colour ------------------------------------------------- */

static uint32_t expand5(uint32_t v) { return (v << 3) | (v >> 2); }
static uint32_t expand6(uint32_t v) { return (v << 2) | (v >> 4); }

static void
bc1_colour(const uint16_t c[2], uint32_t i, bool three, bool punch, float out[4])
{
   uint32_t e[2][3];
   for (int k = 0; k < 2; k++) {
      e[k][0] = expand5(c[k] >> 11);
      e[k][1] = expand6((c[k] >> 5) & 0x3f);
      e[k][2] = expand5(c[k] & 0x1f);
   }
   out[3] = 1.0f;
   for (int ch = 0; ch < 3; ch++) {
      uint32_t v;
      if (i < 2)          v = e[i][ch];
      else if (!three)    v = (i == 2) ? (2 * e[0][ch] + e[1][ch]) / 3 : (e[0][ch] + 2 * e[1][ch]) / 3;
      else if (i == 2)    v = (e[0][ch] + e[1][ch]) / 2;
      else                v = 0;
      out[ch] = (float)v / 255.0f;
   }
   if (three && i == 3 && punch)
      out[3] = 0.0f;
}

static void
bc1_block(uint8_t *blk, uint16_t c0, uint16_t c1, uint32_t b)
{
   memcpy(blk, &c0, 2);
   memcpy(blk + 2, &c1, 2);
   for (uint32_t t = 0; t < 16; t++)
      put_bits(blk + 4, 2 * t, 2, index_of(b, t, 2));
}

/* ---- BC3 alpha / BC4 / BC5 ramps ---------------------------------- */

static void
ramp_block(uint8_t *blk, uint8_t e0, uint8_t e1, uint32_t b)
{
   blk[0] = e0;
   blk[1] = e1;
   for (uint32_t t = 0; t < 16; t++)
      put_bits(blk, 16 + 3 * t, 3, index_of(b, t, 3));
}

/* BC3's alpha in integers. */
static float
alpha_ramp(uint32_t a0, uint32_t a1, uint32_t i)
{
   uint32_t v;
   if (i < 2)          v = i ? a1 : a0;
   else if (a0 > a1)   v = (a0 * (8 - i) + a1 * (i - 1)) / 7;
   else if (i == 6)    v = 0;
   else if (i == 7)    v = 255;
   else                v = (a0 * (6 - i) + a1 * (i - 1)) / 5;
   return (float)v / 255.0f;
}

/* BC4/BC5 in float, the endpoints unorm8 or snorm8. */
static float
red_ramp(uint8_t b0, uint8_t b1, bool sn, uint32_t i)
{
   const float r0 = sn ? fmaxf((float)(int8_t)b0 / 127.0f, -1.0f) : (float)b0 / 255.0f;
   const float r1 = sn ? fmaxf((float)(int8_t)b1 / 127.0f, -1.0f) : (float)b1 / 255.0f;
   if (i < 2)
      return i ? r1 : r0;
   if (r0 > r1)
      return (r0 * (float)(8 - i) + r1 * (float)(i - 1)) / 7.0f;
   if (i == 6)
      return sn ? -1.0f : 0.0f;
   if (i == 7)
      return 1.0f;
   return (r0 * (float)(6 - i) + r1 * (float)(i - 1)) / 5.0f;
}

/* ---- BC6H / BC7 shared interpolation ------------------------------ */

static const uint32_t W4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

static int32_t
lerp64(int32_t a, int32_t b, uint32_t i)
{
   return ((64 - (int32_t)W4[i]) * a + (int32_t)W4[i] * b + 32) >> 6;
}

/* Mode 6: endpoints e[2][4] of 7 bits, p-bits p[2], 4-bit indices. */
static void
bc7_block(uint8_t *blk, const uint32_t e[2][4], const uint32_t p[2], uint32_t b)
{
   memset(blk, 0, 16);
   put_bits(blk, 0, 7, 1u << 6);
   uint32_t pos = 7;
   for (int ch = 0; ch < 4; ch++)
      for (int k = 0; k < 2; k++, pos += 7)
         put_bits(blk, pos, 7, e[k][ch]);
   put_bits(blk, pos++, 1, p[0]);
   put_bits(blk, pos++, 1, p[1]);
   for (uint32_t t = 0; t < 16; t++) {
      const uint32_t n = t ? 4u : 3u;
      put_bits(blk, pos, n, index_of(b, t, 3) * (t ? 2u : 1u) % 16u);
      pos += n;
   }
}

static uint32_t
bc7_index(uint32_t b, uint32_t t)
{
   return index_of(b, t, 3) * (t ? 2u : 1u) % 16u;
}

static float
srgb_to_linear(float c)
{
   return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

/* Mode 10: one region, unsigned 10-bit endpoints r,g,b of e[2][3]. */
static void
bc6_block(uint8_t *blk, const uint32_t e[2][3], uint32_t b)
{
   memset(blk, 0, 16);
   put_bits(blk, 0, 5, 0x03);
   uint32_t pos = 5;
   for (int k = 0; k < 2; k++)
      for (int ch = 0; ch < 3; ch++, pos += 10)
         put_bits(blk, pos, 10, e[k][ch]);
   for (uint32_t t = 0; t < 16; t++) {
      const uint32_t n = t ? 4u : 3u;
      put_bits(blk, pos, n, bc7_index(b, t));
      pos += n;
   }
}

static float
half_to_float(uint16_t h)
{
   const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
   const float v = e ? ldexpf((float)(m | 0x400), (int)e - 25) : ldexpf((float)m, -24);
   return (h & 0x8000) ? -v : v;
}

static float
bc6_channel(uint32_t a, uint32_t b, uint32_t i)
{
   const int32_t u[2] = {
      a == 0 ? 0 : (a == 1023 ? 0xffff : (int32_t)(((a << 15) + 0x4000) >> 9)),
      b == 0 ? 0 : (b == 1023 ? 0xffff : (int32_t)(((b << 15) + 0x4000) >> 9)),
   };
   return half_to_float((uint16_t)((lerp64(u[0], u[1], i) * 31) / 64));
}

/* The two blocks of a case and every texel's expected value. */
static void
build_case(const struct bccase *c, uint8_t blocks[2][16], float expect[][4])
{
   memset(blocks, 0, 2 * 16);
   static const uint16_t c1[2][2] = { { 0xF800, 0x001F }, { 0x07E0, 0x0000 } };
   static const uint16_t c1a[2][2] = { { 0xF800, 0x001F }, { 0x0000, 0x07E0 } };
   static const uint8_t ramp[2][2] = { { 255, 0 }, { 0, 255 } };
   static const uint8_t ramp_sn[2][2] = { { 0x7f, 0x80 }, { 0x80, 0x7f } };
   static const uint32_t e7[2][2][4] = { { { 127, 0, 0, 127 }, { 0, 127, 0, 127 } },
                                         { { 10, 20, 30, 127 }, { 100, 90, 80, 64 } } };
   static const uint32_t p7[2][2] = { { 1, 0 }, { 0, 1 } };
   static const uint32_t e6[2][2][3] = { { { 0, 0, 0 }, { 1023, 512, 0 } },
                                         { { 1023, 1023, 1023 }, { 0, 256, 100 } } };
   for (uint32_t b = 0; b < 2; b++) {
      uint8_t *blk = blocks[b];
      switch (c->kind) {
      case BC1:   bc1_block(blk, c1[b][0], c1[b][1], b); break;
      case BC1A:  bc1_block(blk, c1a[b][0], c1a[b][1], b); break;
      case BC3:   ramp_block(blk, ramp[b][0], ramp[b][1], b);
                  bc1_block(blk + 8, c1[b][0], c1[b][1], b); break;
      case BC4:   ramp_block(blk, ramp[b][0], ramp[b][1], b); break;
      case BC4SN: ramp_block(blk, ramp_sn[b][0], ramp_sn[b][1], b); break;
      case BC5:   ramp_block(blk, ramp[b][0], ramp[b][1], b);
                  ramp_block(blk + 8, ramp[!b][0], ramp[!b][1], b); break;
      case BC6UF: bc6_block(blk, e6[b], b); break;
      case BC7:
      case BC7_SRGB: bc7_block(blk, e7[b], p7[b], b); break;
      }
   }
   for (uint32_t y = 0; y < HEIGHT; y++) {
      for (uint32_t x = 0; x < WIDTH; x++) {
         const uint32_t b = x / 4, t = y * 4 + (x % 4);
         float *o = expect[y * WIDTH + x];
         o[0] = o[1] = o[2] = 0.0f;
         o[3] = 1.0f;
         switch (c->kind) {
         case BC1:
            bc1_colour(c1[b], index_of(b, t, 2), c1[b][0] <= c1[b][1], false, o);
            break;
         case BC1A:
            bc1_colour(c1a[b], index_of(b, t, 2), c1a[b][0] <= c1a[b][1], true, o);
            break;
         case BC3:
            bc1_colour(c1[b], index_of(b, t, 2), false, false, o);
            o[3] = alpha_ramp(ramp[b][0], ramp[b][1], index_of(b, t, 3));
            break;
         case BC4:
            o[0] = red_ramp(ramp[b][0], ramp[b][1], false, index_of(b, t, 3));
            break;
         case BC4SN:
            o[0] = red_ramp(ramp_sn[b][0], ramp_sn[b][1], true, index_of(b, t, 3));
            break;
         case BC5:
            o[0] = red_ramp(ramp[b][0], ramp[b][1], false, index_of(b, t, 3));
            o[1] = red_ramp(ramp[!b][0], ramp[!b][1], false, index_of(b, t, 3));
            break;
         case BC6UF:
            for (int ch = 0; ch < 3; ch++)
               o[ch] = bc6_channel(e6[b][0][ch], e6[b][1][ch], bc7_index(b, t));
            break;
         case BC7:
         case BC7_SRGB:
            for (int ch = 0; ch < 4; ch++) {
               const int32_t a = (int32_t)(((e7[b][0][ch] << 1) | p7[b][0]));
               const int32_t z = (int32_t)(((e7[b][1][ch] << 1) | p7[b][1]));
               const float v = (float)lerp64(a, z, bc7_index(b, t)) / 255.0f;
               o[ch] = (c->kind == BC7_SRGB && ch < 3) ? srgb_to_linear(v) : v;
            }
            break;
         }
      }
   }
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

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "tex_bc.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "tex_bc.frag.spv";

   /* --- instance / device / queue --------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-tex_bc",
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

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures avail;
   vkGetPhysicalDeviceFeatures(pd, &avail);
   if (!avail.textureCompressionBC) {
      printf("FAILED: textureCompressionBC is not supported\n");
      return 1;
   }
   VkPhysicalDeviceFeatures feats = { .textureCompressionBC = VK_TRUE };
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

   /* --- shared render target / pass / readback -------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FB_FORMAT,
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
   if (imt == UINT32_MAX) { fprintf(stderr, "FAILED: no image memory\n"); return 1; }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FB_FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));

   VkAttachmentDescription att = {
      .format = FB_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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

   const VkDeviceSize rb_bytes = (VkDeviceSize)WIDTH * HEIGHT * 16;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = rb_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   uint32_t bmt = find_mem(&mp, bmr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (bmt == UINT32_MAX) { fprintf(stderr, "FAILED: no host memory\n"); return 1; }
   VkMemoryAllocateInfo bmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bmr.size, .memoryTypeIndex = bmt,
   };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, rb, bmem, 0));

   /* --- shared vertex module, layout, sampler, command pool -------- */
   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) return 1;

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
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

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

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

   /* --- run every case -------------------------------------------- */
   unsigned failed = 0;
   for (uint32_t ci = 0; ci < NUM_CASES; ci++) {
      const struct bccase *c = &CASES[ci];
      uint8_t blocks[2][16];
      float expect[HEIGHT * WIDTH][4];
      build_case(c, blocks, expect);
      const VkDeviceSize tex_bytes = 2u * c->block_bytes;
      uint8_t data[32];
      memcpy(data, blocks[0], c->block_bytes);
      memcpy(data + c->block_bytes, blocks[1], c->block_bytes);

      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, c->fmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
         printf("%s: FAIL (not advertised as a sampled image format)\n", c->name);
         failed++;
         continue;
      }

      VkImageCreateInfo tci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = c->fmt,
         .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      };
      VkImage tex;
      CHECK(vkCreateImage(dev, &tci, NULL, &tex));
      VkMemoryRequirements tmr;
      vkGetImageMemoryRequirements(dev, tex, &tmr);
      uint32_t tmt = find_mem(&mp, tmr.memoryTypeBits, 0);
      if (tmt == UINT32_MAX) { fprintf(stderr, "FAILED: no texture memory\n"); return 1; }
      VkMemoryAllocateInfo tmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = tmr.size, .memoryTypeIndex = tmt,
      };
      VkDeviceMemory tmem;
      CHECK(vkAllocateMemory(dev, &tmai, NULL, &tmem));
      CHECK(vkBindImageMemory(dev, tex, tmem, 0));

      VkImageViewCreateInfo tvci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = tex, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = c->fmt,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1, .layerCount = 1,
         },
      };
      VkImageView texview;
      CHECK(vkCreateImageView(dev, &tvci, NULL, &texview));

      VkBufferCreateInfo sbci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = tex_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      };
      VkBuffer sbuf;
      CHECK(vkCreateBuffer(dev, &sbci, NULL, &sbuf));
      VkMemoryRequirements smr;
      vkGetBufferMemoryRequirements(dev, sbuf, &smr);
      uint32_t smt = find_mem(&mp, smr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (smt == UINT32_MAX) { fprintf(stderr, "FAILED: no staging memory\n"); return 1; }
      VkMemoryAllocateInfo smai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = smr.size, .memoryTypeIndex = smt,
      };
      VkDeviceMemory smem;
      CHECK(vkAllocateMemory(dev, &smai, NULL, &smem));
      CHECK(vkBindBufferMemory(dev, sbuf, smem, 0));
      void *sp;
      CHECK(vkMapMemory(dev, smem, 0, tex_bytes, 0, &sp));
      memcpy(sp, data, tex_bytes);
      vkUnmapMemory(dev, smem);

      VkDescriptorPoolSize dps = {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
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
      VkDescriptorImageInfo dii = {
         .sampler = sampler, .imageView = texview,
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      VkWriteDescriptorSet wds = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = dset, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &dii,
      };
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);


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

      /* record: upload texture, render, copy out */
      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));

      VkImageMemoryBarrier to_dst = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = tex,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &to_dst);

      VkBufferImageCopy tcopy = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyBufferToImage(cmd, sbuf, tex,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tcopy);

      VkImageMemoryBarrier to_read = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = tex,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                           0, NULL, 0, NULL, 1, &to_read);

      VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 1, .pClearValues = &clear,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl,
                              0, 1, &dset, 0, NULL);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);

      VkBufferImageCopy region = {
         .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .layerCount = 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rb, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));

      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));

      float *px;
      CHECK(vkMapMemory(dev, bmem, 0, rb_bytes, 0, (void **)&px));
      unsigned bad = 0;
      for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
         for (uint32_t ch = 0; ch < 4; ch++) {
            const float got = px[i * 4 + ch], want = expect[i][ch];
            if (!(fabsf(got - want) <= TOL * fmaxf(1.0f, fabsf(want)))) {
               if (bad++ < 4)
                  printf("  %s texel (%u,%u) channel %u: %g, expected %g\n", c->name,
                         i % WIDTH, i / WIDTH, ch, got, want);
            }
         }
      }
      vkUnmapMemory(dev, bmem);
      printf("%s: %s (%u of %u channels wrong)\n", c->name, bad ? "FAIL" : "pass", bad,
             WIDTH * HEIGHT * 4);
      if (bad) failed++;

      vkDestroyPipeline(dev, pipe, NULL);
      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkFreeMemory(dev, smem, NULL);
      vkDestroyBuffer(dev, sbuf, NULL);
      /* Each case releases its texture before the next one allocates, so a
       * later case's storage can land on a freed address -- the aliasing a
       * descriptor-to-resource match has to stay correct across. */
      vkDestroyImageView(dev, texview, NULL);
      vkFreeMemory(dev, tmem, NULL);
      vkDestroyImage(dev, tex, NULL);
   }

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u case(s))\n", failed);
      return 1;
   }
   printf("PASSED (block-compressed: every texel of every format decodes)\n");
   return 0;
}
