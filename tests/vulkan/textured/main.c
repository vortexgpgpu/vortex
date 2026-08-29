/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Texture-format matrix for the vortexpipe driver.
 *
 * Renders a full-screen quad sampling a 2x2 texture into a 64x64
 * R8G8B8A8_UNORM image, once per texture format, and checks each
 * quadrant against the texel the test itself encoded. The quad covers
 * the frame and the texture coordinate is interpolated across it, so
 * texel (x,y) owns quadrant (x,y) of the output.
 *
 * Three phases:
 *
 *   A  what the driver advertises. vkGetPhysicalDeviceFormatProperties
 *      over the candidate list, asserted against what the device can
 *      actually do. This is the only direction a conforming app can
 *      check: an app cannot discover a format that works but is not
 *      advertised, so a format missing from the advertisement is
 *      invisible to everything except an assertion like this one.
 *
 *   B  sampling. Every advertised colour format is uploaded, sampled
 *      and compared. A format the driver over-advertises fails here
 *      instead of producing wrong pixels.
 *
 *   C  depth formats bound as sampled images. Only the red channel is
 *      compared. Green and blue are the view swizzle's zeros and carry
 *      nothing; the depth itself is quantised to 8 bits on the way
 *      through the sampler, which an 8-bit colour target cannot see, so
 *      this phase is not evidence about depth-sample precision.
 *
 * Expected values are computed here from the source texel bytes -- the
 * encode and the decode are both test-side -- so the check cannot be
 * satisfied by the implementation agreeing with itself. Comparisons
 * carry a one-LSB slack: the output channel is 8-bit unorm, and f16
 * cannot represent k/255 exactly.
 *
 * Integer texture formats (R8G8B8A8_UINT/SINT) need integer-returning
 * samplers, which tests/vulkan/tex_fetch already covers; they appear
 * here in phase A only.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe: it exercises
 * the full vortexpipe graphics pipeline (vertex + fragment stage on
 * Vortex) plus the TEX unit -- vkCmdCopyBufferToImage feeds the
 * sampler, the fragment shader's texture() lowers to vx_tex.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH   64u
#define HEIGHT  64u
#define TEXW     2u
#define TEXH     2u
#define NTEXEL  (TEXW * TEXH)
#define FB_FORMAT  VK_FORMAT_R8G8B8A8_UNORM

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* ---- phase A: what the driver must and must not advertise -------- */

enum { ADV_NO = 0, ADV_YES = 1 };

struct advcase {
   VkFormat    fmt;
   const char *name;
   int         sampled;
   int         color;
   int         depth;
   int         storage;
};

/* The device's real capability, by usage:
 *   sampled  -- the fixed-function set, everything the host decodes to
 *               A8R8G8B8, and the native float formats
 *   color    -- what the output merger can store: the 32-bit orders it
 *               writes through, plus the narrow and sRGB formats it encodes
 *               in software
 *   depth    -- the formats the OM tests and writes
 *   storage  -- the formats the fragment translator emits image_load /
 *               image_store for
 * sRGB is absent from `sampled` on purpose: the host decode converts
 * sRGB to linear at 8-bit output precision, where Vulkan wants the
 * conversion at higher precision ahead of filtering. */
static const struct advcase adv[] = {
   /*                                        sampled  color    depth   storage */
   { VK_FORMAT_R8G8B8A8_UNORM,      "R8G8B8A8_UNORM",
                                             ADV_YES, ADV_YES, ADV_NO,  ADV_YES },
   { VK_FORMAT_B8G8R8A8_UNORM,      "B8G8R8A8_UNORM",
                                             ADV_YES, ADV_YES, ADV_NO,  ADV_NO  },
   { VK_FORMAT_R8_UNORM,            "R8_UNORM",
                                             ADV_YES, ADV_YES, ADV_NO,  ADV_NO  },
   { VK_FORMAT_R8G8_UNORM,          "R8G8_UNORM",
                                             ADV_YES, ADV_YES, ADV_NO,  ADV_NO  },
   { VK_FORMAT_R8G8B8A8_UINT,       "R8G8B8A8_UINT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_R8G8B8A8_SINT,       "R8G8B8A8_SINT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_R5G6B5_UNORM_PACK16, "R5G6B5_PACK16",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_A1R5G5B5_UNORM_PACK16, "A1R5G5B5_PACK16",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_R16_SFLOAT,          "R16_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_YES },
   { VK_FORMAT_R16G16_SFLOAT,       "R16G16_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_R32_SFLOAT,          "R32_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_YES },
   { VK_FORMAT_R32G32_SFLOAT,       "R32G32_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_YES },
   { VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_NO,  ADV_YES },
   { VK_FORMAT_D16_UNORM,           "D16_UNORM",
                                             ADV_YES, ADV_NO,  ADV_YES, ADV_NO  },
   { VK_FORMAT_D32_SFLOAT,          "D32_SFLOAT",
                                             ADV_YES, ADV_NO,  ADV_YES, ADV_NO  },
   { VK_FORMAT_D24_UNORM_S8_UINT,   "D24_UNORM_S8_UINT",
                                             ADV_YES, ADV_NO,  ADV_YES, ADV_NO  },
   /* sRGB renders but does not sample: the merger applies the transfer
    * function on device, in the direction Vulkan specifies on each side of a
    * blend, whereas the host texture decode would convert to linear at 8-bit
    * output precision ahead of filtering. */
   { VK_FORMAT_R8G8B8A8_SRGB,       "R8G8B8A8_SRGB",
                                             ADV_NO,  ADV_YES, ADV_NO,  ADV_NO  },
   { VK_FORMAT_B8G8R8A8_SRGB,       "B8G8R8A8_SRGB",
                                             ADV_NO,  ADV_YES, ADV_NO,  ADV_NO  },
   /* refused, and must stay refused */
   { VK_FORMAT_D32_SFLOAT_S8_UINT,  "D32_SFLOAT_S8_UINT",
                                             ADV_NO,  ADV_NO,  ADV_NO,  ADV_NO  },
   { VK_FORMAT_S8_UINT,             "S8_UINT",
                                             ADV_NO,  ADV_NO,  ADV_NO,  ADV_NO  },
};

static bool
phase_a(VkPhysicalDevice pd)
{
   printf("--- phase A: advertised format support ---\n");
   printf("%-22s %-9s %-9s %-9s %-9s\n",
          "format", "sampled", "color", "depth", "storage");
   unsigned bad = 0;
   for (unsigned i = 0; i < sizeof(adv) / sizeof(adv[0]); i++) {
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, adv[i].fmt, &fp);
      const VkFormatFeatureFlags o = fp.optimalTilingFeatures;
      const int got[4] = {
         (o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? 1 : 0,
         (o & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ? 1 : 0,
         (o & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? 1 : 0,
         (o & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? 1 : 0,
      };
      const int want[4] = { adv[i].sampled, adv[i].color,
                            adv[i].depth, adv[i].storage };
      char cell[4][10];
      for (unsigned k = 0; k < 4; k++) {
         if (got[k] == want[k]) {
            snprintf(cell[k], sizeof(cell[k]), "%s", got[k] ? "yes" : "-");
         } else {
            snprintf(cell[k], sizeof(cell[k]), "%s!=%s",
                     got[k] ? "yes" : "-", want[k] ? "yes" : "-");
            bad++;
         }
      }
      printf("%-22s %-9s %-9s %-9s %-9s\n", adv[i].name,
             cell[0], cell[1], cell[2], cell[3]);
   }
   if (bad) {
      printf("phase A: FAILED (%u advertisement mismatches)\n", bad);
      return false;
   }
   printf("phase A: PASSED\n");
   return true;
}

/* ---- texel encode / expected decode ----------------------------- */

enum { CH_R = 1u, CH_G = 2u, CH_B = 4u, CH_A = 8u };

struct fmtcase {
   VkFormat    fmt;
   const char *name;
   uint32_t    bpt;     /* source bytes per texel */
   bool        depth;
   uint32_t    check;   /* channels compared against the expectation */
};

static const struct fmtcase cases[] = {
   { VK_FORMAT_R8G8B8A8_UNORM,        "R8G8B8A8_UNORM",      4,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_B8G8R8A8_UNORM,        "B8G8R8A8_UNORM",      4,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R8_UNORM,              "R8_UNORM",            1,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R8G8_UNORM,            "R8G8_UNORM",          2,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R5G6B5_UNORM_PACK16,   "R5G6B5_PACK16",       2,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_A1R5G5B5_UNORM_PACK16, "A1R5G5B5_PACK16",     2,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R16_SFLOAT,            "R16_SFLOAT",          2,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R16G16_SFLOAT,         "R16G16_SFLOAT",       4,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R16G16B16A16_SFLOAT,   "R16G16B16A16_SFLOAT", 8,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R32_SFLOAT,            "R32_SFLOAT",          4,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R32G32_SFLOAT,         "R32G32_SFLOAT",       8,  false,
     CH_R | CH_G | CH_B | CH_A },
   { VK_FORMAT_R32G32B32A32_SFLOAT,   "R32G32B32A32_SFLOAT", 16, false,
     CH_R | CH_G | CH_B | CH_A },
   /* depth: red only -- see the header note on (D,D,D,1) vs (D,0,0,1) */
   { VK_FORMAT_D16_UNORM,             "D16_UNORM",           2,  true,  CH_R },
   { VK_FORMAT_D32_SFLOAT,            "D32_SFLOAT",          4,  true,  CH_R },
   { VK_FORMAT_D24_UNORM_S8_UINT,     "D24_UNORM_S8_UINT",   4,  true,  CH_R },
};

/* Source texels, as 8-bit RGBA. Three saturated corners plus a mid-tone, so a
 * decode that only gets the extremes right still fails.
 *
 * Each case rotates the pattern by its own index (`rot`). Without that every
 * case would expect the same four output colours, and a driver that sampled the
 * *previous* case's texture would pass the whole matrix -- which is exactly how
 * the resident-upload cache aliased a recycled resource pointer without any test
 * noticing. */
static const uint8_t src_rgba[NTEXEL][4] = {
   { 255,   0,   0, 255 },
   {   0, 255,   0, 255 },
   {   0,   0, 255, 255 },
   { 132,  66, 198, 255 },
};

/* Source depths, as 8-bit levels spanning the range. Rotated the same way. */
static const uint8_t src_depth[NTEXEL] = { 0, 85, 170, 255 };

/* Round-to-nearest-even f32 -> f16. Only [0,1] values are encoded here,
 * so the subnormal arm just flushes. */
static uint16_t
f32_to_f16(float f)
{
   uint32_t x;
   memcpy(&x, &f, sizeof x);
   const uint32_t sign = (x >> 16) & 0x8000u;
   const int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
   const uint32_t man  = x & 0x7fffffu;
   if (exp <= 0)
      return (uint16_t)sign;
   if (exp >= 31)
      return (uint16_t)(sign | 0x7c00u);
   uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
   const uint32_t rem = man & 0x1fffu;
   if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
      h++;
   return (uint16_t)h;
}

static float
f16_to_f32(uint16_t h)
{
   const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
   const uint32_t e    = (h >> 10) & 0x1fu;
   const uint32_t m    = h & 0x3ffu;
   uint32_t x;
   if (e == 0) {
      x = sign;                                   /* zero / flushed subnormal */
   } else if (e == 31) {
      x = sign | 0x7f800000u | (m << 13);
   } else {
      x = sign | ((e - 15 + 127) << 23) | (m << 13);
   }
   float f;
   memcpy(&f, &x, sizeof f);
   return f;
}

static uint8_t
unorm8(float f)
{
   if (f <= 0.0f) return 0u;
   if (f >= 1.0f) return 255u;
   return (uint8_t)(f * 255.0f + 0.5f);
}

/* Encode one source texel into the case's format at `dst`, and report the
 * RGBA the sampler must return for it. Both halves live here, so what the
 * test asserts is derived from the bytes it wrote, never from the driver. */
static void
encode_texel(const struct fmtcase *c, unsigned i, unsigned rot, uint8_t *dst,
             uint8_t expect[4])
{
   const unsigned t = (i + rot) % NTEXEL;
   const uint8_t *s = src_rgba[t];
   expect[0] = expect[1] = expect[2] = 0u;
   expect[3] = 255u;

   switch (c->fmt) {
   case VK_FORMAT_R8G8B8A8_UNORM:
      dst[0] = s[0]; dst[1] = s[1]; dst[2] = s[2]; dst[3] = s[3];
      memcpy(expect, s, 4);
      break;
   case VK_FORMAT_B8G8R8A8_UNORM:
      dst[0] = s[2]; dst[1] = s[1]; dst[2] = s[0]; dst[3] = s[3];
      memcpy(expect, s, 4);
      break;
   case VK_FORMAT_R8_UNORM:
      dst[0] = s[0];
      expect[0] = s[0];
      break;
   case VK_FORMAT_R8G8_UNORM:
      dst[0] = s[0]; dst[1] = s[1];
      expect[0] = s[0]; expect[1] = s[1];
      break;
   case VK_FORMAT_R5G6B5_UNORM_PACK16: {
      const uint32_t r = (uint32_t)(s[0] * 31 + 127) / 255u;
      const uint32_t g = (uint32_t)(s[1] * 63 + 127) / 255u;
      const uint32_t b = (uint32_t)(s[2] * 31 + 127) / 255u;
      const uint16_t v = (uint16_t)((r << 11) | (g << 5) | b);
      memcpy(dst, &v, 2);
      /* 5/6-bit -> 8-bit by high-bit replication, as both the host decode
       * and the device unpack do. */
      expect[0] = (uint8_t)((r << 3) | (r >> 2));
      expect[1] = (uint8_t)((g << 2) | (g >> 4));
      expect[2] = (uint8_t)((b << 3) | (b >> 2));
      break;
   }
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16: {
      const uint32_t r = (uint32_t)(s[0] * 31 + 127) / 255u;
      const uint32_t g = (uint32_t)(s[1] * 31 + 127) / 255u;
      const uint32_t b = (uint32_t)(s[2] * 31 + 127) / 255u;
      const uint32_t a = s[3] >= 128u ? 1u : 0u;
      const uint16_t v = (uint16_t)((a << 15) | (r << 10) | (g << 5) | b);
      memcpy(dst, &v, 2);
      expect[0] = (uint8_t)((r << 3) | (r >> 2));
      expect[1] = (uint8_t)((g << 3) | (g >> 2));
      expect[2] = (uint8_t)((b << 3) | (b >> 2));
      expect[3] = a ? 255u : 0u;
      break;
   }
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SFLOAT: {
      const unsigned n = c->bpt / 2u;
      for (unsigned k = 0; k < n; k++) {
         const uint16_t h = f32_to_f16((float)s[k] / 255.0f);
         memcpy(dst + k * 2u, &h, 2);
         expect[k] = unorm8(f16_to_f32(h));
      }
      if (n < 4u) expect[3] = 255u;
      break;
   }
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_SFLOAT: {
      const unsigned n = c->bpt / 4u;
      for (unsigned k = 0; k < n; k++) {
         const float f = (float)s[k] / 255.0f;
         memcpy(dst + k * 4u, &f, 4);
         expect[k] = unorm8(f);
      }
      if (n < 4u) expect[3] = 255u;
      break;
   }
   case VK_FORMAT_D16_UNORM: {
      const uint16_t d = (uint16_t)(((uint32_t)src_depth[t] * 65535u + 127u) / 255u);
      memcpy(dst, &d, 2);
      expect[0] = src_depth[t];
      break;
   }
   case VK_FORMAT_D32_SFLOAT: {
      const float f = (float)src_depth[t] / 255.0f;
      memcpy(dst, &f, 4);
      expect[0] = src_depth[t];
      break;
   }
   case VK_FORMAT_D24_UNORM_S8_UINT: {
      /* The depth aspect copies as one 32-bit word per texel with the
       * 24-bit depth in the low bits. The stencil byte is not part of
       * this copy and is left zero. */
      const uint32_t z = (uint32_t)(((uint64_t)src_depth[t] * 16777215u + 127u) / 255u);
      memcpy(dst, &z, 4);
      expect[0] = src_depth[t];
      break;
   }
   default:
      break;
   }
}

/* ---- boilerplate ------------------------------------------------ */

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

/* first memory type satisfying `want`; UINT32_MAX if none. */
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

/* Everything the per-format pass needs from the one-time setup. */
struct ctx {
   VkDevice         dev;
   VkQueue          queue;
   VkCommandBuffer  cmd;
   VkRenderPass     rp;
   VkFramebuffer    fb;
   VkPipeline       pipe;
   VkPipelineLayout pl;
   VkDescriptorSet  dset;
   VkSampler        sampler;
   VkImage          img;       /* colour attachment */
   VkBuffer         rb;        /* readback */
   VkDeviceMemory   rbmem;
   const VkPhysicalDeviceMemoryProperties *mp;
};

/* Upload the 2x2 texture in `c`'s format, render the quad, read it back and
 * compare each quadrant against the expectation `rot` selects. Returns false on
 * a mismatch or an API failure. */
static bool
run_case(struct ctx *k, const struct fmtcase *c, unsigned rot)
{
   const VkImageAspectFlags aspect = c->depth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                              : VK_IMAGE_ASPECT_COLOR_BIT;
   uint8_t expect[NTEXEL][4];
   uint8_t staging[NTEXEL * 16];
   memset(staging, 0, sizeof staging);
   for (unsigned i = 0; i < NTEXEL; i++)
      encode_texel(c, i, rot, staging + (size_t)i * c->bpt, expect[i]);

   /* --- texture image + view ------------------------------------- */
   VkImageCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = c->fmt,
      .extent = { TEXW, TEXH, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage tex;
   if (vkCreateImage(k->dev, &tci, NULL, &tex) != VK_SUCCESS) {
      printf("  %-22s FAILED (vkCreateImage)\n", c->name);
      return false;
   }
   VkMemoryRequirements tmr;
   vkGetImageMemoryRequirements(k->dev, tex, &tmr);
   VkMemoryAllocateInfo tmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = tmr.size,
      .memoryTypeIndex = find_mem(k->mp, tmr.memoryTypeBits, 0),
   };
   VkDeviceMemory tmem;
   if (vkAllocateMemory(k->dev, &tmai, NULL, &tmem) != VK_SUCCESS ||
       vkBindImageMemory(k->dev, tex, tmem, 0) != VK_SUCCESS) {
      printf("  %-22s FAILED (texture memory)\n", c->name);
      return false;
   }
   VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = tex, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = c->fmt,
      .subresourceRange = { aspect, 0, 1, 0, 1 },
   };
   VkImageView texview;
   if (vkCreateImageView(k->dev, &tvci, NULL, &texview) != VK_SUCCESS) {
      printf("  %-22s FAILED (vkCreateImageView)\n", c->name);
      return false;
   }

   /* --- staging buffer ------------------------------------------- */
   const VkDeviceSize texbytes = (VkDeviceSize)NTEXEL * c->bpt;
   VkBufferCreateInfo sbci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = texbytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
   };
   VkBuffer sbuf;
   if (vkCreateBuffer(k->dev, &sbci, NULL, &sbuf) != VK_SUCCESS) {
      printf("  %-22s FAILED (staging buffer)\n", c->name);
      return false;
   }
   VkMemoryRequirements smr;
   vkGetBufferMemoryRequirements(k->dev, sbuf, &smr);
   VkMemoryAllocateInfo smai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = smr.size,
      .memoryTypeIndex = find_mem(k->mp, smr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory smem;
   void *sp;
   if (vkAllocateMemory(k->dev, &smai, NULL, &smem) != VK_SUCCESS ||
       vkBindBufferMemory(k->dev, sbuf, smem, 0) != VK_SUCCESS ||
       vkMapMemory(k->dev, smem, 0, texbytes, 0, &sp) != VK_SUCCESS) {
      printf("  %-22s FAILED (staging memory)\n", c->name);
      return false;
   }
   memcpy(sp, staging, (size_t)texbytes);
   vkUnmapMemory(k->dev, smem);

   /* --- bind it -------------------------------------------------- */
   VkDescriptorImageInfo dii = {
      .sampler = k->sampler, .imageView = texview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = k->dset, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii,
   };
   vkUpdateDescriptorSets(k->dev, 1, &wds, 0, NULL);

   /* --- record + submit ------------------------------------------ */
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   if (vkBeginCommandBuffer(k->cmd, &cbbi) != VK_SUCCESS) {
      printf("  %-22s FAILED (vkBeginCommandBuffer)\n", c->name);
      return false;
   }

   VkImageMemoryBarrier to_dst = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = tex, .subresourceRange = { aspect, 0, 1, 0, 1 },
      .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   vkCmdPipelineBarrier(k->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_dst);

   VkBufferImageCopy tcopy = {
      .imageSubresource = { .aspectMask = aspect, .layerCount = 1 },
      .imageExtent = { TEXW, TEXH, 1 },
   };
   vkCmdCopyBufferToImage(k->cmd, sbuf, tex,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tcopy);

   VkImageMemoryBarrier to_read = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = tex, .subresourceRange = { aspect, 0, 1, 0, 1 },
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
   };
   vkCmdPipelineBarrier(k->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_read);

   VkClearValue clear = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = k->rp, .framebuffer = k->fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(k->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(k->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, k->pipe);
   vkCmdBindDescriptorSets(k->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, k->pl,
                           0, 1, &k->dset, 0, NULL);
   vkCmdDraw(k->cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(k->cmd);

   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(k->cmd, k->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          k->rb, 1, &region);
   if (vkEndCommandBuffer(k->cmd) != VK_SUCCESS) {
      printf("  %-22s FAILED (vkEndCommandBuffer)\n", c->name);
      return false;
   }

   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &k->cmd,
   };
   if (vkQueueSubmit(k->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS ||
       vkQueueWaitIdle(k->queue) != VK_SUCCESS) {
      printf("  %-22s FAILED (submit)\n", c->name);
      return false;
   }

   /* --- compare each quadrant ------------------------------------ */
   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   uint8_t *px;
   if (vkMapMemory(k->dev, k->rbmem, 0, bytes, 0, (void **)&px) != VK_SUCCESS) {
      printf("  %-22s FAILED (readback map)\n", c->name);
      return false;
   }
   /* uv (0,0) is at NDC (-1,-1), which is the top-left in Vulkan's
    * y-down framebuffer, so texel (x,y) owns quadrant (x,y). */
   static const uint32_t qx[NTEXEL] = { WIDTH / 4u, 3u * WIDTH / 4u,
                                        WIDTH / 4u, 3u * WIDTH / 4u };
   static const uint32_t qy[NTEXEL] = { HEIGHT / 4u, HEIGHT / 4u,
                                        3u * HEIGHT / 4u, 3u * HEIGHT / 4u };
   bool ok = true;
   char detail[256];
   detail[0] = '\0';
   for (unsigned i = 0; i < NTEXEL; i++) {
      const uint8_t *p = px + ((size_t)qy[i] * WIDTH + qx[i]) * 4u;
      for (unsigned ch = 0; ch < 4; ch++) {
         if (!(c->check & (1u << ch)))
            continue;
         const int diff = (int)p[ch] - (int)expect[i][ch];
         if (diff > 1 || diff < -1) {
            ok = false;
            snprintf(detail, sizeof detail,
                     "texel %u channel %u: got %u want %u",
                     i, ch, p[ch], expect[i][ch]);
            break;
         }
      }
      if (!ok) break;
   }
   if (!ok) {
      /* Dump every quadrant next to its expectation: which channels moved says
       * far more than the first mismatch alone -- a stale texture, a swapped
       * channel order and a quantisation error look identical otherwise. */
      size_t n = strlen(detail);
      for (unsigned i = 0; i < NTEXEL && n + 1 < sizeof detail; i++) {
         const uint8_t *p = px + ((size_t)qy[i] * WIDTH + qx[i]) * 4u;
         const int w = snprintf(detail + n, sizeof detail - n,
                                " | %u: %u,%u,%u,%u vs %u,%u,%u,%u",
                                i, p[0], p[1], p[2], p[3],
                                expect[i][0], expect[i][1],
                                expect[i][2], expect[i][3]);
         if (w < 0)
            break;
         n += (size_t)w;
         if (n >= sizeof detail)      /* truncated; stop rather than wrap */
            break;
      }
   }
   vkUnmapMemory(k->dev, k->rbmem);

   printf("  %-22s %s%s%s\n", c->name, ok ? "PASSED" : "FAILED",
          ok ? "" : " -- ", detail);

   vkDestroyImageView(k->dev, texview, NULL);
   vkDestroyImage(k->dev, tex, NULL);
   vkFreeMemory(k->dev, tmem, NULL);
   vkDestroyBuffer(k->dev, sbuf, NULL);
   vkFreeMemory(k->dev, smem, NULL);
   return ok;
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "textured.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "textured.frag.spv";

   /* --- instance --------------------------------------------------- */
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-textured",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   /* --- physical device + graphics queue family ------------------- */
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   const bool adv_ok = phase_a(pd);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
   free(qfp);
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   /* --- logical device + queue ------------------------------------ */
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

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   /* --- colour attachment image ----------------------------------- */
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

   /* --- render pass + framebuffer --------------------------------- */
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

   /* --- shader modules -------------------------------------------- */
   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) return 1;

   /* --- descriptor set: one combined image sampler ---------------- */
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

   /* --- graphics pipeline (vertex + fragment -> vortexpipe) ------- */
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
   /* gl_VertexIndex-driven: no vertex buffers, no attributes. */
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

   /* --- host-visible readback buffer ------------------------------ */
   const VkDeviceSize bytes = (VkDeviceSize)WIDTH * HEIGHT * 4;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

   /* --- command buffer -------------------------------------------- */
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

   /* --- phases B and C -------------------------------------------- */
   struct ctx k = {
      .dev = dev, .queue = queue, .cmd = cmd, .rp = rp, .fb = fb,
      .pipe = pipe, .pl = pl, .dset = dset, .sampler = sampler,
      .img = img, .rb = rb, .rbmem = bmem, .mp = &mp,
   };

   const unsigned ncases = (unsigned)(sizeof(cases) / sizeof(cases[0]));
   unsigned passed = 0, failed = 0, skipped = 0;
   bool phase_c_open = true;
   for (unsigned i = 0; i < ncases; i++) {
      if (cases[i].depth && phase_c_open) {
         printf("--- phase C: depth formats bound as sampled images ---\n");
         phase_c_open = false;
      } else if (!cases[i].depth && i == 0) {
         printf("--- phase B: colour format sampling ---\n");
      }
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pd, cases[i].fmt, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
         printf("  %-22s SKIPPED (not advertised as sampled)\n", cases[i].name);
         skipped++;
         continue;
      }
      if (run_case(&k, &cases[i], i)) passed++; else failed++;
   }

   /* cleanup (best-effort; a smoke test exits anyway) */
   vkDestroyCommandPool(dev, cp, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyBuffer(dev, rb, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyImage(dev, img, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (!adv_ok || failed || skipped) {
      printf("FAILED (advertisement %s, %u sampled, %u wrong, %u skipped)\n",
             adv_ok ? "ok" : "wrong", passed, failed, skipped);
      return 1;
   }
   printf("PASSED (%u formats advertised correctly, %u sampled correctly)\n",
          (unsigned)(sizeof(adv) / sizeof(adv[0])), passed);
   return 0;
}
