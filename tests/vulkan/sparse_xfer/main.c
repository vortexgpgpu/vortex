/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Transfers into and out of partially resident sparse images two or more 64 KiB
 * tiles wide, which linear addressing gets wrong.
 *
 * - bufimg: a buffer is copied into a partially bound R32_UINT 3D image, tiles
 *   in a checkerboard, and the image back into a second buffer. Every resident
 *   texel must come back, and must sit in its own tile of the bound memory.
 * - imgimg: a box crossing tile edges is copied between two RGBA8 2D sparse
 *   images bound differently -- the source fully, the destination in a
 *   checkerboard. Each resident destination texel in the box takes its source
 *   texel, and every one outside keeps the memory's fill.
 * - clear: one layer of a partially bound R32_UINT 2D array sparse image is
 *   cleared; its resident texels take the colour and the other layer's keep
 *   their copied values.
 * - blit: a scaled nearest blit from an ordinary RGBA8 image into a box of a
 *   partially bound sparse one. Only on vortexpipe: llvmpipe blits by
 *   rendering, and its rasterizer writes a sparse target linearly.
 *
 * Every check reads the bound memory through a host mapping, at the texel's
 * place in its tile. The order of texels inside a tile is the implementation's
 * own; lavapipe stores x, then y, then z.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE 65536u
#define FILL 0x11u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

struct ctx {
   VkPhysicalDevice pd;
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice        dev;
   VkQueue         queue;
   VkCommandBuffer cb;
};

/* A sparse image, its tiling, and the one memory its bound tiles share. */
struct sp_img {
   VkImage        img;
   VkFormat       format;
   VkImageType    type;
   uint32_t       w, h, d, layers, bpp;
   uint32_t       gw, gh, gd, nx, ny, nz;
   uint32_t      *slot;   /* memory tile of each tile, UINT32_MAX when unbound */
   VkDeviceMemory mem;
   uint8_t       *map;
};

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static bool
alloc_mem(struct ctx *c, VkDeviceSize bytes, uint32_t bits, VkDeviceMemory *mem)
{
   uint32_t mt = find_mem(&c->mp, bits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return false; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = bytes, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(c->dev, &mai, NULL, mem));
   return true;
}

static bool
make_buffer(struct ctx *c, VkDeviceSize bytes, VkBuffer *buf, VkDeviceMemory *mem,
            uint8_t **map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, mem)) {
      return false;
   }
   CHECK(vkBindBufferMemory(c->dev, *buf, *mem, 0));
   CHECK(vkMapMemory(c->dev, *mem, 0, bytes, 0, (void **)map));
   return true;
}

static uint32_t
seed(uint32_t salt, uint32_t i)
{
   return salt ^ (i * 2654435761u);
}

static uint32_t
tile_of(const struct sp_img *s, uint32_t x, uint32_t y, uint32_t z, uint32_t layer)
{
   return ((layer * s->nz + z / s->gd) * s->ny + y / s->gh) * s->nx + x / s->gw;
}

/* The mapped bytes of texel (x, y, z) of `layer`, or NULL when its tile is
 * unbound. */
static uint8_t *
texel_at(const struct sp_img *s, uint32_t x, uint32_t y, uint32_t z, uint32_t layer)
{
   const uint32_t m = s->slot[tile_of(s, x, y, z, layer)];
   if (m == UINT32_MAX) {
      return NULL;
   }
   const uint32_t in = (x % s->gw) + (y % s->gh) * s->gw + (z % s->gd) * s->gw * s->gh;
   return s->map + (size_t)m * TILE + (size_t)in * s->bpp;
}

/* Create a sparse image, bind the tiles `checker` selects (all when it is 2,
 * else those whose x + y + z + layer tile parity equals it), each to the next
 * tile of one memory, and fill that memory with FILL. */
static bool
make_sparse(struct ctx *c, struct sp_img *s, uint32_t checker)
{
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   uint32_t np = 1;
   VkSparseImageFormatProperties sfp;
   vkGetPhysicalDeviceSparseImageFormatProperties(c->pd, s->format, s->type,
      VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &np, &sfp);
   if (np < 1) {
      fprintf(stderr, "FAILED: no sparse residency for format %d\n", (int)s->format);
      return false;
   }
   s->gw = sfp.imageGranularity.width;
   s->gh = sfp.imageGranularity.height;
   s->gd = sfp.imageGranularity.depth;
   if (s->gw * s->gh * s->gd * s->bpp != TILE || s->gw >= s->w) {
      fprintf(stderr, "FAILED: granularity %ux%ux%u is not a tile narrower than the image\n",
              s->gw, s->gh, s->gd);
      return false;
   }
   s->nx = (s->w + s->gw - 1) / s->gw;
   s->ny = (s->h + s->gh - 1) / s->gh;
   s->nz = (s->d + s->gd - 1) / s->gd;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      .imageType = s->type, .format = s->format, .extent = { s->w, s->h, s->d },
      .mipLevels = 1, .arrayLayers = s->layers, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, &s->img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, s->img, &imr);
   const uint32_t n = s->nx * s->ny * s->nz * s->layers;
   s->slot = malloc(n * sizeof *s->slot);
   VkSparseImageMemoryBind *b = calloc(n, sizeof *b);
   uint32_t bound = 0, m = 0;
   for (uint32_t l = 0; l < s->layers; l++) {
      for (uint32_t tz = 0; tz < s->nz; tz++) {
         for (uint32_t ty = 0; ty < s->ny; ty++) {
            for (uint32_t tx = 0; tx < s->nx; tx++) {
               const bool on = checker == 2 || ((tx + ty + tz + l) & 1u) == checker;
               const uint32_t t = ((l * s->nz + tz) * s->ny + ty) * s->nx + tx;
               s->slot[t] = on ? bound++ : UINT32_MAX;
               if (!on) {
                  continue;
               }
               b[m++] = (VkSparseImageMemoryBind){
                  .subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, l },
                  .offset = { (int32_t)(tx * s->gw), (int32_t)(ty * s->gh),
                              (int32_t)(tz * s->gd) },
                  .extent = { s->w - tx * s->gw < s->gw ? s->w - tx * s->gw : s->gw,
                              s->h - ty * s->gh < s->gh ? s->h - ty * s->gh : s->gh,
                              s->d - tz * s->gd < s->gd ? s->d - tz * s->gd : s->gd },
                  .memoryOffset = (VkDeviceSize)s->slot[t] * TILE,
               };
            }
         }
      }
   }
   if (!alloc_mem(c, (VkDeviceSize)bound * TILE, imr.memoryTypeBits, &s->mem)) {
      return false;
   }
   for (uint32_t i = 0; i < m; i++) {
      b[i].memory = s->mem;
   }
   CHECK(vkMapMemory(c->dev, s->mem, 0, VK_WHOLE_SIZE, 0, (void **)&s->map));
   memset(s->map, FILL, (size_t)bound * TILE);
   VkSparseImageMemoryBindInfo ibi = { .image = s->img, .bindCount = m, .pBinds = b };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageBindCount = 1, .pImageBinds = &ibi,
   };
   VkResult r = vkQueueBindSparse(c->queue, 1, &bsi, VK_NULL_HANDLE);
   free(b);
   CHECK(r);
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

static void
free_sparse(struct ctx *c, struct sp_img *s)
{
   vkUnmapMemory(c->dev, s->mem);
   vkDestroyImage(c->dev, s->img, NULL);
   vkFreeMemory(c->dev, s->mem, NULL);
   free(s->slot);
}

static bool
begin(struct ctx *c)
{
   CHECK(vkResetCommandBuffer(c->cb, 0));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(c->cb, &cbbi));
   return true;
}

static bool
submit(struct ctx *c)
{
   CHECK(vkEndCommandBuffer(c->cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &c->cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

static void
barrier(struct ctx *c, VkImage img, uint32_t layers, VkImageLayout from, VkImageLayout to)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers },
   };
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &b);
}

static void
first_bad(char *first, size_t n, unsigned *bad, const char *what, uint32_t x, uint32_t y,
          uint32_t z, uint32_t got, uint32_t want)
{
   if (!(*bad)++) {
      snprintf(first, n, "%s (%u,%u,%u): got 0x%08x, want 0x%08x", what, x, y, z, got, want);
   }
}

/* Buffer -> partially bound 3D image -> buffer. */
static bool
run_bufimg(struct ctx *c)
{
   struct sp_img s = { .format = VK_FORMAT_R32_UINT, .type = VK_IMAGE_TYPE_3D,
                       .w = 40, .h = 34, .d = 18, .layers = 1, .bpp = 4 };
   if (!make_sparse(c, &s, 0)) {
      return false;
   }
   const uint32_t n = s.w * s.h * s.d;
   VkBuffer in, out;
   VkDeviceMemory in_mem, out_mem;
   uint32_t *ip, *op;
   if (!make_buffer(c, n * 4u, &in, &in_mem, (uint8_t **)&ip) ||
       !make_buffer(c, n * 4u, &out, &out_mem, (uint8_t **)&op)) {
      return false;
   }
   for (uint32_t i = 0; i < n; i++) {
      ip[i] = seed(0x3d0c0de5u, i);
   }
   memset(op, 0, n * 4u);
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { s.w, s.h, s.d },
   };
   if (!begin(c)) {
      return false;
   }
   barrier(c, s.img, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyBufferToImage(c->cb, in, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
   barrier(c, s.img, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyImageToBuffer(c->cb, s.img, VK_IMAGE_LAYOUT_GENERAL, out, 1, &r);
   if (!submit(c)) {
      return false;
   }
   unsigned bad = 0, res = 0;
   char first[160] = "";
   for (uint32_t z = 0; z < s.d; z++) {
      for (uint32_t y = 0; y < s.h; y++) {
         for (uint32_t x = 0; x < s.w; x++) {
            const uint32_t i = (z * s.h + y) * s.w + x;
            const uint8_t *t = texel_at(&s, x, y, z, 0);
            if (!t) {
               continue;
            }
            res++;
            uint32_t m;
            memcpy(&m, t, 4);
            if (m != ip[i]) {
               first_bad(first, sizeof first, &bad, "tile memory at", x, y, z, m, ip[i]);
            } else if (op[i] != ip[i]) {
               first_bad(first, sizeof first, &bad, "read back at", x, y, z, op[i], ip[i]);
            }
         }
      }
   }
   vkDestroyBuffer(c->dev, in, NULL);
   vkDestroyBuffer(c->dev, out, NULL);
   vkFreeMemory(c->dev, in_mem, NULL);
   vkFreeMemory(c->dev, out_mem, NULL);
   free_sparse(c, &s);
   if (bad) {
      printf("FAILED (bufimg: %u/%u resident texels wrong; first %s)\n", bad, res, first);
      return false;
   }
   printf("bufimg: %u resident texels of a %ux%ux%u checkerboard copied in and out\n",
          res, s.w, s.h, s.d);
   return true;
}

/* A box across tile edges between two sparse images bound differently. */
static bool
run_imgimg(struct ctx *c)
{
   struct sp_img src = { .format = VK_FORMAT_R8G8B8A8_UNORM, .type = VK_IMAGE_TYPE_2D,
                         .w = 200, .h = 140, .d = 1, .layers = 1, .bpp = 4 };
   struct sp_img dst = src;
   if (!make_sparse(c, &src, 2) || !make_sparse(c, &dst, 1)) {
      return false;
   }
   const uint32_t n = src.w * src.h;
   VkBuffer in;
   VkDeviceMemory in_mem;
   uint32_t *ip;
   if (!make_buffer(c, n * 4u, &in, &in_mem, (uint8_t **)&ip)) {
      return false;
   }
   for (uint32_t i = 0; i < n; i++) {
      ip[i] = seed(0x1a2b3c4du, i);
   }
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { src.w, src.h, 1 },
   };
   /* Source (12, 9), destination (101, 117), 90 x 20: both cross a tile edge. */
   const uint32_t sx = 12, sy = 9, dx = 101, dy = 117, bw = 90, bh = 20;
   VkImageCopy ic = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .srcOffset = { (int32_t)sx, (int32_t)sy, 0 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstOffset = { (int32_t)dx, (int32_t)dy, 0 },
      .extent = { bw, bh, 1 },
   };
   if (!begin(c)) {
      return false;
   }
   barrier(c, src.img, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   barrier(c, dst.img, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyBufferToImage(c->cb, in, src.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
   barrier(c, src.img, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyImage(c->cb, src.img, VK_IMAGE_LAYOUT_GENERAL, dst.img, VK_IMAGE_LAYOUT_GENERAL,
                  1, &ic);
   if (!submit(c)) {
      return false;
   }
   unsigned bad = 0, inside = 0;
   char first[160] = "";
   for (uint32_t y = 0; y < dst.h; y++) {
      for (uint32_t x = 0; x < dst.w; x++) {
         const uint8_t *t = texel_at(&dst, x, y, 0, 0);
         if (!t) {
            continue;
         }
         uint32_t got, want = FILL * 0x01010101u;
         memcpy(&got, t, 4);
         if (x >= dx && x < dx + bw && y >= dy && y < dy + bh) {
            want = ip[(y - dy + sy) * src.w + (x - dx + sx)];
            inside++;
         }
         if (got != want) {
            first_bad(first, sizeof first, &bad, "destination texel", x, y, 0, got, want);
         }
      }
   }
   vkDestroyBuffer(c->dev, in, NULL);
   vkFreeMemory(c->dev, in_mem, NULL);
   free_sparse(c, &src);
   free_sparse(c, &dst);
   if (bad) {
      printf("FAILED (imgimg: %u texels wrong; first %s)\n", bad, first);
      return false;
   }
   printf("imgimg: a %ux%u box across tile edges, %u resident destination texels in it\n",
          bw, bh, inside);
   return true;
}

/* A clear of one layer of a partially bound 2D array. */
static bool
run_clear(struct ctx *c)
{
   struct sp_img s = { .format = VK_FORMAT_R32_UINT, .type = VK_IMAGE_TYPE_2D,
                       .w = 160, .h = 136, .d = 1, .layers = 2, .bpp = 4 };
   if (!make_sparse(c, &s, 0)) {
      return false;
   }
   const uint32_t n = s.w * s.h * 2u;
   VkBuffer in;
   VkDeviceMemory in_mem;
   uint32_t *ip;
   if (!make_buffer(c, n * 4u, &in, &in_mem, (uint8_t **)&ip)) {
      return false;
   }
   for (uint32_t i = 0; i < n; i++) {
      ip[i] = seed(0xc1ea7u, i);
   }
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2 },
      .imageExtent = { s.w, s.h, 1 },
   };
   const VkClearColorValue cv = { .uint32 = { 0x5eed1234u, 0, 0, 0 } };
   const VkImageSubresourceRange cr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1 };
   if (!begin(c)) {
      return false;
   }
   barrier(c, s.img, 2, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyBufferToImage(c->cb, in, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
   barrier(c, s.img, 2, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdClearColorImage(c->cb, s.img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &cr);
   if (!submit(c)) {
      return false;
   }
   unsigned bad = 0, res = 0;
   char first[160] = "";
   for (uint32_t l = 0; l < 2; l++) {
      for (uint32_t y = 0; y < s.h; y++) {
         for (uint32_t x = 0; x < s.w; x++) {
            const uint8_t *t = texel_at(&s, x, y, 0, l);
            if (!t) {
               continue;
            }
            res++;
            uint32_t got;
            memcpy(&got, t, 4);
            const uint32_t want = l == 1 ? cv.uint32[0] : ip[(l * s.h + y) * s.w + x];
            if (got != want) {
               first_bad(first, sizeof first, &bad, "texel", x, y, l, got, want);
            }
         }
      }
   }
   vkDestroyBuffer(c->dev, in, NULL);
   vkFreeMemory(c->dev, in_mem, NULL);
   free_sparse(c, &s);
   if (bad) {
      printf("FAILED (clear: %u/%u resident texels wrong; first %s)\n", bad, res, first);
      return false;
   }
   printf("clear: layer 1 of a two-layer checkerboard cleared, %u resident texels right\n",
          res);
   return true;
}

/* A 2x nearest blit from an ordinary image into a box of a sparse one. */
static bool
run_blit(struct ctx *c)
{
   struct sp_img s = { .format = VK_FORMAT_R8G8B8A8_UNORM, .type = VK_IMAGE_TYPE_2D,
                       .w = 200, .h = 140, .d = 1, .layers = 1, .bpp = 4 };
   if (!make_sparse(c, &s, 1)) {
      return false;
   }
   const uint32_t sw = 64, sh = 32, dx = 70, dy = 100;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { sw, sh, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage src;
   CHECK(vkCreateImage(c->dev, &imci, NULL, &src));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, src, &mr);
   VkDeviceMemory src_mem;
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, &src_mem)) {
      return false;
   }
   CHECK(vkBindImageMemory(c->dev, src, src_mem, 0));
   VkBuffer in;
   VkDeviceMemory in_mem;
   uint32_t *ip;
   if (!make_buffer(c, sw * sh * 4u, &in, &in_mem, (uint8_t **)&ip)) {
      return false;
   }
   for (uint32_t i = 0; i < sw * sh; i++) {
      ip[i] = seed(0xb117u, i);
   }
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { sw, sh, 1 },
   };
   VkImageBlit ib = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .srcOffsets = { { 0, 0, 0 }, { (int32_t)sw, (int32_t)sh, 1 } },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstOffsets = { { (int32_t)dx, (int32_t)dy, 0 },
                      { (int32_t)(dx + 2 * sw), (int32_t)(dy + 2 * sh), 1 } },
   };
   if (!begin(c)) {
      return false;
   }
   barrier(c, src, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   barrier(c, s.img, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdCopyBufferToImage(c->cb, in, src, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
   barrier(c, src, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
   vkCmdBlitImage(c->cb, src, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &ib,
                  VK_FILTER_NEAREST);
   if (!submit(c)) {
      return false;
   }
   unsigned bad = 0, inside = 0;
   char first[160] = "";
   for (uint32_t y = 0; y < s.h; y++) {
      for (uint32_t x = 0; x < s.w; x++) {
         const uint8_t *t = texel_at(&s, x, y, 0, 0);
         if (!t) {
            continue;
         }
         uint32_t got, want = FILL * 0x01010101u;
         memcpy(&got, t, 4);
         if (x >= dx && x < dx + 2 * sw && y >= dy && y < dy + 2 * sh) {
            want = ip[((y - dy) / 2) * sw + (x - dx) / 2];
            inside++;
         }
         if (got != want) {
            first_bad(first, sizeof first, &bad, "destination texel", x, y, 0, got, want);
         }
      }
   }
   vkDestroyBuffer(c->dev, in, NULL);
   vkFreeMemory(c->dev, in_mem, NULL);
   vkDestroyImage(c->dev, src, NULL);
   vkFreeMemory(c->dev, src_mem, NULL);
   free_sparse(c, &s);
   if (bad) {
      printf("FAILED (blit: %u texels wrong; first %s)\n", bad, first);
      return false;
   }
   printf("blit: a 2x blit into a %ux%u box across tile edges, %u resident texels in it\n",
          2 * sw, 2 * sh, inside);
   return true;
}

int
main(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_xfer",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateInstance\n");
      return 1;
   }
   uint32_t npd = 1;
   struct ctx c;
   memset(&c, 0, sizeof c);
   if (vkEnumeratePhysicalDevices(inst, &npd, &c.pd) < 0 || npd == 0) {
      fprintf(stderr, "FAILED: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(c.pd, &props);
   printf("device: %s\n", props.deviceName);
   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(c.pd, &have);
   if (!have.sparseBinding || !have.sparseResidencyImage2D || !have.sparseResidencyImage3D) {
      printf("NotSupported: sparse residency of 2D and 3D images is not offered\n");
      return 0;
   }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs both graphics and sparse binding\n");
      return 0;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures want = {
      .sparseBinding = VK_TRUE,
      .sparseResidencyImage2D = VK_TRUE,
      .sparseResidencyImage3D = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &want,
   };
   if (vkCreateDevice(c.pd, &dci, NULL, &c.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   if (vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS ||
       (cbai.commandPool = cp, vkAllocateCommandBuffers(c.dev, &cbai, &c.cb)) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: command pool or buffer\n");
      return 1;
   }

   const bool vortex = strncmp(props.deviceName, "vortexpipe", 10) == 0;
   const unsigned failed = !run_bufimg(&c) + !run_imgimg(&c) + !run_clear(&c) +
                           (vortex && !run_blit(&c));
   if (failed) {
      printf("FAILED (sparse_xfer: %u cases)\n", failed);
      return 1;
   }

   vkDestroyCommandPool(c.dev, cp, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (sparse_xfer: buffer and image copies, a layer clear and a scaled blit "
          "into partially resident sparse images, each texel in its own tile)\n");
   return 0;
}
