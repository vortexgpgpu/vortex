/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Sampling partially resident sparse textures, and their residency codes.
 *
 * Two RGBA8 sparse textures two or more 64 KiB tiles wide: a 2D array of three
 * levels and two layers, level 0 bound in a checkerboard, level 1 fully, level
 * 2 not at all; and a single-level 3D texture bound in a checkerboard. Their
 * texels are seeded through a buffer copy. Compute dispatches then take, at
 * the texels of every level around each tile's edges and on a sparse lattice
 * over the rest, a sparse texel fetch, a sparse nearest sample at
 * the texel's centre, a sparse linear sample at its corner, whose footprint
 * spans the texels around it, and on level 0 of the array a sparse gather
 * there. Every residency code must match a host reference of the footprint --
 * resident only when every texel it reads lies on a bound tile -- and every
 * resident value must be the texel's.
 *
 * A draw samples level 0 of the array in its fragment shader the same way.
 *
 * Then the bindings change and nothing else: level 0's checkerboard is
 * inverted onto new memory, which the host filled with a second pattern, and
 * level 2 is bound to it too. The same dispatches and the draw must see the new
 * residency and the new memory's texels. The order of texels inside a tile is
 * the implementation's own; lavapipe stores x, then y, then z.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE   65536u
#define RT     32u
#define SALT_A 0x2468ace0u
#define SALT_B 0x13579bdfu

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
   VkSampler       nearest, linear;
   VkDescriptorSetLayout dsl;
   VkPipelineLayout      pl;
};

struct params {
   uint32_t e[4];
   uint32_t level;
   uint32_t count;
};

/* The texels the dispatches sample, a texel list the host fills per level. */
#define MAX_TEXELS 8192u

/* A sparse RGBA8 texture and where each of its tiles is bound in the current
 * phase: memory A or B and the tile within it, or nowhere. */
struct tex {
   VkImage        img;
   VkImageView    view;
   VkImageType    type;
   uint32_t       w, h, d, layers, levels;
   uint32_t       gw, gh, gd;
   uint32_t       ntiles;
   uint32_t      *mem_of;   /* 0: unbound, 1: memory A, 2: memory B */
   uint32_t      *slot;
   uint32_t      *salt;     /* the pattern a bound tile's texels hold */
   VkDeviceMemory mem[2];
   uint8_t       *map[2];
   uint32_t       used[2];
};

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
   if (buf) {
      *out_size = (size_t)sz;
   }
   return buf;
}

static bool
make_module(struct ctx *c, const char *path, VkShaderModule *sm)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return false;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = spv,
   };
   VkResult r = vkCreateShaderModule(c->dev, &smci, NULL, sm);
   free(spv);
   CHECK(r);
   return true;
}

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
make_buffer(struct ctx *c, VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer *buf,
            VkDeviceMemory *mem, void **map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage,
   };
   CHECK(vkCreateBuffer(c->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c->dev, *buf, &mr);
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, mem)) {
      return false;
   }
   CHECK(vkBindBufferMemory(c->dev, *buf, *mem, 0));
   CHECK(vkMapMemory(c->dev, *mem, 0, bytes, 0, map));
   return true;
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
barrier(struct ctx *c, VkImage img, VkImageLayout from, VkImageLayout to)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                            VK_REMAINING_ARRAY_LAYERS },
   };
   vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static uint32_t
seed(uint32_t salt, uint32_t level, uint32_t layer, uint32_t x, uint32_t y, uint32_t z)
{
   return salt ^ (level * 0x9e3779b9u) ^ (layer * 0x7f4a7c15u) ^ (x * 2654435761u) ^
          (y * 40503u) ^ (z * 2246822519u);
}

static uint32_t
lw(const struct tex *t, uint32_t l) { return t->w >> l ? t->w >> l : 1u; }
static uint32_t
lh(const struct tex *t, uint32_t l) { return t->h >> l ? t->h >> l : 1u; }
static uint32_t
ld(const struct tex *t, uint32_t l) { return t->d >> l ? t->d >> l : 1u; }
static uint32_t
tiles_x(const struct tex *t, uint32_t l) { return (lw(t, l) + t->gw - 1) / t->gw; }
static uint32_t
tiles_y(const struct tex *t, uint32_t l) { return (lh(t, l) + t->gh - 1) / t->gh; }
static uint32_t
tiles_z(const struct tex *t, uint32_t l) { return (ld(t, l) + t->gd - 1) / t->gd; }

/* Index of the tile holding (x, y, z) of `layer` of level `l`. */
static uint32_t
tile_id(const struct tex *t, uint32_t l, uint32_t layer, uint32_t x, uint32_t y, uint32_t z)
{
   uint32_t base = 0;
   for (uint32_t k = 0; k < l; k++) {
      base += tiles_x(t, k) * tiles_y(t, k) * tiles_z(t, k) * t->layers;
   }
   return base + ((layer * tiles_z(t, l) + z / t->gd) * tiles_y(t, l) + y / t->gh) *
                 tiles_x(t, l) + x / t->gw;
}

static bool
bound(const struct tex *t, uint32_t l, uint32_t layer, uint32_t x, uint32_t y, uint32_t z)
{
   return t->mem_of[tile_id(t, l, layer, x, y, z)] != 0;
}

/* The texel's value, given its tile's current binding. */
static uint32_t
texel(const struct tex *t, uint32_t l, uint32_t layer, uint32_t x, uint32_t y, uint32_t z)
{
   return seed(t->salt[tile_id(t, l, layer, x, y, z)], l, layer, x, y, z);
}

/* Bind tile (tx, ty, tz) of `layer` of level `l` to memory `m` (1 or 2, at its
 * next tile) or unbind it (0); the bind is added to `b`. */
static void
set_tile(struct tex *t, VkSparseImageMemoryBind *b, uint32_t *n, uint32_t l, uint32_t layer,
         uint32_t tx, uint32_t ty, uint32_t tz, uint32_t m, uint32_t salt)
{
   const uint32_t id = tile_id(t, l, layer, tx * t->gw, ty * t->gh, tz * t->gd);
   t->mem_of[id] = m;
   t->salt[id] = salt;
   uint32_t s = 0;
   if (m) {
      s = t->slot[id] = t->used[m - 1]++;
   }
   const uint32_t ex = lw(t, l) - tx * t->gw, ey = lh(t, l) - ty * t->gh;
   const uint32_t ez = ld(t, l) - tz * t->gd;
   b[(*n)++] = (VkSparseImageMemoryBind){
      .subresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, layer },
      .offset = { (int32_t)(tx * t->gw), (int32_t)(ty * t->gh), (int32_t)(tz * t->gd) },
      .extent = { ex < t->gw ? ex : t->gw, ey < t->gh ? ey : t->gh, ez < t->gd ? ez : t->gd },
      .memory = m ? t->mem[m - 1] : VK_NULL_HANDLE,
      .memoryOffset = (VkDeviceSize)s * TILE,
   };
}

static bool
bind(struct ctx *c, struct tex *t, VkSparseImageMemoryBind *b, uint32_t n)
{
   VkSparseImageMemoryBindInfo ibi = { .image = t->img, .bindCount = n, .pBinds = b };
   VkBindSparseInfo bsi = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .imageBindCount = 1, .pImageBinds = &ibi,
   };
   CHECK(vkQueueBindSparse(c->queue, 1, &bsi, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

/* Phase 0: level 0 in a checkerboard and level 1 fully on memory A, the rest
 * unbound. Phase 1: level 0's other squares and level 2 on memory B, which the
 * host fills with SALT_B's texels first; level 0's first squares unbound. */
static bool
bind_phase(struct ctx *c, struct tex *t, uint32_t phase)
{
   VkSparseImageMemoryBind *b = calloc(t->ntiles, sizeof *b);
   uint32_t n = 0;
   for (uint32_t l = 0; l < t->levels; l++) {
      for (uint32_t layer = 0; layer < t->layers; layer++) {
         for (uint32_t tz = 0; tz < tiles_z(t, l); tz++) {
            for (uint32_t ty = 0; ty < tiles_y(t, l); ty++) {
               for (uint32_t tx = 0; tx < tiles_x(t, l); tx++) {
                  const uint32_t parity = (tx + ty + tz + layer) & 1u;
                  if (phase == 0) {
                     if (l == 0 && parity == 0) {
                        set_tile(t, b, &n, l, layer, tx, ty, tz, 1, SALT_A);
                     } else if (l == 1) {
                        set_tile(t, b, &n, l, layer, tx, ty, tz, 1, SALT_A);
                     }
                  } else if (l == 2 || (l == 0 && parity == 1)) {
                     set_tile(t, b, &n, l, layer, tx, ty, tz, 2, SALT_B);
                  } else if (l == 0) {
                     set_tile(t, b, &n, l, layer, tx, ty, tz, 0, 0);
                  }
               }
            }
         }
      }
   }
   if (phase == 1) {
      /* The new memory's texels, at their places in their tiles. */
      for (uint32_t l = 0; l < t->levels; l++) {
         for (uint32_t layer = 0; layer < t->layers; layer++) {
            for (uint32_t z = 0; z < ld(t, l); z++) {
               for (uint32_t y = 0; y < lh(t, l); y++) {
                  for (uint32_t x = 0; x < lw(t, l); x++) {
                     const uint32_t id = tile_id(t, l, layer, x, y, z);
                     if (t->mem_of[id] != 2) {
                        continue;
                     }
                     const uint32_t v = texel(t, l, layer, x, y, z);
                     const uint32_t in = (x % t->gw) + (y % t->gh) * t->gw +
                                         (z % t->gd) * t->gw * t->gh;
                     memcpy(t->map[1] + (size_t)t->slot[id] * TILE + in * 4u, &v, 4);
                  }
               }
            }
         }
      }
   }
   const bool ok = bind(c, t, b, n);
   free(b);
   return ok;
}

static bool
make_tex(struct ctx *c, struct tex *t)
{
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   uint32_t np = 1;
   VkSparseImageFormatProperties sfp;
   vkGetPhysicalDeviceSparseImageFormatProperties(c->pd, VK_FORMAT_R8G8B8A8_UNORM, t->type,
      VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &np, &sfp);
   if (np < 1) {
      fprintf(stderr, "FAILED: no sparse residency for RGBA8\n");
      return false;
   }
   t->gw = sfp.imageGranularity.width;
   t->gh = sfp.imageGranularity.height;
   t->gd = sfp.imageGranularity.depth;
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      .imageType = t->type, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { t->w, t->h, t->d }, .mipLevels = t->levels, .arrayLayers = t->layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, &t->img));
   uint32_t nreq = 1;
   VkSparseImageMemoryRequirements sreq;
   vkGetImageSparseMemoryRequirements(c->dev, t->img, &nreq, &sreq);
   if (nreq < 1 || sreq.imageMipTailFirstLod < t->levels) {
      fprintf(stderr, "FAILED: a mip tail, which this test does not bind\n");
      return false;
   }
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, t->img, &imr);
   t->ntiles = 0;
   for (uint32_t l = 0; l < t->levels; l++) {
      t->ntiles += tiles_x(t, l) * tiles_y(t, l) * tiles_z(t, l) * t->layers;
   }
   t->mem_of = calloc(t->ntiles, sizeof *t->mem_of);
   t->slot = calloc(t->ntiles, sizeof *t->slot);
   t->salt = calloc(t->ntiles, sizeof *t->salt);
   for (unsigned m = 0; m < 2; m++) {
      if (!alloc_mem(c, (VkDeviceSize)t->ntiles * TILE, imr.memoryTypeBits, &t->mem[m])) {
         return false;
      }
      CHECK(vkMapMemory(c->dev, t->mem[m], 0, VK_WHOLE_SIZE, 0, (void **)&t->map[m]));
   }
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
      .viewType = t->type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                                              : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, t->levels, 0, t->layers },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &t->view));
   if (!bind_phase(c, t, 0)) {
      return false;
   }

   /* Seed every level and layer through a copy; what lands on an unbound tile
    * is never checked. */
   size_t bytes = 0;
   for (uint32_t l = 0; l < t->levels; l++) {
      bytes += (size_t)lw(t, l) * lh(t, l) * ld(t, l) * t->layers * 4u;
   }
   VkBuffer buf;
   VkDeviceMemory bmem;
   uint32_t *p;
   if (!make_buffer(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &buf, &bmem, (void **)&p)) {
      return false;
   }
   VkBufferImageCopy *r = calloc(t->levels * t->layers, sizeof *r);
   size_t at = 0;
   uint32_t nr = 0;
   for (uint32_t l = 0; l < t->levels; l++) {
      for (uint32_t layer = 0; layer < t->layers; layer++) {
         r[nr++] = (VkBufferImageCopy){
            .bufferOffset = at * 4u,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, layer, 1 },
            .imageExtent = { lw(t, l), lh(t, l), ld(t, l) },
         };
         for (uint32_t z = 0; z < ld(t, l); z++) {
            for (uint32_t y = 0; y < lh(t, l); y++) {
               for (uint32_t x = 0; x < lw(t, l); x++) {
                  p[at++] = seed(SALT_A, l, layer, x, y, z);
               }
            }
         }
      }
   }
   bool ok = begin(c);
   if (ok) {
      barrier(c, t->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      vkCmdCopyBufferToImage(c->cb, buf, t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, nr, r);
      barrier(c, t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      ok = submit(c);
   }
   free(r);
   vkDestroyBuffer(c->dev, buf, NULL);
   vkFreeMemory(c->dev, bmem, NULL);
   return ok;
}

static uint32_t
clampi(int32_t v, uint32_t n)
{
   return v < 0 ? 0u : ((uint32_t)v >= n ? n - 1u : (uint32_t)v);
}

static bool
near_edge(uint32_t c, uint32_t tile)
{
   const uint32_t m = c % tile;
   return m <= 1u || m + 2u >= tile;
}

/* The texels of level `l` of `t` a dispatch samples: those beside a tile edge
 * along one axis, every fourth along the others, and a lattice of the rest. */
static uint32_t
pick_texels(const struct tex *t, uint32_t l, uint32_t *list)
{
   const bool is3d = t->type == VK_IMAGE_TYPE_3D;
   const uint32_t w = lw(t, l), h = lh(t, l), slices = is3d ? ld(t, l) : t->layers;
   uint32_t n = 0;
   for (uint32_t z = 0; z < slices; z++) {
      for (uint32_t y = 0; y < h; y++) {
         for (uint32_t x = 0; x < w; x++) {
            const bool ez = is3d && near_edge(z, t->gd);
            const bool pick = (near_edge(x, t->gw) && y % 4u == 0) ||
                              (near_edge(y, t->gh) && x % 4u == 0) ||
                              (ez && x % 4u == 0 && y % 4u == 0) ||
                              (x % 23u == 0 && y % 19u == 0);
            if (!pick || n == MAX_TEXELS) {
               continue;
            }
            list[4u * n + 0] = x;
            list[4u * n + 1] = y;
            list[4u * n + 2] = z;
            list[4u * n + 3] = 0;
            n++;
         }
      }
   }
   return n;
}

/* Check one dispatch's results for level `l` of `t` in its current binding. */
static bool
check_level(const struct tex *t, uint32_t l, uint32_t phase, const uint32_t *list,
            uint32_t count, const uint32_t *out)
{
   const bool is3d = t->type == VK_IMAGE_TYPE_3D;
   const uint32_t w = lw(t, l), h = lh(t, l), d = ld(t, l);
   unsigned bad = 0, res = 0;
   char first[200] = "";
#define BAD(...) do { if (!bad++) snprintf(first, sizeof first, __VA_ARGS__); } while (0)
   for (uint32_t n = 0; n < count; n++) {
      const uint32_t x = list[4u * n], y = list[4u * n + 1], z = list[4u * n + 2];
      const uint32_t *o = out + 8u * n;
      const uint32_t layer = is3d ? 0 : z, zz = is3d ? z : 0;
      const bool r = bound(t, l, layer, x, y, zz);
      const uint32_t v = texel(t, l, layer, x, y, zz);
      res += r;
      if (o[1] != r || o[3] != r) {
         BAD("(%u,%u,%u) fetch/nearest residency %u/%u, want %u", x, y, z, o[1], o[3], r);
         continue;
      }
      if (r && (o[0] != v || o[2] != v)) {
         BAD("(%u,%u,%u) fetch/nearest 0x%08x/0x%08x, want 0x%08x", x, y, z, o[0], o[2], v);
         continue;
      }
      const uint32_t xs[2] = { clampi((int32_t)x - 1, w), x };
      const uint32_t ys[2] = { clampi((int32_t)y - 1, h), y };
      const uint32_t zs[2] = { clampi((int32_t)zz - 1, d), zz };
      bool fp = true;
      for (uint32_t k = 0; k < (is3d ? 8u : 4u); k++) {
         fp = fp && bound(t, l, layer, xs[k & 1], ys[(k >> 1) & 1],
                          is3d ? zs[k >> 2] : 0);
      }
      if (o[4] != fp) {
         BAD("(%u,%u,%u) linear footprint residency %u, want %u", x, y, z, o[4], fp);
         continue;
      }
      if (is3d || l != 0) {
         continue;
      }
      if (o[5] != fp) {
         BAD("(%u,%u,%u) gather residency %u, want %u", x, y, z, o[5], fp);
         continue;
      }
      /* Gather order: (x0, y1), (x1, y1), (x1, y0), (x0, y0). */
      const uint32_t gx[4] = { xs[0], xs[1], xs[1], xs[0] };
      const uint32_t gy[4] = { ys[1], ys[1], ys[0], ys[0] };
      uint32_t want = 0;
      for (uint32_t k = 0; k < 4; k++) {
         want |= (texel(t, l, layer, gx[k], gy[k], 0) & 0xffu) << (8 * k);
      }
      if (fp && o[6] != want) {
         BAD("(%u,%u,%u) gather red 0x%08x, want 0x%08x", x, y, z, o[6], want);
      }
   }
#undef BAD
   if (bad) {
      printf("FAILED (%s level %u, phase %u: %u texels wrong; first %s)\n",
             is3d ? "3d" : "array", l, phase, bad, first);
      return false;
   }
   printf("%s level %u phase %u: %u texels, %u resident, codes and values right\n",
          is3d ? "3d" : "array", l, phase, count, res);
   return true;
}

static bool
run_dispatches(struct ctx *c, struct tex *t, VkPipeline pipe, VkDescriptorSet ds,
               uint32_t *list, const uint32_t *out, uint32_t phase)
{
   const bool is3d = t->type == VK_IMAGE_TYPE_3D;
   for (uint32_t l = 0; l < t->levels; l++) {
      const uint32_t slices = is3d ? ld(t, l) : t->layers;
      const uint32_t count = pick_texels(t, l, list);
      const struct params p = { { lw(t, l), lh(t, l), slices, 0 }, l, count };
      if (!begin(c)) {
         return false;
      }
      vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &ds, 0, NULL);
      vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_ALL, 0, sizeof p, &p);
      vkCmdDispatch(c->cb, (count + 15u) / 16u, 1, 1);
      if (!submit(c) || !check_level(t, l, phase, list, count, out)) {
         return false;
      }
   }
   return true;
}

static bool
make_set(struct ctx *c, VkDescriptorPool dp, const struct tex *t, VkBuffer out,
         VkDeviceSize out_bytes, VkBuffer list, VkDescriptorSet *ds)
{
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &c->dsl,
   };
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, ds));
   VkDescriptorImageInfo dii[2] = {
      { .sampler = c->nearest, .imageView = t->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { .sampler = c->linear, .imageView = t->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
   };
   VkDescriptorBufferInfo dbi = { .buffer = out, .offset = 0, .range = out_bytes };
   VkDescriptorBufferInfo lbi = { .buffer = list, .offset = 0, .range = MAX_TEXELS * 16u };
   VkWriteDescriptorSet w[4] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = *ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = *ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = *ds, .dstBinding = 2,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = *ds, .dstBinding = 3,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &lbi },
   };
   vkUpdateDescriptorSets(c->dev, 4, w, 0, NULL);
   return true;
}

struct draw {
   VkImage        rt;
   VkDeviceMemory rt_mem;
   VkImageView    rt_view;
   VkRenderPass   rp;
   VkFramebuffer  fb;
   VkPipeline     pipe;
   VkBuffer       rb;
   VkDeviceMemory rb_mem;
   uint32_t      *rbp;
};

static bool
make_draw(struct ctx *c, struct draw *g, const char *vs_path, const char *fs_path)
{
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { RT, RT, 1 }, .mipLevels = 1,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c->dev, &imci, NULL, &g->rt));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(c->dev, g->rt, &mr);
   if (!alloc_mem(c, mr.size, mr.memoryTypeBits, &g->rt_mem)) {
      return false;
   }
   CHECK(vkBindImageMemory(c->dev, g->rt, g->rt_mem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g->rt,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &g->rt_view));
   VkAttachmentDescription ad = {
      .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference ar = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sd = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ar,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &ad, .subpassCount = 1, .pSubpasses = &sd,
   };
   CHECK(vkCreateRenderPass(c->dev, &rpci, NULL, &g->rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = g->rp,
      .attachmentCount = 1, .pAttachments = &g->rt_view, .width = RT, .height = RT,
      .layers = 1,
   };
   CHECK(vkCreateFramebuffer(c->dev, &fbci, NULL, &g->fb));
   VkShaderModule vs, fs;
   if (!make_module(c, vs_path, &vs) || !make_module(c, fs_path, &fs)) {
      return false;
   }
   VkPipelineShaderStageCreateInfo st[2] = {
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
   VkViewport vp = { 0, 0, (float)RT, (float)RT, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { RT, RT } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
      .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .layout = c->pl, .renderPass = g->rp,
   };
   CHECK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &g->pipe));
   vkDestroyShaderModule(c->dev, vs, NULL);
   vkDestroyShaderModule(c->dev, fs, NULL);
   return make_buffer(c, RT * RT * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &g->rb, &g->rb_mem,
                      (void **)&g->rbp);
}

/* Draw level 0, layer 1 of `t` into the target and check every pixel. */
static bool
run_draw(struct ctx *c, struct draw *g, const struct tex *t, VkDescriptorSet ds, uint32_t phase)
{
   const struct params p = { { t->w, t->h, t->layers, 0 }, 0, 0 };
   if (!begin(c)) {
      return false;
   }
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = g->rp,
      .framebuffer = g->fb, .renderArea = { { 0, 0 }, { RT, RT } },
   };
   vkCmdBeginRenderPass(c->cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g->pipe);
   vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, c->pl, 0, 1, &ds, 0, NULL);
   vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_ALL, 0, sizeof p, &p);
   vkCmdDraw(c->cb, 3, 1, 0, 0);
   vkCmdEndRenderPass(c->cb);
   VkBufferImageCopy r = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { RT, RT, 1 },
   };
   vkCmdCopyImageToBuffer(c->cb, g->rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->rb, 1, &r);
   if (!submit(c)) {
      return false;
   }
   unsigned bad = 0, res = 0;
   char first[160] = "";
   for (uint32_t py = 0; py < RT; py++) {
      for (uint32_t px = 0; px < RT; px++) {
         const uint32_t x = 8u * px + 3u, y = (129u * py) / 31u;
         const bool rr = bound(t, 0, 1, x, y, 0);
         const uint32_t got = g->rbp[py * RT + px];
         const uint32_t want = (rr ? 0xff000000u : 0u) |
                               (rr ? (texel(t, 0, 1, x, y, 0) & 0xffffffu) : (got & 0xffffffu));
         res += rr;
         if (got != want && !bad++) {
            snprintf(first, sizeof first, "pixel (%u,%u) <- texel (%u,%u): 0x%08x, want 0x%08x",
                     px, py, x, y, got, want);
         }
      }
   }
   if (bad) {
      printf("FAILED (draw, phase %u: %u pixels wrong; first %s)\n", phase, bad, first);
      return false;
   }
   printf("draw phase %u: %u pixels, %u on resident texels, right\n", phase, RT * RT, res);
   return true;
}

int
main(int argc, char **argv)
{
   if (argc < 5) {
      fprintf(stderr, "FAILED: usage: %s st_arr.comp.spv st_3d.comp.spv st_draw.vert.spv "
              "st_draw.frag.spv\n", argv[0]);
      return 1;
   }
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_tex", .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
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
   if (!have.shaderResourceResidency || !have.sparseBinding ||
       !have.sparseResidencyImage2D || !have.sparseResidencyImage3D) {
      printf("NotSupported: sparse residency of 2D and 3D images or shader residency\n");
      return 0;
   }
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs graphics, compute and sparse binding\n");
      return 0;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures want = {
      .shaderResourceResidency = VK_TRUE, .sparseBinding = VK_TRUE,
      .sparseResidencyImage2D = VK_TRUE, .sparseResidencyImage3D = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &want,
   };
   if (vkCreateDevice(c.pd, &dci, NULL, &c.dev) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkCreateDevice\n");
      return 1;
   }
   vkGetDeviceQueue(c.dev, qf, 0, &c.queue);
   vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 8.0f,
   };
   VkDescriptorSetLayoutBinding dslb[4] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
      { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4, .pBindings = dslb,
   };
   VkPushConstantRange pcr = { VK_SHADER_STAGE_ALL, 0, sizeof(struct params) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &c.dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   if (vkCreateSampler(c.dev, &sci, NULL, &c.nearest) != VK_SUCCESS ||
       (sci.magFilter = sci.minFilter = VK_FILTER_LINEAR,
        vkCreateSampler(c.dev, &sci, NULL, &c.linear)) != VK_SUCCESS ||
       vkCreateDescriptorSetLayout(c.dev, &dslci, NULL, &c.dsl) != VK_SUCCESS ||
       vkCreatePipelineLayout(c.dev, &plci, NULL, &c.pl) != VK_SUCCESS ||
       vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS ||
       vkCreateDescriptorPool(c.dev, &dpci, NULL, &dp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: sampler, layout or pool creation\n");
      return 1;
   }
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   if (vkAllocateCommandBuffers(c.dev, &cbai, &c.cb) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkAllocateCommandBuffers\n");
      return 1;
   }

   VkPipeline cpipe[2];
   for (unsigned i = 0; i < 2; i++) {
      VkShaderModule sm;
      if (!make_module(&c, argv[1 + i], &sm)) {
         return 1;
      }
      VkComputePipelineCreateInfo cpi = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
         .layout = c.pl,
      };
      if (vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &cpipe[i]) !=
          VK_SUCCESS) {
         fprintf(stderr, "FAILED: compute pipeline\n");
         return 1;
      }
      vkDestroyShaderModule(c.dev, sm, NULL);
   }

   struct tex arr = { .type = VK_IMAGE_TYPE_2D, .w = 260, .h = 130, .d = 1, .layers = 2,
                      .levels = 3 };
   struct tex vol = { .type = VK_IMAGE_TYPE_3D, .w = 40, .h = 34, .d = 18, .layers = 1,
                      .levels = 1 };
   struct draw g;
   memset(&g, 0, sizeof g);
   const VkDeviceSize out_bytes = (VkDeviceSize)MAX_TEXELS * 8 * 4;
   VkBuffer out, list;
   VkDeviceMemory out_mem, list_mem;
   uint32_t *outp, *listp;
   VkDescriptorSet ds_arr, ds_vol;
   if (!make_tex(&c, &arr) || !make_tex(&c, &vol) ||
       !make_buffer(&c, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &out_mem,
                    (void **)&outp) ||
       !make_buffer(&c, MAX_TEXELS * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &list, &list_mem,
                    (void **)&listp) ||
       !make_set(&c, dp, &arr, out, out_bytes, list, &ds_arr) ||
       !make_set(&c, dp, &vol, out, out_bytes, list, &ds_vol) ||
       !make_draw(&c, &g, argv[3], argv[4])) {
      return 1;
   }

   for (uint32_t phase = 0; phase < 2; phase++) {
      if (phase == 1 && (!bind_phase(&c, &arr, 1) || !bind_phase(&c, &vol, 1))) {
         return 1;
      }
      if (!run_dispatches(&c, &arr, cpipe[0], ds_arr, listp, outp, phase) ||
          !run_dispatches(&c, &vol, cpipe[1], ds_vol, listp, outp, phase) ||
          !run_draw(&c, &g, &arr, ds_arr, phase)) {
         return 1;
      }
   }

   printf("PASSED (sparse_tex: sparse fetches, samples and gathers of a partially resident "
          "2D array and 3D texture, codes by footprint, values by tile, across a rebind)\n");
   return 0;
}
