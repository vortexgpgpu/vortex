/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Storage access to partially resident sparse images.
 *
 * Every case is a sparse image at least two 64 KiB tiles wide, so a texel
 * addressed linearly lands in the wrong tile. Its tiles are bound in a
 * checkerboard, one tile per bind, to consecutive tiles of memory A. A compute
 * dispatch stores every texel's seeded value; a second one sparse-loads every
 * texel, checks the plain load agrees, and updates each resident texel -- an
 * atomic add on R32_UINT, a store on the other formats. Each case is one way the
 * driver reaches the texel: 2D, 3D (z through the tiles) and 2D array views;
 * texels of 1, 4 and 8 bytes; the packed RGBA8 encode; the per-channel 16-bit
 * encode; and a view declared without a format.
 *
 * The host then checks every texel's residency code against the checkerboard,
 * every resident texel's value, and, through a mapping of memory A, that each
 * resident texel's update sits in the memory of its own tile, at its place in
 * the tile. The order of texels inside a tile is the implementation's own;
 * lavapipe stores x, then y, then z. On vortexpipe a non-resident texel must
 * load as zero, even though the first dispatch stored to it.
 *
 * Then the binding is inverted: the tiles bound so far are unbound, and the
 * others are bound to memory B, which the host has filled with a second
 * pattern. Without any other call, the next dispatch must see the new
 * residency and read memory B's texels, and its updates must land in memory B.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL     4u
#define TILE      65536u
#define SALT_A    0x13572468u
#define SALT_B    0x8badf00du

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return false;                                                \
   }                                                               \
} while (0)

enum sp_update { UPD_ADD7, UPD_XOR };

struct sp_case {
   const char     *name;
   VkFormat        format;
   VkImageType     type;
   VkImageViewType view;
   uint32_t        w, h, d, layers;
   uint32_t        bpp;
   enum sp_update  update;
};

/* In the order of the SHADERS list, which is the order of argv. */
static const struct sp_case cases[] = {
   { "r32ui_2d",     VK_FORMAT_R32_UINT,           VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
     160, 136, 1, 1, 4, UPD_ADD7 },
   { "r32ui_3d",     VK_FORMAT_R32_UINT,           VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
     40, 34, 18, 1, 4, UPD_ADD7 },
   { "r32ui_array",  VK_FORMAT_R32_UINT,           VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
     160, 8, 1, 3, 4, UPD_ADD7 },
   { "rgba8_2d",     VK_FORMAT_R8G8B8A8_UNORM,     VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
     136, 130, 1, 1, 4, UPD_XOR },
   { "rgba8_array",  VK_FORMAT_R8G8B8A8_UNORM,     VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
     160, 4, 1, 4, 4, UPD_XOR },
   { "rg16ui_3d",    VK_FORMAT_R16G16_UINT,        VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
     34, 33, 17, 1, 4, UPD_XOR },
   { "r8ui_2d",      VK_FORMAT_R8_UINT,            VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
     260, 4, 1, 1, 1, UPD_XOR },
   { "rgba16ui_array", VK_FORMAT_R16G16B16A16_UINT, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
     130, 66, 1, 2, 8, UPD_XOR },
   { "noformat_2d",  VK_FORMAT_R32_UINT,           VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
     136, 130, 1, 1, 4, UPD_ADD7 },
};
#define NUM_CASES (sizeof(cases) / sizeof(cases[0]))

struct ctx {
   VkPhysicalDevice pd;
   VkPhysicalDeviceMemoryProperties mp;
   VkDevice        dev;
   VkQueue         queue;
   VkCommandBuffer cb;
   VkDescriptorSetLayout dsl;
   VkPipelineLayout      pl;
   bool            zero_unbound;
};

struct params {
   uint32_t e[4];
   uint32_t mode;
   uint32_t salt;
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
make_buffer(struct ctx *c, VkDeviceSize bytes, VkBufferUsageFlags usage,
            VkBuffer *buf, VkDeviceMemory *mem)
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
   return true;
}

/* The seeded value of texel p, as the shaders compute it. */
static uint32_t
seed(uint32_t salt, uint32_t x, uint32_t y, uint32_t z)
{
   return salt ^ (x * 2654435761u) ^ (y * 40503u) ^ (z * 2246822519u);
}

/* What a sparse load of a texel holding `v` packs into its result word. */
static uint32_t
packed_of(const struct sp_case *k, uint32_t v)
{
   return k->bpp == 1 ? (v & 0xffu) : v;
}

static uint32_t
updated(const struct sp_case *k, uint32_t v)
{
   if (k->update == UPD_ADD7) {
      return v + 7u;
   }
   return k->bpp == 1 ? (v ^ 0x5au) & 0xffu : v ^ 0x5a5a5a5au;
}

/* The bytes of a texel whose packed value is `v`. */
static void
texel_bytes(const struct sp_case *k, uint32_t v, uint8_t out[8])
{
   const uint32_t w[2] = { v, ~v };
   memcpy(out, w, sizeof w);
}

struct tiling {
   uint32_t gw, gh, gd;      /* tile size in texels */
   uint32_t nx, ny, nz, nl;  /* tiles per axis, layers */
};

static uint32_t
tile_parity(uint32_t tx, uint32_t ty, uint32_t tz, uint32_t l)
{
   return (tx + ty + tz + l) & 1u;
}

/* Tile number in the order tiles are given memory: x, then y, then z, then
 * layer; the n-th bound tile of a phase takes tile n of its memory. */
static uint32_t
tile_index(const struct tiling *g, uint32_t tx, uint32_t ty, uint32_t tz, uint32_t l)
{
   return ((l * g->nz + tz) * g->ny + ty) * g->nx + tx;
}

/* Byte offset of texel (x, y, z) inside its tile. */
static uint32_t
in_tile(const struct sp_case *k, const struct tiling *g, uint32_t x, uint32_t y, uint32_t z)
{
   return ((x % g->gw) + (y % g->gh) * g->gw + (z % g->gd) * g->gw * g->gh) * k->bpp;
}

/* For each tile, the memory tile it is bound to in `phase` (0 or 1), or
 * UINT32_MAX when it is unbound then. */
static uint32_t *
phase_slots(const struct tiling *g, uint32_t phase, uint32_t *count)
{
   const uint32_t n = g->nx * g->ny * g->nz * g->nl;
   uint32_t *slot = malloc(n * sizeof *slot);
   uint32_t next = 0;
   for (uint32_t l = 0; l < g->nl; l++) {
      for (uint32_t tz = 0; tz < g->nz; tz++) {
         for (uint32_t ty = 0; ty < g->ny; ty++) {
            for (uint32_t tx = 0; tx < g->nx; tx++) {
               slot[tile_index(g, tx, ty, tz, l)] =
                  tile_parity(tx, ty, tz, l) == phase ? next++ : UINT32_MAX;
            }
         }
      }
   }
   *count = next;
   return slot;
}

static uint32_t
min_u32(uint32_t a, uint32_t b)
{
   return a < b ? a : b;
}

/* Bind every tile of `phase` to `mem` and unbind every tile of the other. */
static bool
bind_phase(struct ctx *c, VkImage img, const struct sp_case *k, const struct tiling *g,
           const uint32_t *slot, VkDeviceMemory mem)
{
   const uint32_t n = g->nx * g->ny * g->nz * g->nl;
   VkSparseImageMemoryBind *b = calloc(n, sizeof *b);
   const uint32_t depth = k->type == VK_IMAGE_TYPE_3D ? k->d : 1u;
   uint32_t m = 0;
   for (uint32_t l = 0; l < g->nl; l++) {
      for (uint32_t tz = 0; tz < g->nz; tz++) {
         for (uint32_t ty = 0; ty < g->ny; ty++) {
            for (uint32_t tx = 0; tx < g->nx; tx++) {
               const uint32_t s = slot[tile_index(g, tx, ty, tz, l)];
               b[m++] = (VkSparseImageMemoryBind){
                  .subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, l },
                  .offset = { (int32_t)(tx * g->gw), (int32_t)(ty * g->gh),
                              (int32_t)(tz * g->gd) },
                  .extent = { min_u32(g->gw, k->w - tx * g->gw),
                              min_u32(g->gh, k->h - ty * g->gh),
                              min_u32(g->gd, depth - tz * g->gd) },
                  .memory = s == UINT32_MAX ? VK_NULL_HANDLE : mem,
                  .memoryOffset = s == UINT32_MAX ? 0 : (VkDeviceSize)s * TILE,
               };
            }
         }
      }
   }
   VkSparseImageMemoryBindInfo ibi = { .image = img, .bindCount = m, .pBinds = b };
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

static bool
dispatch(struct ctx *c, VkPipeline pipe, VkDescriptorSet ds, const struct sp_case *k,
         uint32_t mode, uint32_t salt)
{
   const uint32_t slices = k->type == VK_IMAGE_TYPE_3D ? k->d : k->layers;
   const struct params p = { { k->w, k->h, slices, 0 }, mode, salt };
   CHECK(vkResetCommandBuffer(c->cb, 0));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(c->cb, &cbbi));
   vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &ds, 0, NULL);
   vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
   vkCmdDispatch(c->cb, (k->w + LOCAL - 1) / LOCAL, (k->h + LOCAL - 1) / LOCAL, slices);
   CHECK(vkEndCommandBuffer(c->cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &c->cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   return true;
}

/* Check one sparse-load pass of `phase` against memory `mem` (mapped at
 * `map`): residency, the loaded values (seeded with `salt`), and the updates
 * the pass left in memory. */
static bool
check_pass(struct ctx *c, const struct sp_case *k, const struct tiling *g,
           const uint32_t *slot, uint32_t phase, uint32_t salt,
           const uint32_t *out, const uint8_t *map)
{
   const uint32_t slices = k->type == VK_IMAGE_TYPE_3D ? k->d : k->layers;
   const bool is3d = k->type == VK_IMAGE_TYPE_3D;
   uint32_t bad_code = 0, bad_value = 0, bad_mem = 0, bad_zero = 0, n_res = 0;
   char first[256] = "";
   for (uint32_t z = 0; z < slices; z++) {
      for (uint32_t y = 0; y < k->h; y++) {
         for (uint32_t x = 0; x < k->w; x++) {
            const uint32_t i = (z * k->h + y) * k->w + x;
            const uint32_t tz = is3d ? z / g->gd : 0, l = is3d ? 0 : z;
            const uint32_t s = slot[tile_index(g, x / g->gw, y / g->gh, tz, l)];
            const bool want_res = s != UINT32_MAX;
            const uint32_t v = packed_of(k, seed(salt, x, y, z));
            const uint32_t got = out[2u * i], flags = out[2u * i + 1u];
            if ((flags & 1u) != (want_res ? 1u : 0u) || (flags & 2u)) {
               if (!bad_code++ && !bad_value && !bad_mem) {
                  snprintf(first, sizeof first, "texel (%u,%u,%u): flags 0x%x, want resident=%u "
                           "and the plain load agreeing", x, y, z, flags, want_res);
               }
               continue;
            }
            if (!want_res) {
               if (c->zero_unbound && got != 0u && !bad_zero++ && !bad_code && !bad_value &&
                   !bad_mem) {
                  snprintf(first, sizeof first, "texel (%u,%u,%u) on an unbound tile loaded "
                           "0x%08x, want 0", x, y, z, got);
               }
               continue;
            }
            n_res++;
            if (got != v) {
               if (!bad_value++ && !bad_code && !bad_mem) {
                  snprintf(first, sizeof first, "texel (%u,%u,%u): loaded 0x%08x, want 0x%08x",
                           x, y, z, got, v);
               }
               continue;
            }
            uint8_t want[8], have[8];
            texel_bytes(k, updated(k, v), want);
            const uint32_t at = s * TILE + in_tile(k, g, x, y, is3d ? z : 0);
            memcpy(have, map + at, k->bpp);
            if (memcmp(have, want, k->bpp)) {
               uint32_t hw = 0;
               memcpy(&hw, have, k->bpp < 4 ? k->bpp : 4);
               if (!bad_mem++ && !bad_code && !bad_value) {
                  snprintf(first, sizeof first, "texel (%u,%u,%u): memory tile %u + 0x%x holds "
                           "0x%08x, want 0x%08x", x, y, z, s, at % TILE, hw, updated(k, v));
               }
            }
         }
      }
   }
   if (bad_code || bad_value || bad_mem || bad_zero) {
      printf("FAILED (%s, phase %u: %u residency, %u value, %u memory, %u unbound-zero "
             "mismatches; first %s)\n", k->name, phase, bad_code, bad_value, bad_mem, bad_zero,
             first);
      return false;
   }
   printf("%s phase %u: %u resident texels loaded and updated in their tiles, residency "
          "of all %u right\n", k->name, phase, n_res, k->w * k->h * slices);
   return true;
}

static bool
run_case(struct ctx *c, const struct sp_case *k, const char *spv_path)
{
   const uint32_t slices = k->type == VK_IMAGE_TYPE_3D ? k->d : k->layers;
   const uint32_t texels = k->w * k->h * slices;
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

   uint32_t np = 0;
   vkGetPhysicalDeviceSparseImageFormatProperties(c->pd, k->format, k->type,
      VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &np, NULL);
   VkSparseImageFormatProperties sfp;
   if (np < 1) {
      printf("%s: NotSupported (no sparse residency for this format)\n", k->name);
      return true;
   }
   np = 1;
   vkGetPhysicalDeviceSparseImageFormatProperties(c->pd, k->format, k->type,
      VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &np, &sfp);
   struct tiling g = {
      .gw = sfp.imageGranularity.width, .gh = sfp.imageGranularity.height,
      .gd = sfp.imageGranularity.depth,
   };
   if (g.gw * g.gh * g.gd * k->bpp != TILE || g.gw >= k->w) {
      printf("FAILED (%s: granularity %ux%ux%u is not a 64 KiB tile narrower than "
             "the image)\n", k->name, g.gw, g.gh, g.gd);
      return false;
   }
   g.nx = (k->w + g.gw - 1) / g.gw;
   g.ny = (k->h + g.gh - 1) / g.gh;
   g.nz = k->type == VK_IMAGE_TYPE_3D ? (k->d + g.gd - 1) / g.gd : 1u;
   g.nl = k->layers;

   size_t sz = 0;
   uint32_t *spv = read_spirv(spv_path, &sz);
   if (!spv) {
      return false;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   free(spv);
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = c->pl,
   };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   vkDestroyShaderModule(c->dev, sm, NULL);

   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      .imageType = k->type, .format = k->format,
      .extent = { k->w, k->h, k->type == VK_IMAGE_TYPE_3D ? k->d : 1u },
      .mipLevels = 1, .arrayLayers = k->layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   CHECK(vkCreateImage(c->dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c->dev, img, &imr);

   uint32_t na = 0, nb = 0;
   uint32_t *slot_a = phase_slots(&g, 0, &na);
   uint32_t *slot_b = phase_slots(&g, 1, &nb);
   VkDeviceMemory mem_a, mem_b;
   if (!alloc_mem(c, (VkDeviceSize)na * TILE, imr.memoryTypeBits, &mem_a) ||
       !alloc_mem(c, (VkDeviceSize)(nb ? nb : 1) * TILE, imr.memoryTypeBits, &mem_b)) {
      return false;
   }
   uint8_t *map_a, *map_b;
   CHECK(vkMapMemory(c->dev, mem_a, 0, VK_WHOLE_SIZE, 0, (void **)&map_a));
   CHECK(vkMapMemory(c->dev, mem_b, 0, VK_WHOLE_SIZE, 0, (void **)&map_b));
   memset(map_a, 0xee, (size_t)na * TILE);

   /* Memory B holds the second pattern before it is bound. */
   const bool is3d = k->type == VK_IMAGE_TYPE_3D;
   for (uint32_t z = 0; z < slices; z++) {
      for (uint32_t y = 0; y < k->h; y++) {
         for (uint32_t x = 0; x < k->w; x++) {
            const uint32_t s = slot_b[tile_index(&g, x / g.gw, y / g.gh,
                                                 is3d ? z / g.gd : 0, is3d ? 0 : z)];
            if (s == UINT32_MAX) {
               continue;
            }
            uint8_t b[8];
            texel_bytes(k, packed_of(k, seed(SALT_B, x, y, z)), b);
            memcpy(map_b + s * TILE + in_tile(k, &g, x, y, is3d ? z : 0), b, k->bpp);
         }
      }
   }

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = k->view, .format = k->format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, k->layers },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c->dev, &ivci, NULL, &view));
   const VkDeviceSize out_bytes = (VkDeviceSize)texels * 2u * sizeof(uint32_t);
   VkBuffer out;
   VkDeviceMemory out_mem;
   if (!make_buffer(c, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out, &out_mem)) {
      return false;
   }
   uint32_t *outp;
   CHECK(vkMapMemory(c->dev, out_mem, 0, out_bytes, 0, (void **)&outp));

   VkDescriptorPoolSize dps[2] = {
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  .descriptorCount = 1 },
      { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c->dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &c->dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(c->dev, &dsai, &ds));
   VkDescriptorImageInfo dii = { .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorBufferInfo dbi = { .buffer = out, .offset = 0, .range = out_bytes };
   VkWriteDescriptorSet wds[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dii },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c->dev, 2, wds, 0, NULL);

   bool ok = bind_phase(c, img, k, &g, slot_a, mem_a) &&
             dispatch(c, pipe, ds, k, 0, SALT_A) &&
             dispatch(c, pipe, ds, k, 1, SALT_A) &&
             check_pass(c, k, &g, slot_a, 0, SALT_A, outp, map_a) &&
             bind_phase(c, img, k, &g, slot_b, mem_b) &&
             dispatch(c, pipe, ds, k, 1, SALT_B) &&
             check_pass(c, k, &g, slot_b, 1, SALT_B, outp, map_b);

   vkUnmapMemory(c->dev, out_mem);
   vkUnmapMemory(c->dev, mem_a);
   vkUnmapMemory(c->dev, mem_b);
   vkDestroyDescriptorPool(c->dev, dp, NULL);
   vkDestroyPipeline(c->dev, pipe, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyImage(c->dev, img, NULL);
   vkFreeMemory(c->dev, mem_a, NULL);
   vkFreeMemory(c->dev, mem_b, NULL);
   vkDestroyBuffer(c->dev, out, NULL);
   vkFreeMemory(c->dev, out_mem, NULL);
   free(slot_a);
   free(slot_b);
   return ok;
}

int
main(int argc, char **argv)
{
   if (argc < 1 + (int)NUM_CASES) {
      fprintf(stderr, "FAILED: usage: %s <%u shader .spv files>\n", argv[0],
              (unsigned)NUM_CASES);
      return 1;
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-sparse_image",
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
   c.zero_unbound = strncmp(props.deviceName, "vortexpipe", 10) == 0;

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(c.pd, &have);
   if (!have.shaderResourceResidency || !have.sparseBinding ||
       !have.sparseResidencyImage2D || !have.sparseResidencyImage3D ||
       !have.shaderStorageImageReadWithoutFormat || !have.shaderStorageImageWriteWithoutFormat) {
      printf("NotSupported: sparse residency of 2D and 3D images, shader residency or "
             "format-less storage access is not offered by this device\n");
      return 0;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (qfp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) { qf = i; break; }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("NotSupported: no queue family runs both compute and sparse binding\n");
      return 0;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceFeatures want = {
      .shaderResourceResidency = VK_TRUE,
      .sparseBinding = VK_TRUE,
      .sparseResidencyImage2D = VK_TRUE,
      .sparseResidencyImage3D = VK_TRUE,
      .shaderStorageImageReadWithoutFormat = VK_TRUE,
      .shaderStorageImageWriteWithoutFormat = VK_TRUE,
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

   VkDescriptorSetLayoutBinding dslb[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = dslb,
   };
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(struct params),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &c.dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool cp;
   if (vkCreateDescriptorSetLayout(c.dev, &dslci, NULL, &c.dsl) != VK_SUCCESS ||
       vkCreatePipelineLayout(c.dev, &plci, NULL, &c.pl) != VK_SUCCESS ||
       vkCreateCommandPool(c.dev, &cpci, NULL, &cp) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: layout or command pool creation\n");
      return 1;
   }
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   if (vkAllocateCommandBuffers(c.dev, &cbai, &c.cb) != VK_SUCCESS) {
      fprintf(stderr, "FAILED: vkAllocateCommandBuffers\n");
      return 1;
   }

   unsigned failed = 0;
   for (unsigned i = 0; i < NUM_CASES; i++) {
      failed += !run_case(&c, &cases[i], argv[1 + i]);
   }
   if (failed) {
      printf("FAILED (sparse_image: %u of %u cases)\n", failed, (unsigned)NUM_CASES);
      return 1;
   }

   vkDestroyCommandPool(c.dev, cp, NULL);
   vkDestroyPipelineLayout(c.dev, c.pl, NULL);
   vkDestroyDescriptorSetLayout(c.dev, c.dsl, NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("PASSED (sparse_image: %u partially resident sparse images, stored, sparse-loaded "
          "and updated in their own tiles, then rebound to new memory)\n",
          (unsigned)NUM_CASES);
   return 0;
}
