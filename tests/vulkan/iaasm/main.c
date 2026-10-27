/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Input assembly on the device. Every case draws its primitives twice: once
 * as the application issues them -- an index buffer of any width read from
 * any first index with any vertex offset, strips, fans and lists cut by
 * primitive restart, instances -- and once as a plain non-indexed list whose
 * vertices the host lays out by the specification's assembly rules: each
 * primitive in draw order, its vertices in the winding order of its place in
 * its strip or fan, rotated as the provoking-vertex mode puts them. The two
 * images must be identical, bit for bit.
 *
 * The triangles overlap and the later ones cover the earlier, so the order of
 * the primitives shows. A fragment's colour mixes its flat input -- which
 * vertex is provoking -- with the smooth one in another channel order, which
 * every vertex of the primitive weighs into, and the vertex shader replaces the
 * flat colour with magenta when gl_VertexIndex is not the index the vertex
 * record was meant to be read at. Some cases cull back faces, so a triangle of
 * a strip with the wrong winding is lost.
 *
 * The lattice case is a strip of more than two thousand positions over the
 * whole target, cut into rows by restarts plus some at random, in 32-bit
 * indices offset by a vertex offset.
 *
 * Then the statistics: pipeline statistics around an instanced restart strip,
 * and around the same draw under rasterizer discard, which leaves only its
 * counts to make; the input assembler's primitive count is what the restart
 * leaves, times the instances.
 *
 * With --cut N it draws only N strips whose restarts leave every run too short
 * for a triangle, and checks the target keeps its clear colour. The device
 * finds that out in the restart scan and skips the rest of each draw, which
 * the run's retired-instruction count shows (tests/vulkan/iacut).
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     64u
#define HEIGHT    64u
#define FB_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define IMG_BYTES (WIDTH * HEIGHT * 4u)

/* The vertex table: NRAND scattered vertices, then a LAT x LAT lattice. */
#define NRAND     96u
#define LAT       33u
#define NTABLE    (NRAND + LAT * LAT)
#define MAX_POS   2400u
#define MAX_REF   (3u * MAX_POS)

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      printf("FAILED (iaasm: %s -> VkResult %d)\n", #x, (int)_r);  \
      return 1;                                                    \
   }                                                               \
} while (0)

struct vtx {
   float x, y;
   float r, g, b;
   float id;   /* the gl_VertexIndex this record is read at */
};

/* ---- the cases -------------------------------------------------------- */

enum { IDX_NONE = 0, IDX_U8 = 1, IDX_U16 = 2, IDX_U32 = 4 };

struct ia_case {
   const char *name;
   VkPrimitiveTopology topo;
   uint32_t isz;            /* IDX_* */
   bool restart, last, cull;
   uint32_t count;          /* index positions, or vertices */
   uint32_t first;          /* firstIndex, or firstVertex */
   int32_t  base;           /* vertexOffset */
   uint32_t instances, first_instance;
   uint32_t restart_pct;    /* positions that hold the restart index; 100: every third */
   bool lattice;
};

static const struct ia_case cases[] = {
   { "list u16",            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  IDX_U16, false, false, false, 30, 0, 0, 1, 0, 0, false },
   { "list u32 offset",     VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  IDX_U32, false, true,  true,  31, 5, -3, 1, 0, 0, false },
   { "list u8 base",        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  IDX_U8,  false, false, false, 24, 2, 7, 1, 0, 0, false },
   { "strip u16",           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U16, false, false, true,  20, 0, 0, 1, 0, 0, false },
   { "strip u16 last",      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U16, false, true,  true,  21, 3, 0, 1, 0, 0, false },
   { "strip u32 restart",   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U32, true,  false, false, 40, 1, 2, 1, 0, 22, false },
   { "strip u8 restart x3", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U8,  true,  true,  true,  37, 0, 0, 3, 1, 18, false },
   { "fan u16 restart",     VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,   IDX_U16, true,  false, false, 33, 4, -1, 1, 0, 15, false },
   { "fan u32 restart last", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,  IDX_U32, true,  true,  true,  29, 0, 0, 2, 0, 12, false },
   { "line strip u16 restart", VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,  IDX_U16, true,  false, false, 35, 0, 0, 1, 0, 20, false },
   { "line list u32 restart",  VK_PRIMITIVE_TOPOLOGY_LINE_LIST,   IDX_U32, true,  true,  false, 34, 2, 0, 1, 0, 15, false },
   { "points u8 restart",   VK_PRIMITIVE_TOPOLOGY_POINT_LIST,     IDX_U8,  true,  false, false, 40, 0, 3, 1, 0, 25, false },
   { "list u16 restart x2", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  IDX_U16, true,  false, true,  41, 6, 0, 2, 1, 10, false },
   { "strip unindexed last", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_NONE, false, true, true,  18, 4, 0, 1, 0, 0, false },
   { "fan unindexed x2",    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,   IDX_NONE, false, false, false, 12, 9, 0, 2, 0, 0, false },
   { "strip runs too short", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U16, true, false, false, 16, 0, 0, 1, 0, 100, false },
   { "lattice u32 restart", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, IDX_U32, true,  false, true,  0, 7, 11, 1, 0, 1, true },
};
#define NCASES (sizeof(cases) / sizeof(cases[0]))

static uint32_t
rng(uint32_t *s)
{
   *s = *s * 1664525u + 1013904223u;
   return *s >> 8;
}

static uint32_t
restart_value(uint32_t isz)
{
   return isz == IDX_U8 ? 0xffu : isz == IDX_U16 ? 0xffffu : 0xffffffffu;
}

/* The case's index positions (or, unindexed, its vertex numbers), into idx;
 * returns how many. */
static uint32_t
case_positions(const struct ia_case *c, uint32_t seed, uint32_t *idx)
{
   const uint32_t rs = restart_value(c->isz);
   if (c->lattice) {
      /* Row pairs of the lattice, each a run: (r, col), (r + 1, col) for every
       * column, then a restart; and a few more restarts at random. */
      uint32_t n = 0;
      for (uint32_t r = 0; r + 1 < LAT; r++) {
         for (uint32_t col = 0; col < LAT; col++) {
            for (uint32_t k = 0; k < 2; k++) {
               idx[n++] = NRAND + (r + k) * LAT + col - (uint32_t)c->base;
            }
         }
         idx[n++] = rs;
      }
      for (uint32_t k = 0; k < 12; k++) {
         idx[rng(&seed) % n] = rs;
      }
      return n;
   }
   for (uint32_t p = 0; p < c->count; p++) {
      if (c->restart && (c->restart_pct == 100u ? p % 3u == 2u
                                                 : rng(&seed) % 100u < c->restart_pct)) {
         idx[p] = rs;
         continue;
      }
      const uint32_t lo = c->base < 0 ? (uint32_t)-c->base : 0u;
      const uint32_t hi = NRAND - (c->base > 0 ? (uint32_t)c->base : 0u);
      idx[p] = lo + rng(&seed) % (hi - lo);
   }
   if (c->restart && c->restart_pct < 100u && c->count > 4) {
      /* A restart at either end and two in a row. */
      idx[0] = rs;
      idx[c->count - 1] = rs;
      idx[c->count / 2] = rs;
      idx[c->count / 2 + 1] = rs;
   }
   return c->count;
}

static uint32_t
per_prim(VkPrimitiveTopology t)
{
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return 1;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return 2;
   default:
      return 3;
   }
}

static VkPrimitiveTopology
list_of(VkPrimitiveTopology t)
{
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return t;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
   default:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   }
}

/* The table vertex of `pos`'s index, or of vertex `pos` unindexed. */
static uint32_t
table_vertex(const struct ia_case *c, const uint32_t *idx, uint32_t pos)
{
   if (c->isz == IDX_NONE) {
      return c->first + pos;
   }
   return idx[pos] + (uint32_t)c->base;
}

/* The case's primitives as a list of table vertices, by the specification's
 * rules; returns how many vertices. */
static uint32_t
assemble(const struct ia_case *c, const uint32_t *idx, uint32_t n, uint32_t *out)
{
   const uint32_t rs = restart_value(c->isz);
   const uint32_t per = per_prim(c->topo);
   uint32_t w = 0, run = 0;
#define TV(pos) table_vertex(c, idx, (pos))
   for (uint32_t p = 0; p < n; p++) {
      if (c->restart && idx[p] == rs) {
         run = p + 1;
         continue;
      }
      const uint32_t k = p - run;
      switch (c->topo) {
      case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
         if (k % per == per - 1) {
            for (uint32_t j = per; j-- > 0;) {
               out[w++] = TV(p - j);
            }
         }
         break;
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
         if (k >= 1) {
            out[w++] = TV(p - 1);
            out[w++] = TV(p);
         }
         break;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: {
         if (k < 2) {
            break;
         }
         const uint32_t t = k - 2;
         if (!(t & 1u)) {
            out[w++] = TV(p - 2); out[w++] = TV(p - 1); out[w++] = TV(p);
         } else if (!c->last) {   /* (t, t + 2, t + 1) */
            out[w++] = TV(p - 2); out[w++] = TV(p); out[w++] = TV(p - 1);
         } else {                 /* (t + 1, t, t + 2) */
            out[w++] = TV(p - 1); out[w++] = TV(p - 2); out[w++] = TV(p);
         }
         break;
      }
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
         if (k < 2) {
            break;
         }
         if (!c->last) {          /* (t + 1, t + 2, 0) */
            out[w++] = TV(p - 1); out[w++] = TV(p); out[w++] = TV(run);
         } else {                 /* (0, t + 1, t + 2) */
            out[w++] = TV(run); out[w++] = TV(p - 1); out[w++] = TV(p);
         }
         break;
      default:
         break;
      }
   }
#undef TV
   return w;
}

static void
make_table(struct vtx *t)
{
   uint32_t s = 0x1234567u;
   for (uint32_t v = 0; v < NRAND; v++) {
      t[v].x = -0.9f + 1.8f * (float)(rng(&s) % 1024u) / 1023.0f;
      t[v].y = -0.9f + 1.8f * (float)(rng(&s) % 1024u) / 1023.0f;
      t[v].r = (float)(rng(&s) % 256u) / 255.0f;
      t[v].g = (float)(rng(&s) % 256u) / 255.0f;
      t[v].b = (float)(rng(&s) % 256u) / 255.0f;
   }
   for (uint32_t r = 0; r < LAT; r++) {
      for (uint32_t col = 0; col < LAT; col++) {
         struct vtx *v = &t[NRAND + r * LAT + col];
         v->x = -1.0f + 2.0f * (float)col / (float)(LAT - 1) + 0.013f;
         v->y = -1.0f + 2.0f * (float)r / (float)(LAT - 1) + 0.007f;
         v->r = (float)(rng(&s) % 256u) / 255.0f;
         v->g = (float)(rng(&s) % 256u) / 255.0f;
         v->b = (float)(rng(&s) % 256u) / 255.0f;
      }
   }
   for (uint32_t v = 0; v < NTABLE; v++) {
      t[v].id = (float)v;
   }
}

/* ---- Vulkan helpers ----------------------------------------------------- */

static uint32_t *
read_spirv(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
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

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) {
      return VK_NULL_HANDLE;
   }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
      sm = VK_NULL_HANDLE;
   }
   free(spv);
   return sm;
}

static uint32_t
find_mem(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
         VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   return UINT32_MAX;
}

static VkResult
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize bytes,
            VkBufferUsageFlags usage, VkBuffer *out_buf, VkDeviceMemory *out_mem,
            void **out_map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = usage,
   };
   VkResult r = vkCreateBuffer(dev, &bci, NULL, out_buf);
   if (r != VK_SUCCESS) {
      return r;
   }
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *out_buf, &mr);
   const uint32_t mt = find_mem(mp, mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mt == UINT32_MAX) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   r = vkAllocateMemory(dev, &mai, NULL, out_mem);
   if (r != VK_SUCCESS) {
      return r;
   }
   r = vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
   if (r != VK_SUCCESS) {
      return r;
   }
   return vkMapMemory(dev, *out_mem, 0, bytes, 0, out_map);
}

struct pipe_key {
   VkPrimitiveTopology topo;
   bool restart, last, cull, discard;
};

static VkResult
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl, VkShaderModule vs,
              VkShaderModule fs, struct pipe_key k, VkPipeline *out)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkVertexInputBindingDescription bind = { 0, sizeof(struct vtx), VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription attrs[3] = {
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct vtx, x) },
      { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct vtx, r) },
      { 2, 0, VK_FORMAT_R32_SFLOAT, offsetof(struct vtx, id) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind,
      .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attrs,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = k.topo, .primitiveRestartEnable = k.restart,
   };
   VkViewport vpt = { 0, 0, (float)WIDTH, (float)HEIGHT, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { WIDTH, HEIGHT } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vpt, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
      .provokingVertexMode = k.last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                    : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .pNext = &pv,
      .rasterizerDiscardEnable = k.discard,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = k.cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
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
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp,
   };
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static bool
has_extension(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
   VkExtensionProperties *e = calloc(n ? n : 1, sizeof(*e));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
   bool found = false;
   for (uint32_t i = 0; i < n && !found; i++) {
      found = strcmp(e[i].extensionName, name) == 0;
   }
   free(e);
   return found;
}

static VkIndexType
index_type(uint32_t isz)
{
   return isz == IDX_U8 ? VK_INDEX_TYPE_UINT8_KHR
        : isz == IDX_U16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
}

static void
store_indices(uint8_t *dst, uint32_t isz, uint32_t first, const uint32_t *idx, uint32_t n)
{
   for (uint32_t p = 0; p < n; p++) {
      const uint32_t v = idx[p];
      uint8_t *at = dst + (size_t)(first + p) * isz;
      if (isz == IDX_U8) {
         *at = (uint8_t)v;
      } else if (isz == IDX_U16) {
         const uint16_t h = (uint16_t)v;
         memcpy(at, &h, 2);
      } else {
         memcpy(at, &v, 4);
      }
   }
}

int
main(int argc, char **argv)
{
   const char *vs_path = argc > 1 ? argv[1] : "iaasm.vert.spv";
   const char *fs_path = argc > 2 ? argv[2] : "iaasm.frag.spv";
   const char *exts[] = {
      VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME,
      VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME,
      VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
   };

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "iaasm",
      .apiVersion = VK_API_VERSION_1_2,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   const VkResult er = vkEnumeratePhysicalDevices(inst, &npd, &pd);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || npd == 0) {
      printf("FAILED (iaasm: no physical device)\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   for (unsigned i = 0; i < sizeof exts / sizeof exts[0]; i++) {
      if (!has_extension(pd, exts[i])) {
         printf("FAILED (iaasm: %s not supported)\n", exts[i]);
         return 1;
      }
   }
   VkPhysicalDeviceIndexTypeUint8FeaturesKHR u8_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR,
      .indexTypeUint8 = VK_TRUE,
   };
   VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT lr_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
      .pNext = &u8_on, .primitiveTopologyListRestart = VK_TRUE,
   };
   VkPhysicalDeviceProvokingVertexFeaturesEXT pv_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
      .pNext = &lr_on, .provokingVertexLast = VK_TRUE,
   };
   VkPhysicalDeviceFeatures feats = { .pipelineStatisticsQuery = VK_TRUE, .largePoints = VK_TRUE };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &pv_on,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = sizeof exts / sizeof exts[0], .ppEnabledExtensionNames = exts,
      .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      printf("FAILED (iaasm: cannot load %s or %s)\n", vs_path, fs_path);
      return 1;
   }

   /* --- target ----------------------------------------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = FB_FORMAT, .extent = { WIDTH, HEIGHT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &imci, NULL, &img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(dev, img, &imr);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = imr.size,
      .memoryTypeIndex = find_mem(&mp, imr.memoryTypeBits, 0),
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FB_FORMAT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkAttachmentDescription att = {
      .format = FB_FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference attref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &attref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub,
   };
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 1, .pAttachments = &view, .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* --- data --------------------------------------------------------------- */
   VkBuffer tbuf, ibuf, rbuf, pxbuf;
   VkDeviceMemory tmem, imem2, rmem, pxmem;
   struct vtx *table, *refmap;
   uint8_t *imap, *pxmap;
   const VkDeviceSize ref_slot = (VkDeviceSize)MAX_REF * sizeof(struct vtx);
   const VkDeviceSize idx_slot = (VkDeviceSize)(MAX_POS + 16u) * 4u;
   CHECK(host_buffer(dev, &mp, NTABLE * sizeof(struct vtx), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     &tbuf, &tmem, (void **)&table));
   CHECK(host_buffer(dev, &mp, idx_slot * NCASES, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     &ibuf, &imem2, (void **)&imap));
   CHECK(host_buffer(dev, &mp, ref_slot * NCASES, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     &rbuf, &rmem, (void **)&refmap));
   CHECK(host_buffer(dev, &mp, (VkDeviceSize)IMG_BYTES * 2u * NCASES,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT, &pxbuf, &pxmem, (void **)&pxmap));
   make_table(table);
   memset(imap, 0x5a, idx_slot * NCASES);

   uint32_t *idx = malloc(MAX_POS * 4u);
   uint32_t *list = malloc(MAX_REF * 4u);
   uint32_t npos[NCASES], nref[NCASES];
   for (unsigned c = 0; c < NCASES; c++) {
      const struct ia_case *k = &cases[c];
      npos[c] = case_positions(k, 0x9e3779b9u * (c + 1u), idx);
      if (k->isz != IDX_NONE) {
         store_indices(imap + idx_slot * c, k->isz, k->first, idx, npos[c]);
      }
      nref[c] = assemble(k, idx, npos[c], list);
      struct vtx *ref = (struct vtx *)((uint8_t *)refmap + ref_slot * c);
      for (uint32_t j = 0; j < nref[c]; j++) {
         ref[j] = table[list[j]];
         ref[j].id = (float)j;
      }
   }

   /* --- pipelines ------------------------------------------------------------ */
   VkPipeline dpipe[NCASES], rpipe[NCASES];
   for (unsigned c = 0; c < NCASES; c++) {
      const struct ia_case *k = &cases[c];
      const struct pipe_key dk = { k->topo, k->restart, k->last, k->cull, false };
      const struct pipe_key rk = { list_of(k->topo), false, k->last, k->cull, false };
      CHECK(make_pipeline(dev, rp, pl, vs, fs, dk, &dpipe[c]));
      CHECK(make_pipeline(dev, rp, pl, vs, fs, rk, &rpipe[c]));
   }

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0,
   };
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkClearValue clear = { .color = { .float32 = { 51.0f / 255.0f, 102.0f / 255.0f,
                                                  153.0f / 255.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } }, .clearValueCount = 1, .pClearValues = &clear,
   };
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   const float inst_dx = 0.15f;
   const uint32_t clear_px = 0xff996633u;

   if (argc > 4 && strcmp(argv[3], "--cut") == 0) {
      const uint32_t ndraw = (uint32_t)atoi(argv[4]);
      unsigned cc = 0;
      while (cc < NCASES && strcmp(cases[cc].name, "strip runs too short") != 0) {
         cc++;
      }
      if (cc == NCASES || nref[cc] != 0) {
         printf("FAILED (iaasm: no fully cut case)\n");
         return 1;
      }
      const VkDeviceSize zero = 0;
      CHECK(vkBeginCommandBuffer(cmd, &cbbi));
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inst_dx, &inst_dx);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dpipe[cc]);
      vkCmdBindVertexBuffers(cmd, 0, 1, &tbuf, &zero);
      vkCmdBindIndexBuffer(cmd, ibuf, idx_slot * cc, index_type(cases[cc].isz));
      for (uint32_t d = 0; d < ndraw; d++) {
         vkCmdDrawIndexed(cmd, npos[cc], cases[cc].instances, cases[cc].first, cases[cc].base,
                          cases[cc].first_instance);
      }
      vkCmdEndRenderPass(cmd);
      VkBufferImageCopy region = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { WIDTH, HEIGHT, 1 },
      };
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pxbuf, 1, &region);
      CHECK(vkEndCommandBuffer(cmd));
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));
      uint32_t drawn = 0;
      for (uint32_t p = 0; p < WIDTH * HEIGHT; p++) {
         uint32_t g;
         memcpy(&g, pxmap + 4u * p, 4);
         drawn += g != clear_px;
      }
      free(idx);
      free(list);
      vkDestroyDevice(dev, NULL);
      vkDestroyInstance(inst, NULL);
      if (drawn) {
         printf("FAILED (iaasm --cut: %u px drawn by draws that assemble nothing)\n", drawn);
         return 1;
      }
      printf("PASSED (iaasm --cut: %u fully cut restart draws drew nothing)\n", ndraw);
      return 0;
   }

   /* --- images ---------------------------------------------------------------- */
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   for (unsigned c = 0; c < NCASES; c++) {
      const struct ia_case *k = &cases[c];
      for (unsigned pass = 0; pass < 2; pass++) {
         vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
         vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inst_dx, &inst_dx);
         if (pass == 0) {
            const VkDeviceSize zero = 0;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dpipe[c]);
            vkCmdBindVertexBuffers(cmd, 0, 1, &tbuf, &zero);
            if (k->isz != IDX_NONE) {
               vkCmdBindIndexBuffer(cmd, ibuf, idx_slot * c, index_type(k->isz));
               vkCmdDrawIndexed(cmd, npos[c], k->instances, k->first, k->base, k->first_instance);
            } else {
               vkCmdDraw(cmd, npos[c], k->instances, k->first, k->first_instance);
            }
         } else if (nref[c]) {
            const VkDeviceSize off = ref_slot * c;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rpipe[c]);
            vkCmdBindVertexBuffers(cmd, 0, 1, &rbuf, &off);
            vkCmdDraw(cmd, nref[c], k->instances, 0, k->first_instance);
         }
         vkCmdEndRenderPass(cmd);
         VkBufferImageCopy region = {
            .bufferOffset = (VkDeviceSize)IMG_BYTES * (2u * c + pass),
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { WIDTH, HEIGHT, 1 },
         };
         vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pxbuf, 1, &region);
      }
   }
   CHECK(vkEndCommandBuffer(cmd));
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned failed = 0;
   for (unsigned c = 0; c < NCASES; c++) {
      const uint8_t *got = pxmap + (size_t)IMG_BYTES * 2u * c;
      const uint8_t *want = got + IMG_BYTES;
      uint32_t diff = 0, drawn = 0;
      for (uint32_t p = 0; p < WIDTH * HEIGHT; p++) {
         uint32_t g, w;
         memcpy(&g, got + 4u * p, 4);
         memcpy(&w, want + 4u * p, 4);
         diff += g != w;
         drawn += w != clear_px;
      }
      /* A case drawing nothing must agree with a reference drawing nothing. */
      const bool ok = diff == 0 && (drawn > 0 || nref[c] == 0);
      printf("%-24s %s  %u primitives, %u px drawn, %u px differ\n", cases[c].name,
             ok ? "pass" : "FAIL", nref[c] / per_prim(cases[c].topo), drawn, diff);
      failed += !ok;
   }

   /* --- statistics --------------------------------------------------------------
    * The restart cases' instanced strip, rasterized and under rasterizer
    * discard: the input assembler made the reference's primitives each time. */
   const unsigned sc = 6;   /* "strip u8 restart x3" */
   const uint64_t want_prims = (uint64_t)(nref[sc] / 3u) * cases[sc].instances;
   VkQueryPoolCreateInfo qpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS, .queryCount = 2,
      .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT,
   };
   VkQueryPool qp;
   CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
   VkPipeline discard_pipe;
   {
      const struct ia_case *k = &cases[sc];
      const struct pipe_key dk = { k->topo, k->restart, k->last, k->cull, true };
      CHECK(make_pipeline(dev, rp, pl, vs, fs, dk, &discard_pipe));
   }
   CHECK(vkResetCommandBuffer(cmd, 0));
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   vkCmdResetQueryPool(cmd, qp, 0, 2);
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof inst_dx, &inst_dx);
   {
      const VkDeviceSize zero = 0;
      const struct ia_case *k = &cases[sc];
      vkCmdBindVertexBuffers(cmd, 0, 1, &tbuf, &zero);
      vkCmdBindIndexBuffer(cmd, ibuf, idx_slot * sc, index_type(k->isz));
      for (unsigned q = 0; q < 2; q++) {
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, q ? discard_pipe : dpipe[sc]);
         vkCmdBeginQuery(cmd, qp, q, 0);
         vkCmdDrawIndexed(cmd, npos[sc], k->instances, k->first, k->base, k->first_instance);
         vkCmdEndQuery(cmd, qp, q);
      }
   }
   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   uint64_t stats[2][2] = { { 0 } };
   CHECK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof stats, stats, sizeof stats[0],
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   for (unsigned q = 0; q < 2; q++) {
      /* Clipping counts only what reaches it; under discard that is implementation-defined. */
      const bool ok = stats[q][0] == want_prims && (q == 1 || stats[q][1] == want_prims);
      printf("statistics %-13s %s  input-assembly primitives %llu, clipping invocations %llu, "
             "want %llu\n", q ? "discard" : "rasterized", ok ? "pass" : "FAIL",
             (unsigned long long)stats[q][0], (unsigned long long)stats[q][1],
             (unsigned long long)want_prims);
      failed += !ok;
   }

   free(idx);
   free(list);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (failed) {
      printf("FAILED (%u check(s))\n", failed);
      return 1;
   }
   printf("PASSED (input assembly: %u cases of index widths, strips, fans, lists and "
          "restarts, instanced, both provoking modes, culled; restart statistics)\n",
          (unsigned)NCASES);
   return 0;
}
