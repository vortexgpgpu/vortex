/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Transform feedback (VK_EXT_transform_feedback) from the vertex stage.
 *
 * Every captured value is exact in fp32 (xfb.vert) -- vectors, a matrix, an
 * array, a struct and component-packed floats, most read by no
 * fragment input -- so the expected contents of
 * both buffers are rebuilt on the host from the vertex and instance index of
 * each captured vertex, in the order the specification gives: primitives in
 * draw order, instances one after another, strips and fans split into
 * triangles in their winding order, rotated so the provoking vertex comes first
 * or last. Both buffers start filled with a sentinel, and every byte is
 * compared: the holes inside a vertex and everything past the last captured
 * vertex must still hold it.
 *
 *   topology  list, strip and fan in both provoking-vertex modes, an indexed
 *             strip cut by primitive restart; that strip, then a restart draw
 *             whose runs are all too short to make a triangle (it captures
 *             nothing, and must still leave the filled sizes where the strip
 *             left them), then the strip instanced from instance 1, all in one
 *             capture; the too-short draw on its own, an
 *             instanced list starting at instance 1, an instanced strip,
 *             points, a line strip and a line list, and the four adjacency
 *             topologies -- the triangle strip in both modes -- of which only
 *             each primitive's own vertices are captured, never the adjacent
 *             ones; each in its own
 *             Begin/End over its own buffer ranges, the filled sizes saved to
 *             counter buffers at End
 *   overflow  four triangles into ranges with room for three in buffer 0 and
 *             one word short of three in buffer 1: two are written to both,
 *             none of the rest anywhere; points drawn next still fit and are
 *             written after them. A primitive with too little space left in
 *             any buffer is recorded in none and not counted as written
 *             (vertexpostproc.adoc, "Transform Feedback Capture"); the rule
 *             is per primitive, so a later one that fits is both.
 *   empty     buffer 1 bound with size 0: no primitive fits, so nothing is
 *             recorded in either buffer, and the draw still runs
 *   many      4608 vertices, more than one launch of every configuration has
 *             threads, with both faces culled: the vertex stage runs alone
 *             and its capture is still complete
 *   huge      33000 points under rasterizer discard, more primitives than a
 *             rasterizer pass takes: the draw runs the vertex stage alone and
 *             captures every point
 *   unbound   buffer 1 not bound: buffer 0 is still captured
 *   resume    three segments in three command buffers, each resuming from the
 *             counter buffer the previous one saved
 *   first     a counter saved and restored for buffer 1 alone
 *             (firstCounterBuffer 1): buffer 1 resumes, buffer 0 restarts at 0
 *   discard   points, lines and triangles under rasterizer discard, in a pass
 *             of their own that must keep its clear colour
 *   query     a stream query around a draw outside capture and the overflow
 *             segment (written 4, needed 6), read 64-bit with
 *             vkGetQueryPoolResults and vkCmdCopyQueryPoolResults, which must
 *             agree; a second around one list case read 32-bit; and a
 *             primitives-generated query around the overflow segment
 *             (generated 7, the outside draw included)
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     32u
#define HEIGHT    32u
#define FORMAT    VK_FORMAT_R8G8B8A8_UNORM
#define SENTINEL  0xDEADBEEFu
#define STRIDE0   96u
#define STRIDE1   24u
/* Each case captures into its own 4 KiB range of each buffer. */
#define SLICE     4096u
#define NSLICES   26u
/* The many-vertex and huge cases capture into buffers of their own. */
#define NBIG      4608u
#define NHUGE     33000u
#define BUF_BYTES (SLICE * NSLICES)

#define CHECK(x) do {                                                    \
   VkResult _r = (x);                                                    \
   if (_r != VK_SUCCESS) {                                               \
      printf("FAILED (xfb: %s -> VkResult %d)\n", #x, (int)_r);          \
      return 1;                                                          \
   }                                                                     \
} while (0)

/* ---- expected capture ------------------------------------------------ */

struct vtx { uint32_t v, i; };

static void
rec0(uint8_t *b, struct vtx x)
{
   const float a[2] = { (float)x.v, (float)x.i + 0.25f };
   const uint32_t id = x.v * 16u + x.i;
   const float only = (float)x.v + 0.5f;
   memcpy(b + 4, a, sizeof a);
   memcpy(b + 16, &id, 4);
   memcpy(b + 24, &only, 4);
   const float m[4] = { (float)x.v, (float)x.v + 1.0f, (float)x.i, -(float)x.v };
   const float arr[2] = { 2.0f * (float)x.v, 3.0f * (float)x.v + (float)x.i };
   memcpy(b + 32, m, sizeof m);
   memcpy(b + 52, arr, sizeof arr);
   const float blk[4] = { 5.0f * (float)x.v, 5.0f * (float)x.v + 1.0f, 7.0f * (float)x.i,
                          (float)x.v - (float)x.i };
   memcpy(b + 64, blk, sizeof blk);
}

static void
rec1(uint8_t *b, struct vtx x)
{
   const float r[6] = {
      -0.75f + 0.0625f * (float)(x.v % 16u) + 0.03125f * (float)x.i,
      -0.75f + 0.25f * (float)(x.v / 16u), 0.5f, 1.0f,
      1.0f + (float)(x.v % 4u), 2.0f + (float)x.v,
   };
   memcpy(b, r, sizeof r);
}

/* A draw as the input assembler sees it. */
struct draw {
   VkPrimitiveTopology topo;
   bool last;                /* provoking vertex last */
   uint32_t count;           /* vertices, or indices when idx is set */
   const uint16_t *idx;      /* restart value 0xffff cuts a strip */
   uint32_t first_vertex;
   uint32_t instances, first_instance;
   uint32_t first_index;     /* of idx in the index buffer */
};

/* The captured vertices of `d` in capture order; returns how many. */
static uint32_t
expect_draw(const struct draw *d, struct vtx *out)
{
   uint32_t n = 0;
   for (uint32_t inst = 0; inst < d->instances; inst++) {
      const uint32_t i = d->first_instance + inst;
      uint32_t run = 0;   /* first position of the current strip/fan run */
      for (uint32_t p = 0; p < d->count; p++) {
         if (d->idx && d->idx[p] == 0xffffu) {
            run = p + 1;
            continue;
         }
         const uint32_t k = p - run;   /* position within the run */
#define V(pos) ((struct vtx){ d->idx ? d->idx[pos] : d->first_vertex + (pos), i })
         switch (d->topo) {
         case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
            out[n++] = V(p);
            break;
         case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
            if (k % 2u == 1u) {
               out[n++] = V(p - 1);
               out[n++] = V(p);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
            if (k >= 1) {
               out[n++] = V(p - 1);
               out[n++] = V(p);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
            if (k % 3u == 2u) {
               out[n++] = V(p - 2);
               out[n++] = V(p - 1);
               out[n++] = V(p);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: {
            if (k < 2) {
               break;
            }
            const uint32_t t = k - 2;   /* triangle (t, t+1+t%2, t+2-t%2) of the run */
            if (!(t & 1u)) {
               out[n++] = V(p - 2); out[n++] = V(p - 1); out[n++] = V(p);
            } else if (!d->last) {
               out[n++] = V(p - 2); out[n++] = V(p); out[n++] = V(p - 1);
            } else {
               out[n++] = V(p - 1); out[n++] = V(p - 2); out[n++] = V(p);
            }
            break;
         }
         case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:   /* (a, 1, 2, b) */
            if (k % 4u == 3u) {
               out[n++] = V(p - 2);
               out[n++] = V(p - 1);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:  /* segment (t+1, t+2) */
            if (k >= 3) {
               out[n++] = V(p - 2);
               out[n++] = V(p - 1);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:   /* 0, 2, 4 of six */
            if (k % 6u == 5u) {
               out[n++] = V(p - 5); out[n++] = V(p - 3); out[n++] = V(p - 1);
            }
            break;
         case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: {
            /* Triangle t closes at position 2t+5; its own vertices are 2t,
             * 2t+2 and 2t+4, an odd one wound back like a strip's. */
            if (k < 5 || (k - 5u) % 2u) {
               break;
            }
            const uint32_t t = (k - 5u) / 2u, a = p - 5, b = p - 3, c = p - 1;
            if (!(t & 1u)) {
               out[n++] = V(a); out[n++] = V(b); out[n++] = V(c);
            } else if (!d->last) {
               out[n++] = V(a); out[n++] = V(c); out[n++] = V(b);
            } else {
               out[n++] = V(b); out[n++] = V(a); out[n++] = V(c);
            }
            break;
         }
         case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
            if (k < 2) {
               break;
            }
            if (!d->last) {   /* (t+1, t+2, 0) */
               out[n++] = V(p - 1); out[n++] = V(p); out[n++] = V(run);
            } else {          /* (0, t+1, t+2) */
               out[n++] = V(run); out[n++] = V(p - 1); out[n++] = V(p);
            }
            break;
         default:
            break;
         }
#undef V
      }
   }
   return n;
}

/* Write vertices [0, n) at byte `at0` of buffer 0 and `at1` of buffer 1. */
static void
expect_put(uint8_t *e0, uint8_t *e1, uint32_t at0, uint32_t at1,
           const struct vtx *x, uint32_t n)
{
   for (uint32_t k = 0; k < n; k++) {
      if (e0) {
         rec0(e0 + at0 + k * STRIDE0, x[k]);
      }
      if (e1) {
         rec1(e1 + at1 + k * STRIDE1, x[k]);
      }
   }
}

static uint32_t
first_diff(const uint8_t *got, const uint8_t *want, uint32_t bytes)
{
   for (uint32_t b = 0; b < bytes; b++) {
      if (got[b] != want[b]) {
         return b;
      }
   }
   return UINT32_MAX;
}

/* ---- Vulkan helpers -------------------------------------------------- */

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
host_buffer(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
            VkDeviceSize bytes, VkBufferUsageFlags usage,
            VkBuffer *out_buf, VkDeviceMemory *out_mem, void **out_map)
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
   uint32_t mt = find_mem(mp, mr.memoryTypeBits,
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
   bool last, restart, discard, cull_all;
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
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = k.topo, .primitiveRestartEnable = k.restart,
   };
   VkViewport vp = { 0, 0, (float)WIDTH, (float)HEIGHT, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { WIDTH, HEIGHT } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc,
   };
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
      .provokingVertexMode = k.last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                    : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = &pv,
      .rasterizerDiscardEnable = k.discard,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = k.cull_all ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
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
   return vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, out);
}

static bool
has_device_extension(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   if (vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS) {
      return false;
   }
   VkExtensionProperties *exts = calloc(n, sizeof(*exts));
   bool found = false;
   if (exts && vkEnumerateDeviceExtensionProperties(pd, NULL, &n, exts) == VK_SUCCESS) {
      for (uint32_t i = 0; i < n && !found; i++) {
         found = strcmp(exts[i].extensionName, name) == 0;
      }
   }
   free(exts);
   return found;
}

/* ---- cases ------------------------------------------------------------ */

/* The index buffer: a strip cut once, then one whose every run is too short. */
static const uint16_t index_data[] = {
   0, 1, 2, 3, 0xffff, 4, 5, 6, 7,
   0, 1, 0xffff, 2, 3, 0xffff, 0xffff, 4,
};
#define RESTART_IDX index_data
#define SHORT_IDX   (index_data + 9)

enum {
   C_LIST, C_STRIP_F, C_STRIP_L, C_FAN_F, C_FAN_L, C_RESTART, C_INST, C_POINTS,
   C_LINES, C_INST_STRIP, C_LINE_LIST, C_RESTART_INST, C_RESTART_SHORT, C_LINES_ADJ,
   C_LINE_STRIP_ADJ, C_TRIS_ADJ, C_TRI_STRIP_ADJ_F, C_TRI_STRIP_ADJ_L, NTOPO,
};

static const char *const topo_names[NTOPO] = {
   "list", "strip", "strip-last", "fan", "fan-last", "restart", "instanced",
   "points", "line-strip", "instanced-strip", "line-list", "restart-instanced",
   "restart-short", "line-list-adjacency", "line-strip-adjacency", "triangle-list-adjacency",
   "triangle-strip-adjacency", "triangle-strip-adjacency-last",
};

static const struct draw topo_draws[NTOPO] = {
   [C_LIST]    = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 6, NULL, 0, 1, 0 },
   [C_STRIP_F] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 6, NULL, 0, 1, 0 },
   [C_STRIP_L] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, true, 6, NULL, 0, 1, 0 },
   [C_FAN_F]   = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, false, 5, NULL, 0, 1, 0 },
   [C_FAN_L]   = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, true, 5, NULL, 0, 1, 0 },
   [C_RESTART] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 9, RESTART_IDX, 0, 1, 0 },
   [C_INST]    = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 3, NULL, 2, 2, 1 },
   [C_POINTS]  = { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, 4, NULL, 0, 1, 0 },
   [C_LINES]   = { VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, false, 4, NULL, 0, 1, 0 },
   [C_INST_STRIP] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 5, NULL, 0, 3, 0 },
   [C_LINE_LIST]  = { VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, 6, NULL, 0, 1, 0 },
   [C_RESTART_INST]  = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 9, RESTART_IDX, 0, 2, 1 },
   [C_RESTART_SHORT] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 8, SHORT_IDX, 0, 1, 0, 9 },
   [C_LINES_ADJ]  = { VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY, false, 8, NULL, 0, 1, 0 },
   [C_LINE_STRIP_ADJ] = { VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, false, 6, NULL, 0, 1, 0 },
   [C_TRIS_ADJ]   = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY, false, 12, NULL, 0, 1, 0 },
   [C_TRI_STRIP_ADJ_F] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, false, 10, NULL,
                           0, 1, 0 },
   [C_TRI_STRIP_ADJ_L] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, true, 10, NULL,
                           0, 1, 0 },
};

/* Slices of the capture buffers. */
enum { S_OVF = NTOPO, S_UNBOUND, S_RESUME, S_DISCARD, S_FIRST, S_EMPTY, S_MANY, S_HUGE };

/* Counter-buffer words: two per slice (buffer 0, buffer 1). */
#define CTR(slice, b) (((slice) * 2u + (b)) * 4u)

static const struct draw ovf_tris   = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 12, NULL, 0, 1, 0 };
static const struct draw ovf_points = { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, 2, NULL, 20, 1, 0 };
static const struct draw outside    = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 3, NULL, 40, 1, 0 };
/* Room for three triangles in buffer 0, one word short of three in buffer 1. */
#define OVF_SIZE0 (9u * STRIDE0)
#define OVF_SIZE1 (9u * STRIDE1 - 4u)

static const struct draw resume_draws[3] = {
   { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 3, NULL, 0, 1, 0 },
   { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false, 4, NULL, 3, 1, 0 },
   { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, 2, NULL, 7, 1, 0 },
};

static const struct draw many_tris = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, NBIG, NULL, 0, 1, 0 };
static const struct draw huge_points = { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, NHUGE, NULL, 0, 1, 0 };

static const struct draw discard_draws[3] = {
   { VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, 3, NULL, 0, 1, 0 },
   { VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, false, 3, NULL, 3, 1, 0 },
   { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, 3, NULL, 6, 1, 0 },
};

static struct pipe_key
key_of(const struct draw *d, bool discard, bool cull_all)
{
   return (struct pipe_key){ d->topo, d->last, d->idx != NULL, discard, cull_all };
}

int
main(int argc, char **argv)
{
   const char *vs_path = (argc > 1) ? argv[1] : "xfb.vert.spv";
   const char *fs_path = (argc > 2) ? argv[2] : "xfb.frag.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-xfb", .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) {
      printf("FAILED (xfb: no physical device)\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   const char *exts[] = {
      VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
      VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
      VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME,
   };
   for (unsigned e = 0; e < 3; e++) {
      if (!has_device_extension(pd, exts[e])) {
         printf("FAILED (xfb: %s not advertised)\n", exts[e]);
         return 1;
      }
   }
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDeviceProvokingVertexFeaturesEXT pv_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
      .pNext = &xfb_have,
   };
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq_have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
      .pNext = &pv_have,
   };
   VkPhysicalDeviceFeatures2 have = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &pgq_have,
   };
   vkGetPhysicalDeviceFeatures2(pd, &have);
   VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProvokingVertexPropertiesEXT pv_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT,
      .pNext = &xfb_props,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pv_props,
   };
   vkGetPhysicalDeviceProperties2(pd, &props2);
   if (!xfb_have.transformFeedback || !pv_have.provokingVertexLast ||
       !pv_have.transformFeedbackPreservesProvokingVertex ||
       !pgq_have.primitivesGeneratedQuery || !have.features.shaderClipDistance ||
       !have.features.geometryShader ||
       !xfb_props.transformFeedbackQueries || xfb_props.maxTransformFeedbackBuffers < 2 ||
       !pv_props.transformFeedbackPreservesTriangleFanProvokingVertex) {
      printf("FAILED (xfb: a required feature is missing: transformFeedback %u, "
             "provokingVertexLast %u, preserves provoking %u/%u, queries %u, "
             "buffers %u, geometryShader %u)\n", xfb_have.transformFeedback,
             pv_have.provokingVertexLast,
             pv_have.transformFeedbackPreservesProvokingVertex,
             pv_props.transformFeedbackPreservesTriangleFanProvokingVertex,
             xfb_props.transformFeedbackQueries, xfb_props.maxTransformFeedbackBuffers,
             have.features.geometryShader);
      return 1;
   }
   /* Without geometry streams Vulkan requires exactly one stream. */
   if (!xfb_have.geometryStreams && xfb_props.maxTransformFeedbackStreams != 1) {
      printf("FAILED (xfb: maxTransformFeedbackStreams %u without geometryStreams, "
             "must be 1)\n", xfb_props.maxTransformFeedbackStreams);
      return 1;
   }

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf && qf == UINT32_MAX; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qf = i;
      }
   }
   free(qfp);
   if (qf == UINT32_MAX) {
      printf("FAILED (xfb: no graphics queue)\n");
      return 1;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
      .transformFeedback = VK_TRUE,
   };
   VkPhysicalDeviceProvokingVertexFeaturesEXT pv_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
      .pNext = &xfb_on, .provokingVertexLast = VK_TRUE,
      .transformFeedbackPreservesProvokingVertex = VK_TRUE,
   };
   VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT pgq_on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
      .pNext = &pv_on, .primitivesGeneratedQuery = VK_TRUE,
   };
   /* The adjacency topologies need geometryShader, even drawn without one. */
   VkPhysicalDeviceFeatures feats = { .shaderClipDistance = VK_TRUE, .geometryShader = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &pgq_on,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts,
      .pEnabledFeatures = &feats,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);

   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)
      vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT");
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb = (PFN_vkCmdBeginTransformFeedbackEXT)
      vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT");
   PFN_vkCmdEndTransformFeedbackEXT end_xfb = (PFN_vkCmdEndTransformFeedbackEXT)
      vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT");
   if (!bind_xfb || !begin_xfb || !end_xfb) {
      printf("FAILED (xfb: transform-feedback entry points missing)\n");
      return 1;
   }

   VkShaderModule vs = load_module(dev, vs_path);
   VkShaderModule fs = load_module(dev, fs_path);
   if (!vs || !fs) {
      printf("FAILED (xfb: cannot load %s / %s)\n", vs_path, fs_path);
      return 1;
   }

   /* --- colour target, one render pass ------------------------------- */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = FORMAT, .extent = { WIDTH, HEIGHT, 1 },
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = FORMAT,
      .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .levelCount = 1, .layerCount = 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   VkAttachmentDescription att = {
      .format = FORMAT, .samples = VK_SAMPLE_COUNT_1_BIT,
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
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp, .attachmentCount = 1, .pAttachments = &view,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkPipeline topo_pipe[NTOPO], ovf_pipe[2], outside_pipe, resume_pipe[3], discard_pipe[3];
   VkPipeline many_pipe, huge_pipe;
   CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&many_tris, false, true), &many_pipe));
   CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&huge_points, true, false), &huge_pipe));
   for (unsigned c = 0; c < NTOPO; c++) {
      CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&topo_draws[c], false, false), &topo_pipe[c]));
   }
   CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&ovf_tris, false, false), &ovf_pipe[0]));
   CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&ovf_points, false, false), &ovf_pipe[1]));
   CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&outside, false, false), &outside_pipe));
   for (unsigned c = 0; c < 3; c++) {
      CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&resume_draws[c], false, false), &resume_pipe[c]));
      CHECK(make_pipeline(dev, rp, pl, vs, fs, key_of(&discard_draws[c], true, false), &discard_pipe[c]));
   }

   /* --- buffers -------------------------------------------------------- */
   const VkBufferUsageFlags xfb_usage = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
   VkBuffer xbuf[2], ctrbuf, ibuf, pxbuf, qbuf;
   VkDeviceMemory xmem[2], ctrmem, imem2, pxmem, qmem;
   uint8_t *xmap[2], *ctrmap, *imap, *pxmap, *qmap;
   for (unsigned b = 0; b < 2; b++) {
      CHECK(host_buffer(dev, &mp, BUF_BYTES, xfb_usage, &xbuf[b], &xmem[b], (void **)&xmap[b]));
      for (uint32_t w = 0; w < BUF_BYTES / 4u; w++) {
         ((uint32_t *)xmap[b])[w] = SENTINEL;
      }
   }
   /* The many-vertex and huge captures, each with a sentinel tail past its
    * last vertex. */
   const struct draw *const big_draws[2] = { &many_tris, &huge_points };
   const uint32_t big_slices[2] = { S_MANY, S_HUGE };
   const char *const big_names[2] = { "many", "huge" };
   uint32_t big_bytes[2][2];
   VkBuffer bigbuf[2][2];
   VkDeviceMemory bigmem[2][2];
   uint8_t *bigmap[2][2];
   for (unsigned c = 0; c < 2; c++) {
      big_bytes[c][0] = big_draws[c]->count * STRIDE0 + 256u;
      big_bytes[c][1] = big_draws[c]->count * STRIDE1 + 256u;
      for (unsigned b = 0; b < 2; b++) {
         CHECK(host_buffer(dev, &mp, big_bytes[c][b], xfb_usage, &bigbuf[c][b], &bigmem[c][b],
                           (void **)&bigmap[c][b]));
         for (uint32_t w = 0; w < big_bytes[c][b] / 4u; w++) {
            ((uint32_t *)bigmap[c][b])[w] = SENTINEL;
         }
      }
   }
   const uint32_t ctr_bytes = NSLICES * 8u;
   CHECK(host_buffer(dev, &mp, ctr_bytes, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
                     &ctrbuf, &ctrmem, (void **)&ctrmap));
   memset(ctrmap, 0xff, ctr_bytes);
   CHECK(host_buffer(dev, &mp, sizeof index_data, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     &ibuf, &imem2, (void **)&imap));
   memcpy(imap, index_data, sizeof index_data);
   const VkDeviceSize px_bytes = (VkDeviceSize)WIDTH * HEIGHT * 4u;
   CHECK(host_buffer(dev, &mp, px_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &pxbuf, &pxmem,
                     (void **)&pxmap));
   /* Query copies: the stream query 64-bit (written, needed), then the 32-bit
    * one, then the primitives-generated query. */
   const VkDeviceSize q_bytes = 64u;
   CHECK(host_buffer(dev, &mp, q_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &qbuf, &qmem,
                     (void **)&qmap));
   memset(qmap, 0xff, q_bytes);

   VkQueryPoolCreateInfo xqpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, .queryCount = 2,
   };
   VkQueryPool xqp;
   CHECK(vkCreateQueryPool(dev, &xqpci, NULL, &xqp));
   VkQueryPoolCreateInfo pqpci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT, .queryCount = 1,
   };
   VkQueryPool pqp;
   CHECK(vkCreateQueryPool(dev, &pqpci, NULL, &pqp));

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cmpci, NULL, &cp));
   VkCommandBuffer cmd[5];
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 5,
   };
   CHECK(vkAllocateCommandBuffers(dev, &cbai, cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkClearValue black = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } };
   VkClearValue blue  = { .color = { .float32 = { 0.0f, 0.0f, 1.0f, 1.0f } } };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp, .framebuffer = fb, .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
      .clearValueCount = 1, .pClearValues = &black,
   };

#define BIND(cb, slice, n, size0, size1) do {                                     \
      const VkDeviceSize off_[2] = { (VkDeviceSize)(slice) * SLICE,               \
                                     (VkDeviceSize)(slice) * SLICE };             \
      const VkDeviceSize size_[2] = { (size0), (size1) };                         \
      bind_xfb((cb), 0, (n), xbuf, off_, size_);                                  \
   } while (0)
#define DRAW(cb, d) do {                                                          \
      if ((d)->idx) {                                                             \
         vkCmdBindIndexBuffer((cb), ibuf, 0, VK_INDEX_TYPE_UINT16);               \
         vkCmdDrawIndexed((cb), (d)->count, (d)->instances, (d)->first_index, 0,  \
                          (d)->first_instance);                                   \
      } else {                                                                    \
         vkCmdDraw((cb), (d)->count, (d)->instances, (d)->first_vertex,           \
                   (d)->first_instance);                                          \
      }                                                                           \
   } while (0)

   /* --- 1: every topology, overflow, unbound buffer, queries --------- */
   const VkBuffer ctrs[2] = { ctrbuf, ctrbuf };
   CHECK(vkBeginCommandBuffer(cmd[0], &cbbi));
   vkCmdResetQueryPool(cmd[0], xqp, 0, 2);
   vkCmdResetQueryPool(cmd[0], pqp, 0, 1);
   vkCmdBeginRenderPass(cmd[0], &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   for (unsigned c = 0; c < NTOPO; c++) {
      const VkDeviceSize ctr_off[2] = { CTR(c, 0), CTR(c, 1) };
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, topo_pipe[c]);
      BIND(cmd[0], c, 2, SLICE, SLICE);
      if (c == C_LIST) {
         vkCmdBeginQuery(cmd[0], xqp, 1, 0);
      }
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      if (c == C_RESTART_INST) {
         DRAW(cmd[0], &topo_draws[C_RESTART]);
         DRAW(cmd[0], &topo_draws[C_RESTART_SHORT]);
      }
      DRAW(cmd[0], &topo_draws[c]);
      end_xfb(cmd[0], 0, 2, ctrs, ctr_off);
      if (c == C_LIST) {
         vkCmdEndQuery(cmd[0], xqp, 1);
      }
   }
   {
      const VkDeviceSize ctr_off[2] = { CTR(S_OVF, 0), CTR(S_OVF, 1) };
      vkCmdBeginQuery(cmd[0], xqp, 0, 0);
      vkCmdBeginQuery(cmd[0], pqp, 0, 0);
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, outside_pipe);
      DRAW(cmd[0], &outside);
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, ovf_pipe[0]);
      BIND(cmd[0], S_OVF, 2, OVF_SIZE0, OVF_SIZE1);
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      DRAW(cmd[0], &ovf_tris);
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, ovf_pipe[1]);
      DRAW(cmd[0], &ovf_points);
      end_xfb(cmd[0], 0, 2, ctrs, ctr_off);
      vkCmdEndQuery(cmd[0], pqp, 0);
      vkCmdEndQuery(cmd[0], xqp, 0);
   }
   {
      const VkDeviceSize ctr_off[1] = { CTR(S_UNBOUND, 0) };
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, topo_pipe[C_LIST]);
      BIND(cmd[0], S_UNBOUND, 1, SLICE, SLICE);
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      DRAW(cmd[0], &topo_draws[C_LIST]);
      end_xfb(cmd[0], 0, 1, ctrs, ctr_off);
   }
   {
      const VkDeviceSize ctr_off[2] = { CTR(S_FIRST, 0), CTR(S_FIRST, 1) };
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, resume_pipe[0]);
      BIND(cmd[0], S_FIRST, 2, SLICE, SLICE);
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      DRAW(cmd[0], &resume_draws[0]);
      end_xfb(cmd[0], 1, 1, ctrs, &ctr_off[1]);
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, resume_pipe[2]);
      begin_xfb(cmd[0], 1, 1, ctrs, &ctr_off[1]);
      DRAW(cmd[0], &resume_draws[2]);
      end_xfb(cmd[0], 0, 2, ctrs, ctr_off);
   }
   {
      const VkDeviceSize ctr_off[2] = { CTR(S_EMPTY, 0), CTR(S_EMPTY, 1) };
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, topo_pipe[C_LIST]);
      BIND(cmd[0], S_EMPTY, 2, SLICE, 0);
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      DRAW(cmd[0], &topo_draws[C_LIST]);
      end_xfb(cmd[0], 0, 2, ctrs, ctr_off);
   }
   for (unsigned c = 0; c < 2; c++) {
      const VkDeviceSize ctr_off[2] = { CTR(big_slices[c], 0), CTR(big_slices[c], 1) };
      const VkDeviceSize zero[2] = { 0, 0 };
      const VkDeviceSize whole[2] = { VK_WHOLE_SIZE, VK_WHOLE_SIZE };
      vkCmdBindPipeline(cmd[0], VK_PIPELINE_BIND_POINT_GRAPHICS, c ? huge_pipe : many_pipe);
      bind_xfb(cmd[0], 0, 2, bigbuf[c], zero, whole);
      begin_xfb(cmd[0], 0, 0, NULL, NULL);
      DRAW(cmd[0], big_draws[c]);
      end_xfb(cmd[0], 0, 2, ctrs, ctr_off);
   }
   vkCmdEndRenderPass(cmd[0]);
   vkCmdCopyQueryPoolResults(cmd[0], xqp, 0, 1, qbuf, 0, 16,
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   vkCmdCopyQueryPoolResults(cmd[0], xqp, 1, 1, qbuf, 16, 8, VK_QUERY_RESULT_WAIT_BIT);
   vkCmdCopyQueryPoolResults(cmd[0], pqp, 0, 1, qbuf, 24, 8,
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   VkBufferImageCopy region = {
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd[0], img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pxbuf, 1, &region);
   CHECK(vkEndCommandBuffer(cmd[0]));

   /* --- 2-4: one segment each, resumed through the counter buffer ----- */
   for (unsigned s = 0; s < 3; s++) {
      VkCommandBuffer cb = cmd[1 + s];
      const VkDeviceSize ctr_off[2] = { CTR(S_RESUME, 0), CTR(S_RESUME, 1) };
      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, resume_pipe[s]);
      BIND(cb, S_RESUME, 2, SLICE, SLICE);
      begin_xfb(cb, 0, s ? 2u : 0u, s ? ctrs : NULL, s ? ctr_off : NULL);
      DRAW(cb, &resume_draws[s]);
      end_xfb(cb, 0, 2, ctrs, ctr_off);
      vkCmdEndRenderPass(cb);
      CHECK(vkEndCommandBuffer(cb));
   }

   /* --- 5: discard, in a pass of its own ------------------------------ */
   {
      VkRenderPassBeginInfo rpbi2 = rpbi;
      rpbi2.pClearValues = &blue;
      const VkDeviceSize ctr_off[2] = { CTR(S_DISCARD, 0), CTR(S_DISCARD, 1) };
      CHECK(vkBeginCommandBuffer(cmd[4], &cbbi));
      vkCmdBeginRenderPass(cmd[4], &rpbi2, VK_SUBPASS_CONTENTS_INLINE);
      BIND(cmd[4], S_DISCARD, 2, SLICE, SLICE);
      begin_xfb(cmd[4], 0, 0, NULL, NULL);
      for (unsigned s = 0; s < 3; s++) {
         vkCmdBindPipeline(cmd[4], VK_PIPELINE_BIND_POINT_GRAPHICS, discard_pipe[s]);
         DRAW(cmd[4], &discard_draws[s]);
      }
      end_xfb(cmd[4], 0, 2, ctrs, ctr_off);
      vkCmdEndRenderPass(cmd[4]);
      CHECK(vkEndCommandBuffer(cmd[4]));
   }

   /* The first pass's image is read back before the discard pass reuses it. */
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1 };
   si.pCommandBuffers = &cmd[0];
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   long painted = 0;
   for (uint32_t p = 0; p < WIDTH * HEIGHT; p++) {
      painted += pxmap[p * 4u] == 255u && pxmap[p * 4u + 2u] == 0u;
   }
   for (unsigned c = 1; c < 5; c++) {
      si.pCommandBuffers = &cmd[c];
      CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));
   }
   /* The discard pass leaves the clear colour in the attachment. */
   CHECK(vkResetCommandBuffer(cmd[0], 0));
   CHECK(vkBeginCommandBuffer(cmd[0], &cbbi));
   vkCmdCopyImageToBuffer(cmd[0], img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pxbuf, 1, &region);
   CHECK(vkEndCommandBuffer(cmd[0]));
   si.pCommandBuffers = &cmd[0];
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   long not_blue = 0;
   for (uint32_t p = 0; p < WIDTH * HEIGHT; p++) {
      not_blue += !(pxmap[p * 4u] == 0u && pxmap[p * 4u + 1u] == 0u && pxmap[p * 4u + 2u] == 255u);
   }

   uint64_t q64[3] = { 0 };
   uint32_t q32 = 0;
   CHECK(vkGetQueryPoolResults(dev, xqp, 0, 1, 16, q64, 16,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   uint32_t q32pair[2] = { 0 };
   CHECK(vkGetQueryPoolResults(dev, xqp, 1, 1, 8, q32pair, 8, VK_QUERY_RESULT_WAIT_BIT));
   CHECK(vkGetQueryPoolResults(dev, pqp, 0, 1, 8, &q64[2], 8,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   q32 = q32pair[0];
   uint64_t copied64[2], copied_pg;
   uint32_t copied32[2];
   memcpy(copied64, qmap, 16);
   memcpy(copied32, qmap + 16, 8);
   memcpy(&copied_pg, qmap + 24, 8);

   /* --- expected contents ----------------------------------------------- */
   uint8_t *want[2];
   uint32_t want_ctr[NSLICES][2];
   memset(want_ctr, 0xff, sizeof want_ctr);
   for (unsigned b = 0; b < 2; b++) {
      want[b] = malloc(BUF_BYTES);
      for (uint32_t w = 0; w < BUF_BYTES / 4u; w++) {
         ((uint32_t *)want[b])[w] = SENTINEL;
      }
   }
   struct vtx x[64];
   for (unsigned c = 0; c < NTOPO; c++) {
      uint32_t at = 0;
      if (c == C_RESTART_INST) {
         at = expect_draw(&topo_draws[C_RESTART], x);
         expect_put(want[0], want[1], c * SLICE, c * SLICE, x, at);
      }
      const uint32_t n = expect_draw(&topo_draws[c], x);
      expect_put(want[0], want[1], c * SLICE + at * STRIDE0, c * SLICE + at * STRIDE1, x, n);
      want_ctr[c][0] = (at + n) * STRIDE0;
      want_ctr[c][1] = (at + n) * STRIDE1;
   }
   {
      /* Two of the four triangles fit buffer 1; the points fit after them. */
      uint32_t n = expect_draw(&ovf_tris, x);
      expect_put(want[0], want[1], S_OVF * SLICE, S_OVF * SLICE, x, 6);
      (void)n;
      n = expect_draw(&ovf_points, x);
      expect_put(want[0], want[1], S_OVF * SLICE + 6u * STRIDE0, S_OVF * SLICE + 6u * STRIDE1,
                 x, n);
      want_ctr[S_OVF][0] = (6u + n) * STRIDE0;
      want_ctr[S_OVF][1] = (6u + n) * STRIDE1;
   }
   {
      const uint32_t n = expect_draw(&topo_draws[C_LIST], x);
      expect_put(want[0], NULL, S_UNBOUND * SLICE, 0, x, n);
      want_ctr[S_UNBOUND][0] = n * STRIDE0;
   }
   uint32_t at = 0;
   for (unsigned s = 0; s < 3; s++) {
      const uint32_t n = expect_draw(&resume_draws[s], x);
      expect_put(want[0], want[1], S_RESUME * SLICE + at * STRIDE0,
                 S_RESUME * SLICE + at * STRIDE1, x, n);
      at += n;
   }
   want_ctr[S_RESUME][0] = at * STRIDE0;
   want_ctr[S_RESUME][1] = at * STRIDE1;
   at = 0;
   for (unsigned s = 0; s < 3; s++) {
      const uint32_t n = expect_draw(&discard_draws[s], x);
      expect_put(want[0], want[1], S_DISCARD * SLICE + at * STRIDE0,
                 S_DISCARD * SLICE + at * STRIDE1, x, n);
      at += n;
   }
   want_ctr[S_DISCARD][0] = at * STRIDE0;
   want_ctr[S_DISCARD][1] = at * STRIDE1;
   {
      /* Buffer 0 restarts at 0 and keeps the list's last vertex past the
       * points; buffer 1 holds both draws back to back. */
      const uint32_t n0 = expect_draw(&resume_draws[0], x);
      expect_put(want[0], want[1], S_FIRST * SLICE, S_FIRST * SLICE, x, n0);
      const uint32_t n1 = expect_draw(&resume_draws[2], x);
      expect_put(want[0], NULL, S_FIRST * SLICE, 0, x, n1);
      expect_put(NULL, want[1], 0, S_FIRST * SLICE + n0 * STRIDE1, x, n1);
      want_ctr[S_FIRST][0] = n1 * STRIDE0;
      want_ctr[S_FIRST][1] = (n0 + n1) * STRIDE1;
   }
   want_ctr[S_EMPTY][0] = 0;
   want_ctr[S_EMPTY][1] = 0;
   want_ctr[S_MANY][0] = NBIG * STRIDE0;
   want_ctr[S_MANY][1] = NBIG * STRIDE1;
   want_ctr[S_HUGE][0] = NHUGE * STRIDE0;
   want_ctr[S_HUGE][1] = NHUGE * STRIDE1;

   /* --- compare -------------------------------------------------------- */
   static const char *const slice_names[NSLICES] = {
      [S_OVF] = "overflow", [S_UNBOUND] = "unbound", [S_RESUME] = "resume",
      [S_DISCARD] = "discard", [S_FIRST] = "first-counter", [S_EMPTY] = "empty",
      [S_MANY] = "many", [S_HUGE] = "huge",
   };
   unsigned bad = 0;
   for (unsigned sl = 0; sl < NSLICES; sl++) {
      const char *name = sl < NTOPO ? topo_names[sl] : slice_names[sl];
      for (unsigned b = 0; b < 2; b++) {
         const uint32_t d = first_diff(xmap[b] + sl * SLICE, want[b] + sl * SLICE, SLICE);
         if (d != UINT32_MAX) {
            const uint32_t stride = b ? STRIDE1 : STRIDE0;
            uint32_t gw, ww;
            memcpy(&gw, xmap[b] + sl * SLICE + (d & ~3u), 4);
            memcpy(&ww, want[b] + sl * SLICE + (d & ~3u), 4);
            printf("  %s: buffer %u differs first at byte %u (vertex %u, byte %u): "
                   "0x%08x, want 0x%08x\n", name ? name : "?", b, d, d / stride,
                   d % stride, gw, ww);
            bad++;
         }
         uint32_t ctr;
         memcpy(&ctr, ctrmap + CTR(sl, b), 4);
         if (ctr != want_ctr[sl][b]) {
            printf("  %s: buffer %u counter %u, want %u\n", name ? name : "?", b, ctr,
                   want_ctr[sl][b]);
            bad++;
         }
      }
   }
   for (unsigned c = 0; c < 2; c++) {
      struct vtx *bx = malloc(big_draws[c]->count * sizeof(*bx));
      uint8_t *bw[2] = { malloc(big_bytes[c][0]), malloc(big_bytes[c][1]) };
      for (unsigned b = 0; b < 2; b++) {
         for (uint32_t w = 0; w < big_bytes[c][b] / 4u; w++) {
            ((uint32_t *)bw[b])[w] = SENTINEL;
         }
      }
      const uint32_t n = expect_draw(big_draws[c], bx);
      expect_put(bw[0], bw[1], 0, 0, bx, n);
      for (unsigned b = 0; b < 2; b++) {
         const uint32_t d = first_diff(bigmap[c][b], bw[b], big_bytes[c][b]);
         if (d != UINT32_MAX) {
            const uint32_t stride = b ? STRIDE1 : STRIDE0;
            printf("  %s: buffer %u differs first at vertex %u, byte %u\n", big_names[c], b,
                   d / stride, d % stride);
            bad++;
         }
         free(bw[b]);
      }
      free(bx);
   }
   /* Outside capture nothing counts; the overflow segment needs 4 + 2 and
    * writes 2 + 2. The primitives-generated query counts all three draws. */
   const uint64_t want_q[2] = { 4u, 6u };
   if (q64[0] != want_q[0] || q64[1] != want_q[1]) {
      printf("  stream query: written %llu needed %llu, want %llu %llu\n",
             (unsigned long long)q64[0], (unsigned long long)q64[1],
             (unsigned long long)want_q[0], (unsigned long long)want_q[1]);
      bad++;
   }
   if (copied64[0] != q64[0] || copied64[1] != q64[1]) {
      printf("  stream query: copied %llu %llu, read %llu %llu\n",
             (unsigned long long)copied64[0], (unsigned long long)copied64[1],
             (unsigned long long)q64[0], (unsigned long long)q64[1]);
      bad++;
   }
   if (q32 != 2u || q32pair[1] != 2u || copied32[0] != 2u || copied32[1] != 2u) {
      printf("  32-bit stream query: read %u %u, copied %u %u, want 2 2\n", q32, q32pair[1],
             copied32[0], copied32[1]);
      bad++;
   }
   if (q64[2] != 7u || copied_pg != 7u) {
      printf("  primitives generated: read %llu, copied %llu, want 7\n",
             (unsigned long long)q64[2], (unsigned long long)copied_pg);
      bad++;
   }
   if (painted == 0) {
      printf("  the capturing draws painted no pixel\n");
      bad++;
   }
   if (not_blue) {
      printf("  discard pass: %ld of %u pixels are not the clear colour\n", not_blue,
             WIDTH * HEIGHT);
      bad++;
   }

   for (unsigned b = 0; b < 2; b++) {
      free(want[b]);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   if (bad) {
      printf("FAILED (xfb: %u checks wrong)\n", bad);
      return 1;
   }
   printf("PASSED (xfb: %u topologies, overflow, unbound buffer, resume, firstCounterBuffer, "
          "empty range, %u vertices, discard and queries byte-exact; %ld px painted)\n",
          (unsigned)NTOPO, NBIG, painted);
   return 0;
}
