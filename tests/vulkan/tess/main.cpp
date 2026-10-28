/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Tessellation on the device.
 *
 * Every case draws patches through a control and an evaluation shader into a
 * cleared 64x64 target, and compares the image pixel for pixel with the same
 * primitives drawn as a plain list. The host makes that list: it tessellates
 * each patch with gfx_tess.h -- the tessellator the device runs, which
 * tests/regression/gfx_tess_kernel checks bit for bit against the reference
 * llvmpipe runs -- evaluates the evaluation shader's arithmetic, which is exact,
 * and draws the result through the vertex stage alone. The two images agree
 * only when the device made the same points, the same primitives in the same
 * winding and, under blending, in the same order.
 *
 *   static_*   levels known when the pipeline is made (specialization
 *              constants): triangles, quads and isolines in every spacing,
 *              both windings, and point mode
 *   dynamic_*  levels from each patch's first control point, different per
 *              patch, including a patch every level of which is 1, and
 *              patches a zero, negative or NaN outer level discards
 *   pass       a control shader that writes per-vertex and per-patch outputs
 *              and reads its neighbours' after barrier()
 *   tri_in     three input control points, four output ones
 *   order      overlapping patches blended in order, coloured by
 *              gl_PrimitiveID, over two instances
 *   many       64 patches of varying levels
 *   gs_bound   a geometry shader of 32 invocations, each with room for 126
 *              triangles, after 160 instances of a patch with levels known
 *              only on the device (all but the last discarded): a draw-wide
 *              bound past 2^32 primitives, far from what the draw makes
 *   gs_*       a geometry shader after the tessellation stages, passing each
 *              triangle, line or point on; gs_wide with evaluation records
 *              wider than the geometry stage's, over many domain points
 *   xfb_*      the evaluation stage's output captured by transform feedback,
 *              compared value for value with the reference list in order, and
 *              with room for one primitive fewer than the draw makes
 *   barrier*   rounds of neighbour reads between barriers, at the top level
 *              and in a loop
 *   cp16/cp32  16 and 32 control points in and out
 *   stats_*    the control and evaluation stages' pipeline statistics, with
 *              levels known on the host and on the device, and with constant
 *              outer levels but no inner ones, nothing rasterized and through
 *              a geometry stage
 *   batches_*  draws in many batches and ranges of a batch, through a small
 *              VORTEXPIPE_TS_BATCH_BYTES: levels on the host, growing levels
 *              on the device, a capture, instances
 *   multiview  one draw seen by two views
 *   handoff    a capture across draws on the device, llvmpipe and the device
 *              again. Run on its own (-c handoff), with STRICT 0.
 *   levels_max 2048 patches at level 64, every level known only on the device,
 *              through 32-vec4 evaluation records: 8.65M domain points and
 *              4.7 GB of records, more than the device holds at once, each
 *              point marking its own slot of a storage buffer with nothing
 *              rasterized. Run on its own (-c levels_max).
 *
 * The fragment shader halves the colour of a back-facing triangle, so a
 * triangle wound the wrong way changes the image.
 *
 * -c <name> runs one case. TESS_BITE=1 builds every expectation the
 * way a wrong implementation would draw it (the other winding, the last
 * primitive of each patch dropped, points moved by a pixel); every case must
 * then fail.
 */

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gfx_tess.h"

#define W 64u
#define H 64u

#define CHECK(x) do {                                                     \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      printf("FAILED (tess: %s -> VkResult %d)\n", #x, (int)_r);           \
      exit(1);                                                            \
   }                                                                      \
} while (0)

static bool bite;

/* snprintf that fails the test when the text does not fit. */
static void
format(char *dst, size_t n, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   const int r = vsnprintf(dst, n, fmt, ap);
   va_end(ap);
   if (r < 0 || (size_t)r >= n) {
      printf("FAILED (tess: name \"%s\" does not fit)\n", fmt);
      exit(1);
   }
}

/* ---- Vulkan ---------------------------------------------------------- */

struct cp_vertex {
   float x, y;
   float lo[4];
   float li[2];
};

struct ref_vertex {
   float x, y;
   float col[4];
};

static struct {
   VkInstance inst;
   VkPhysicalDevice phys;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
   VkCommandPool pool;
   VkImage img;
   VkDeviceMemory img_mem;
   VkImageView view;
   VkRenderPass rp;
   VkFramebuffer fb;
   VkBuffer rb;
   VkDeviceMemory rb_mem;
   void *rb_map;
   VkPipelineLayout layout;
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb;
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb;
   PFN_vkCmdEndTransformFeedbackEXT end_xfb;
} vk;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk.mp.memoryTypeCount; i++) {
      if ((bits & (1u << i)) && (vk.mp.memoryTypes[i].propertyFlags & want) == want) {
         return i;
      }
   }
   printf("FAILED (tess: no memory type)\n");
   exit(1);
}

static VkShaderModule
module(const char *name)
{
   FILE *f = fopen(name, "rb");
   if (!f) {
      printf("FAILED (tess: cannot open %s)\n", name);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   std::vector<uint32_t> code((size_t)sz / 4u);
   if (fread(code.data(), 1, (size_t)sz, f) != (size_t)sz) {
      printf("FAILED (tess: cannot read %s)\n", name);
      exit(1);
   }
   fclose(f);
   VkShaderModuleCreateInfo ci = {};
   ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   ci.codeSize = (size_t)sz;
   ci.pCode = code.data();
   VkShaderModule m;
   CHECK(vkCreateShaderModule(vk.dev, &ci, NULL, &m));
   return m;
}

static void
buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *b, VkDeviceMemory *m, void **map)
{
   VkBufferCreateInfo ci = {};
   ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   ci.size = size;
   ci.usage = usage;
   CHECK(vkCreateBuffer(vk.dev, &ci, NULL, b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(vk.dev, *b, &mr);
   VkMemoryAllocateInfo ai = {};
   ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   ai.allocationSize = mr.size;
   ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, m));
   CHECK(vkBindBufferMemory(vk.dev, *b, *m, 0));
   CHECK(vkMapMemory(vk.dev, *m, 0, size, 0, map));
}

static void
init(void)
{
   VkApplicationInfo app = {};
   app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
   app.apiVersion = VK_API_VERSION_1_1;
   VkInstanceCreateInfo ici = {};
   ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
   ici.pApplicationInfo = &app;
   CHECK(vkCreateInstance(&ici, NULL, &vk.inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(vk.inst, &n, &vk.phys);
   if (!n) {
      printf("FAILED (tess: no device)\n");
      exit(1);
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(vk.phys, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.mp);

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(vk.phys, &have);
   if (!have.tessellationShader || !have.shaderTessellationAndGeometryPointSize ||
       !have.geometryShader || !have.vertexPipelineStoresAndAtomics ||
       !have.pipelineStatisticsQuery) {
      printf("FAILED (tess: device lacks tessellationShader, geometryShader, their point "
             "size, vertexPipelineStoresAndAtomics or pipelineStatisticsQuery)\n");
      exit(1);
   }
   /* A device with tessellation takes patches of 32 control points. */
   if (props.limits.maxTessellationPatchSize < 32) {
      printf("FAILED (tess: maxTessellationPatchSize %u with tessellationShader)\n",
             props.limits.maxTessellationPatchSize);
      exit(1);
   }

   uint32_t nq = 8;
   VkQueueFamilyProperties qp[8];
   vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &nq, qp);
   for (vk.qf = 0; vk.qf < nq && !(qp[vk.qf].queueFlags & VK_QUEUE_GRAPHICS_BIT); vk.qf++) {
   }
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {};
   qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
   qci.queueFamilyIndex = vk.qf;
   qci.queueCount = 1;
   qci.pQueuePriorities = &prio;
   VkPhysicalDeviceFeatures feat = {};
   feat.tessellationShader = VK_TRUE;
   feat.geometryShader = VK_TRUE;
   feat.shaderTessellationAndGeometryPointSize = VK_TRUE;
   feat.vertexPipelineStoresAndAtomics = VK_TRUE;
   feat.pipelineStatisticsQuery = VK_TRUE;
   VkPhysicalDeviceMultiviewFeatures mv = {};
   mv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
   mv.multiview = VK_TRUE;
   mv.multiviewTessellationShader = VK_TRUE;
   VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT lr = {};
   lr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT;
   lr.pNext = &mv;
   lr.primitiveTopologyPatchListRestart = VK_TRUE;
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xf = {};
   xf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT;
   xf.pNext = &lr;
   xf.transformFeedback = VK_TRUE;
   const char *ext[] = { VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
                         VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {};
   dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
   dci.pNext = &xf;
   dci.enabledExtensionCount = 2;
   dci.ppEnabledExtensionNames = ext;
   dci.queueCreateInfoCount = 1;
   dci.pQueueCreateInfos = &qci;
   dci.pEnabledFeatures = &feat;
   CHECK(vkCreateDevice(vk.phys, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, vk.qf, 0, &vk.queue);
   vk.bind_xfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdBindTransformFeedbackBuffersEXT");
   vk.begin_xfb = (PFN_vkCmdBeginTransformFeedbackEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdBeginTransformFeedbackEXT");
   vk.end_xfb = (PFN_vkCmdEndTransformFeedbackEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdEndTransformFeedbackEXT");

   VkCommandPoolCreateInfo pci = {};
   pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
   pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
   pci.queueFamilyIndex = vk.qf;
   CHECK(vkCreateCommandPool(vk.dev, &pci, NULL, &vk.pool));

   VkImageCreateInfo imci = {};
   imci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
   imci.imageType = VK_IMAGE_TYPE_2D;
   imci.format = VK_FORMAT_R8G8B8A8_UNORM;
   imci.extent = { W, H, 1 };
   imci.mipLevels = 1;
   imci.arrayLayers = 1;
   imci.samples = VK_SAMPLE_COUNT_1_BIT;
   imci.tiling = VK_IMAGE_TILING_OPTIMAL;
   imci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   CHECK(vkCreateImage(vk.dev, &imci, NULL, &vk.img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk.dev, vk.img, &mr);
   VkMemoryAllocateInfo ai = {};
   ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   ai.allocationSize = mr.size;
   ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, 0);
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &vk.img_mem));
   CHECK(vkBindImageMemory(vk.dev, vk.img, vk.img_mem, 0));
   VkImageViewCreateInfo vci = {};
   vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
   vci.image = vk.img;
   vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
   vci.format = VK_FORMAT_R8G8B8A8_UNORM;
   vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, &vk.view));

   VkAttachmentDescription att = {};
   att.format = VK_FORMAT_R8G8B8A8_UNORM;
   att.samples = VK_SAMPLE_COUNT_1_BIT;
   att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {};
   sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   sub.colorAttachmentCount = 1;
   sub.pColorAttachments = &ref;
   VkRenderPassCreateInfo rpci = {};
   rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
   rpci.attachmentCount = 1;
   rpci.pAttachments = &att;
   rpci.subpassCount = 1;
   rpci.pSubpasses = &sub;
   CHECK(vkCreateRenderPass(vk.dev, &rpci, NULL, &vk.rp));
   VkFramebufferCreateInfo fci = {};
   fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
   fci.renderPass = vk.rp;
   fci.attachmentCount = 1;
   fci.pAttachments = &vk.view;
   fci.width = W;
   fci.height = H;
   fci.layers = 1;
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &vk.fb));
   buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &vk.rb, &vk.rb_mem, &vk.rb_map);

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 8 };
   VkPipelineLayoutCreateInfo plci = {};
   plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
   plci.pushConstantRangeCount = 1;
   plci.pPushConstantRanges = &pcr;
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &vk.layout));
}

/* ---- cases ------------------------------------------------------------ */

enum domain { TRI, QUAD, ISO };

struct tess_case {
   const char *name;
   const char *tese;              /* the evaluation shader's SPIR-V */
   const char *tesc;              /* NULL: tess.tesc */
   enum domain dom;
   uint32_t spacing;              /* GFX_TESS_SPACING_* */
   bool ccw;                      /* VertexOrderCcw */
   bool points;                   /* PointMode */
   bool dyn;                      /* levels from the patches' first control points */
   float lo[4], li[2];            /* the static levels */
   std::vector<cp_vertex> cps;    /* the patches' control points */
   uint32_t patch_size;           /* input control points per patch */
   uint32_t instances;            /* 0: one */
   bool last_live;                /* every instance but the last discarded */
   bool blend;                    /* blend in order, coloured by gl_PrimitiveID */
   bool pass;                     /* pass.tesc / pass.tese */
   const char *geom;              /* a geometry shader after them, or NULL */
   bool xfb;                      /* capture the evaluation stage's colour */
   uint32_t xfb_short;            /* the capture's room short by this many primitives */
   uint32_t rounds;               /* barrier.tesc: neighbour rounds after pass's values */
   bool discard;                  /* nothing rasterized */
   bool stats;                    /* the stages' pipeline statistics, checked */
   const char *batch_bytes;       /* VORTEXPIPE_TS_BATCH_BYTES for the draw, or NULL */
   /* Drawn indexed when not empty, with primitive restart when RESTART cuts
    * the patch list into runs, each run's incomplete last patch dropped. */
   std::vector<uint32_t> indices;
};

static const uint32_t RESTART = 0xffffffffu;

/* levels_max's patches: the fewest whose evaluation records, made as one
 * batch, are more than a 32-bit device can allocate at once (955 fit). */
static const uint32_t LEVELS_MAX_PATCHES = 956;

/* The control points of the patches the draw assembles. */
static std::vector<cp_vertex>
assembled(const tess_case &c)
{
   if (c.indices.empty()) {
      return c.cps;
   }
   std::vector<cp_vertex> out, run;
   for (uint32_t i : c.indices) {
      if (i == RESTART) {
         run.clear();
         continue;
      }
      run.push_back(c.cps[i]);
      if (run.size() == c.patch_size) {
         out.insert(out.end(), run.begin(), run.end());
         run.clear();
      }
   }
   return out;
}

/* Everything init made, in reverse: the driver reports its work placement as
 * the device goes. */
static void
teardown(void)
{
   vkDeviceWaitIdle(vk.dev);
   vkDestroyPipelineLayout(vk.dev, vk.layout, NULL);
   vkUnmapMemory(vk.dev, vk.rb_mem);
   vkDestroyBuffer(vk.dev, vk.rb, NULL);
   vkFreeMemory(vk.dev, vk.rb_mem, NULL);
   vkDestroyFramebuffer(vk.dev, vk.fb, NULL);
   vkDestroyRenderPass(vk.dev, vk.rp, NULL);
   vkDestroyImageView(vk.dev, vk.view, NULL);
   vkDestroyImage(vk.dev, vk.img, NULL);
   vkFreeMemory(vk.dev, vk.img_mem, NULL);
   vkDestroyCommandPool(vk.dev, vk.pool, NULL);
   vkDestroyDevice(vk.dev, NULL);
   vkDestroyInstance(vk.inst, NULL);
}

/* Grid cell (i, j) of n x n over the target, as four control points
 * counter-clockwise from the lower-left corner in NDC: multiples of 1/64. */
static void
cell(std::vector<cp_vertex> &cps, unsigned i, unsigned j, unsigned n, const float lo[4],
     const float li[2], float shrink = 0.0f)
{
   const float s = 2.0f / (float)n;
   const float x0 = -1.0f + s * (float)i + shrink, y0 = -1.0f + s * (float)j + shrink;
   const float x1 = x0 + s - 2.0f * shrink, y1 = y0 + s - 2.0f * shrink;
   const float xs[4] = { x0, x1, x1, x0 }, ys[4] = { y0, y0, y1, y1 };
   for (unsigned k = 0; k < 4; k++) {
      cp_vertex v = {};
      v.x = xs[k];
      v.y = ys[k];
      memcpy(v.lo, lo, sizeof v.lo);
      memcpy(v.li, li, sizeof v.li);
      cps.push_back(v);
   }
}

static uint32_t
spacing_mode(const tess_case &c)
{
   const uint32_t d = c.dom == TRI ? GFX_TESS_DOMAIN_TRI : c.dom == QUAD ? GFX_TESS_DOMAIN_QUAD
                                                                       : GFX_TESS_DOMAIN_ISOLINE;
   /* The pipelines take Vulkan's default domain origin, the upper left, under
    * which a counter-clockwise vertex order is the reference tessellator's
    * "counter-clockwise" output. */
   const uint32_t out = c.points ? GFX_TESS_OUT_POINT : c.dom == ISO ? GFX_TESS_OUT_LINE
                      : (c.ccw == bite) ? GFX_TESS_OUT_TRI_CW : GFX_TESS_OUT_TRI_CCW;
   return GFX_TESS_MODE(d, c.spacing, out);
}

/* The control points of patch p and the instance step the vertex shader adds. */
static const float INST_STEP[2] = { 0.0f, 0.0f };

/* The reference: each patch tessellated on the host and evaluated as the
 * evaluation shader does, every patch's primitives in order. */
static std::vector<ref_vertex>
reference(const tess_case &c, VkPrimitiveTopology *topo, uint64_t made[2] = NULL)
{
   std::vector<ref_vertex> out;
   if (made) {
      made[0] = made[1] = 0;
   }
   const uint32_t mode = spacing_mode(c);
   const uint32_t per = c.patch_size;
   const std::vector<cp_vertex> cps = assembled(c);
   const uint32_t npatch = (uint32_t)cps.size() / per;
   const uint32_t inst = c.instances ? c.instances : 1u;
   *topo = c.points ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
         : c.dom == ISO ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   for (uint32_t in = 0; in < inst; in++) {
      for (uint32_t p = 0; p < npatch; p++) {
         const cp_vertex *cp = &cps[p * per];
         float levels[6];
         const bool dead = c.last_live && in + 1u != inst;
         for (unsigned k = 0; k < 4; k++) {
            levels[k] = dead ? 0.0f : c.dyn || c.pass ? cp[0].lo[k] : c.lo[k];
         }
         for (unsigned k = 0; k < 2; k++) {
            levels[4 + k] = c.dyn || c.pass ? cp[0].li[k] : c.li[k];
         }
         gfx_tess_patch_t pd;
         gfx_tess::process(mode, levels, levels + 4, &pd);
         if (pd.flags & GFX_TESS_CULLED) {
            continue;
         }
         if (made) {
            made[0] += pd.num_points + (bite ? 1u : 0u);
            made[1] += pd.num_prims - (bite && pd.num_prims ? 1u : 0u);
         }
         float px[4], py[4];
         for (unsigned k = 0; k < 4; k++) {
            if (k < per) {
               px[k] = cp[k].x + (float)in * INST_STEP[0];
               py[k] = cp[k].y + (float)in * INST_STEP[1];
            } else {
               px[k] = px[0] + (px[2] - px[1]);
               py[k] = py[0] + (py[2] - py[1]);
            }
         }
         std::vector<ref_vertex> pts(pd.num_points);
         for (uint32_t i = 0; i < pd.num_points; i++) {
            gfx_tess::fxp_t fu, fv;
            gfx_tess::point(pd, i, &fu, &fv);
            const float u = gfx_tess::to_float(fu), v = gfx_tess::to_float(fv);
            ref_vertex &r = pts[i];
            r.x = px[0] + u * (px[1] - px[0]) + v * (px[3] - px[0]);
            r.y = py[0] + u * (py[1] - py[0]) + v * (py[3] - py[0]);
            if (c.pass) {
               float val[4], next[4];
               for (unsigned k = 0; k < 4; k++) {
                  val[k] = px[k] * 2.0f + (float)k * 0.125f;
               }
               for (uint32_t r = 0; r < c.rounds; r++) {
                  float nv[4];
                  for (unsigned k = 0; k < 4; k++) {
                     nv[k] = val[k] + val[(k + 1) & 3] * 0.5f;
                  }
                  memcpy(val, nv, sizeof val);
               }
               for (unsigned k = 0; k < 4; k++) {
                  next[k] = val[(k + 1) & 3];
               }
               const float sum = val[0] + val[1] + val[2] + val[3];
               auto fract = [](float f) { return f - floorf(f); };
               r.col[0] = fract(val[1] - val[0]);
               r.col[1] = fract(next[2] + next[3]);
               r.col[2] = fract(sum);
               r.col[3] = 1.0f;
            } else if (c.blend) {
               const uint32_t id = p;
               r.col[0] = (float)(id & 7) * 0.125f;
               r.col[1] = (float)((id >> 3) & 7) * 0.125f;
               r.col[2] = u;
               r.col[3] = 0.5f;
            } else {
               r.col[0] = u;
               r.col[1] = v;
               r.col[2] = c.dom == TRI ? 1.0f - u - v : 0.5f;
               r.col[3] = 1.0f;
            }
            if (bite && c.points) {
               r.x += 2.0f / (float)W;
            }
         }
         if (c.points) {
            out.insert(out.end(), pts.begin(), pts.end());
            continue;
         }
         const uint32_t vpp = c.dom == ISO ? 2u : 3u;
         std::vector<uint32_t> idx((size_t)pd.num_prims * vpp);
         const gfx_tess::sink dst = { idx.data(), vpp, 0, 0 };
         for (uint32_t s = 0; s < pd.num_units; s++) {
            gfx_tess::unit(pd, s, dst);
         }
         const uint32_t keep = bite && pd.num_prims ? pd.num_prims - 1u : pd.num_prims;
         for (uint32_t k = 0; k < keep * vpp; k++) {
            out.push_back(pts[idx[k]]);
         }
      }
   }
   return out;
}

/* The render pass pipelines are made for, when not vk.rp. */
static VkRenderPass pipeline_rp;

static VkPipeline
pipeline(const char *vert, const char *tesc, const char *tese, VkPrimitiveTopology topo,
         uint32_t patch_size, bool blend, const tess_case *c)
{
   VkShaderModule vs = module(vert), fs = module("tess.frag.spv");
   VkShaderModule tcs = tesc ? module(tesc) : VK_NULL_HANDLE;
   VkShaderModule tes = tese ? module(tese) : VK_NULL_HANDLE;
   VkShaderModule gs = c && c->geom ? module(c->geom) : VK_NULL_HANDLE;
   const VkSpecializationMapEntry map[8] = {
      { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 }, { 3, 12, 4 }, { 4, 16, 4 }, { 5, 20, 4 },
      { 6, 24, 4 }, { 7, 28, 4 },
   };
   uint32_t spec[8] = {};
   spec[7] = UINT32_MAX;
   if (c) {
      spec[0] = c->dyn ? 1u : 0u;
      memcpy(&spec[1], c->lo, 16);
      memcpy(&spec[5], c->li, 8);
      if (c->last_live) {
         spec[7] = c->instances - 1u;
      }
   }
   VkSpecializationInfo si = { 8, map, sizeof spec, spec };
   VkPipelineShaderStageCreateInfo st[5] = {};
   unsigned ns = 0;
   st[ns].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   st[ns].stage = VK_SHADER_STAGE_VERTEX_BIT;
   st[ns].module = vs;
   st[ns].pName = "main";
   st[ns++].pSpecializationInfo = &si;
   st[ns].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   st[ns].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   st[ns].module = fs;
   st[ns++].pName = "main";
   if (tcs) {
      st[ns].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st[ns].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
      st[ns].module = tcs;
      st[ns].pName = "main";
      st[ns++].pSpecializationInfo = &si;
      st[ns].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st[ns].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
      st[ns].module = tes;
      st[ns++].pName = "main";
   }
   if (gs) {
      st[ns].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st[ns].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
      st[ns].module = gs;
      st[ns++].pName = "main";
   }
   VkVertexInputBindingDescription bind = { 0, tcs ? (uint32_t)sizeof(cp_vertex)
                                                   : (uint32_t)sizeof(ref_vertex),
                                            VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription attr[3] = {
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
      { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8 },
      { 2, 0, VK_FORMAT_R32G32_SFLOAT, 24 },
   };
   VkPipelineVertexInputStateCreateInfo vi = {};
   vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
   vi.vertexBindingDescriptionCount = 1;
   vi.pVertexBindingDescriptions = &bind;
   vi.vertexAttributeDescriptionCount = tcs ? 3u : 2u;
   vi.pVertexAttributeDescriptions = attr;
   VkPipelineInputAssemblyStateCreateInfo ia = {};
   ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
   ia.topology = tcs ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : topo;
   ia.primitiveRestartEnable =
      c && std::count(c->indices.begin(), c->indices.end(), RESTART) ? VK_TRUE : VK_FALSE;
   VkPipelineTessellationStateCreateInfo ts = {};
   ts.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
   ts.patchControlPoints = patch_size;
   VkViewport vp = { 0, 0, W, H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {};
   vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
   vps.viewportCount = 1;
   vps.pViewports = &vp;
   vps.scissorCount = 1;
   vps.pScissors = &sc;
   VkPipelineRasterizationStateCreateInfo rs = {};
   rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
   rs.polygonMode = VK_POLYGON_MODE_FILL;
   rs.cullMode = VK_CULL_MODE_NONE;
   rs.rasterizerDiscardEnable = c && c->discard ? VK_TRUE : VK_FALSE;
   rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rs.lineWidth = 1.0f;
   VkPipelineMultisampleStateCreateInfo ms = {};
   ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState cba = {};
   cba.colorWriteMask = 0xf;
   if (blend) {
      cba.blendEnable = VK_TRUE;
      cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      cba.colorBlendOp = VK_BLEND_OP_ADD;
      cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba.alphaBlendOp = VK_BLEND_OP_ADD;
   }
   VkPipelineColorBlendStateCreateInfo cb = {};
   cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
   cb.attachmentCount = 1;
   cb.pAttachments = &cba;
   VkGraphicsPipelineCreateInfo gp = {};
   gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
   gp.stageCount = ns;
   gp.pStages = st;
   gp.pVertexInputState = &vi;
   gp.pInputAssemblyState = &ia;
   gp.pTessellationState = tcs ? &ts : NULL;
   gp.pViewportState = &vps;
   gp.pRasterizationState = &rs;
   gp.pMultisampleState = &ms;
   gp.pColorBlendState = &cb;
   gp.layout = vk.layout;
   gp.renderPass = pipeline_rp ? pipeline_rp : vk.rp;
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
   vkDestroyShaderModule(vk.dev, vs, NULL);
   vkDestroyShaderModule(vk.dev, fs, NULL);
   if (tcs) {
      vkDestroyShaderModule(vk.dev, tcs, NULL);
      vkDestroyShaderModule(vk.dev, tes, NULL);
   }
   if (gs) {
      vkDestroyShaderModule(vk.dev, gs, NULL);
   }
   return p;
}

/* Draw `n` vertices of `data` (`stride` bytes each) through `p`, `inst`
 * instances, into a cleared target, and read it back into img. With
 * `indices`, the draw is of them, 32-bit. */
static void
render(VkPipeline p, const void *data, uint32_t stride, uint32_t n, uint32_t inst,
       uint32_t *img, VkBuffer xfb = VK_NULL_HANDLE, VkDeviceSize xfb_size = 0,
       VkQueryPool stats = VK_NULL_HANDLE, VkQueryPool xq = VK_NULL_HANDLE,
       const std::vector<uint32_t> *indices = NULL)
{
   VkBuffer vb = VK_NULL_HANDLE, ib = VK_NULL_HANDLE;
   VkDeviceMemory vm = VK_NULL_HANDLE, im = VK_NULL_HANDLE;
   if (n) {
      void *map;
      buffer((VkDeviceSize)stride * n, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &map);
      memcpy(map, data, (size_t)stride * n);
   }
   if (indices && !indices->empty()) {
      void *map;
      buffer(indices->size() * 4u, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib, &im, &map);
      memcpy(map, indices->data(), indices->size() * 4u);
   }
   VkCommandBufferAllocateInfo cai = {};
   cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cai.commandPool = vk.pool;
   cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cai.commandBufferCount = 1;
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {};
   bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   if (stats) {
      vkCmdResetQueryPool(cmd, stats, 0, 1);
   }
   if (xq) {
      vkCmdResetQueryPool(cmd, xq, 0, 1);
   }
   VkClearValue clear = {};
   VkRenderPassBeginInfo rbi = {};
   rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   rbi.renderPass = vk.rp;
   rbi.framebuffer = vk.fb;
   rbi.renderArea = { { 0, 0 }, { W, H } };
   rbi.clearValueCount = 1;
   rbi.pClearValues = &clear;
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   if (stats) {
      vkCmdBeginQuery(cmd, stats, 0, 0);
   }
   if (xq) {
      vkCmdBeginQuery(cmd, xq, 0, 0);
   }
   if (n) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
      vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, INST_STEP);
      VkDeviceSize off = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
      if (xfb) {
         VkDeviceSize xoff = 0;
         vk.bind_xfb(cmd, 0, 1, &xfb, &xoff, &xfb_size);
         vk.begin_xfb(cmd, 0, 0, NULL, NULL);
      }
      if (ib) {
         vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
         vkCmdDrawIndexed(cmd, (uint32_t)indices->size(), inst, 0, 0, 0);
      } else {
         vkCmdDraw(cmd, n, inst, 0, 0);
      }
      if (xfb) {
         vk.end_xfb(cmd, 0, 0, NULL, NULL);
      }
   }
   if (xq) {
      vkCmdEndQuery(cmd, xq, 0);
   }
   if (stats) {
      vkCmdEndQuery(cmd, stats, 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy reg = {};
   reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
   reg.imageExtent = { W, H, 1 };
   vkCmdCopyImageToBuffer(cmd, vk.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.rb, 1, &reg);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {};
   si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   si.commandBufferCount = 1;
   si.pCommandBuffers = &cmd;
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   memcpy(img, vk.rb_map, W * H * 4);
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   if (vb) {
      vkUnmapMemory(vk.dev, vm);
      vkDestroyBuffer(vk.dev, vb, NULL);
      vkFreeMemory(vk.dev, vm, NULL);
   }
   if (ib) {
      vkUnmapMemory(vk.dev, im);
      vkDestroyBuffer(vk.dev, ib, NULL);
      vkFreeMemory(vk.dev, im, NULL);
   }
}

static unsigned
run_case(const tess_case &c)
{
   VkPrimitiveTopology topo;
   uint64_t made[2];
   const std::vector<ref_vertex> ref = reference(c, &topo, made);
   const uint32_t inst = c.instances ? c.instances : 1u;
   std::vector<uint32_t> got(W * H), want(W * H);

   char tese[64];
   format(tese, sizeof tese, "%s.spv", c.tese);
   const char *tesc = c.tesc ? c.tesc : c.pass ? "pass.tesc.spv" : "tess.tesc.spv";
   VkPipeline tp = pipeline("tess.vert.spv", tesc, tese, topo, c.patch_size, c.blend, &c);
   /* The capture: room for every reference vertex and a few more, which must
    * stay untouched; with xfb_short the bound range holds that many primitives
    * fewer than the draw makes. */
   VkBuffer xb = VK_NULL_HANDLE;
   VkDeviceMemory xm = VK_NULL_HANDLE;
   float *xmap = NULL;
   const size_t xfloats = 4u * (ref.size() + 4u);
   const uint32_t vpp = c.points ? 1u : c.dom == ISO ? 2u : 3u;
   const size_t kept = ref.size() - std::min((size_t)c.xfb_short * vpp, ref.size());
   VkQueryPool xq = VK_NULL_HANDLE, sq = VK_NULL_HANDLE;
   if (c.xfb) {
      void *map;
      buffer(xfloats * 4u, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, &xb, &xm, &map);
      xmap = (float *)map;
      for (size_t k = 0; k < xfloats; k++) {
         xmap[k] = -7.0f;
      }
      VkQueryPoolCreateInfo qci = {};
      qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      qci.queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT;
      qci.queryCount = 1;
      CHECK(vkCreateQueryPool(vk.dev, &qci, NULL, &xq));
   }
   const VkQueryPipelineStatisticFlags sbits =
      (c.geom ? VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT : 0u) |
      VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
   if (c.stats) {
      VkQueryPoolCreateInfo qci = {};
      qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      qci.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
      qci.queryCount = 1;
      qci.pipelineStatistics = sbits;
      CHECK(vkCreateQueryPool(vk.dev, &qci, NULL, &sq));
   }
   if (c.batch_bytes) {
      setenv("VORTEXPIPE_TS_BATCH_BYTES", c.batch_bytes, 1);
   }
   render(tp, c.cps.data(), sizeof(cp_vertex), (uint32_t)c.cps.size(), inst, got.data(), xb,
          kept * 16u, sq, xq, &c.indices);
   unsetenv("VORTEXPIPE_TS_BATCH_BYTES");
   vkDestroyPipeline(vk.dev, tp, NULL);
   unsigned qbad = 0;
   if (sq) {
      /* In bit order: geometry invocations, control patches, evaluation
       * invocations. */
      uint64_t r[3] = { 0, 0, 0 };
      CHECK(vkGetQueryPoolResults(vk.dev, sq, 0, 1, sizeof r, r, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      const uint64_t patches = (uint64_t)(assembled(c).size() / c.patch_size) * inst;
      const uint64_t want_q[3] = { made[1], patches, made[0] };
      const char *const what[3] = { "geometry invocations", "control patches",
                                    "evaluation invocations" };
      for (unsigned k = c.geom ? 0u : 1u, i = 0; k < 3; k++, i++) {
         if (r[i] != want_q[k]) {
            printf("  %s: %s %llu, expected %llu\n", c.name, what[k], (unsigned long long)r[i],
                   (unsigned long long)want_q[k]);
            qbad++;
         }
      }
      vkDestroyQueryPool(vk.dev, sq, NULL);
   }
   if (xq) {
      uint64_t r[2] = { 0, 0 };
      CHECK(vkGetQueryPoolResults(vk.dev, xq, 0, 1, sizeof r, r, 16,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      const uint64_t prims = ref.size() / vpp;
      if (r[0] != kept / vpp || r[1] != prims) {
         printf("  %s: %llu primitives written of %llu needed, expected %zu of %llu\n",
                c.name, (unsigned long long)r[0], (unsigned long long)r[1], kept / vpp,
                (unsigned long long)prims);
         qbad++;
      }
      vkDestroyQueryPool(vk.dev, xq, NULL);
   }

   VkPipeline rp = pipeline("ref.vert.spv", NULL, NULL, topo, 0, c.blend, NULL);
   render(rp, ref.data(), sizeof(ref_vertex), (uint32_t)ref.size(), 1, want.data());
   vkDestroyPipeline(vk.dev, rp, NULL);

   unsigned bad = qbad, covered = 0;
   if (c.discard) {
      std::fill(want.begin(), want.end(), 0u);
   }
   for (unsigned k = 0; k < W * H; k++) {
      covered += want[k] != 0;
      if (got[k] != want[k] && bad++ < 4) {
         printf("  %s: pixel (%u,%u) = %08x, expected %08x\n", c.name, k % W, k / W,
                got[k], want[k]);
      }
   }
   if (!covered && !bad && !c.discard && !(c.indices.size() && ref.empty())) {
      printf("  %s: the expected image is empty\n", c.name);
      bad = 1;
   }
   if (c.xfb) {
      unsigned xbad = 0;
      for (size_t v = 0; v < ref.size() + 4u; v++) {
         for (unsigned k = 0; k < 4; k++) {
            const float want_f = v < kept ? ref[v].col[k] : -7.0f;
            if (memcmp(&xmap[4 * v + k], &want_f, 4) != 0 && xbad++ < 4) {
               printf("  %s: captured vertex %zu component %u = %g, expected %g\n", c.name, v,
                      k, xmap[4 * v + k], want_f);
            }
         }
      }
      bad += xbad;
      vkUnmapMemory(vk.dev, xm);
      vkDestroyBuffer(vk.dev, xb, NULL);
      vkFreeMemory(vk.dev, xm, NULL);
   }
   printf("%s: %s (%zu reference vertices)\n", c.name, bad ? "FAIL" : "ok", ref.size());
   return bad;
}

/* A one-pass image of `layers` layers and its views, for multiview. */
struct layered {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   VkRenderPass rp;
   VkFramebuffer fb;
};

static layered
layered_make(uint32_t layers, uint32_t view_mask)
{
   layered l = {};
   VkImageCreateInfo imci = {};
   imci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
   imci.imageType = VK_IMAGE_TYPE_2D;
   imci.format = VK_FORMAT_R8G8B8A8_UNORM;
   imci.extent = { W, H, 1 };
   imci.mipLevels = 1;
   imci.arrayLayers = layers;
   imci.samples = VK_SAMPLE_COUNT_1_BIT;
   imci.tiling = VK_IMAGE_TILING_OPTIMAL;
   imci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   CHECK(vkCreateImage(vk.dev, &imci, NULL, &l.img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk.dev, l.img, &mr);
   VkMemoryAllocateInfo ai = {};
   ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   ai.allocationSize = mr.size;
   ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, 0);
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &l.mem));
   CHECK(vkBindImageMemory(vk.dev, l.img, l.mem, 0));
   VkImageViewCreateInfo vci = {};
   vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
   vci.image = l.img;
   vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   vci.format = VK_FORMAT_R8G8B8A8_UNORM;
   vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, &l.view));
   VkAttachmentDescription att = {};
   att.format = VK_FORMAT_R8G8B8A8_UNORM;
   att.samples = VK_SAMPLE_COUNT_1_BIT;
   att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {};
   sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   sub.colorAttachmentCount = 1;
   sub.pColorAttachments = &ref;
   VkRenderPassMultiviewCreateInfo mv = {};
   mv.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
   mv.subpassCount = 1;
   mv.pViewMasks = &view_mask;
   VkRenderPassCreateInfo rpci = {};
   rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
   rpci.pNext = &mv;
   rpci.attachmentCount = 1;
   rpci.pAttachments = &att;
   rpci.subpassCount = 1;
   rpci.pSubpasses = &sub;
   CHECK(vkCreateRenderPass(vk.dev, &rpci, NULL, &l.rp));
   VkFramebufferCreateInfo fci = {};
   fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
   fci.renderPass = l.rp;
   fci.attachmentCount = 1;
   fci.pAttachments = &l.view;
   fci.width = W;
   fci.height = H;
   fci.layers = 1;
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &l.fb));
   return l;
}

static void
layered_free(layered &l)
{
   vkDestroyFramebuffer(vk.dev, l.fb, NULL);
   vkDestroyRenderPass(vk.dev, l.rp, NULL);
   vkDestroyImageView(vk.dev, l.view, NULL);
   vkDestroyImage(vk.dev, l.img, NULL);
   vkFreeMemory(vk.dev, l.mem, NULL);
}

static unsigned
compare(const char *name, const uint32_t *got, const uint32_t *want)
{
   unsigned bad = 0;
   for (unsigned k = 0; k < W * H; k++) {
      if (got[k] != want[k] && bad++ < 4) {
         printf("  %s: pixel (%u,%u) = %08x, expected %08x\n", name, k % W, k / W, got[k],
                want[k]);
      }
   }
   return bad;
}

/* multiview: one draw seen by two views, the evaluation stage moving view 1's
 * patches half the target right; each layer must be the reference of its
 * view. */
static unsigned
run_multiview(void)
{
   tess_case c = {};
   c.name = "multiview";
   c.tese = "quad_odd_ccw_view.tese";
   c.dom = QUAD;
   c.spacing = GFX_TESS_SPACING_ODD;
   c.ccw = true;
   c.dyn = true;
   for (unsigned j = 0; j < 4; j++) {
      const float lo[4] = { 2.0f + (float)j, 3.5f, 1.5f + (float)j, 4.0f };
      const float li[2] = { 3.0f + 0.5f * (float)j, 2.0f };
      cell(c.cps, 0, j, 4, lo, li, 1.0f / 64.0f);
   }
   c.patch_size = 4;
   VkPrimitiveTopology topo;
   const std::vector<ref_vertex> ref0 = reference(c, &topo);
   std::vector<ref_vertex> ref1 = ref0;
   for (ref_vertex &r : ref1) {
      r.x += 0.5f;
   }

   layered l = layered_make(2, 0x3u);
   pipeline_rp = l.rp;
   VkPipeline tp = pipeline("tess.vert.spv", "tess.tesc.spv", "quad_odd_ccw_view.tese.spv", topo,
                            4, false, &c);
   pipeline_rp = VK_NULL_HANDLE;
   void *vmap;
   VkBuffer vb, rb;
   VkDeviceMemory vm, rm;
   buffer(c.cps.size() * sizeof(cp_vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &vmap);
   memcpy(vmap, c.cps.data(), c.cps.size() * sizeof(cp_vertex));
   void *rmap;
   buffer(2u * W * H * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rm, &rmap);
   VkCommandBufferAllocateInfo cai = {};
   cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cai.commandPool = vk.pool;
   cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cai.commandBufferCount = 1;
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {};
   bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkClearValue clear = {};
   VkRenderPassBeginInfo rbi = {};
   rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   rbi.renderPass = l.rp;
   rbi.framebuffer = l.fb;
   rbi.renderArea = { { 0, 0 }, { W, H } };
   rbi.clearValueCount = 1;
   rbi.pClearValues = &clear;
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tp);
   vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, INST_STEP);
   VkDeviceSize off = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
   vkCmdDraw(cmd, (uint32_t)c.cps.size(), 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy reg = {};
   reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2 };
   reg.imageExtent = { W, H, 1 };
   vkCmdCopyImageToBuffer(cmd, l.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &reg);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {};
   si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   si.commandBufferCount = 1;
   si.pCommandBuffers = &cmd;
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   std::vector<uint32_t> got(2u * W * H), want(W * H);
   memcpy(got.data(), rmap, 2u * W * H * 4u);
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, tp, NULL);

   unsigned bad = 0;
   VkPipeline rp = pipeline("ref.vert.spv", NULL, NULL, topo, 0, false, NULL);
   for (unsigned v = 0; v < 2; v++) {
      const std::vector<ref_vertex> &ref = v ? ref1 : ref0;
      render(rp, ref.data(), sizeof(ref_vertex), (uint32_t)ref.size(), 1, want.data());
      char name[32];
      format(name, sizeof name, "multiview view %u", v);
      bad += compare(name, got.data() + (size_t)v * W * H, want.data());
   }
   vkDestroyPipeline(vk.dev, rp, NULL);
   vkDestroyBuffer(vk.dev, vb, NULL);
   vkFreeMemory(vk.dev, vm, NULL);
   vkDestroyBuffer(vk.dev, rb, NULL);
   vkFreeMemory(vk.dev, rm, NULL);
   layered_free(l);
   printf("multiview: %s (%zu reference vertices a view)\n", bad ? "FAIL" : "ok", ref0.size());
   return bad;
}

/* handoff: one capture across three tessellated draws -- on the device, on
 * llvmpipe (a geometry stage's capture after tessellation, which runs there),
 * on the device again. The captured stream must be the three draws' in order:
 * llvmpipe resuming from the sizes the device reached, the device from where
 * llvmpipe left them. Run with STRICT 0, on its own (-c handoff). */
static unsigned
run_handoff(void)
{
   tess_case c = {};
   c.name = "handoff";
   c.tese = "quad_equal_ccw_xfb.tese";
   c.dom = QUAD;
   c.spacing = GFX_TESS_SPACING_EQUAL;
   c.ccw = true;
   c.dyn = true;
   c.xfb = true;
   for (unsigned j = 0; j < 4; j++) {
      const float lo[4] = { 2.0f, 3.0f + (float)j, 2.0f, 1.0f + (float)j };
      const float li[2] = { 2.0f + (float)j, 3.0f };
      cell(c.cps, j & 1, j >> 1, 2, lo, li, 1.0f / 32.0f);
   }
   c.patch_size = 4;
   VkPrimitiveTopology topo;
   const std::vector<ref_vertex> ref = reference(c, &topo);
   tess_case g = c;
   g.geom = "gs_xfb.geom.spv";
   VkPipeline pa = pipeline("tess.vert.spv", "tess.tesc.spv", "quad_equal_ccw_xfb.tese.spv", topo,
                            4, false, &c);
   VkPipeline pb = pipeline("tess.vert.spv", "tess.tesc.spv", "quad_equal_ccw.tese.spv", topo, 4,
                            false, &g);
   const size_t nv = 3u * ref.size(), xfloats = 4u * (nv + 4u);
   VkBuffer xb, vb;
   VkDeviceMemory xm, vm;
   void *xmap, *vmap;
   buffer(xfloats * 4u, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT, &xb, &xm, &xmap);
   float *xf = (float *)xmap;
   for (size_t k = 0; k < xfloats; k++) {
      xf[k] = -7.0f;
   }
   buffer(c.cps.size() * sizeof(cp_vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &vmap);
   memcpy(vmap, c.cps.data(), c.cps.size() * sizeof(cp_vertex));
   VkCommandBufferAllocateInfo cai = {};
   cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cai.commandPool = vk.pool;
   cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cai.commandBufferCount = 1;
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {};
   bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkClearValue clear = {};
   VkRenderPassBeginInfo rbi = {};
   rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   rbi.renderPass = vk.rp;
   rbi.framebuffer = vk.fb;
   rbi.renderArea = { { 0, 0 }, { W, H } };
   rbi.clearValueCount = 1;
   rbi.pClearValues = &clear;
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   VkDeviceSize off = 0, xsize = xfloats * 4u;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
   vk.bind_xfb(cmd, 0, 1, &xb, &off, &xsize);
   vk.begin_xfb(cmd, 0, 0, NULL, NULL);
   const VkPipeline order[3] = { pa, pb, pa };
   for (VkPipeline p : order) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
      vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, INST_STEP);
      vkCmdDraw(cmd, (uint32_t)c.cps.size(), 1, 0, 0);
   }
   vk.end_xfb(cmd, 0, 0, NULL, NULL);
   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {};
   si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   si.commandBufferCount = 1;
   si.pCommandBuffers = &cmd;
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   unsigned bad = 0;
   for (size_t v = 0; v < nv + 4u; v++) {
      for (unsigned k = 0; k < 4; k++) {
         const float want_f = v < nv ? ref[v % ref.size()].col[k] : -7.0f;
         if (memcmp(&xf[4 * v + k], &want_f, 4) != 0 && bad++ < 4) {
            printf("  handoff: captured vertex %zu (draw %zu) component %u = %g, expected %g\n",
                   v, v / ref.size(), k, xf[4 * v + k], want_f);
         }
      }
   }
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, pa, NULL);
   vkDestroyPipeline(vk.dev, pb, NULL);
   vkDestroyBuffer(vk.dev, xb, NULL);
   vkFreeMemory(vk.dev, xm, NULL);
   vkDestroyBuffer(vk.dev, vb, NULL);
   vkFreeMemory(vk.dev, vm, NULL);
   printf("handoff: %s (%zu reference vertices a draw)\n", bad ? "FAIL" : "ok", ref.size());
   return bad;
}

/* levels_max: every patch's every domain point evaluated, once, at its own
 * place, however many batches and ranges of a batch the draw takes: `npatch`
 * patches at level 64. */
static unsigned
run_levels_max(uint32_t npatch)
{
   const uint32_t per = 65u * 65u;
   const float lo[4] = { 64.0f, 64.0f, 64.0f, 64.0f }, li[2] = { 64.0f, 64.0f };
   std::vector<cp_vertex> cps;
   for (uint32_t k = 0; k < npatch; k++) {
      cell(cps, k % 64u, k / 64u, 64, lo, li);
   }
   const size_t nslot = (size_t)npatch * per;
   VkBuffer sb;
   VkDeviceMemory sm;
   void *smap;
   buffer(nslot * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &sb, &sm, &smap);
   memset(smap, 0, nslot * 4u);
   VkBuffer vb;
   VkDeviceMemory vm;
   void *vmap;
   buffer(cps.size() * sizeof(cp_vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &vmap);
   memcpy(vmap, cps.data(), cps.size() * sizeof(cp_vertex));

   VkDescriptorSetLayoutBinding bind = {};
   bind.binding = 0;
   bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   bind.descriptorCount = 1;
   bind.stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
   VkDescriptorSetLayoutCreateInfo dlci = {};
   dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
   dlci.bindingCount = 1;
   dlci.pBindings = &bind;
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(vk.dev, &dlci, NULL, &dsl));
   VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = {};
   dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
   dpci.maxSets = 1;
   dpci.poolSizeCount = 1;
   dpci.pPoolSizes = &psz;
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(vk.dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dai = {};
   dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
   dai.descriptorPool = dp;
   dai.descriptorSetCount = 1;
   dai.pSetLayouts = &dsl;
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(vk.dev, &dai, &ds));
   VkDescriptorBufferInfo dbi = { sb, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds = {};
   wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
   wds.dstSet = ds;
   wds.descriptorCount = 1;
   wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   wds.pBufferInfo = &dbi;
   vkUpdateDescriptorSets(vk.dev, 1, &wds, 0, NULL);
   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT, 0, 8 };
   VkPipelineLayoutCreateInfo plci = {};
   plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
   plci.setLayoutCount = 1;
   plci.pSetLayouts = &dsl;
   plci.pushConstantRangeCount = 1;
   plci.pPushConstantRanges = &pcr;
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &pl));

   VkShaderModule vs = module("tess.vert.spv"), tcs = module("tess.tesc.spv");
   VkShaderModule tes = module("max.tese.spv");
   const VkSpecializationMapEntry map[2] = { { 0, 0, 4 }, { 7, 4, 4 } };
   const uint32_t spec[2] = { 1u, UINT32_MAX };
   VkSpecializationInfo si = { 2, map, sizeof spec, spec };
   VkPipelineShaderStageCreateInfo st[3] = {};
   const VkShaderStageFlagBits stage[3] = { VK_SHADER_STAGE_VERTEX_BIT,
                                            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT };
   const VkShaderModule mod[3] = { vs, tcs, tes };
   for (unsigned k = 0; k < 3; k++) {
      st[k].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st[k].stage = stage[k];
      st[k].module = mod[k];
      st[k].pName = "main";
      st[k].pSpecializationInfo = &si;
   }
   VkVertexInputBindingDescription vbd = { 0, (uint32_t)sizeof(cp_vertex),
                                           VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription attr[3] = {
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
      { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8 },
      { 2, 0, VK_FORMAT_R32G32_SFLOAT, 24 },
   };
   VkPipelineVertexInputStateCreateInfo vi = {};
   vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
   vi.vertexBindingDescriptionCount = 1;
   vi.pVertexBindingDescriptions = &vbd;
   vi.vertexAttributeDescriptionCount = 3;
   vi.pVertexAttributeDescriptions = attr;
   VkPipelineInputAssemblyStateCreateInfo ia = {};
   ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
   ia.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   VkPipelineTessellationStateCreateInfo ts = {};
   ts.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
   ts.patchControlPoints = 4;
   VkViewport vp = { 0, 0, W, H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {};
   vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
   vps.viewportCount = 1;
   vps.pViewports = &vp;
   vps.scissorCount = 1;
   vps.pScissors = &sc;
   VkPipelineRasterizationStateCreateInfo rs = {};
   rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
   rs.rasterizerDiscardEnable = VK_TRUE;
   rs.polygonMode = VK_POLYGON_MODE_FILL;
   rs.cullMode = VK_CULL_MODE_NONE;
   rs.lineWidth = 1.0f;
   VkPipelineMultisampleStateCreateInfo ms = {};
   ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkGraphicsPipelineCreateInfo gp = {};
   gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
   gp.stageCount = 3;
   gp.pStages = st;
   gp.pVertexInputState = &vi;
   gp.pInputAssemblyState = &ia;
   gp.pTessellationState = &ts;
   gp.pViewportState = &vps;
   gp.pRasterizationState = &rs;
   gp.pMultisampleState = &ms;
   gp.layout = pl;
   gp.renderPass = vk.rp;
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
   for (VkShaderModule m : mod) {
      vkDestroyShaderModule(vk.dev, m, NULL);
   }

   VkCommandBufferAllocateInfo cai = {};
   cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cai.commandPool = vk.pool;
   cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cai.commandBufferCount = 1;
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {};
   bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkClearValue clear = {};
   VkRenderPassBeginInfo rbi = {};
   rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   rbi.renderPass = vk.rp;
   rbi.framebuffer = vk.fb;
   rbi.renderArea = { { 0, 0 }, { W, H } };
   rbi.clearValueCount = 1;
   rbi.pClearValues = &clear;
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, NULL);
   vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, INST_STEP);
   VkDeviceSize off = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
   vkCmdDraw(cmd, (uint32_t)cps.size(), 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkMemoryBarrier mb = {};
   mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
   mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo sub = {};
   sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   sub.commandBufferCount = 1;
   sub.pCommandBuffers = &cmd;
   CHECK(vkQueueSubmit(vk.queue, 1, &sub, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));

   const uint32_t *seen = (const uint32_t *)smap;
   unsigned bad = 0;
   for (size_t k = 0; k < nslot; k++) {
      const uint32_t want = (uint32_t)k + (bite ? 2u : 1u);
      if (seen[k] != want && bad++ < 4) {
         printf("  levels_max: patch %zu point (%zu, %zu) = %u, expected %u\n", k / per,
                k % 65u, k / 65u % 65u, seen[k], want);
      }
   }
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyPipelineLayout(vk.dev, pl, NULL);
   vkDestroyDescriptorPool(vk.dev, dp, NULL);
   vkDestroyDescriptorSetLayout(vk.dev, dsl, NULL);
   vkDestroyBuffer(vk.dev, vb, NULL);
   vkFreeMemory(vk.dev, vm, NULL);
   vkDestroyBuffer(vk.dev, sb, NULL);
   vkFreeMemory(vk.dev, sm, NULL);
   printf("levels_max: %s (%zu domain points)\n", bad ? "FAIL" : "ok", nslot);
   return bad;
}

int
main(int argc, char **argv)
{
   /* -c <name>: that case alone; -n <patches>: levels_max's patches. */
   const char *only = NULL;
   uint32_t max_patches = LEVELS_MAX_PATCHES;
   for (int k = 1; k + 1 < argc; k++) {
      if (strcmp(argv[k], "-c") == 0) {
         only = argv[k + 1];
      }
      if (strcmp(argv[k], "-n") == 0) {
         max_patches = (uint32_t)atoi(argv[k + 1]);
      }
   }
   bite = getenv("TESS_BITE") && atoi(getenv("TESS_BITE"));
   init();

   std::vector<tess_case> cases;
   static const char *const SPACE[3] = { "equal", "odd", "even" };
   static const char *const DOM[3] = { "tri", "quad", "iso" };
   const float LO[3][4] = { { 3.0f, 5.5f, 2.0f, 4.25f }, { 2.75f, 1.5f, 6.25f, 3.0f },
                            { 4.0f, 2.25f, 3.5f, 5.0f } };
   const float LI[3][2] = { { 3.75f, 2.5f }, { 5.0f, 1.75f }, { 2.5f, 4.5f } };

   /* Static levels: each domain, spacing and winding, in four cells. */
   for (int d = 0; d < 3; d++) {
      for (uint32_t s = 0; s < 3; s++) {
         for (int w = 0; w < (d == ISO ? 1 : 2); w++) {
            tess_case c = {};
            char name[64], tese[64];
            format(name, sizeof name, "static_%s_%s%s", DOM[d], SPACE[s],
                     d == ISO ? "" : w ? "_ccw" : "_cw");
            format(tese, sizeof tese, "%s_%s%s.tese", DOM[d], SPACE[s],
                     d == ISO ? "" : w ? "_ccw" : "_cw");
            c.name = strdup(name);
            c.tese = strdup(tese);
            c.dom = (enum domain)d;
            c.spacing = s;
            c.ccw = w != 0;
            memcpy(c.lo, LO[s], sizeof c.lo);
            memcpy(c.li, LI[s], sizeof c.li);
            for (unsigned k = 0; k < 4; k++) {
               cell(c.cps, k & 1, k >> 1, 2, c.lo, c.li, 1.0f / 32.0f);
            }
            c.patch_size = 4;
            cases.push_back(c);
         }
      }
   }
   /* Point mode, static. */
   for (int d = 0; d < 3; d++) {
      tess_case c = {};
      char name[64], tese[64];
      format(name, sizeof name, "static_%s_points", DOM[d]);
      format(tese, sizeof tese, "%s_points.tese", DOM[d]);
      c.name = strdup(name);
      c.tese = strdup(tese);
      c.dom = (enum domain)d;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.points = true;
      memcpy(c.lo, LO[1], sizeof c.lo);
      memcpy(c.li, LI[1], sizeof c.li);
      cell(c.cps, 0, 0, 1, c.lo, c.li, 1.0f / 16.0f);
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* Dynamic levels: different per patch, the minimum patch, and patches the
    * outer levels discard. */
   const float DLO[9][4] = {
      { 1.0f, 1.0f, 1.0f, 1.0f }, { 2.5f, 7.0f, 3.25f, 1.5f }, { 0.0f, 3.0f, 3.0f, 3.0f },
      { 6.0f, 6.0f, 6.0f, 6.0f }, { 3.0f, -1.0f, 4.0f, 2.0f }, { 1.75f, 4.5f, 9.0f, 2.0f },
      { 3.0f, 3.0f, NAN, 3.0f },  { 12.0f, 2.0f, 5.5f, 7.25f }, { 4.0f, 8.0f, 2.0f, 16.0f },
   };
   const float DLI[9][2] = {
      { 1.0f, 1.0f }, { 4.5f, 2.0f }, { 3.0f, 3.0f }, { 6.0f, 6.0f }, { 2.0f, 2.0f },
      { 1.0f, 7.5f }, { 3.0f, 3.0f }, { 9.0f, 3.25f }, { 5.0f, 5.0f },
   };
   for (int d = 0; d < 3; d++) {
      for (uint32_t s = 0; s < 3; s++) {
         tess_case c = {};
         char name[64], tese[64];
         format(name, sizeof name, "dynamic_%s_%s", DOM[d], SPACE[s]);
         format(tese, sizeof tese, "%s_%s%s.tese", DOM[d], SPACE[s], d == ISO ? "" : "_ccw");
         c.name = strdup(name);
         c.tese = strdup(tese);
         c.dom = (enum domain)d;
         c.spacing = s;
         c.ccw = true;
         c.dyn = true;
         for (unsigned k = 0; k < 9; k++) {
            cell(c.cps, k % 3, k / 3, 3, DLO[k], DLI[k], 1.0f / 64.0f);
         }
         c.patch_size = 4;
         cases.push_back(c);
      }
   }
   {
      tess_case c = {};
      c.name = "dynamic_quad_points";
      c.tese = "quad_points.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.points = true;
      c.dyn = true;
      for (unsigned k = 0; k < 4; k++) {
         cell(c.cps, k & 1, k >> 1, 2, DLO[k + 1], DLI[k + 1], 1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* The control stage's outputs, barrier and neighbour reads. */
   {
      tess_case c = {};
      c.name = "pass";
      c.tese = "pass.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_EQUAL;
      c.ccw = true;
      c.pass = true;
      for (unsigned k = 0; k < 4; k++) {
         cell(c.cps, k & 1, k >> 1, 2, DLO[k + 5 > 8 ? 8 : k + 5], DLI[k + 1], 1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* Three input control points make four. */
   {
      tess_case c = {};
      c.name = "tri_in";
      c.tese = "quad_even_ccw.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_EVEN;
      c.ccw = true;
      c.dyn = true;
      std::vector<cp_vertex> four;
      for (unsigned k = 0; k < 4; k++) {
         cell(four, k & 1, k >> 1, 2, DLO[k + 1], DLI[k + 1], 1.0f / 32.0f);
      }
      for (size_t i = 0; i < four.size(); i++) {
         if (i % 4 != 3) {
            c.cps.push_back(four[i]);
         }
      }
      c.patch_size = 3;
      cases.push_back(c);
   }
   /* Overlapping patches blended in order, by gl_PrimitiveID, two instances. */
   {
      tess_case c = {};
      c.name = "order";
      c.tese = "quad_id.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      c.blend = true;
      for (unsigned k = 0; k < 6; k++) {
         const float lo[4] = { 1.0f + (float)k, 3.5f, 2.0f + 0.5f * (float)k, 5.0f };
         const float li[2] = { 2.5f + (float)k, 3.0f };
         cell(c.cps, 0, 0, 1, lo, li, (float)k / 32.0f);
      }
      c.patch_size = 4;
      c.instances = 2;
      cases.push_back(c);
   }
   /* Many patches, every one its own levels. */
   {
      tess_case c = {};
      c.name = "many";
      c.tese = "tri_odd_cw.tese";
      c.dom = TRI;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = false;
      c.dyn = true;
      for (unsigned k = 0; k < 64; k++) {
         const float lo[4] = { 1.0f + (float)(k % 5), 1.0f + (float)(k % 7) * 0.5f,
                               2.0f + (float)(k % 3), 1.0f };
         const float li[2] = { 1.0f + (float)(k % 4) * 1.25f, 1.0f };
         cell(c.cps, k % 8, k / 8, 8, lo, li, 0.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }

   /* A draw-wide bound past 2^32 primitives: every patch at the most a
    * patch makes, times every invocation's declared most. */
   {
      tess_case c = {};
      c.name = "gs_bound";
      c.tese = "quad_odd_ccw.tese";
      c.geom = "gs_amp.geom.spv";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      cell(c.cps, 0, 0, 1, DLO[7], DLI[7], 1.0f / 16.0f);
      c.patch_size = 4;
      c.instances = 160;
      c.last_live = true;
      cases.push_back(c);
   }

   /* A geometry shader after the tessellation stages, for each primitive. */
   static const struct { const char *name, *tese, *geom; enum domain dom; bool points; }
   GSC[3] = {
      { "gs_quad", "quad_odd_ccw.tese", "gs_tri.geom.spv", QUAD, false },
      { "gs_iso", "iso_even.tese", "gs_line.geom.spv", ISO, false },
      { "gs_points", "tri_points.tese", "gs_point.geom.spv", TRI, true },
   };
   for (const auto &g : GSC) {
      tess_case c = {};
      c.name = g.name;
      c.tese = g.tese;
      c.geom = g.geom;
      c.dom = g.dom;
      c.spacing = g.points || g.dom == QUAD ? GFX_TESS_SPACING_ODD : GFX_TESS_SPACING_EVEN;
      c.ccw = true;
      c.points = g.points;
      c.dyn = true;
      for (unsigned k = 0; k < 4; k++) {
         cell(c.cps, k & 1, k >> 1, 2, DLO[k + 1], DLI[k + 1], 1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* Evaluation records wider than the geometry stage's own, over enough
    * domain points that a buffer sized by the narrower stride would overflow. */
   {
      tess_case c = {};
      c.name = "gs_wide";
      c.tese = "quad_odd_ccw_wide.tese";
      c.geom = "gs_tri.geom.spv";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      const float wlo[4][4] = { { 15.0f, 13.5f, 16.0f, 14.25f }, { 16.0f, 12.0f, 15.5f, 16.0f },
                                { 13.0f, 16.0f, 14.0f, 15.0f }, { 16.0f, 16.0f, 16.0f, 16.0f } };
      const float wli[4][2] = { { 15.0f, 16.0f }, { 14.5f, 13.0f }, { 16.0f, 15.25f },
                                { 16.0f, 16.0f } };
      for (unsigned k = 0; k < 4; k++) {
         cell(c.cps, k & 1, k >> 1, 2, wlo[k], wli[k], 1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* The evaluation stage captured, static and dynamic levels. */
   for (int k = 0; k < 2; k++) {
      tess_case c = {};
      c.name = k ? "xfb_iso_dynamic" : "xfb_quad_static";
      c.tese = k ? "iso_even_xfb.tese" : "quad_equal_ccw_xfb.tese";
      c.dom = k ? ISO : QUAD;
      c.spacing = k ? GFX_TESS_SPACING_EVEN : GFX_TESS_SPACING_EQUAL;
      c.ccw = true;
      c.dyn = k != 0;
      c.xfb = true;
      memcpy(c.lo, LO[0], sizeof c.lo);
      memcpy(c.li, LI[0], sizeof c.li);
      for (unsigned j = 0; j < 4; j++) {
         cell(c.cps, j & 1, j >> 1, 2, k ? DLO[j + 4] : c.lo, k ? DLI[j + 4] : c.li,
              1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }

   /* The capture with room for one primitive fewer than the draw makes. */
   for (size_t k = 0, xfb0 = cases.size() - 2; k < 2; k++) {
      tess_case c = cases[xfb0 + k];
      c.name = k ? "xfb_overflow_dynamic" : "xfb_overflow_static";
      c.xfb_short = 1;
      cases.push_back(c);
   }
   /* Rounds of neighbour reads between barriers: at the top level, several
    * patches a CTA, and in a loop, one. */
   for (int k = 0; k < 2; k++) {
      tess_case c = {};
      c.name = k ? "barrier_nested" : "barrier";
      c.tese = "pass.tese";
      c.tesc = k ? "barrier_nested.tesc.spv" : "barrier.tesc.spv";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_EQUAL;
      c.ccw = true;
      c.pass = true;
      c.rounds = 3;
      for (unsigned j = 0; j < 16; j++) {
         cell(c.cps, j & 3, j >> 2, 4, DLO[j % 9], DLI[(j + 3) % 9], 1.0f / 64.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* 16 and 32 control points in and out: two patches a warp, and one. */
   for (uint32_t m : { 16u, 32u }) {
      tess_case c = {};
      c.name = m == 16u ? "cp16" : "cp32";
      c.tese = "quad_odd_ccw.tese";
      c.tesc = m == 16u ? "tess_m16.tesc.spv" : "tess_m32.tesc.spv";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      for (unsigned j = 0; j < 4; j++) {
         std::vector<cp_vertex> four;
         cell(four, j & 1, j >> 1, 2, DLO[j + 1], DLI[j + 1], 1.0f / 32.0f);
         for (uint32_t i = 0; i < m; i++) {
            c.cps.push_back(four[i & 3]);
         }
      }
      c.patch_size = m;
      cases.push_back(c);
   }
   /* The stages' pipeline statistics, with levels known on the host and only
    * on the device, culled patches among them. */
   for (int k = 0; k < 2; k++) {
      tess_case c = {};
      c.name = k ? "stats_dynamic" : "stats_static";
      c.tese = "quad_even_cw.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_EVEN;
      c.ccw = false;
      c.dyn = k != 0;
      c.stats = true;
      memcpy(c.lo, LO[2], sizeof c.lo);
      memcpy(c.li, LI[2], sizeof c.li);
      for (unsigned j = 0; j < 9; j++) {
         cell(c.cps, j % 3, j / 3, 3, k ? DLO[j] : c.lo, k ? DLI[j] : c.li, 1.0f / 64.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* Constant outer levels and no inner ones, which the quad domain reads: the
    * levels are the device's, nothing rasterized -- counted all the same --
    * and through a geometry stage. */
   for (int k = 0; k < 2; k++) {
      tess_case c = {};
      c.name = k ? "stats_outer_gs" : "stats_outer_discard";
      c.tese = "quad_odd_ccw.tese";
      c.tesc = "outer.tesc.spv";
      c.geom = k ? "gs_tri.geom.spv" : NULL;
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      c.discard = k == 0;
      c.stats = true;
      const float lo[4] = { 3.0f, 2.0f, 4.0f, 5.0f }, li[2] = { 0.0f, 0.0f };
      for (unsigned j = 0; j < 4; j++) {
         cell(c.cps, j & 1, j >> 1, 2, lo, li, 1.0f / 32.0f);
      }
      c.patch_size = 4;
      cases.push_back(c);
   }
   /* Draws in many batches and, with the levels the device's, ranges of a
    * batch: a budget of a few patches. Levels known on the host; levels that
    * grow patch by patch, so the records grow and move; a capture, its sizes
    * flipping halves at each commit; instances, gl_PrimitiveID wrapping. */
   {
      tess_case c = {};
      c.name = "batches_static";
      c.tese = "tri_odd_ccw.tese";
      c.dom = TRI;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      memcpy(c.lo, LO[1], sizeof c.lo);
      memcpy(c.li, LI[1], sizeof c.li);
      for (unsigned j = 0; j < 16; j++) {
         cell(c.cps, j & 3, j >> 2, 4, c.lo, c.li, 1.0f / 64.0f);
      }
      c.patch_size = 4;
      c.batch_bytes = "2048";
      cases.push_back(c);
   }
   {
      tess_case c = {};
      c.name = "batches_dynamic";
      c.tese = "quad_even_ccw.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_EVEN;
      c.ccw = true;
      c.dyn = true;
      for (unsigned j = 0; j < 16; j++) {
         const float g = 2.0f + 0.75f * (float)j;
         const float lo[4] = { g, g + 1.0f, g * 0.5f + 1.0f, g }, li[2] = { g, g + 0.5f };
         cell(c.cps, j & 3, j >> 2, 4, lo, li, 1.0f / 64.0f);
      }
      c.patch_size = 4;
      c.batch_bytes = "2048";
      cases.push_back(c);
   }
   {
      tess_case c = cases[cases.size() - 1];
      c.name = "batches_dynamic_gs";
      c.geom = "gs_tri.geom.spv";
      cases.push_back(c);
   }
   /* An indexed patch list, its patches out of order, two instances. */
   {
      tess_case c = {};
      c.name = "indexed";
      c.tese = "quad_id.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      c.blend = true;
      c.stats = true;
      for (unsigned k = 0; k < 4; k++) {
         const float lo[4] = { 3.0f + (float)k, 2.5f, 1.0f + (float)k, 4.0f };
         const float li[2] = { 2.0f + (float)k, 3.0f };
         cell(c.cps, 0, 0, 1, lo, li, (float)k / 24.0f);
      }
      c.indices = { 12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7 };
      c.patch_size = 4;
      c.instances = 2;
      cases.push_back(c);
   }
   /* Patch lists cut by a restart index, whose patches only the device
    * counts: runs ending in an incomplete patch, instances coloured by
    * gl_PrimitiveID; levels every patch shares, which only place the patches
    * when the host knows them too; and every index a restart. */
   {
      tess_case c = {};
      c.name = "restart";
      c.tese = "quad_id.tese";
      c.dom = QUAD;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.dyn = true;
      c.blend = true;
      c.stats = true;
      for (unsigned k = 0; k < 6; k++) {
         const float lo[4] = { 2.0f + (float)k, 3.5f, 1.0f + 0.75f * (float)k, 4.0f };
         const float li[2] = { 1.5f + (float)k, 2.0f };
         cell(c.cps, 0, 0, 1, lo, li, (float)k / 32.0f);
      }
      c.indices = { 0, 1, 2, 3, RESTART, 4, 5, RESTART, 4, 5, 6, 7, 8, 9, 10, 11, 12, RESTART,
                    RESTART, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, RESTART, 20, 21, 22, 23 };
      c.patch_size = 4;
      c.instances = 2;
      cases.push_back(c);
   }
   {
      tess_case c = {};
      c.name = "restart_static";
      c.tese = "tri_odd_ccw.tese";
      c.dom = TRI;
      c.spacing = GFX_TESS_SPACING_ODD;
      c.ccw = true;
      c.stats = true;
      memcpy(c.lo, LO[1], sizeof c.lo);
      memcpy(c.li, LI[1], sizeof c.li);
      for (unsigned j = 0; j < 4; j++) {
         cell(c.cps, j & 1, j >> 1, 2, c.lo, c.li, 1.0f / 32.0f);
      }
      c.indices = { 0, 1, 2, RESTART, 4, 5, 6, 7, 8, 9, 10, RESTART, 12, 13, 14, 15 };
      c.patch_size = 4;
      cases.push_back(c);
   }
   {
      tess_case c = cases.back();
      c.name = "restart_all";
      c.indices.assign(16, RESTART);
      cases.push_back(c);
   }

   for (size_t i = 0, n = cases.size(); i < n; i++) {
      if (!strcmp(cases[i].name, "xfb_iso_dynamic") ||
          !strcmp(cases[i].name, "xfb_quad_static")) {
         tess_case c = cases[i];
         c.name = c.dyn ? "batches_xfb_dynamic" : "batches_xfb_static";
         c.batch_bytes = "1024";
         cases.push_back(c);
      }
      if (!strcmp(cases[i].name, "order")) {
         tess_case c = cases[i];
         c.name = "batches_instanced";
         c.instances = 3;
         c.batch_bytes = "1024";
         cases.push_back(c);
      }
      if (!strcmp(cases[i].name, "restart")) {
         tess_case c = cases[i];
         c.name = "batches_restart";
         c.batch_bytes = "1024";
         cases.push_back(c);
      }
   }

   /* Cases whose levels the host knows: the run's vortexpipe-tess line must
    * attribute as many draws to that path (TESS_STATIC in the Makefile). */
   unsigned failed = 0, ran = 0, host_levels = 0;
   for (const tess_case &c : cases) {
      if (only && strcmp(only, c.name) != 0) {
         continue;
      }
      failed += run_case(c) ? 1u : 0u;
      ran++;
      host_levels += !c.dyn && !c.pass && !c.tesc &&
                     !std::count(c.indices.begin(), c.indices.end(), RESTART);
   }
   if (!only || strcmp(only, "multiview") == 0) {
      failed += run_multiview() ? 1u : 0u;
      ran++;
   }
   if (only && strcmp(only, "handoff") == 0) {
      failed += run_handoff() ? 1u : 0u;
      ran++;
   }
   if (only && strcmp(only, "levels_max") == 0) {
      failed += run_levels_max(max_patches) ? 1u : 0u;
      ran++;
   }
   teardown();
   if (failed || !ran) {
      printf("FAILED (tess: %u of %u cases)\n", failed, ran);
      return 1;
   }
   printf("PASSED (tess: %u cases, %u with levels known on the host)\n", ran, host_levels);
   return 0;
}
