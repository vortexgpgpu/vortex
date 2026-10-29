/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Task and mesh shaders on the device.
 *
 * Every mesh workgroup w emits primitives k = 0, 1, ... each in screen cell
 * prim_cell(w, k) (mesh_common.glsl) and colour prim_colour(w, k), so many
 * workgroups' primitives land in the same cells. Each case is drawn twice into
 * a cleared 64x64 target: once plainly, where a cell shows the last primitive
 * drawn into it, and once with a depth test every later fragment fails, where
 * it shows the first. The host builds both images from the order the
 * specification gives -- task, then mesh workgroup, then primitive index --
 * so a primitive drawn out of order, lost, duplicated or misplaced changes a
 * pixel. Triangles are compared pixel for pixel; a line or point is checked
 * for its colour on every pixel of its cell it covers.
 *
 *   tri       one workgroup, one triangle (static count)
 *   static    a 2x2x2 grid, 32 triangles per workgroup (static count)
 *   dyn       a 3x2x2 grid, 0 to 32 triangles per workgroup
 *   line      lines, the counts of dyn
 *   point     points, the counts of dyn
 *   cull      dyn with every odd primitive culled (CullPrimitiveEXT)
 *   shared    dyn's triangles built through workgroup memory and a barrier
 *   task      tasks launching 0, 1, 24 and 3 mesh workgroups, which read their
 *             numbering from the task payload
 *   taskfix   tasks that all launch two mesh workgroups
 *   perprim   per-primitive colour, gl_PrimitiveID and gl_ViewportIndex
 *             (four viewports, each scissored to a quadrant)
 *   indirect  dyn's grid through vkCmdDrawMeshTasksIndirectEXT -- a zero grid
 *             between two draws, a 32-byte stride, gl_DrawID numbering each
 *             draw's workgroups -- and through the count variant with a count
 *             below the maximum
 *   indwin    window's grid through the count variant: the device sizes its
 *             windows from the grid it reads
 *   indover   the same over a 120x60 grid, more workgroups than the windows
 *             recorded for an indirect draw hold: the rest is drawn after
 *             (MESH_BIG=1)
 *   big       a 40x30 grid of 32 points each, more than a front-end pass holds
 *   bigdyn    the same with a run-time count of 30, whose total is read back
 *             (big and bigdyn take ~150M SimX cycles each: they run alone, with
 *             MESH_BIG=1)
 *   window    workgroups whose declared outputs need more memory than one
 *             window of them may take, so the draw runs in several
 *   maxmem    the largest mesh workgroup the device advertises, with all the
 *             workgroup memory it advertises
 *   partial   a static count whose odd primitives' indices are left unwritten,
 *             drawn after a draw that wrote them all: the device makes them
 *             degenerate, so they draw nothing (the specification leaves them
 *             undefined; llvmpipe draws them from the indices left behind)
 *   taskbig   2100 tasks with the largest payload, more than one task chunk
 *   taskwin   tasks whose mesh workgroups take several mesh windows
 *   discard   cull's draw with rasterizer discard: nothing drawn
 *   layer     dyn's triangles each on the layer its gl_Layer names, into a
 *             two-layer framebuffer
 *   queries   task and mesh invocations and mesh primitives generated, exact,
 *             for task, taskfix, cull, static, window and discard
 *
 * MESH_CASE selects cases by name, a comma-separated list.
 *
 * MESH_BITE=1 builds every expectation the way a wrong implementation would
 * draw it (the primitives in reverse order, or the counts off by one); every
 * case must then fail.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64u
#define H 64u
#define CLEAR 0x00000000u
#define CELLS 16u
#define MAX_PRIMS 65536u
#define LAYERS 2u

#define CHECK(x) do {                                                     \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      printf("FAILED (mesh: %s -> VkResult %d)\n", #x, (int)_r);           \
      exit(1);                                                            \
   }                                                                      \
} while (0)

static bool bite;

/* ---- the numbering the shaders use (mesh_common.glsl) -------------------- */

static uint32_t
wg_prims(uint32_t w)
{
   return (w * 7u + 3u) % 33u;
}

static uint32_t
prim_cell(uint32_t w, uint32_t k)
{
   return (w * 13u + k * 5u) % CELLS;
}

static uint32_t
prim_colour(uint32_t w, uint32_t k)
{
   return 0xff000000u | (((w << 8) | k) + 1u);
}

static uint32_t
big_pixel(uint32_t w, uint32_t k)
{
   return ((w * 32u + k) * 37u) % 4096u;
}

/* task.task's grids: task t launches task_grid(t). */
static const uint32_t TASK_GRID[4][3] = { { 0, 1, 1 }, { 1, 1, 1 }, { 2, 3, 4 }, { 3, 1, 1 } };

/* The mesh workgroups task t of taskbig.task and taskwin.task launches. */
static uint32_t
taskbig_launch(uint32_t t)
{
   return t % 3u;
}

static uint32_t
taskwin_launch(uint32_t t)
{
   return 250u + 50u * (t % 3u);
}

/* ---- host reference ---------------------------------------------------- */

enum topo { TOPO_TRI, TOPO_LINE, TOPO_POINT, TOPO_PIXEL };
enum model { M_ONE, M_STATIC, M_DYN, M_PERPRIM, M_PC, M_FAT, M_TASKBIG };

/* mcase.flags */
#define C_MAXMEM  (1u << 0)   /* local size and workgroup memory at the limits */
#define C_PARTIAL (1u << 1)   /* two draws, the second leaving odd indices unwritten */
#define C_DISCARD (1u << 2)   /* rasterizer discard */
#define C_LAYER   (1u << 3)   /* the two-layer framebuffer */

struct prim {
   uint32_t cell;      /* TOPO_PIXEL: the pixel */
   uint32_t colour;
   uint32_t half;      /* perprim: 0 the upper-left triangle of the quad, 1 the other */
   uint32_t viewport;  /* perprim: the quadrant it is scissored to */
   uint32_t layer;     /* layer: the framebuffer layer */
};

struct px { double x, y; };

static double
edge(struct px a, struct px b, double x, double y)
{
   return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

static void
fill_tri(uint32_t *img, const struct px v[3], uint32_t c, int quadrant)
{
   for (unsigned j = 0; j < H; j++) {
      for (unsigned i = 0; i < W; i++) {
         if (quadrant >= 0 && (int)((i / 32u) + 2u * (j / 32u)) != quadrant)
            continue;
         const double x = i + 0.5, y = j + 0.5;
         const double e0 = edge(v[0], v[1], x, y), e1 = edge(v[1], v[2], x, y),
                      e2 = edge(v[2], v[0], x, y);
         if (e0 == 0 || e1 == 0 || e2 == 0) {
            const bool in = (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
            if (in) {
               printf("FAILED (mesh: test geometry puts pixel (%u,%u) on an edge)\n", i, j);
               exit(1);
            }
            continue;
         }
         if ((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0))
            img[j * W + i] = c;
      }
   }
}

static struct px
cell_px(uint32_t cell, double x, double y)
{
   const struct px p = { (cell % 4u) * 16.0 + x, (cell / 4u) * 16.0 + y };
   return p;
}

/* The colour a perprim primitive's fragment writes (perprim.frag). Biting, its
 * gl_PrimitiveID is the primitive's index in the draw's list instead. */
static uint32_t
perprim_colour(const struct prim *p, uint32_t w, uint32_t k, uint32_t index)
{
   const uint32_t vert = p->half ? 0x10u : 0x00u;   /* its provoking vertex's value */
   const uint32_t id = (bite ? index : 1000u * w + k) & 0xffu;
   return (p->colour & 0xffffu) | (id << 16) | (vert << 24);
}

/* Draw one primitive into the reference. */
static void
ref_prim(uint32_t *img, enum topo topo, const struct prim *p, uint32_t colour, bool perprim)
{
   if (topo == TOPO_PIXEL) {
      img[p->cell] = colour;
      return;
   }
   if (topo != TOPO_TRI)
      return;
   if (perprim) {
      const struct px q[4] = { cell_px(p->cell, 2, 2), cell_px(p->cell, 14, 2),
                               cell_px(p->cell, 2, 8), cell_px(p->cell, 14, 8) };
      const struct px t0[3] = { q[0], q[1], q[2] }, t1[3] = { q[1], q[3], q[2] };
      fill_tri(img, p->half ? t1 : t0, colour, (int)p->viewport);
      return;
   }
   const struct px v[3] = { cell_px(p->cell, 2, 2), cell_px(p->cell, 14, 2),
                            cell_px(p->cell, 2, 8) };
   fill_tri(img, v, colour, -1);
}

/* ---- Vulkan ---------------------------------------------------------- */

static struct {
   VkInstance inst;
   VkPhysicalDevice phys;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
   VkCommandPool pool;
   VkImage img, depth, limg, ldepth;
   VkDeviceMemory img_mem, depth_mem, limg_mem, ldepth_mem;
   VkImageView view, depth_view, lview, ldepth_view;
   VkRenderPass rp;
   VkFramebuffer fb, lfb;
   uint32_t max_mesh_inv, max_mesh_shared;
   VkBuffer rb;
   VkDeviceMemory rb_mem;
   void *rb_map;
   VkPipelineLayout layout;
   VkBuffer ind;
   VkDeviceMemory ind_mem;
   uint32_t *ind_map;
   PFN_vkCmdDrawMeshTasksEXT draw;
   PFN_vkCmdDrawMeshTasksIndirectEXT draw_ind;
   PFN_vkCmdDrawMeshTasksIndirectCountEXT draw_ind_count;
} vk;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk.mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (vk.mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   printf("FAILED (mesh: no memory type)\n");
   exit(1);
}

static VkShaderModule
module(const char *name)
{
   char path[128];
   snprintf(path, sizeof path, "%s.spv", name);
   FILE *f = fopen(path, "rb");
   if (!f) {
      printf("FAILED (mesh: cannot open %s)\n", path);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc((size_t)sz);
   if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
      printf("FAILED (mesh: cannot read %s)\n", path);
      exit(1);
   }
   fclose(f);
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = (size_t)sz, .pCode = code };
   VkShaderModule m;
   CHECK(vkCreateShaderModule(vk.dev, &ci, NULL, &m));
   free(code);
   return m;
}

static void
buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *b, VkDeviceMemory *m, void **map)
{
   VkBufferCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = usage };
   CHECK(vkCreateBuffer(vk.dev, &ci, NULL, b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(vk.dev, *b, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, m));
   CHECK(vkBindBufferMemory(vk.dev, *b, *m, 0));
   CHECK(vkMapMemory(vk.dev, *m, 0, size, 0, map));
}

static void
image(VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, uint32_t layers,
      VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = layers,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = usage };
   CHECK(vkCreateImage(vk.dev, &ci, NULL, img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk.dev, *img, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits, 0) };
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, mem));
   CHECK(vkBindImageMemory(vk.dev, *img, *mem, 0));
   VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = *img,
                                 .viewType = layers > 1u ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                         : VK_IMAGE_VIEW_TYPE_2D,
                                 .format = fmt,
                                 .subresourceRange = { aspect, 0, 1, 0, layers } };
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, view));
}

static void
init(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   CHECK(vkCreateInstance(&ici, NULL, &vk.inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(vk.inst, &n, &vk.phys);
   if (!n) {
      printf("FAILED (mesh: no device)\n");
      exit(1);
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(vk.phys, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.mp);
   VkPhysicalDeviceMeshShaderPropertiesEXT mprops = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT };
   VkPhysicalDeviceProperties2 props2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                          .pNext = &mprops };
   vkGetPhysicalDeviceProperties2(vk.phys, &props2);
   vk.max_mesh_inv = mprops.maxMeshWorkGroupInvocations;
   vk.max_mesh_shared = mprops.maxMeshSharedMemorySize;

   VkPhysicalDeviceMeshShaderFeaturesEXT have_mesh = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
   VkPhysicalDeviceFeatures2 have = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      .pNext = &have_mesh };
   vkGetPhysicalDeviceFeatures2(vk.phys, &have);
   if (!have_mesh.taskShader || !have_mesh.meshShader || !have_mesh.meshShaderQueries ||
       !have.features.multiViewport || !have.features.pipelineStatisticsQuery) {
      printf("FAILED (mesh: device lacks task or mesh shaders, their queries, multiple "
             "viewports or pipeline statistics)\n");
      exit(1);
   }

   uint32_t nq = 8;
   VkQueueFamilyProperties qp[8];
   vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &nq, qp);
   for (vk.qf = 0; vk.qf < nq && !(qp[vk.qf].queueFlags & VK_QUEUE_GRAPHICS_BIT); vk.qf++)
      ;
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = vk.qf, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .taskShader = VK_TRUE, .meshShader = VK_TRUE, .meshShaderQueries = VK_TRUE };
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &mesh,
      .drawIndirectCount = VK_TRUE };
   VkPhysicalDeviceFeatures2 feat = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      .pNext = &v12 };
   feat.features.multiViewport = VK_TRUE;
   feat.features.pipelineStatisticsQuery = VK_TRUE;
   const char *ext[] = { VK_EXT_MESH_SHADER_EXTENSION_NAME };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feat,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext };
   CHECK(vkCreateDevice(vk.phys, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, vk.qf, 0, &vk.queue);
   vk.draw = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(vk.dev, "vkCmdDrawMeshTasksEXT");
   vk.draw_ind = (PFN_vkCmdDrawMeshTasksIndirectEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdDrawMeshTasksIndirectEXT");
   vk.draw_ind_count = (PFN_vkCmdDrawMeshTasksIndirectCountEXT)
      vkGetDeviceProcAddr(vk.dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   if (!vk.draw || !vk.draw_ind || !vk.draw_ind_count) {
      printf("FAILED (mesh: no vkCmdDrawMeshTasks*EXT)\n");
      exit(1);
   }

   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = vk.qf };
   CHECK(vkCreateCommandPool(vk.dev, &pci, NULL, &vk.pool));

   image(VK_FORMAT_R8G8B8A8_UNORM,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         VK_IMAGE_ASPECT_COLOR_BIT, 1, &vk.img, &vk.img_mem, &vk.view);
   image(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
         VK_IMAGE_ASPECT_DEPTH_BIT, 1, &vk.depth, &vk.depth_mem, &vk.depth_view);
   image(VK_FORMAT_R8G8B8A8_UNORM,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         VK_IMAGE_ASPECT_COLOR_BIT, LAYERS, &vk.limg, &vk.limg_mem, &vk.lview);
   image(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
         VK_IMAGE_ASPECT_DEPTH_BIT, LAYERS, &vk.ldepth, &vk.ldepth_mem, &vk.ldepth_view);

   VkAttachmentDescription att[2] = {
      { .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = VK_FORMAT_D32_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                .colorAttachmentCount = 1, .pColorAttachments = &cref,
                                .pDepthStencilAttachment = &dref };
   VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 2, .pAttachments = att,
                                   .subpassCount = 1, .pSubpasses = &sub };
   CHECK(vkCreateRenderPass(vk.dev, &rpci, NULL, &vk.rp));
   VkImageView views[2] = { vk.view, vk.depth_view };
   VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                   .renderPass = vk.rp, .attachmentCount = 2,
                                   .pAttachments = views, .width = W, .height = H,
                                   .layers = 1 };
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &vk.fb));
   VkImageView lviews[2] = { vk.lview, vk.ldepth_view };
   fci.pAttachments = lviews;
   fci.layers = LAYERS;
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &vk.lfb));
   buffer(LAYERS * W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &vk.rb, &vk.rb_mem, &vk.rb_map);
   void *map;
   buffer(256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &vk.ind, &vk.ind_mem, &map);
   vk.ind_map = map;

   VkPushConstantRange pcr = { VK_SHADER_STAGE_MESH_BIT_EXT, 0, 4 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &vk.layout));
}

struct mcase {
   const char *name;
   const char *task;          /* NULL: no task stage */
   const char *mesh;
   const char *frag;          /* NULL: flat.frag */
   uint32_t grid[3];
   enum topo topo;
   enum model model;
   bool cull;                 /* odd primitives culled */
   int tasks;                 /* 0: none; 1: task.task's grids; 2: two per task */
   int indirect;              /* 0: direct; 1: indirect; 2: indirect count */
   uint32_t dyn;              /* the shader's DYN specialization constant */
   uint32_t pc;               /* its push constant: M_PC's primitives per workgroup */
   bool big;                  /* run with MESH_BIG=1 only */
   uint32_t flags;            /* C_* */
   uint32_t ts_inv, ms_inv;   /* queries: invocations per task and mesh workgroup */
};

static VkPipeline
pipeline(const struct mcase *c, bool first)
{
   VkShaderModule ms = module(c->mesh), fs = module(c->frag ? c->frag : "flat.frag");
   VkShaderModule ts = c->task ? module(c->task) : VK_NULL_HANDLE;
   /* DYN; with C_MAXMEM, the local size and the words of workgroup memory. */
   const uint32_t spec[3] = { c->dyn, vk.max_mesh_inv, vk.max_mesh_shared / 4u };
   VkSpecializationMapEntry sme[3] = { { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 } };
   VkSpecializationInfo spi = { (c->flags & C_MAXMEM) ? 3u : 1u, sme, sizeof spec, spec };
   VkPipelineShaderStageCreateInfo st[3] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_MESH_BIT_EXT, .module = ms, .pName = "main",
        .pSpecializationInfo = &spi },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_TASK_BIT_EXT, .module = ts, .pName = "main" },
   };
   const bool perprim = c->model == M_PERPRIM;
   VkViewport vp[4];
   VkRect2D sc[4];
   for (unsigned i = 0; i < 4; i++) {
      vp[i] = (VkViewport){ 0, 0, (float)W, (float)H, 0, 1 };
      sc[i] = perprim ? (VkRect2D){ { (int32_t)(32u * (i % 2u)), (int32_t)(32u * (i / 2u)) },
                                    { 32u, 32u } }
                      : (VkRect2D){ { 0, 0 }, { W, H } };
   }
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = perprim ? 4u : 1u, .pViewports = vp,
      .scissorCount = perprim ? 4u : 1u, .pScissors = sc };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .rasterizerDiscardEnable = (c->flags & C_DISCARD) != 0,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo msi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = first, .depthWriteEnable = first,
      .depthCompareOp = VK_COMPARE_OP_LESS };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = c->task ? 3u : 2u, .pStages = st,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &msi,
      .pDepthStencilState = &ds, .pColorBlendState = &cb, .layout = vk.layout,
      .renderPass = vk.rp };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gci, NULL, &p));
   vkDestroyShaderModule(vk.dev, ms, NULL);
   vkDestroyShaderModule(vk.dev, fs, NULL);
   if (ts)
      vkDestroyShaderModule(vk.dev, ts, NULL);
   return p;
}

/* The mesh workgroups of a case's draw, numbered as its shaders number them,
 * in the order the specification gives: the tasks in order, every task's mesh
 * workgroups in order; an indirect draw's draws in order, the second after a
 * zero grid and so numbered as draw 2. */
static uint32_t
case_workgroups(const struct mcase *c, uint32_t *wl)
{
   uint32_t n = 0;
   if (c->tasks == 1) {
      for (uint32_t t = 0; t < c->grid[0]; t++)
         n += TASK_GRID[t % 4u][0] * TASK_GRID[t % 4u][1] * TASK_GRID[t % 4u][2];
   } else if (c->tasks == 2) {
      n = 2u * c->grid[0];
   } else if (c->tasks == 3 || c->tasks == 4) {
      for (uint32_t t = 0; t < c->grid[0]; t++)
         n += c->tasks == 3 ? taskbig_launch(t) : taskwin_launch(t);
   } else {
      n = c->grid[0] * c->grid[1] * c->grid[2];
   }
   for (uint32_t w = 0; wl && w < n; w++)
      wl[w] = w;
   if (c->indirect == 1) {
      for (uint32_t w = 0; wl && w < n; w++)
         wl[n + w] = 2u * n + w;
      n *= 2u;
   }
   return n;
}

static uint32_t
case_prims(const struct mcase *c, struct prim *out)
{
   static uint32_t wl[MAX_PRIMS];
   const uint32_t wgs = case_workgroups(c, wl);
   uint32_t n = 0;
   /* partial: the draw writing every primitive, then the one writing the even
    * ones in the other colour. */
   const uint32_t draws = (c->flags & C_PARTIAL) ? 2u : 1u;
   for (uint32_t i = 0; i < wgs * draws; i++) {
      const uint32_t w = wl[i % wgs], d = i / wgs;
      const uint32_t np = c->model == M_ONE ? 1u : c->model == M_STATIC ? 32u
                        : c->model == M_PERPRIM ? 2u * (w % 3u + 1u)
                        : c->model == M_PC ? c->pc
                        : c->model == M_FAT ? wg_prims(w) % 4u
                        : c->model == M_TASKBIG ? wg_prims(w) % 3u : wg_prims(w);
      for (uint32_t k = 0; k < np; k++) {
         if (c->cull && (k & 1u))
            continue;
         if (d == 1u && (k & 1u))
            continue;
         /* A perprim quad's two triangles share the cell of its first. */
         const uint32_t cell = c->model == M_ONE ? 5u
                             : c->topo == TOPO_PIXEL ? big_pixel(w, k)
                             : prim_cell(w, c->model == M_PERPRIM ? k & ~1u : k);
         struct prim p = { cell, prim_colour(w, k) ^ (d == 1u ? 0x00800000u : 0u), k & 1u,
                           k % 4u, (c->flags & C_LAYER) ? (w + k) & 1u : 0u };
         if (c->model == M_PERPRIM)
            p.colour = perprim_colour(&p, w, k, n);
         if (n >= MAX_PRIMS) {
            printf("FAILED (mesh: %s makes more primitives than the test holds)\n", c->name);
            exit(1);
         }
         out[n++] = p;
      }
   }
   return n;
}

/* What the draw leaves, layer by layer: the first or the last primitive of
 * every pixel, the draw order reversed when biting; nothing under rasterizer
 * discard, unless biting. */
static void
reference(const struct mcase *c, bool first, uint32_t *img, uint32_t cell_col[CELLS],
          bool cell_used[CELLS])
{
   static struct prim prims[MAX_PRIMS];
   const uint32_t n = (c->flags & C_DISCARD) && !bite ? 0u : case_prims(c, prims);
   for (unsigned i = 0; i < LAYERS * W * H; i++)
      img[i] = CLEAR;
   memset(cell_used, 0, CELLS * sizeof(bool));
   const bool rev = first != bite;
   for (uint32_t i = 0; i < n; i++) {
      const struct prim *p = &prims[rev ? n - 1u - i : i];
      if (c->model == M_ONE && bite)
         p = &(struct prim){ 5u, prim_colour(0u, 1u), 0u, 0u, 0u };
      ref_prim(img + p->layer * W * H, c->topo, p, p->colour, c->model == M_PERPRIM);
      if (c->topo != TOPO_PIXEL) {
         cell_col[p->cell] = p->colour;
         cell_used[p->cell] = true;
      }
   }
}

/* Record a case's draw, with its grids in the indirect buffer for an indirect
 * draw: the grid, then a zero grid, then the grid again for the plain form;
 * the count form reads one of a maximum of three. */
static void
record_draw(VkCommandBuffer cmd, const struct mcase *c)
{
   if (c->flags & C_PARTIAL) {
      const uint32_t all = 1u, evens = 0u;
      vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_MESH_BIT_EXT, 0, 4, &all);
      vk.draw(cmd, c->grid[0], c->grid[1], c->grid[2]);
      vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_MESH_BIT_EXT, 0, 4, &evens);
      vk.draw(cmd, c->grid[0], c->grid[1], c->grid[2]);
      return;
   }
   vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_MESH_BIT_EXT, 0, 4, &c->pc);
   if (!c->indirect) {
      vk.draw(cmd, c->grid[0], c->grid[1], c->grid[2]);
      return;
   }
   uint32_t *g = vk.ind_map;
   memset(g, 0, 256);
   for (unsigned k = 0; k < 3; k++) {
      g[k] = c->grid[k];
      g[16 + k] = c->grid[k];
   }
   g[8] = 0;
   g[60] = 1;   /* the draw count, at byte 240 */
   if (c->indirect == 1)
      vk.draw_ind(cmd, vk.ind, 0, 3, 32);
   else
      vk.draw_ind_count(cmd, vk.ind, 0, vk.ind, 240, 3, 32);
}

/* Is `name` one of the comma-separated names of `list`? */
static bool
case_named(const char *list, const char *name)
{
   const size_t n = strlen(name);
   for (const char *p = list; *p;) {
      const char *e = strchr(p, ',');
      const size_t len = e ? (size_t)(e - p) : strlen(p);
      if (len == n && !strncmp(p, name, n))
         return true;
      if (!e)
         break;
      p = e + 1;
   }
   return false;
}

static bool
run_case(const struct mcase *c)
{
   bool ok = true;
   for (int f = 0; f < 2 && ok; f++) {
      const bool first = f == 1;
      VkPipeline p = pipeline(c, first);
      VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 1 };
      VkCommandBuffer cmd;
      CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
      VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
      CHECK(vkBeginCommandBuffer(cmd, &bi));
      VkClearValue clr[2] = { { .color = { .uint32 = { 0, 0, 0, 0 } } },
                              { .depthStencil = { 1.0f, 0 } } };
      const bool layered = (c->flags & C_LAYER) != 0;
      VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                    .renderPass = vk.rp, .framebuffer = layered ? vk.lfb : vk.fb,
                                    .renderArea = { { 0, 0 }, { W, H } },
                                    .clearValueCount = 2, .pClearValues = clr };
      vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
      record_draw(cmd, c);
      vkCmdEndRenderPass(cmd);
      const uint32_t layers = layered ? LAYERS : 1u;
      VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers },
                                .imageExtent = { W, H, 1 } };
      vkCmdCopyImageToBuffer(cmd, layered ? vk.limg : vk.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             vk.rb, 1, &reg);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                          .pCommandBuffers = &cmd };
      CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(vk.queue));
      vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
      vkDestroyPipeline(vk.dev, p, NULL);

      static uint32_t ref[LAYERS * W * H];
      uint32_t cell_col[CELLS];
      bool cell_used[CELLS];
      reference(c, first, ref, cell_col, cell_used);
      const uint32_t *got = vk.rb_map;
      unsigned bad = 0, first_bad = 0;
      if (c->topo == TOPO_TRI || c->topo == TOPO_PIXEL) {
         for (unsigned i = 0; i < layers * W * H; i++) {
            if (got[i] != ref[i] && !bad++)
               first_bad = i;
         }
      } else {
         /* Every covered pixel of a cell shows the cell's primitive, and a cell
          * with one is covered somewhere. */
         unsigned covered[CELLS] = { 0 };
         for (unsigned i = 0; i < W * H; i++) {
            const unsigned cell = (i % W) / 16u + 4u * ((i / W) / 16u);
            if (got[i] == CLEAR)
               continue;
            covered[cell]++;
            if ((!cell_used[cell] || got[i] != cell_col[cell]) && !bad++)
               first_bad = i;
         }
         for (unsigned k = 0; k < CELLS; k++) {
            if (cell_used[k] && !covered[k] && !bad++)
               first_bad = (k % 4u) * 16u + (k / 4u) * 16u * W;
         }
      }
      if (bad) {
         printf("  %s (%s): %u pixels wrong, first (%u,%u) of layer %u = %08x, expected %08x\n",
                c->name, first ? "first" : "last", bad, first_bad % W, (first_bad / W) % H,
                first_bad / (W * H), got[first_bad], ref[first_bad]);
         ok = false;
      }
   }
   return ok;
}

/* Pipeline statistics and mesh primitives generated over one draw. */
static bool
run_queries(const struct mcase *c)
{
   VkQueryPool qp[2];
   VkQueryPoolCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                 .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS, .queryCount = 1,
                                 .pipelineStatistics =
                                    VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
                                    VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT };
   CHECK(vkCreateQueryPool(vk.dev, &qci, NULL, &qp[0]));
   qci.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
   qci.pipelineStatistics = 0;
   CHECK(vkCreateQueryPool(vk.dev, &qci, NULL, &qp[1]));
   VkPipeline p = pipeline(c, false);
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = vk.pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   vkCmdResetQueryPool(cmd, qp[0], 0, 1);
   vkCmdResetQueryPool(cmd, qp[1], 0, 1);
   VkClearValue clr[2] = { { .color = { .uint32 = { 0, 0, 0, 0 } } },
                           { .depthStencil = { 1.0f, 0 } } };
   VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                 .renderPass = vk.rp, .framebuffer = vk.fb,
                                 .renderArea = { { 0, 0 }, { W, H } },
                                 .clearValueCount = 2, .pClearValues = clr };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   vkCmdBeginQuery(cmd, qp[0], 0, 0);
   vkCmdBeginQuery(cmd, qp[1], 0, 0);
   record_draw(cmd, c);
   vkCmdEndQuery(cmd, qp[1], 0);
   vkCmdEndQuery(cmd, qp[0], 0);
   vkCmdEndRenderPass(cmd);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   uint64_t stats[2] = { 0, 0 }, prims = 0;
   CHECK(vkGetQueryPoolResults(vk.dev, qp[0], 0, 1, sizeof stats, stats, sizeof stats,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   CHECK(vkGetQueryPoolResults(vk.dev, qp[1], 0, 1, sizeof prims, &prims, sizeof prims,
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkDestroyQueryPool(vk.dev, qp[0], NULL);
   vkDestroyQueryPool(vk.dev, qp[1], NULL);

   static struct prim scratch[MAX_PRIMS];
   const uint64_t want_task = c->task ? (uint64_t)c->grid[0] * c->grid[1] * c->grid[2] * c->ts_inv
                                      : 0u;
   const uint64_t want_mesh = (uint64_t)case_workgroups(c, NULL) * c->ms_inv;
   const uint64_t want_prims = case_prims(c, scratch) + (bite ? 1u : 0u);
   if (stats[0] != want_task || stats[1] != want_mesh || prims != want_prims) {
      printf("  %s (queries): task %llu mesh %llu primitives %llu, expected %llu %llu %llu\n",
             c->name, (unsigned long long)stats[0], (unsigned long long)stats[1],
             (unsigned long long)prims, (unsigned long long)want_task,
             (unsigned long long)want_mesh, (unsigned long long)want_prims);
      return false;
   }
   return true;
}

int
main(void)
{
   bite = getenv("MESH_BITE") && atoi(getenv("MESH_BITE"));
   init();
   static const struct mcase cases[] = {
      { "tri",      NULL, "tri.mesh",         NULL, { 1, 1, 1 }, TOPO_TRI,   M_ONE,    false, 0, 0 },
      { "static",   NULL, "grid_static.mesh", NULL, { 2, 2, 2 }, TOPO_TRI,   M_STATIC, false, 0, 0 },
      { "dyn",      NULL, "grid_dyn.mesh",    NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    false, 0, 0 },
      { "line",     NULL, "grid_line.mesh",   NULL, { 3, 2, 2 }, TOPO_LINE,  M_DYN,    false, 0, 0 },
      { "point",    NULL, "grid_point.mesh",  NULL, { 3, 2, 2 }, TOPO_POINT, M_DYN,    false, 0, 0 },
      { "cull",     NULL, "grid_cull.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    true,  0, 0 },
      { "shared",   NULL, "shared.mesh",      NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    false, 0, 0 },
      { "task",     "task.task", "task.mesh", NULL, { 4, 1, 1 }, TOPO_TRI,   M_DYN,    false, 1, 0 },
      { "taskfix",  "task_fixed.task", "task.mesh", NULL, { 5, 1, 1 }, TOPO_TRI, M_DYN, false, 2, 0 },
      { "perprim",  NULL, "perprim.mesh", "perprim.frag", { 3, 2, 1 }, TOPO_TRI, M_PERPRIM, false, 0, 0 },
      { "indirect", NULL, "grid_draw.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    false, 0, 1 },
      { "indcount", NULL, "grid_draw.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    false, 0, 2 },
      { "indwin",   NULL, "fat.mesh",         NULL, { 50, 26, 1 }, TOPO_TRI,   M_FAT,    false, 0, 2 },
      { "indover",  NULL, "fat.mesh",         NULL, { 120, 60, 1 }, TOPO_TRI,  M_FAT,    false, 0, 2,
        0, 0, true },
      { "big",      NULL, "big.mesh",         NULL, { 40, 30, 1 }, TOPO_PIXEL, M_STATIC, false, 0, 0,
        0, 0, true },
      { "bigdyn",   NULL, "big.mesh",         NULL, { 40, 30, 1 }, TOPO_PIXEL, M_PC,     false, 0, 0,
        1, 30, true },
      { "window",   NULL, "fat.mesh",         NULL, { 50, 26, 1 }, TOPO_TRI,   M_FAT,    false, 0, 0 },
      { "maxmem",   NULL, "maxmem.mesh",      NULL, { 2, 1, 1 }, TOPO_TRI,   M_PC,     false, 0, 0,
        0, 8, false, C_MAXMEM },
      { "partial",  NULL, "partial.mesh",     NULL, { 2, 2, 1 }, TOPO_TRI,   M_STATIC, false, 0, 0,
        0, 0, false, C_PARTIAL },
      { "taskbig",  "taskbig.task", "taskbig.mesh", NULL, { 2100, 1, 1 }, TOPO_TRI, M_TASKBIG, false,
        3, 0 },
      { "taskwin",  "taskwin.task", "taskwin.mesh", NULL, { 5, 1, 1 }, TOPO_TRI, M_FAT, false, 4, 0 },
      { "discard",  NULL, "grid_cull.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    true,  0, 0,
        0, 0, false, C_DISCARD },
      { "layer",    NULL, "layer.mesh",       NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    false, 0, 0,
        0, 0, false, C_LAYER },
   };
   /* With their shaders' invocations per task and mesh workgroup. */
   static const struct mcase qcases[] = {
      { "task",     "task.task", "task.mesh", NULL, { 4, 1, 1 }, TOPO_TRI,   M_DYN,    false, 1, 0,
        0, 0, false, 0, 4, 8 },
      { "taskfix",  "task_fixed.task", "task.mesh", NULL, { 5, 1, 1 }, TOPO_TRI, M_DYN, false, 2, 0,
        0, 0, false, 0, 8, 8 },
      { "cull",     NULL, "grid_cull.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    true,  0, 0,
        0, 0, false, 0, 0, 8 },
      { "static",   NULL, "grid_static.mesh", NULL, { 2, 2, 2 }, TOPO_TRI,   M_STATIC, false, 0, 0,
        0, 0, false, 0, 0, 32 },
      { "window",   NULL, "fat.mesh",         NULL, { 50, 26, 1 }, TOPO_TRI,   M_FAT,    false, 0, 0,
        0, 0, false, 0, 0, 32 },
      { "discard",  NULL, "grid_cull.mesh",   NULL, { 3, 2, 2 }, TOPO_TRI,   M_DYN,    true,  0, 0,
        0, 0, false, C_DISCARD, 0, 8 },
   };
   const char *only = getenv("MESH_CASE");
   const bool big = getenv("MESH_BIG") && atoi(getenv("MESH_BIG"));
   unsigned fails = 0, runs = 0;
   for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
      if ((only && !case_named(only, cases[i].name)) || (!only && cases[i].big != big))
         continue;
      runs++;
      const bool ok = run_case(&cases[i]);
      printf("%s: %s\n", cases[i].name, ok ? "ok" : "MISMATCH");
      fails += !ok;
   }
   const char *qonly = getenv("MESH_QCASE");
   for (unsigned i = 0; i < sizeof qcases / sizeof qcases[0]; i++) {
      if ((only && !case_named(only, "queries")) || (!only && big) ||
          (qonly && strcmp(qonly, qcases[i].name)))
         continue;
      runs++;
      const bool ok = run_queries(&qcases[i]);
      printf("queries %s: %s\n", qcases[i].name, ok ? "ok" : "MISMATCH");
      fails += !ok;
   }
   vkDeviceWaitIdle(vk.dev);
   vkDestroyPipelineLayout(vk.dev, vk.layout, NULL);
   vkDestroyBuffer(vk.dev, vk.ind, NULL);
   vkFreeMemory(vk.dev, vk.ind_mem, NULL);
   vkDestroyBuffer(vk.dev, vk.rb, NULL);
   vkFreeMemory(vk.dev, vk.rb_mem, NULL);
   vkDestroyFramebuffer(vk.dev, vk.fb, NULL);
   vkDestroyFramebuffer(vk.dev, vk.lfb, NULL);
   vkDestroyRenderPass(vk.dev, vk.rp, NULL);
   vkDestroyImageView(vk.dev, vk.view, NULL);
   vkDestroyImageView(vk.dev, vk.depth_view, NULL);
   vkDestroyImageView(vk.dev, vk.lview, NULL);
   vkDestroyImageView(vk.dev, vk.ldepth_view, NULL);
   vkDestroyImage(vk.dev, vk.img, NULL);
   vkDestroyImage(vk.dev, vk.depth, NULL);
   vkDestroyImage(vk.dev, vk.limg, NULL);
   vkDestroyImage(vk.dev, vk.ldepth, NULL);
   vkFreeMemory(vk.dev, vk.img_mem, NULL);
   vkFreeMemory(vk.dev, vk.depth_mem, NULL);
   vkFreeMemory(vk.dev, vk.limg_mem, NULL);
   vkFreeMemory(vk.dev, vk.ldepth_mem, NULL);
   vkDestroyCommandPool(vk.dev, vk.pool, NULL);
   vkDestroyDevice(vk.dev, NULL);
   vkDestroyInstance(vk.inst, NULL);
   if (fails || !runs) {
      printf("FAILED (mesh: %u of %u cases)\n", fails, runs);
      return 1;
   }
   printf("PASSED (mesh: %u cases)\n", runs);
   return 0;
}
