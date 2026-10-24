/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Geometry shaders on the device.
 *
 * Every case draws into a cleared 64x64 target and is compared pixel for pixel
 * with an image the host builds from what the specification says the geometry
 * stage produces: its output strips split into triangles that keep their
 * winding, each drawn in the colour of its provoking vertex, culled by the
 * same facing rule. Vertices sit on whole pixels and every triangle's slanted
 * edge runs two pixels across per pixel down (or the reverse), so no pixel
 * centre lies on an edge and coverage needs no tie rule; the host rasterizer
 * aborts the test if a centre ever does.
 *
 *   pass_*   triangles, lines and points through a shader that passes them on
 *            with their colours' channels swapped, so a draw that skipped the
 *            stage would not match; triangles in both provoking-vertex modes
 *   strip_*  from each input triangle: a three-cell strip (odd triangles wound
 *            back), drawn back-face culled so a triangle wound the wrong way
 *            disappears; a two-vertex strip that never completes; a separate
 *            triangle written over the incomplete strip's records; a lone
 *            last vertex. Every vertex has its own colour, so each triangle
 *            shows which vertex provoked it; both provoking-vertex modes
 *   quad     points expanded into quads
 *   point6   points drawn 6 pixels wide by the geometry stage
 *   polyline triangles from the geometry stage in polygon mode LINE, compared
 *            with the same triangles drawn without it
 *
 * GEOM_BITE=1 builds every expectation the way a wrong implementation would
 * draw it (the stage skipped, the other provoking vertex, odd triangles culled,
 * the incomplete strip drawn); every case must then fail.
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

#define CHECK(x) do {                                                     \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      printf("FAILED (geometry: %s -> VkResult %d)\n", #x, (int)_r);       \
      exit(1);                                                            \
   }                                                                      \
} while (0)

static bool bite;

/* ---- host reference ---------------------------------------------------- */

/* A colour as the target stores it: R in the low byte. */
static uint32_t
rgba(unsigned r, unsigned g, unsigned b, unsigned a)
{
   return r | (g << 8) | (b << 16) | (a << 24);
}

/* The channel swap pass_* applies (vcol.bgra). */
static uint32_t
swap_rb(uint32_t c)
{
   return (c & 0xff00ff00u) | ((c >> 16) & 0xffu) | ((c & 0xffu) << 16);
}

struct px { double x, y; };

/* Vulkan's signed area in framebuffer coordinates (y down). */
static double
area(const struct px v[3])
{
   double a = 0;
   for (int i = 0; i < 3; i++) {
      const struct px *p = &v[i], *q = &v[(i + 1) % 3];
      a += p->x * q->y - q->x * p->y;
   }
   return -0.5 * a;
}

static double
edge(struct px a, struct px b, double x, double y)
{
   return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

static void
fill_tri(uint32_t *img, const struct px v[3], uint32_t c)
{
   for (unsigned j = 0; j < H; j++) {
      for (unsigned i = 0; i < W; i++) {
         const double x = i + 0.5, y = j + 0.5;
         const double e0 = edge(v[0], v[1], x, y), e1 = edge(v[1], v[2], x, y),
                      e2 = edge(v[2], v[0], x, y);
         if (e0 == 0 || e1 == 0 || e2 == 0) {
            const bool in = (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
            if (in) {
               printf("FAILED (geometry: test geometry puts pixel (%u,%u) on an edge)\n", i, j);
               exit(1);
            }
            continue;
         }
         if ((e0 > 0 && e1 > 0 && e2 > 0) || (e0 < 0 && e1 < 0 && e2 < 0))
            img[j * W + i] = c;
      }
   }
}

/* Pixels whose centres lie in [x0, x1) x [y0, y1). */
static void
fill_rect(uint32_t *img, double x0, double y0, double x1, double y1, uint32_t c)
{
   for (unsigned j = 0; j < H; j++)
      for (unsigned i = 0; i < W; i++)
         if (i + 0.5 > x0 && i + 0.5 < x1 && j + 0.5 > y0 && j + 0.5 < y1)
            img[j * W + i] = c;
}

/* A triangle as the rasterizer takes it: culled by facing, drawn in its
 * provoking vertex's colour. cull_back with a counter-clockwise front face. */
static void
draw_tri(uint32_t *img, struct px a, struct px b, struct px c, uint32_t col, bool cull_back)
{
   const struct px v[3] = { a, b, c };
   if (cull_back && !(area(v) > 0))
      return;
   fill_tri(img, v, col);
}

/* ---- Vulkan ---------------------------------------------------------- */

struct gs_vertex {
   float x, y;       /* NDC */
   uint32_t rgba;
   uint32_t n;
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
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkBuffer ssbo;
   VkDeviceMemory ssbo_mem;
   uint32_t *ssbo_map;
   VkImage tex;
   VkDeviceMemory tex_mem;
   VkImageView tex_view;
   VkSampler sampler;
} vk;

/* The per-point counts the texture holds, red channel, texel i for input
 * point i. */
static const uint8_t TEX_COUNTS[16] = { 2, 6, 0, 4, 1, 5, 3, 0, 6, 2, 1, 0, 4, 3, 5, 1 };

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk.mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (vk.mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   printf("FAILED (geometry: no memory type)\n");
   exit(1);
}

static VkShaderModule
module(const char *name)
{
   FILE *f = fopen(name, "rb");
   if (!f) {
      printf("FAILED (geometry: cannot open %s)\n", name);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc((size_t)sz);
   if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
      printf("FAILED (geometry: cannot read %s)\n", name);
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
init(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   CHECK(vkCreateInstance(&ici, NULL, &vk.inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(vk.inst, &n, &vk.phys);
   if (!n) {
      printf("FAILED (geometry: no device)\n");
      exit(1);
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(vk.phys, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.mp);

   VkPhysicalDeviceFeatures have;
   vkGetPhysicalDeviceFeatures(vk.phys, &have);
   if (!have.geometryShader || !have.shaderTessellationAndGeometryPointSize ||
       !have.fillModeNonSolid) {
      printf("FAILED (geometry: device lacks geometryShader, geometry point size or "
             "non-solid fill)\n");
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
   VkPhysicalDeviceFeatures feat = { .geometryShader = VK_TRUE,
                                     .shaderTessellationAndGeometryPointSize = VK_TRUE,
                                     .fillModeNonSolid = VK_TRUE };
   VkPhysicalDeviceProvokingVertexFeaturesEXT pvf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
      .provokingVertexLast = VK_TRUE };
   const char *ext[] = { VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &pvf,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext,
                              .pEnabledFeatures = &feat };
   CHECK(vkCreateDevice(vk.phys, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, vk.qf, 0, &vk.queue);

   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = vk.qf };
   CHECK(vkCreateCommandPool(vk.dev, &pci, NULL, &vk.pool));

   VkImageCreateInfo imci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
   CHECK(vkCreateImage(vk.dev, &imci, NULL, &vk.img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk.dev, vk.img, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits, 0) };
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &vk.img_mem));
   CHECK(vkBindImageMemory(vk.dev, vk.img, vk.img_mem, 0));
   VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = vk.img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                 .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, &vk.view));

   VkAttachmentDescription att = { .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                   .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                   .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                   .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
   VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                .colorAttachmentCount = 1, .pColorAttachments = &ref };
   VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 1, .pAttachments = &att,
                                   .subpassCount = 1, .pSubpasses = &sub };
   CHECK(vkCreateRenderPass(vk.dev, &rpci, NULL, &vk.rp));
   VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                   .renderPass = vk.rp, .attachmentCount = 1,
                                   .pAttachments = &vk.view, .width = W, .height = H,
                                   .layers = 1 };
   CHECK(vkCreateFramebuffer(vk.dev, &fci, NULL, &vk.fb));
   buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &vk.rb, &vk.rb_mem, &vk.rb_map);

   /* The geometry stage's storage buffer and texture, and its push constants. */
   VkDescriptorSetLayoutBinding binds[2] = {
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_GEOMETRY_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_GEOMETRY_BIT, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = binds };
   CHECK(vkCreateDescriptorSetLayout(vk.dev, &dslci, NULL, &vk.dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_GEOMETRY_BIT, 0, 8 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1, .pSetLayouts = &vk.dsl,
                                       .pushConstantRangeCount = 1,
                                       .pPushConstantRanges = &pcr };
   CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &vk.layout));

   void *map;
   buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk.ssbo, &vk.ssbo_mem, &map);
   vk.ssbo_map = map;

   VkImageCreateInfo tci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                             .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                             .extent = { 16, 1, 1 }, .mipLevels = 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT,
                             .tiling = VK_IMAGE_TILING_OPTIMAL,
                             .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT };
   CHECK(vkCreateImage(vk.dev, &tci, NULL, &vk.tex));
   vkGetImageMemoryRequirements(vk.dev, vk.tex, &mr);
   ai.allocationSize = mr.size;
   ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, 0);
   CHECK(vkAllocateMemory(vk.dev, &ai, NULL, &vk.tex_mem));
   CHECK(vkBindImageMemory(vk.dev, vk.tex, vk.tex_mem, 0));
   vci.image = vk.tex;
   CHECK(vkCreateImageView(vk.dev, &vci, NULL, &vk.tex_view));
   VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                               .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                               .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
   CHECK(vkCreateSampler(vk.dev, &sci, NULL, &vk.sampler));

   /* Upload the counts and leave the texture shader-readable. */
   VkBuffer stage;
   VkDeviceMemory stage_mem;
   buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stage, &stage_mem, &map);
   for (unsigned i = 0; i < 16; i++)
      ((uint32_t *)map)[i] = rgba(TEX_COUNTS[i], 0, 0, 255);
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = vk.pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkImageMemoryBarrier to_dst = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = vk.tex,
                                   .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &to_dst);
   VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { 16, 1, 1 } };
   vkCmdCopyBufferToImage(cmd, stage, vk.tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
   VkImageMemoryBarrier to_ro = to_dst;
   to_ro.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   to_ro.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   to_ro.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   to_ro.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &to_ro);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyBuffer(vk.dev, stage, NULL);
   vkFreeMemory(vk.dev, stage_mem, NULL);

   VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
                                  { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps };
   CHECK(vkCreateDescriptorPool(vk.dev, &dpci, NULL, &vk.dpool));
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = vk.dpool, .descriptorSetCount = 1,
                                        .pSetLayouts = &vk.dsl };
   CHECK(vkAllocateDescriptorSets(vk.dev, &dsai, &vk.set));
   VkDescriptorBufferInfo dbi = { vk.ssbo, 0, 16 };
   VkDescriptorImageInfo dii = { vk.sampler, vk.tex_view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   VkWriteDescriptorSet wr[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk.set, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk.set, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
   };
   vkUpdateDescriptorSets(vk.dev, 2, wr, 0, NULL);
}

struct gs_case {
   const char *name;
   const char *geom;              /* NULL: no geometry stage */
   VkPrimitiveTopology topo;
   const struct gs_vertex *v;
   uint32_t nv;
   VkCullModeFlags cull;
   VkPolygonMode poly;
   bool last;                     /* provoking vertex last */
   uint32_t pc[2];                /* push constants */
   const char *vert, *frag;       /* NULL: gs.vert, flat.frag */
   const uint16_t *idx;           /* indexed draw, 0xffff restarting with `restart` */
   uint32_t nidx;
   bool restart;
   uint32_t instances;            /* 0: one */
};

static VkPipeline
pipeline(const struct gs_case *c)
{
   char vname[64], fname[64];
   snprintf(vname, sizeof vname, "%s.spv", c->vert ? c->vert : "gs.vert");
   snprintf(fname, sizeof fname, "%s.spv", c->frag ? c->frag : "flat.frag");
   VkShaderModule vs = module(vname), fs = module(fname);
   VkShaderModule gs = c->geom ? module(c->geom) : VK_NULL_HANDLE;
   VkPipelineShaderStageCreateInfo st[3] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_GEOMETRY_BIT, .module = gs, .pName = "main" },
   };
   VkVertexInputBindingDescription bind = { 0, sizeof(struct gs_vertex),
                                            VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription attr[3] = {
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct gs_vertex, x) },
      { 1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(struct gs_vertex, rgba) },
      { 2, 0, VK_FORMAT_R32_UINT, offsetof(struct gs_vertex, n) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind,
      .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attr };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = c->topo, .primitiveRestartEnable = c->restart };
   VkViewport vp = { 0, 0, W, H, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
      .provokingVertexMode = c->last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                     : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .pNext = &pv,
      .polygonMode = c->poly, .cullMode = c->cull,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkGraphicsPipelineCreateInfo gp = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = c->geom ? 3u : 2u, .pStages = st, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = vk.layout,
      .renderPass = vk.rp };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
   vkDestroyShaderModule(vk.dev, vs, NULL);
   vkDestroyShaderModule(vk.dev, fs, NULL);
   if (gs)
      vkDestroyShaderModule(vk.dev, gs, NULL);
   return p;
}

/* Draw case `c` and read the target back into img. */
static void
render(const struct gs_case *c, uint32_t *img)
{
   VkPipeline p = pipeline(c);
   VkBuffer vb;
   VkDeviceMemory vm;
   void *map;
   buffer(sizeof(struct gs_vertex) * c->nv, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vm, &map);
   memcpy(map, c->v, sizeof(struct gs_vertex) * c->nv);

   memset(vk.ssbo_map, 0, 16);
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = vk.pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(vk.dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   VkClearValue clear = { .color = { .float32 = { 0, 0, 0, 0 } } };
   VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                 .renderPass = vk.rp, .framebuffer = vk.fb,
                                 .renderArea = { { 0, 0 }, { W, H } },
                                 .clearValueCount = 1, .pClearValues = &clear };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.layout, 0, 1, &vk.set, 0,
                           NULL);
   vkCmdPushConstants(cmd, vk.layout, VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof c->pc, c->pc);
   VkDeviceSize off = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
   VkBuffer ib = VK_NULL_HANDLE;
   VkDeviceMemory im = VK_NULL_HANDLE;
   const uint32_t inst = c->instances ? c->instances : 1u;
   if (c->idx) {
      void *imap;
      buffer(2u * c->nidx, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib, &im, &imap);
      memcpy(imap, c->idx, 2u * c->nidx);
      vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, c->nidx, inst, 0, 0, 0);
   } else {
      vkCmdDraw(cmd, c->nv, inst, 0, 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, vk.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.rb, 1, &reg);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(vk.queue));
   memcpy(img, vk.rb_map, W * H * 4);
   vkFreeCommandBuffers(vk.dev, vk.pool, 1, &cmd);
   vkDestroyPipeline(vk.dev, p, NULL);
   vkUnmapMemory(vk.dev, vm);
   vkDestroyBuffer(vk.dev, vb, NULL);
   vkFreeMemory(vk.dev, vm, NULL);
   if (ib) {
      vkUnmapMemory(vk.dev, im);
      vkDestroyBuffer(vk.dev, ib, NULL);
      vkFreeMemory(vk.dev, im, NULL);
   }
}

/* Compare two images; returns the mismatch count and prints the first few. */
static unsigned
compare(const char *name, const uint32_t *got, const uint32_t *want)
{
   unsigned bad = 0, covered = 0;
   for (unsigned k = 0; k < W * H; k++) {
      covered += want[k] != CLEAR;
      if (got[k] != want[k] && bad++ < 4)
         printf("  %s: pixel (%u,%u) = %08x, expected %08x\n", name, k % W, k / W,
                got[k], want[k]);
   }
   if (!covered && !bad) {
      printf("  %s: the expected image is empty\n", name);
      return 1;
   }
   return bad;
}

/* ---- cases ------------------------------------------------------------ */

#define C0 0xff0000ffu
#define C1 0xff00ff00u
#define C2 0xffff0000u
#define C3 0xff00ffffu
#define C4 0xffff00ffu
#define C5 0xffffff00u

#define NDC(p) ((float)((p) / 32.0 - 1.0))
#define PAL_OF(i) ((i) % 6 == 0 ? C0 : (i) % 6 == 1 ? C1 : (i) % 6 == 2 ? C2 : \
                   (i) % 6 == 3 ? C3 : (i) % 6 == 4 ? C4 : C5)
#define V(px, py, col) { NDC(px), NDC(py), (col), 0 }

/* Two cells split corner to corner, 16x8 and 32x16, each vertex its colour. */
static const struct gs_vertex TRIS[] = {
   V(4, 4, C0),   V(4, 12, C1),  V(20, 4, C2),
   V(20, 4, C3),  V(4, 12, C4),  V(20, 12, C5),
   V(24, 24, C1), V(24, 40, C3), V(56, 24, C5),
   V(56, 24, C0), V(24, 40, C2), V(56, 40, C4),
};

static void
expect_tris(uint32_t *img, const struct gs_vertex *v, unsigned n, bool last, bool through)
{
   for (unsigned t = 0; t + 2 < n; t += 3) {
      struct px p[3];
      for (int k = 0; k < 3; k++)
         p[k] = (struct px){ (v[t + k].x + 1.0) * 32.0, (v[t + k].y + 1.0) * 32.0 };
      const uint32_t c = v[t + (last ? 2 : 0)].rgba;
      draw_tri(img, p[0], p[1], p[2], through ? swap_rb(c) : c, false);
   }
}

/* Horizontal and vertical lines along pixel centres, ends on pixel edges. */
static const struct gs_vertex LINES[] = {
   V(4, 30.5, C0),  V(28, 30.5, C1),
   V(40.5, 4, C2),  V(40.5, 36, C3),
   V(60, 50.5, C4), V(10, 50.5, C5),
};

static const struct gs_vertex POINTS[] = {
   V(8, 8, C0), V(20, 8, C3), V(32, 48, C5),
};

/* The strip shader's output for an input triangle at o with cells dx x dy. */
static uint32_t
tint(int k)
{
   const int c = k + 1;
   return rgba((c & 1) ? 255 : 0, ((c >> 1) & 1) ? 255 : 0, ((c >> 2) & 1) ? 255 : 0,
               ((c >> 3) & 1) ? 255 : 0);
}

static void
expect_strip(uint32_t *img, struct px o, struct px dx, struct px dy, bool last)
{
   struct px v[12];
   int k = 0;
   for (int c = 0; c < 3; c++) {
      v[k++] = (struct px){ o.x + c * dx.x, o.y + c * dx.y };
      v[k++] = (struct px){ o.x + c * dx.x + dy.x, o.y + c * dx.y + dy.y };
   }
   /* Triangle i of the strip: odd ones wound back, provoking i or i + 2. */
   for (int i = 0; i < 4; i++) {
      const bool odd = i & 1;
      const int pk = last ? i + 2 : i;
      const uint32_t col = tint(bite ? (last ? i : i + 2) : pk);
      if (bite && odd)
         continue;
      if (!odd)
         draw_tri(img, v[i], v[i + 1], v[i + 2], col, true);
      else if (last)
         draw_tri(img, v[i + 1], v[i], v[i + 2], col, true);
      else
         draw_tri(img, v[i], v[i + 2], v[i + 1], col, true);
   }
   if (bite) {
      /* The incomplete strip drawn as a triangle with the next vertex. */
      struct px a = { o.x + 4 * dx.x, o.y + 4 * dx.y }, b = { a.x + dy.x, a.y + dy.y };
      draw_tri(img, a, b, (struct px){ a.x + dx.x, a.y + dx.y }, tint(6), false);
   }
   struct px a = { o.x + 2 * dy.x, o.y + 2 * dy.y };
   struct px b = { o.x + 3 * dy.x, o.y + 3 * dy.y };
   struct px c = { a.x + dx.x, a.y + dx.y };
   draw_tri(img, a, b, c, tint(last ? 10 : 8), true);
}

/* Each input triangle (o, o + dx, o + dy) -- the strip shader reads only its
 * corners. */
static const struct gs_vertex STRIP_IN[] = {
   V(4, 4, 0), V(12, 4, 0), V(4, 8, 0),
   V(4, 40, 0), V(8, 40, 0), V(4, 48, 0),
};

static const struct gs_vertex QUAD_IN[] = {
   V(8, 8, C1), V(30, 20, C4), V(48, 50, C2),
};

/* Sixteen points, two rows of eight six pixels apart, so the quads of
 * consecutive points overlap and the last one drawn must win. */
static struct gs_vertex DYN_IN[16];
static const uint32_t DYN_N[16] = { 3, 0, 6, 1, 0, 2, 5, 4, 0, 6, 1, 3, 2, 0, 5, 1 };

static void
dyn_points(const uint32_t *n)
{
   for (unsigned i = 0; i < 16; i++) {
      const double x = 6 + 6 * (i % 8), y = 8 + 30 * (i / 8);
      DYN_IN[i] = (struct gs_vertex){ NDC(x), NDC(y), C0, n ? n[i] : 0 };
   }
}

static uint32_t
tint15(unsigned q)
{
   const unsigned c = q % 15u + 1u;
   return rgba((c & 1) ? 255 : 0, ((c >> 1) & 1) ? 255 : 0, ((c >> 2) & 1) ? 255 : 0,
               ((c >> 3) & 1) ? 255 : 0);
}

/* dyn.geom's quads, in the order the specification draws them: input point by
 * input point, each point's in emission order. */
static void
expect_dyn(uint32_t *img, const uint32_t *n)
{
   for (unsigned o = 0; o < 16; o++) {
      /* Reversed thread order is what a mis-ordered list would draw. */
      const unsigned i = bite ? 15u - o : o;
      const double x = (DYN_IN[i].x + 1.0) * 32.0, y = (DYN_IN[i].y + 1.0) * 32.0;
      for (unsigned q = 0; q < n[i]; q++)
         fill_rect(img, x + 2 * q - 4, y + q - 2, x + 2 * q + 4, y + q + 2, tint15(i * 7 + q));
   }
}

/* The primitives an adjacency topology makes of `n` vertices (indices `idx`,
 * 0xffff restarting, or none), as the specification lists each one's
 * vertices. */
static void
adj_decompose(VkPrimitiveTopology topo, const uint16_t *idx, unsigned n,
              unsigned out[][6], unsigned *np)
{
   unsigned run[32], m = 0;
   *np = 0;
   for (unsigned i = 0; i <= n; i++) {
      if (i < n && !(idx && idx[i] == 0xffff)) {
         run[m++] = idx ? idx[i] : i;
         continue;
      }
      switch (topo) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
         for (unsigned p = 0; p + 3 < m; p += 4, (*np)++)
            for (unsigned k = 0; k < 4; k++)
               out[*np][k] = run[p + k];
         break;
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         for (unsigned p = 0; p + 3 < m; p++, (*np)++)
            for (unsigned k = 0; k < 4; k++)
               out[*np][k] = run[p + k];
         break;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
         for (unsigned p = 0; p + 5 < m; p += 6, (*np)++)
            for (unsigned k = 0; k < 6; k++)
               out[*np][k] = run[p + k];
         break;
      default: {
         const unsigned t = m >= 4 ? (m - 4) / 2 : 0;
         for (unsigned i2 = 0; i2 < t; i2++, (*np)++) {
            const unsigned a = 2 * i2;
            unsigned o[6];
            if (t == 1) {
               const unsigned q[6] = { 0, 1, 2, 5, 4, 3 };
               memcpy(o, q, sizeof q);
            } else if (i2 == 0) {
               const unsigned q[6] = { a, a + 1, a + 2, a + 6, a + 4, a + 3 };
               memcpy(o, q, sizeof q);
            } else if (i2 < t - 1) {
               const unsigned q1[6] = { a, a + 3, a + 4, a + 6, a + 2, a - 2 };
               const unsigned q0[6] = { a, a - 2, a + 2, a + 6, a + 4, a + 3 };
               memcpy(o, (i2 & 1) ? q1 : q0, sizeof q0);
            } else {
               const unsigned q1[6] = { a, a + 3, a + 4, a + 5, a + 2, a - 2 };
               const unsigned q0[6] = { a, a - 2, a + 2, a + 5, a + 4, a + 3 };
               memcpy(o, (i2 & 1) ? q1 : q0, sizeof q0);
            }
            for (unsigned k = 0; k < 6; k++)
               out[*np][k] = run[o[k]];
         }
      } break;
      }
      m = 0;
   }
}

static int failures;

/* The storage buffer the geometry stage counted into: one per invocation, and
 * its quads. */
static void
check_counts(const char *name, uint32_t invocations, uint32_t quads)
{
   const uint32_t got_i = vk.ssbo_map[0], got_q = vk.ssbo_map[1];
   const uint32_t want_i = bite ? invocations + 1 : invocations;
   if (got_i != want_i || got_q != quads) {
      printf("  %s: counted %u invocations and %u quads, expected %u and %u\n", name, got_i,
             got_q, want_i, quads);
      printf("%s_counts: FAILED\n", name);
      failures++;
   } else {
      printf("%s_counts: passed\n", name);
   }
}

static void
run(const struct gs_case *c, const uint32_t *want)
{
   uint32_t got[W * H];
   render(c, got);
   const unsigned bad = compare(c->name, got, want);
   printf("%s: %s\n", c->name, bad ? "FAILED" : "passed");
   failures += bad != 0;
}

int
main(void)
{
   bite = getenv("GEOM_BITE") && atoi(getenv("GEOM_BITE"));
   init();
   uint32_t want[W * H], ref[W * H];

   for (int last = 0; last < 2; last++) {
      const struct gs_case c = { last ? "pass_tri_last" : "pass_tri_first", "pass_tri.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, TRIS, 12,
                                 VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, last };
      memset(want, 0, sizeof want);
      expect_tris(want, TRIS, 12, bite ? !last : last, !bite);
      run(&c, want);
   }
   {
      const struct gs_case c = { "pass_line", "pass_line.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_LINE_LIST, LINES, 6,
                                 VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false };
      memset(want, 0, sizeof want);
      const uint32_t s = bite ? 0 : 1;
      fill_rect(want, 4, 30, 28, 31, s ? swap_rb(C0) : C0);
      fill_rect(want, 40, 4, 41, 36, s ? swap_rb(C2) : C2);
      fill_rect(want, 10, 50, 60, 51, s ? swap_rb(C4) : C4);
      run(&c, want);
   }
   {
      const struct gs_case c = { "pass_point", "pass_point.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_POINT_LIST, POINTS, 3,
                                 VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false };
      memset(want, 0, sizeof want);
      const double h = bite ? 0.5 : 3.0;
      for (unsigned i = 0; i < 3; i++) {
         const double x = (POINTS[i].x + 1.0) * 32.0, y = (POINTS[i].y + 1.0) * 32.0;
         fill_rect(want, x - h, y - h, x + h, y + h, bite ? POINTS[i].rgba
                                                          : swap_rb(POINTS[i].rgba));
      }
      run(&c, want);
   }
   for (int last = 0; last < 2; last++) {
      const struct gs_case c = { last ? "strip_last" : "strip_first", "strip.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, STRIP_IN, 6,
                                 VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, last };
      memset(want, 0, sizeof want);
      for (unsigned t = 0; t < 2; t++) {
         const struct px o = { (STRIP_IN[3 * t].x + 1.0) * 32.0, (STRIP_IN[3 * t].y + 1.0) * 32.0 };
         const struct px a = { (STRIP_IN[3 * t + 1].x + 1.0) * 32.0,
                               (STRIP_IN[3 * t + 1].y + 1.0) * 32.0 };
         const struct px b = { (STRIP_IN[3 * t + 2].x + 1.0) * 32.0,
                               (STRIP_IN[3 * t + 2].y + 1.0) * 32.0 };
         expect_strip(want, o, (struct px){ a.x - o.x, a.y - o.y },
                      (struct px){ b.x - o.x, b.y - o.y }, last);
      }
      run(&c, want);
   }
   {
      const struct gs_case c = { "quad", "quad.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                 QUAD_IN, 3, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false };
      memset(want, 0, sizeof want);
      for (unsigned i = 0; i < 3; i++) {
         const double x = (QUAD_IN[i].x + 1.0) * 32.0, y = (QUAD_IN[i].y + 1.0) * 32.0;
         if (bite)
            fill_rect(want, x - 0.5, y - 0.5, x + 0.5, y + 0.5, QUAD_IN[i].rgba);
         else
            fill_rect(want, x - 4, y - 2, x + 4, y + 2, QUAD_IN[i].rgba);
      }
      run(&c, want);
   }
   {
      /* The geometry stage's triangles as lines must be the draw's own. */
      const struct gs_case c = { "polyline", "pass_tri.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, TRIS, 12,
                                 VK_CULL_MODE_NONE, VK_POLYGON_MODE_LINE, false };
      struct gs_vertex sw[12];
      for (unsigned i = 0; i < 12; i++) {
         sw[i] = TRIS[i];
         sw[i].rgba = bite ? TRIS[i].rgba : swap_rb(TRIS[i].rgba);
      }
      const struct gs_case r = { "polyline_ref", NULL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sw,
                                 12, VK_CULL_MODE_NONE, VK_POLYGON_MODE_LINE, false };
      render(&r, ref);
      run(&c, ref);
   }

   /* Counts known only once the shader has run: from the vertex, from a
    * push constant, from a texture, and all zero. */
   {
      uint32_t n[16];
      const char *names[4] = { "dyn_attr", "dyn_push", "dyn_tex", "dyn_zero" };
      for (unsigned v = 0; v < 4; v++) {
         for (unsigned i = 0; i < 16; i++)
            n[i] = v == 0 ? DYN_N[i] : v == 1 ? 2u + (i % 2) : v == 2 ? TEX_COUNTS[i] : 0u;
         dyn_points(v == 3 ? n : DYN_N);
         const struct gs_case c = { names[v], "dyn.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                    DYN_IN, 16, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false,
                                    { v == 3 ? 0u : v, 2u } };
         memset(want, 0, sizeof want);
         expect_dyn(want, n);
         uint32_t got[W * H];
         render(&c, got);
         unsigned bad = 0, quads = 0;
         for (unsigned k = 0; k < W * H; k++) {
            if (got[k] != want[k] && bad++ < 4)
               printf("  %s: pixel (%u,%u) = %08x, expected %08x\n", c.name, k % W, k / W,
                      got[k], want[k]);
         }
         for (unsigned i = 0; i < 16; i++)
            quads += n[i];
         printf("%s: %s\n", c.name, bad ? "FAILED" : "passed");
         failures += bad != 0;
         check_counts(c.name, 16, quads);
      }
   }
   /* Strips of a varying number of cells, culled and in both provoking
    * modes: their triangles staged per thread and ordered on the device. */
   for (int last = 0; last < 2; last++) {
      static const uint32_t cells[4] = { 6, 0, 3, 1 };
      struct gs_vertex in[4];
      for (unsigned i = 0; i < 4; i++)
         in[i] = (struct gs_vertex){ NDC(4), NDC(4 + 14 * i), C0, cells[i] };
      const struct gs_case c = { last ? "dynstrip_last" : "dynstrip_first", "dynstrip.geom.spv",
                                 VK_PRIMITIVE_TOPOLOGY_POINT_LIST, in, 4, VK_CULL_MODE_BACK_BIT,
                                 VK_POLYGON_MODE_FILL, last };
      memset(want, 0, sizeof want);
      for (unsigned i = 0; i < 4; i++) {
         const double x = 4, y = 4 + 14 * i;
         for (unsigned t = 0; t < 2 * cells[i]; t++) {
            struct px v[3];
            for (unsigned k = 0; k < 3; k++)
               v[k] = (struct px){ x + 8 * ((t + k) / 2), y + 4 * ((t + k) % 2) };
            const bool odd = t & 1;
            const unsigned pk = (last ? t + 2 : t);
            const uint32_t col = tint15(i * 3 + (bite ? (last ? t : t + 2) : pk));
            if (!odd)
               draw_tri(want, v[0], v[1], v[2], col, true);
            else if (last)
               draw_tri(want, v[1], v[0], v[2], col, true);
            else
               draw_tri(want, v[0], v[2], v[1], col, true);
         }
      }
      run(&c, want);
   }
   {
      static const uint32_t pts[8] = { 4, 0, 8, 1, 3, 0, 2, 7 };
      struct gs_vertex in[8];
      for (unsigned i = 0; i < 8; i++)
         in[i] = (struct gs_vertex){ NDC(4), NDC(4 + 8 * i), PAL_OF(i), pts[i] };
      const struct gs_case c = { "dynpoint", "dynpoint.geom.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                 in, 8, VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false };
      memset(want, 0, sizeof want);
      for (unsigned i = 0; i < 8; i++)
         for (unsigned k = 0; k < pts[i] + (bite ? 1u : 0u); k++)
            fill_rect(want, 3 + 4 * k, 3 + 8 * i, 5 + 4 * k, 5 + 8 * i, in[i].rgba);
      run(&c, want);
   }


   /* GS instancing: invocations run input primitive by input primitive, each
    * primitive's in invocation order. */
   {
      static const char *const shaders[4] = { "inst1.geom.spv", "inst2.geom.spv",
                                              "inst8.geom.spv", "inst32.geom.spv" };
      static const unsigned ninv[4] = { 1, 2, 8, 32 };
      static const double at[3][2] = { { 12, 12 }, { 20, 18 }, { 16, 28 } };
      struct gs_vertex in[3];
      for (unsigned p = 0; p < 3; p++)
         in[p] = (struct gs_vertex){ NDC(at[p][0]), NDC(at[p][1]), C0, 0 };
      for (unsigned v = 0; v < 4; v++) {
         char name[32];
         snprintf(name, sizeof name, "inst%u", ninv[v]);
         const struct gs_case c = { name, shaders[v], VK_PRIMITIVE_TOPOLOGY_POINT_LIST, in, 3,
                                    VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, false };
         memset(want, 0, sizeof want);
         /* The wrong order runs invocation by invocation across the
          * primitives. */
         for (unsigned a = 0; a < (bite ? ninv[v] : 3u); a++) {
            for (unsigned b = 0; b < (bite ? 3u : ninv[v]); b++) {
               const unsigned p = bite ? b : a, i = bite ? a : b;
               const double x = at[p][0] + i, y = at[p][1] + i;
               fill_rect(want, x - 8, y - 4, x + 8, y + 4, tint15(p * 5 + i));
            }
         }
         if (bite && ninv[v] == 1)
            fill_rect(want, 0, 0, 1, 1, C1);
         run(&c, want);
      }
   }

   /* Adjacency into the geometry stage: every input vertex, in the order the
    * specification gives each topology, whatever the provoking-vertex mode;
    * strips cut by primitive restart, primitive ids running on across it. */
   {
      struct gs_vertex grid[21];
      for (unsigned v = 0; v < 21; v++)
         grid[v] = (struct gs_vertex){ NDC(4 + 8 * (v % 7)), NDC(4 + 8 * (v / 7)), C0, 0 };
      static const uint16_t lstrip[] = { 0, 1, 2, 3, 4, 5, 6, 0xffff, 7, 8, 9, 10, 11, 12 };
      static const uint16_t tstrip[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xffff,
                                         10, 11, 12, 13, 14, 15, 16, 17 };
      const struct {
         const char *name, *geom;
         VkPrimitiveTopology topo;
         const uint16_t *idx;
         unsigned n, per;
      } adj[4] = {
         { "adj_lines", "adj_lines.geom.spv", VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
           NULL, 12, 4 },
         { "adj_linestrip", "adj_lines.geom.spv",
           VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, lstrip, 14, 4 },
         { "adj_tris", "adj_tris.geom.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
           NULL, 18, 6 },
         { "adj_tristrip", "adj_tris.geom.spv",
           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, tstrip, 19, 6 },
      };
      for (unsigned a = 0; a < 4; a++) {
         for (int last = 0; last < 2; last++) {
            char name[40];
            snprintf(name, sizeof name, "%s_%s", adj[a].name, last ? "last" : "first");
            struct gs_case c = { name, adj[a].geom, adj[a].topo, grid, adj[a].idx ? 21 : adj[a].n,
                                 VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, last };
            c.idx = adj[a].idx;
            c.nidx = adj[a].idx ? adj[a].n : 0;
            c.restart = adj[a].idx != NULL;
            memset(want, 0, sizeof want);
            unsigned prims[16][6], np = 0;
            adj_decompose(adj[a].topo, adj[a].idx, adj[a].n, prims, &np);
            for (unsigned p = 0; p < np; p++) {
               for (unsigned k = 0; k < adj[a].per; k++) {
                  /* The wrong order hands the shader its vertices reversed. */
                  const unsigned v = prims[p][bite ? adj[a].per - 1 - k : k];
                  const double x = 4 + 8 * (v % 7), y = 4 + 8 * (v / 7);
                  fill_rect(want, x - 1, y - 1, x + 1, y + 1, tint15(p * adj[a].per + k));
               }
            }
            run(&c, want);
         }
      }
   }

   /* Adjacency drawn without a geometry stage: the primitive is the group's
    * main vertices, flat-shaded from its provoking vertex. */
   for (int last = 0; last < 2; last++) {
      struct gs_vertex v[16];
      /* A triangle strip with adjacency over a band of 8x4 cells; the adjacent
       * vertices lie off the cells and are never drawn. */
      for (unsigned k = 0; k < 6; k++) {
         v[2 * k] = (struct gs_vertex){ NDC(4 + 8 * (k / 2)), NDC(8 + 4 * (k % 2)), PAL_OF(2 * k), 0 };
         v[2 * k + 1] = (struct gs_vertex){ NDC(60), NDC(60), PAL_OF(2 * k + 1), 0 };
      }
      struct gs_case c = { last ? "vsadj_tristrip_last" : "vsadj_tristrip_first", NULL,
                           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, v, 12,
                           VK_CULL_MODE_BACK_BIT, VK_POLYGON_MODE_FILL, last };
      memset(want, 0, sizeof want);
      unsigned prims[16][6], np = 0;
      adj_decompose(c.topo, NULL, 12, prims, &np);
      for (unsigned p = 0; p < np; p++) {
         struct px q[3];
         for (unsigned k = 0; k < 3; k++)
            q[k] = (struct px){ (v[prims[p][2 * k]].x + 1.0) * 32.0,
                                (v[prims[p][2 * k]].y + 1.0) * 32.0 };
         /* The provoking vertex is 2p or 2p + 4 of the strip, wherever the
          * group lists it. */
         const unsigned pv = bite ? prims[p][2] : last ? 2 * p + 4 : 2 * p;
         draw_tri(want, q[0], q[1], q[2], v[pv].rgba, true);
      }
      run(&c, want);

      /* A line strip with adjacency along one row, and a line list with
       * adjacency down one column. */
      static const double lx[6] = { 60, 4, 16, 28, 40, 60 };
      for (unsigned k = 0; k < 6; k++)
         v[k] = (struct gs_vertex){ NDC(lx[k]), NDC(k == 0 || k == 5 ? 60 : 30.5), PAL_OF(k), 0 };
      static const double ly[8] = { 60, 4, 20, 60, 60, 24, 40, 60 };
      for (unsigned k = 0; k < 8; k++)
         v[6 + k] = (struct gs_vertex){ NDC(k % 4 == 0 || k % 4 == 3 ? 60 : 50.5), NDC(ly[k]),
                                        PAL_OF(k + 2), 0 };
      struct gs_case l = { last ? "vsadj_lines_last" : "vsadj_lines_first", NULL,
                           VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, v, 6,
                           VK_CULL_MODE_NONE, VK_POLYGON_MODE_FILL, last };
      uint32_t got[W * H], want2[W * H];
      memset(want, 0, sizeof want);
      for (unsigned i = 0; i < 3; i++)
         fill_rect(want, lx[i + 1], 30, lx[i + 2], 31,
                   v[(bite ? !last : last) ? i + 2 : i + 1].rgba);
      render(&l, got);
      unsigned bad = compare(l.name, got, want);
      struct gs_case ll = l;
      ll.name = last ? "vsadj_linelist_last" : "vsadj_linelist_first";
      ll.topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
      ll.v = v + 6;
      ll.nv = 8;
      memset(want2, 0, sizeof want2);
      for (unsigned i = 0; i < 2; i++)
         fill_rect(want2, 50, ly[4 * i + 1], 51, ly[4 * i + 2],
                   v[6 + 4 * i + ((bite ? !last : last) ? 2 : 1)].rgba);
      printf("%s: %s\n", l.name, bad ? "FAILED" : "passed");
      failures += bad != 0;
      run(&ll, want2);
   }

   /* gl_PrimitiveID in the fragment stage: numbered per instance, running on
    * across a restart, one per triangle in polygon mode, and as a geometry
    * shader writes it. */
   {
      /* Two 16x8 cells split corner to corner. */
      static const struct gs_vertex cells[] = {
         V(4, 4, 0), V(4, 12, 0), V(20, 4, 0), V(20, 4, 0), V(4, 12, 0), V(20, 12, 0),
         V(24, 4, 0), V(24, 12, 0), V(40, 4, 0), V(40, 4, 0), V(24, 12, 0), V(40, 12, 0),
      };
      static const struct gs_vertex strip[] = {
         V(4, 16, 0), V(4, 20, 0), V(12, 16, 0), V(12, 20, 0),
         V(24, 16, 0), V(24, 20, 0), V(32, 16, 0), V(32, 20, 0),
      };
      static const uint16_t strip_idx[] = { 0, 1, 2, 3, 0xffff, 4, 5, 6, 7 };
      /* The same cells' corners on pixel centres, for polygon mode POINT. */
      struct gs_vertex corners[12];
      for (unsigned k = 0; k < 12; k++) {
         corners[k] = cells[k];
         corners[k].x += 0.5f / 32.0f;
         corners[k].y += 0.5f / 32.0f;
      }
      for (unsigned v = 0; v < 4; v++) {
         static const char *const names[4] = { "primid_inst", "primid_restart",
                                               "primid_polypoint", "primid_gs" };
         struct gs_case c = { names[v], v == 3 ? "primid.geom.spv" : NULL,
                              v == 1 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
                                     : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                              v == 1 ? strip : v == 2 ? corners : cells, v == 1 ? 8 : 12,
                              VK_CULL_MODE_NONE,
                              v == 2 ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL, false };
         c.vert = "primid.vert";
         c.frag = "primid.frag";
         c.instances = v == 0 ? 2 : 1;
         if (v == 1) {
            c.idx = strip_idx;
            c.nidx = 9;
            c.restart = true;
         }
         memset(want, 0, sizeof want);
         const struct gs_vertex *src = c.v;
         if (v == 1) {
            for (unsigned t = 0; t < 4; t++) {
               const unsigned b = 4 * (t / 2), i = t % 2;
               struct px q[3];
               for (unsigned k = 0; k < 3; k++)
                  q[k] = (struct px){ (src[b + i + k].x + 1.0) * 32.0, (src[b + i + k].y + 1.0) * 32.0 };
               draw_tri(want, q[0], q[1], q[2], tint15(bite ? t % 2 : t), false);
            }
         } else if (v == 2) {
            /* Polygon mode POINT: each vertex a pixel, in its triangle's id. */
            for (unsigned t = 0; t < 4; t++)
               for (unsigned k = 0; k < 3; k++) {
                  const double x = (src[3 * t + k].x + 1.0) * 32.0, y = (src[3 * t + k].y + 1.0) * 32.0;
                  fill_rect(want, x - 0.5, y - 0.5, x + 0.5, y + 0.5, tint15(bite ? 3 * t + k : t));
               }
         } else {
            for (unsigned inst = 0; inst < c.instances; inst++)
               for (unsigned t = 0; t < 4; t++) {
                  struct px q[3];
                  for (unsigned k = 0; k < 3; k++)
                     q[k] = (struct px){ (src[3 * t + k].x + 1.0) * 32.0,
                                         (src[3 * t + k].y + 1.0) * 32.0 + 32.0 * inst };
                  const unsigned id = v == 3 ? 3 * t + 1 : t;
                  draw_tri(want, q[0], q[1], q[2], tint15(bite ? id + 4 * inst + 1 : id), false);
               }
         }
         run(&c, want);
      }
   }

   if (failures) {
      printf("FAILED (geometry: %d case%s)\n", failures, failures == 1 ? "" : "s");
      return 1;
   }
   printf("PASSED\n");
   return 0;
}
