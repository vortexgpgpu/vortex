/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * draw3d -- Vulkan port of tests/regression/draw3d (Phase 6 milestone).
 *
 * Replays a CGLTrace scene through the Vulkan API on the vortexpipe
 * driver: each draw call becomes a vkCmdDraw of a de-indexed,
 * interleaved vertex buffer, with the trace's depth / blend / texture
 * state mapped onto Vulkan pipeline state. The native draw3d test
 * drives the Vortex RASTER/TEX/OM units directly through vx_dcr_write;
 * this port reaches the very same units, but through Mesa + lavapipe +
 * vortexpipe -- the milestone that says the Vulkan graphics path is
 * complete.
 *
 * The CGLTrace vertices are already clip-space, so the vertex shader
 * is a passthrough; vortexpipe feeds gl_Position straight into
 * graphics::Binning, exactly as native draw3d does.
 *
 *   draw3d [-t trace] [-r reference] [-w width] [-h height] [-o out]
 */

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unistd.h>

#include <cgltrace.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* interleaved vertex fed to the VS: clip-space position + colour + uv */
struct Vertex {
   float pos[4];
   float color[4];
   float uv[2];
};

static VkDevice         dev  = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties memprops;

static uint32_t
find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

/* a device buffer + its memory, optionally filled from `src` */
static bool
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkMemoryPropertyFlags props, const void *src,
            VkBuffer *buf, VkDeviceMemory *mem)
{
   VkBufferCreateInfo bci = {};
   bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   bci.size = size; bci.usage = usage;
   if (vkCreateBuffer(dev, &bci, NULL, buf) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, *buf, &mr);
   uint32_t mt = find_mem(mr.memoryTypeBits, props);
   if (mt == UINT32_MAX) return false;
   VkMemoryAllocateInfo mai = {};
   mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
   if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS)
      return false;
   vkBindBufferMemory(dev, *buf, *mem, 0);
   if (src) {
      void *p;
      if (vkMapMemory(dev, *mem, 0, size, 0, &p) != VK_SUCCESS)
         return false;
      memcpy(p, src, size);
      vkUnmapMemory(dev, *mem);
   }
   return true;
}

static VkShaderModule
load_module(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "FAILED: cannot open %s\n", path); return VK_NULL_HANDLE; }
   fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
   std::vector<uint32_t> code((sz + 3) / 4);
   if (fread(code.data(), 1, sz, f) != (size_t)sz) { fclose(f); return VK_NULL_HANDLE; }
   fclose(f);
   VkShaderModuleCreateInfo smci = {};
   smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   smci.codeSize = sz; smci.pCode = code.data();
   VkShaderModule m = VK_NULL_HANDLE;
   vkCreateShaderModule(dev, &smci, NULL, &m);
   return m;
}

/* one draw call translated to Vulkan objects */
struct DrawCall {
   VkBuffer       vbuf;
   VkDeviceMemory vmem;
   uint32_t       vertex_count;
   VkPipeline     pipeline;
   VkDescriptorSet dset;
};

int
main(int argc, char **argv)
{
   const char *trace_file = "triangle.cgltrace";
   const char *ref_file   = NULL;
   const char *out_file   = "draw3d_out.png";
   const char *vs_path    = "draw3d.vert.spv";
   const char *fs_path    = "draw3d.frag.spv";
   uint32_t width = 128, height = 128;

   for (int c; (c = getopt(argc, argv, "t:r:w:h:o:")) != -1; ) {
      switch (c) {
      case 't': trace_file = optarg; break;
      case 'r': ref_file   = optarg; break;
      case 'o': out_file   = optarg; break;
      case 'w': width      = atoi(optarg); break;
      case 'h': height     = atoi(optarg); break;
      default: break;
      }
   }
   /* positional SPIR-V paths, as the other tests/vulkan suites pass */
   if (optind < argc) vs_path = argv[optind++];
   if (optind < argc) fs_path = argv[optind++];

   /* --- load the CGLTrace scene ----------------------------------- */
   CGLTrace trace;
   if (trace.load(trace_file) != 0) {
      fprintf(stderr, "FAILED: cannot load trace %s\n", trace_file);
      return 1;
   }
   printf("trace: %s -- %zu draw calls, %zu textures, %ux%u\n",
          trace_file, trace.drawcalls.size(), trace.textures.size(),
          width, height);

   /* --- instance / device / queue --------------------------------- */
   VkApplicationInfo app = {};
   app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
   app.pApplicationName = "vortexpipe-draw3d";
   app.apiVersion = VK_API_VERSION_1_1;
   VkInstanceCreateInfo ici = {};
   ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
   ici.pApplicationInfo = &app;
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);
   vkGetPhysicalDeviceMemoryProperties(pd, &memprops);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   std::vector<VkQueueFamilyProperties> qfp(nqf);
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp.data());
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
   if (qf == UINT32_MAX) { fprintf(stderr, "FAILED: no graphics queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {};
   qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
   qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
   VkDeviceCreateInfo dci = {};
   dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
   dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   const VkFormat CFMT = VK_FORMAT_R8G8B8A8_UNORM;
   const VkFormat DFMT = VK_FORMAT_D32_SFLOAT;

   /* --- colour + depth attachments -------------------------------- */
   auto make_image = [&](VkFormat fmt, VkImageUsageFlags usage,
                         VkImage *img, VkDeviceMemory *mem, VkImageView *view,
                         VkImageAspectFlags aspect) -> bool {
      VkImageCreateInfo ici2 = {};
      ici2.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ici2.imageType = VK_IMAGE_TYPE_2D; ici2.format = fmt;
      ici2.extent = { width, height, 1 };
      ici2.mipLevels = 1; ici2.arrayLayers = 1;
      ici2.samples = VK_SAMPLE_COUNT_1_BIT;
      ici2.tiling = VK_IMAGE_TILING_OPTIMAL; ici2.usage = usage;
      if (vkCreateImage(dev, &ici2, NULL, img) != VK_SUCCESS) return false;
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, *img, &mr);
      VkMemoryAllocateInfo mai = {};
      mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize = mr.size;
      mai.memoryTypeIndex = find_mem(mr.memoryTypeBits, 0);
      if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS) return false;
      vkBindImageMemory(dev, *img, *mem, 0);
      VkImageViewCreateInfo ivci = {};
      ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ivci.image = *img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = fmt;
      ivci.subresourceRange = { aspect, 0, 1, 0, 1 };
      return vkCreateImageView(dev, &ivci, NULL, view) == VK_SUCCESS;
   };

   VkImage cimg, dimg; VkDeviceMemory cmem, dmem; VkImageView cview, dview;
   if (!make_image(CFMT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   &cimg, &cmem, &cview, VK_IMAGE_ASPECT_COLOR_BIT) ||
       !make_image(DFMT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                   &dimg, &dmem, &dview, VK_IMAGE_ASPECT_DEPTH_BIT)) {
      fprintf(stderr, "FAILED: attachment images\n");
      return 1;
   }

   /* --- render pass (colour + depth, both cleared) ---------------- */
   VkAttachmentDescription att[2] = {};
   att[0].format = CFMT; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
   att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   att[1].format = DFMT; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
   att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
   att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {};
   sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   sub.colorAttachmentCount = 1; sub.pColorAttachments = &cref;
   sub.pDepthStencilAttachment = &dref;
   VkRenderPassCreateInfo rpci = {};
   rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
   rpci.attachmentCount = 2; rpci.pAttachments = att;
   rpci.subpassCount = 1; rpci.pSubpasses = &sub;
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fbviews[2] = { cview, dview };
   VkFramebufferCreateInfo fbci = {};
   fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
   fbci.renderPass = rp; fbci.attachmentCount = 2; fbci.pAttachments = fbviews;
   fbci.width = width; fbci.height = height; fbci.layers = 1;
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* --- shader modules + pipeline/descriptor layout --------------- */
   VkShaderModule vs = load_module(vs_path);
   VkShaderModule fs = load_module(fs_path);
   if (!vs || !fs) return 1;

   VkDescriptorSetLayoutBinding dslb = {};
   dslb.binding = 0;
   dslb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   dslb.descriptorCount = 1;
   dslb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   VkDescriptorSetLayoutCreateInfo dslci = {};
   dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
   dslci.bindingCount = 1; dslci.pBindings = &dslb;
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {};
   plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
   plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   /* one shared 1x1 white texture for untextured draw calls */
   const uint32_t white = 0xffffffffu;
   VkBuffer wstage; VkDeviceMemory wsmem;
   if (!make_buffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &white, &wstage, &wsmem)) {
      fprintf(stderr, "FAILED: white staging\n"); return 1;
   }

   VkDescriptorPoolSize dps = {};
   dps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   dps.descriptorCount = (uint32_t)trace.drawcalls.size() + 1;
   VkDescriptorPoolCreateInfo dpci = {};
   dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
   dpci.maxSets = (uint32_t)trace.drawcalls.size() + 1;
   dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));

   /* --- command buffer -------------------------------------------- */
   VkCommandPoolCreateInfo cpci = {};
   cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
   cpci.queueFamilyIndex = qf;
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {};
   cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cbai.commandBufferCount = 1;
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {};
   cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   /* a sampled image: 1x1 white, or a draw call's texture */
   auto make_texture = [&](uint32_t tw, uint32_t th, const void *rgba,
                           VkBuffer stage, VkImageView *view) -> bool {
      VkImage timg; VkDeviceMemory tmem;
      VkImageCreateInfo tci = {};
      tci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      tci.imageType = VK_IMAGE_TYPE_2D; tci.format = CFMT;
      tci.extent = { tw, th, 1 }; tci.mipLevels = 1; tci.arrayLayers = 1;
      tci.samples = VK_SAMPLE_COUNT_1_BIT; tci.tiling = VK_IMAGE_TILING_OPTIMAL;
      tci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      if (vkCreateImage(dev, &tci, NULL, &timg) != VK_SUCCESS) return false;
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, timg, &mr);
      VkMemoryAllocateInfo mai = {};
      mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize = mr.size;
      mai.memoryTypeIndex = find_mem(mr.memoryTypeBits, 0);
      if (vkAllocateMemory(dev, &mai, NULL, &tmem) != VK_SUCCESS) return false;
      vkBindImageMemory(dev, timg, tmem, 0);
      (void)rgba;
      VkImageMemoryBarrier b = {};
      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image = timg;
      b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &b);
      VkBufferImageCopy bic = {};
      bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      bic.imageExtent = { tw, th, 1 };
      vkCmdCopyBufferToImage(cmd, stage, timg,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
      b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                           0, NULL, 0, NULL, 1, &b);
      VkImageViewCreateInfo ivci = {};
      ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ivci.image = timg; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = CFMT;
      ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      return vkCreateImageView(dev, &ivci, NULL, view) == VK_SUCCESS;
   };

   VkImageView white_view;
   if (!make_texture(1, 1, &white, wstage, &white_view)) {
      fprintf(stderr, "FAILED: white texture\n"); return 1;
   }

   VkSamplerCreateInfo sci = {};
   sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
   sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
   sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
   sci.addressModeU = sci.addressModeV = sci.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_REPEAT;
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

   /* --- translate each draw call ---------------------------------- */
   std::vector<DrawCall> dcs;
   std::vector<VkBuffer>       tex_stages;   /* keep staging alive */
   std::vector<VkDeviceMemory> tex_smem;

   for (size_t d = 0; d < trace.drawcalls.size(); d++) {
      const CGLTrace::drawcall_t &dc = trace.drawcalls[d];
      const CGLTrace::states_t   &st = dc.states;
      DrawCall out = {};

      /* de-index the triangle list into an interleaved vertex buffer */
      std::vector<Vertex> verts;
      verts.reserve(dc.primitives.size() * 3);
      for (const auto &prim : dc.primitives) {
         const uint32_t idx[3] = { prim.i0, prim.i1, prim.i2 };
         for (int k = 0; k < 3; k++) {
            const CGLTrace::vertex_t &v = dc.vertices.at(idx[k]);
            Vertex o;
            o.pos[0] = v.pos.x; o.pos[1] = v.pos.y;
            o.pos[2] = v.pos.z; o.pos[3] = v.pos.w;
            o.color[0] = v.color.r; o.color[1] = v.color.g;
            o.color[2] = v.color.b; o.color[3] = v.color.a;
            o.uv[0] = v.texcoord.u; o.uv[1] = v.texcoord.v;
            verts.push_back(o);
         }
      }
      out.vertex_count = (uint32_t)verts.size();
      if (out.vertex_count == 0)
         continue;
      if (!make_buffer(verts.size() * sizeof(Vertex),
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       verts.data(), &out.vbuf, &out.vmem)) {
         fprintf(stderr, "FAILED: vertex buffer\n"); return 1;
      }

      /* texture: this draw call's, or the shared 1x1 white one */
      VkImageView tview = white_view;
      if (st.texture_enabled && trace.textures.count(dc.texture_id)) {
         const CGLTrace::texture_t &tx = trace.textures.at(dc.texture_id);
         std::vector<uint8_t> rgba((size_t)tx.width * tx.height * 4, 0xff);
         /* the common cgltrace texture format is A8R8G8B8 (B,G,R,A in
          * memory) -- swizzle to the R8G8B8A8 Vulkan upload. */
         if (tx.format == FORMAT_A8R8G8B8 &&
             tx.pixels.size() >= (size_t)tx.width * tx.height * 4) {
            for (size_t i = 0; i < (size_t)tx.width * tx.height; i++) {
               rgba[i*4+0] = tx.pixels[i*4+2];
               rgba[i*4+1] = tx.pixels[i*4+1];
               rgba[i*4+2] = tx.pixels[i*4+0];
               rgba[i*4+3] = tx.pixels[i*4+3];
            }
         }
         VkBuffer ts; VkDeviceMemory tsm;
         if (make_buffer(rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         rgba.data(), &ts, &tsm)) {
            tex_stages.push_back(ts); tex_smem.push_back(tsm);
            VkImageView v;
            if (make_texture(tx.width, tx.height, rgba.data(), ts, &v))
               tview = v;
         }
      }

      /* descriptor set bound to the chosen texture */
      VkDescriptorSetAllocateInfo dsai = {};
      dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      dsai.descriptorPool = dpool;
      dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
      CHECK(vkAllocateDescriptorSets(dev, &dsai, &out.dset));
      VkDescriptorImageInfo dii = {};
      dii.sampler = sampler; dii.imageView = tview;
      dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkWriteDescriptorSet wds = {};
      wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wds.dstSet = out.dset; wds.dstBinding = 0; wds.descriptorCount = 1;
      wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wds.pImageInfo = &dii;
      vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

      /* fragment mode: colour / texture / modulate (draw3d's rule) */
      int32_t mode = 0;
      if (st.texture_enabled) {
         bool modulate = (st.texture_envmode == CGLTrace::ENVMODE_MODULATE)
                      && st.color_enabled;
         mode = modulate ? 2 : 1;
      }

      /* --- pipeline --------------------------------------------- */
      VkSpecializationMapEntry sme = { 0, 0, sizeof(int32_t) };
      VkSpecializationInfo spec = {};
      spec.mapEntryCount = 1; spec.pMapEntries = &sme;
      spec.dataSize = sizeof(int32_t); spec.pData = &mode;

      VkPipelineShaderStageCreateInfo stages[2] = {};
      stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      stages[0].module = vs; stages[0].pName = "main";
      stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[1].module = fs; stages[1].pName = "main";
      stages[1].pSpecializationInfo = &spec;

      VkVertexInputBindingDescription vib = { 0, sizeof(Vertex),
                                              VK_VERTEX_INPUT_RATE_VERTEX };
      VkVertexInputAttributeDescription via[3] = {
         { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(Vertex, pos) },
         { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(Vertex, color) },
         { 2, 0, VK_FORMAT_R32G32_SFLOAT,       (uint32_t)offsetof(Vertex, uv) },
      };
      VkPipelineVertexInputStateCreateInfo vi = {};
      vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
      vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = via;

      VkPipelineInputAssemblyStateCreateInfo ia = {};
      ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

      VkViewport vp = { 0, 0, (float)width, (float)height, 0.0f, 1.0f };
      VkRect2D sc = { { 0, 0 }, { width, height } };
      VkPipelineViewportStateCreateInfo vps = {};
      vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      vps.viewportCount = 1; vps.pViewports = &vp;
      vps.scissorCount = 1; vps.pScissors = &sc;

      VkPipelineRasterizationStateCreateInfo rs = {};
      rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
      rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

      VkPipelineMultisampleStateCreateInfo ms = {};
      ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

      VkPipelineDepthStencilStateCreateInfo ds = {};
      ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
      ds.depthTestEnable  = st.depth_test ? VK_TRUE : VK_FALSE;
      ds.depthWriteEnable = st.depth_writemask ? VK_TRUE : VK_FALSE;
      /* CGLTrace::ecompare and VkCompareOp share the GL ordering */
      ds.depthCompareOp = (VkCompareOp)st.depth_func;

      VkPipelineColorBlendAttachmentState cba = {};
      cba.blendEnable = st.blend_enabled ? VK_TRUE : VK_FALSE;
      cba.srcColorBlendFactor = cba.srcAlphaBlendFactor =
         st.blend_enabled ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE;
      cba.dstColorBlendFactor = cba.dstAlphaBlendFactor =
         st.blend_enabled ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                          : VK_BLEND_FACTOR_ZERO;
      cba.colorBlendOp = cba.alphaBlendOp = VK_BLEND_OP_ADD;
      cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo cb = {};
      cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      cb.attachmentCount = 1; cb.pAttachments = &cba;

      VkGraphicsPipelineCreateInfo gpci = {};
      gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      gpci.stageCount = 2; gpci.pStages = stages;
      gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
      gpci.pViewportState = &vps; gpci.pRasterizationState = &rs;
      gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
      gpci.pColorBlendState = &cb;
      gpci.layout = pl; gpci.renderPass = rp; gpci.subpass = 0;
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci,
                                      NULL, &out.pipeline));
      dcs.push_back(out);
   }

   /* --- record the render pass ------------------------------------ */
   VkClearValue clears[2];
   clears[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
   clears[1].depthStencil = { 1.0f, 0 };
   VkRenderPassBeginInfo rpbi = {};
   rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
   rpbi.renderPass = rp; rpbi.framebuffer = fb;
   rpbi.renderArea = { { 0, 0 }, { width, height } };
   rpbi.clearValueCount = 2; rpbi.pClearValues = clears;
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   for (const DrawCall &dc : dcs) {
      VkDeviceSize zero = 0;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dc.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl,
                              0, 1, &dc.dset, 0, NULL);
      vkCmdBindVertexBuffers(cmd, 0, 1, &dc.vbuf, &zero);
      vkCmdDraw(cmd, dc.vertex_count, 1, 0, 0);
   }
   vkCmdEndRenderPass(cmd);

   /* --- copy the colour image to a host-visible buffer ------------ */
   const VkDeviceSize cbytes = (VkDeviceSize)width * height * 4;
   VkBuffer rb; VkDeviceMemory rbm;
   if (!make_buffer(cbytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    NULL, &rb, &rbm)) {
      fprintf(stderr, "FAILED: readback buffer\n"); return 1;
   }
   VkBufferImageCopy region = {};
   region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
   region.imageExtent = { width, height, 1 };
   vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          rb, 1, &region);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = {};
   si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* --- save + verify --------------------------------------------- */
   uint8_t *px;
   CHECK(vkMapMemory(dev, rbm, 0, cbytes, 0, (void **)&px));
   std::vector<uint8_t> image(px, px + cbytes);
   vkUnmapMemory(dev, rbm);

   /* The framebuffer is R8G8B8A8; cocogfx SaveImage / CompareImages
    * (FORMAT_A8R8G8B8) want B8G8R8A8 byte order. Native draw3d writes
    * its colour buffer bottom-up, so flip vertically to match the
    * reference PNGs. */
   std::vector<uint8_t> bgra((size_t)width * height * 4);
   for (uint32_t y = 0; y < height; y++) {
      const uint8_t *src = image.data() + (size_t)(height - 1 - y) * width * 4;
      uint8_t       *dst = bgra.data()  + (size_t)y * width * 4;
      for (uint32_t x = 0; x < width; x++) {
         dst[x*4+0] = src[x*4+2];   /* B */
         dst[x*4+1] = src[x*4+1];   /* G */
         dst[x*4+2] = src[x*4+0];   /* R */
         dst[x*4+3] = src[x*4+3];   /* A */
      }
   }
   if (SaveImage(out_file, FORMAT_A8R8G8B8, bgra.data(),
                 width, height, (int32_t)width * 4) != 0) {
      fprintf(stderr, "FAILED: cannot save %s\n", out_file);
      return 1;
   }
   printf("wrote %s\n", out_file);

   unsigned colored = 0;
   for (uint32_t i = 0; i < width * height; i++)
      if (image[i*4] || image[i*4+1] || image[i*4+2])
         colored++;

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (ref_file) {
      int errors = CompareImages(out_file, ref_file, FORMAT_A8R8G8B8);
      if (errors != 0) {
         printf("FAILED (%d pixels differ from %s)\n", errors, ref_file);
         return 1;
      }
      printf("PASSED (draw3d: %u draw calls, %u/%u pixels covered, "
             "matches %s)\n", (unsigned)dcs.size(), colored, width * height,
             ref_file);
   } else {
      if (colored == 0) {
         printf("FAILED (nothing rendered)\n");
         return 1;
      }
      printf("PASSED (draw3d: %u draw calls, %u/%u pixels covered)\n",
             (unsigned)dcs.size(), colored, width * height);
   }
   return 0;
}
