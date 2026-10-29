/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Descriptor robustness, in two parts, each on its own device.
 *
 * Texel buffer stores under robustBufferAccess alone (robustImageAccess off):
 * a storage texel buffer view covers buffer A; buffer B is bound right after A
 * in the same memory. A compute shader stores past the view's end, far enough
 * to reach B. Robust buffer access lets those stores land in A or nowhere,
 * never in B: every word of B must keep its seed.
 *
 * A null storage image under nullDescriptor: an r32ui storage image slot holds
 * a live image, then is rewritten to VK_NULL_HANDLE, as is a slot read without
 * a format; a multisample slot is null from the start. A compute shader sizes,
 * loads and stores through them: the size and the sample count must be 0,
 * either load (0, 0, 0, 0), and the old image must be left as it was.
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define A_WORDS   64u
#define B_SEED    0x1234u
#define IMG_W     4u
#define IMG_SEED  5u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

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

static const VkMemoryPropertyFlags HOST =
   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

struct ctx {
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue queue;
   uint32_t qf;
   VkPhysicalDeviceMemoryProperties mp;
};

static int
make_device(VkPhysicalDevice pd, const void *features_chain,
            const VkPhysicalDeviceFeatures *core, const char *const *exts, uint32_t next,
            struct ctx *c)
{
   c->pd = pd;
   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
   c->qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++)
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c->qf = i; break; }
   free(qfp);
   if (c->qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = c->qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = features_chain,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = next, .ppEnabledExtensionNames = exts,
      .pEnabledFeatures = core,
   };
   CHECK(vkCreateDevice(pd, &dci, NULL, &c->dev));
   vkGetDeviceQueue(c->dev, c->qf, 0, &c->queue);
   vkGetPhysicalDeviceMemoryProperties(pd, &c->mp);
   return 0;
}

static int
make_pipeline(const struct ctx *c, const VkDescriptorSetLayoutBinding *b, uint32_t nb,
              const uint32_t *spv, size_t spv_size, VkDescriptorSetLayout *dsl,
              VkPipelineLayout *pl, VkPipeline *pipe)
{
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nb, .pBindings = b,
   };
   CHECK(vkCreateDescriptorSetLayout(c->dev, &dslci, NULL, dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = dsl,
   };
   CHECK(vkCreatePipelineLayout(c->dev, &plci, NULL, pl));
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(c->dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = *pl,
   };
   CHECK(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, NULL, pipe));
   vkDestroyShaderModule(c->dev, sm, NULL);
   return 0;
}

/* Record `rec` into a one-shot command buffer, submit it and wait. */
static int
run_once(const struct ctx *c, void (*rec)(VkCommandBuffer, void *), void *data)
{
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(c->dev, &cpi, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(c->dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   rec(cb, data);
   VkMemoryBarrier toh = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &toh, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb,
   };
   CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(c->queue));
   vkDestroyCommandPool(c->dev, cp, NULL);
   return 0;
}

/* ---- part 1: texel buffer stores under robustBufferAccess only ---------- */

struct texbuf_rec {
   VkPipeline pipe;
   VkPipelineLayout pl;
   VkDescriptorSet ds;
   uint32_t groups;
};

static void
texbuf_record(VkCommandBuffer cb, void *data)
{
   const struct texbuf_rec *r = data;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pl, 0, 1, &r->ds, 0, NULL);
   vkCmdDispatch(cb, r->groups, 1, 1);
}

static int
texbuf_part(VkPhysicalDevice pd, const uint32_t *spv, size_t spv_size)
{
   VkPhysicalDeviceFeatures core = { .robustBufferAccess = VK_TRUE };
   struct ctx c;
   if (make_device(pd, NULL, &core, NULL, 0, &c))
      return 1;

   /* A and B in one allocation, B at the first offset it may take after A. */
   const VkDeviceSize a_bytes = A_WORDS * 4u;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a_bytes,
      .usage = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
   };
   VkBuffer a, b;
   CHECK(vkCreateBuffer(c.dev, &bci, NULL, &a));
   CHECK(vkCreateBuffer(c.dev, &bci, NULL, &b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(c.dev, a, &mr);
   const VkDeviceSize b_off = (a_bytes + mr.alignment - 1) / mr.alignment * mr.alignment;
   const VkDeviceSize total = b_off + a_bytes;
   const uint32_t mt = find_mem(&c.mp, mr.memoryTypeBits, HOST);
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = total, .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(c.dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(c.dev, a, mem, 0));
   CHECK(vkBindBufferMemory(c.dev, b, mem, b_off));
   uint32_t *m;
   CHECK(vkMapMemory(c.dev, mem, 0, total, 0, (void **)&m));
   for (VkDeviceSize i = 0; i < total / 4u; i++)
      m[i] = B_SEED;
   vkUnmapMemory(c.dev, mem);

   VkBufferViewCreateInfo bvci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = a,
      .format = VK_FORMAT_R32_UINT, .range = VK_WHOLE_SIZE,
   };
   VkBufferView view;
   CHECK(vkCreateBufferView(c.dev, &bvci, NULL, &view));
   VkDescriptorSetLayoutBinding bind = {
      0, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL,
   };
   struct texbuf_rec r;
   VkDescriptorSetLayout dsl;
   if (make_pipeline(&c, &bind, 1, spv, spv_size, &dsl, &r.pl, &r.pipe))
      return 1;
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c.dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   CHECK(vkAllocateDescriptorSets(c.dev, &dsai, &r.ds));
   VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = r.ds, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .pTexelBufferView = &view,
   };
   vkUpdateDescriptorSets(c.dev, 1, &w, 0, NULL);
   /* Enough invocations to store past B's last word. */
   r.groups = (uint32_t)(total / 4u);
   if (run_once(&c, texbuf_record, &r))
      return 1;

   CHECK(vkMapMemory(c.dev, mem, 0, total, 0, (void **)&m));
   unsigned a_bad = 0, b_bad = 0;
   for (uint32_t i = 0; i < A_WORDS; i++)
      a_bad += m[i] != 0xdeadu;
   const uint32_t *bw = m + b_off / 4u;
   uint32_t first = UINT32_MAX;
   for (uint32_t i = 0; i < A_WORDS; i++) {
      if (bw[i] != B_SEED) {
         if (first == UINT32_MAX)
            first = i;
         b_bad++;
      }
   }
   vkUnmapMemory(c.dev, mem);
   if (a_bad || b_bad) {
      printf("FAILED (texel buffer stores: %u/%u words of the view not stored, %u/%u words "
             "of the next buffer overwritten%s)\n", a_bad, A_WORDS, b_bad, A_WORDS,
             b_bad ? ", past the view's end" : "");
      return 1;
   }
   printf("texel buffer stores: %u stored, past the end none reached the next buffer "
          "(at %u bytes)\n", A_WORDS, (unsigned)b_off);

   vkDestroyDescriptorPool(c.dev, dp, NULL);
   vkDestroyPipeline(c.dev, r.pipe, NULL);
   vkDestroyPipelineLayout(c.dev, r.pl, NULL);
   vkDestroyDescriptorSetLayout(c.dev, dsl, NULL);
   vkDestroyBufferView(c.dev, view, NULL);
   vkDestroyBuffer(c.dev, a, NULL);
   vkDestroyBuffer(c.dev, b, NULL);
   vkFreeMemory(c.dev, mem, NULL);
   vkDestroyDevice(c.dev, NULL);
   return 0;
}

/* ---- part 2: a storage image slot rewritten to null --------------------- */

struct null_rec {
   VkPipeline pipe;
   VkPipelineLayout pl;
   VkDescriptorSet ds;
   VkImage img;
   VkBuffer readback;
};

static void
seed_record(VkCommandBuffer cb, void *data)
{
   const struct null_rec *r = data;
   VkImageMemoryBarrier tog = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = r->img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, NULL, 0, NULL, 1, &tog);
   VkClearColorValue cv = { .uint32 = { IMG_SEED, IMG_SEED, IMG_SEED, IMG_SEED } };
   VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   vkCmdClearColorImage(cb, r->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
}

static void
null_record(VkCommandBuffer cb, void *data)
{
   const struct null_rec *r = data;
   VkMemoryBarrier tos = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &tos, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pl, 0, 1, &r->ds, 0, NULL);
   vkCmdDispatch(cb, 1, 1, 1);
   VkMemoryBarrier tot = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 1, &tot, 0, NULL, 0, NULL);
   VkBufferImageCopy bic = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { IMG_W, IMG_W, 1 },
   };
   vkCmdCopyImageToBuffer(cb, r->img, VK_IMAGE_LAYOUT_GENERAL, r->readback, 1, &bic);
}

static int
null_part(VkPhysicalDevice pd, const uint32_t *spv, size_t spv_size)
{
   VkPhysicalDeviceRobustness2FeaturesEXT r2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 f2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &r2,
   };
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   if (!r2.nullDescriptor || !f2.features.shaderStorageImageMultisample ||
       !f2.features.shaderStorageImageReadWithoutFormat) {
      fprintf(stderr, "FAILED: no nullDescriptor, multisample storage images or "
              "storage reads without a format\n");
      return 1;
   }
   VkPhysicalDeviceFeatures core = {
      .shaderStorageImageMultisample = VK_TRUE,
      .shaderStorageImageReadWithoutFormat = VK_TRUE,
   };
   VkPhysicalDeviceRobustness2FeaturesEXT r2on = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .nullDescriptor = VK_TRUE,
   };
   const char *ext = "VK_EXT_robustness2";
   struct ctx c;
   if (make_device(pd, &r2on, &core, &ext, 1, &c))
      return 1;

   struct null_rec r;
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .extent = { IMG_W, IMG_W, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   CHECK(vkCreateImage(c.dev, &ici, NULL, &r.img));
   VkMemoryRequirements imr;
   vkGetImageMemoryRequirements(c.dev, r.img, &imr);
   uint32_t imt = find_mem(&c.mp, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (imt == UINT32_MAX)
      imt = find_mem(&c.mp, imr.memoryTypeBits, 0);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = imr.size, .memoryTypeIndex = imt,
   };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(c.dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(c.dev, r.img, imem, 0));
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = r.img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(c.dev, &ivci, NULL, &view));

   VkBuffer out;
   VkDeviceMemory out_mem, rb_mem;
   VkBufferCreateInfo obci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4u * 16u,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   CHECK(vkCreateBuffer(c.dev, &obci, NULL, &out));
   VkBufferCreateInfo rbci = obci;
   rbci.size = IMG_W * IMG_W * 4u;
   rbci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
   CHECK(vkCreateBuffer(c.dev, &rbci, NULL, &r.readback));
   VkBuffer bufs[2] = { out, r.readback };
   VkDeviceMemory *mems[2] = { &out_mem, &rb_mem };
   for (int i = 0; i < 2; i++) {
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(c.dev, bufs[i], &mr);
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size, .memoryTypeIndex = find_mem(&c.mp, mr.memoryTypeBits, HOST),
      };
      CHECK(vkAllocateMemory(c.dev, &mai, NULL, mems[i]));
      CHECK(vkBindBufferMemory(c.dev, bufs[i], *mems[i], 0));
   }

   VkDescriptorSetLayoutBinding b[4] = {
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
   };
   VkDescriptorSetLayout dsl;
   if (make_pipeline(&c, b, 4, spv, spv_size, &dsl, &r.pl, &r.pipe))
      return 1;
   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(c.dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   CHECK(vkAllocateDescriptorSets(c.dev, &dsai, &r.ds));
   /* The formatted and the typeless slots hold the live image first, then are
    * rewritten to null; the multisample slot is null from the start. */
   VkDescriptorImageInfo live = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorImageInfo none = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL };
   VkDescriptorBufferInfo dbi = { out, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w[4] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = r.ds, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &live },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = r.ds, .dstBinding = 3,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &live },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = r.ds, .dstBinding = 2,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &none },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = r.ds, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi },
   };
   vkUpdateDescriptorSets(c.dev, 4, w, 0, NULL);
   w[0].pImageInfo = &none;
   w[1].pImageInfo = &none;
   vkUpdateDescriptorSets(c.dev, 2, w, 0, NULL);

   uint32_t *o;
   CHECK(vkMapMemory(c.dev, out_mem, 0, 64u, 0, (void **)&o));
   memset(o, 0xab, 64u);
   vkUnmapMemory(c.dev, out_mem);
   if (run_once(&c, seed_record, &r) || run_once(&c, null_record, &r))
      return 1;

   CHECK(vkMapMemory(c.dev, out_mem, 0, 64u, 0, (void **)&o));
   const bool size_ok = o[0] == 0 && o[1] == 0 && o[8] == 0;
   const bool load_ok = !o[4] && !o[5] && !o[6] && !o[7] && !o[12] && !o[13] && !o[14] &&
                        !o[15];
   printf("null storage image: size (%u, %u), samples %u, load (%u %u %u %u), "
          "typeless load (%u %u %u %u)\n", o[0], o[1], o[8], o[4], o[5], o[6], o[7],
          o[12], o[13], o[14], o[15]);
   vkUnmapMemory(c.dev, out_mem);
   uint32_t *t;
   CHECK(vkMapMemory(c.dev, rb_mem, 0, IMG_W * IMG_W * 4u, 0, (void **)&t));
   unsigned changed = 0;
   for (uint32_t i = 0; i < IMG_W * IMG_W; i++)
      changed += t[i] != IMG_SEED;
   vkUnmapMemory(c.dev, rb_mem);
   if (!size_ok || !load_ok || changed) {
      printf("FAILED (null storage image rewritten over a live one: size or samples %s, "
             "loads %s, %u texels of the old image changed)\n", size_ok ? "0" : "not 0",
             load_ok ? "0" : "not 0", changed);
      return 1;
   }

   vkDestroyDescriptorPool(c.dev, dp, NULL);
   vkDestroyPipeline(c.dev, r.pipe, NULL);
   vkDestroyPipelineLayout(c.dev, r.pl, NULL);
   vkDestroyDescriptorSetLayout(c.dev, dsl, NULL);
   vkDestroyImageView(c.dev, view, NULL);
   vkDestroyImage(c.dev, r.img, NULL);
   vkFreeMemory(c.dev, imem, NULL);
   for (int i = 0; i < 2; i++) {
      vkDestroyBuffer(c.dev, bufs[i], NULL);
      vkFreeMemory(c.dev, *mems[i], NULL);
   }
   vkDestroyDevice(c.dev, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *tb_path = (argc > 1) ? argv[1] : "texbuf_store.comp.spv";
   const char *ni_path = (argc > 2) ? argv[2] : "null_image.comp.spv";
   size_t tb_size = 0, ni_size = 0;
   uint32_t *tb_spv = read_spirv(tb_path, &tb_size);
   uint32_t *ni_spv = read_spirv(ni_path, &ni_size);
   if (!tb_spv || !ni_spv)
      return 1;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-robust-desc",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo inci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&inci, NULL, &inst));
   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   /* Both parts run, so a failure in one does not hide the other. */
   const int bad = texbuf_part(pd, tb_spv, tb_size) | null_part(pd, ni_spv, ni_size);
   if (bad)
      return 1;

   free(tb_spv);
   free(ni_spv);
   vkDestroyInstance(inst, NULL);
   printf("PASSED (robust_desc: texel buffer stores kept inside their buffer under "
          "robustBufferAccess alone; a storage image slot rewritten to null sizes 0, "
          "reads 0 and leaves the old image intact)\n");
   return 0;
}
