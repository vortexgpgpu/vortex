/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vkCmdDispatchIndirect, and dispatches under VK_EXT_conditional_rendering.
 *
 * In the GPU cases a shader writes the workgroup counts (and the predicate
 * word) into the buffer the dispatch then reads, inside one command buffer, so
 * the counts exist only in device memory when the dispatch is recorded. The
 * marking shader counts every workgroup it runs as and records the grid, so a
 * dispatch that ran the wrong grid, ran a workgroup twice, skipped one, or ran
 * at all when it should not have, each fails differently.
 *
 *   gpu 3x2x2 / offset            counts written by a shader, at offset 0 / 32
 *   gpu zero x / y / z            a zero count dispatches nothing
 *   gpu large                     2048 workgroups
 *   host 5x1x1                    counts written through a mapping
 *   cond gpu 0 / 1 [inverted]     a direct dispatch under a predicate the
 *                                 shader wrote, normal and inverted
 *   cond indirect gpu 0 / 1       an indirect dispatch under a predicate
 *
 * Run against lavapipe with GALLIUM_DRIVER=vortexpipe.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HITS      4096u
#define META      16u
#define ARGS      64u

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

enum cond_mode { COND_NONE, COND_NORMAL, COND_INVERTED };

struct dcase {
   const char    *name;
   bool           gpu_args;   /* counts written by args.comp, else by the host */
   bool           indirect;   /* vkCmdDispatchIndirect, else vkCmdDispatch */
   uint32_t       off;        /* word offset of the record */
   uint32_t       x, y, z;
   enum cond_mode cond;
   uint32_t       pred;
};

static const struct dcase cases[] = {
   { "gpu 3x2x2",               true,  true,  0u,  3u, 2u, 2u, COND_NONE,     0u },
   { "gpu offset 4x1x3",        true,  true,  8u,  4u, 1u, 3u, COND_NONE,     0u },
   { "gpu zero x",              true,  true,  0u,  0u, 4u, 4u, COND_NONE,     0u },
   { "gpu zero y",              true,  true,  0u,  4u, 0u, 4u, COND_NONE,     0u },
   { "gpu zero z",              true,  true,  0u,  4u, 4u, 0u, COND_NONE,     0u },
   { "gpu large 1024x2x1",      true,  true,  0u,  1024u, 2u, 1u, COND_NONE,  0u },
   { "host 5x1x1",              false, true,  0u,  5u, 1u, 1u, COND_NONE,     0u },
   { "cond gpu 0",              true,  false, 0u,  2u, 2u, 1u, COND_NORMAL,   0u },
   { "cond gpu 1",              true,  false, 0u,  2u, 2u, 1u, COND_NORMAL,   1u },
   { "cond gpu 0 inverted",     true,  false, 0u,  2u, 2u, 1u, COND_INVERTED, 0u },
   { "cond gpu 7 inverted",     true,  false, 0u,  2u, 2u, 1u, COND_INVERTED, 7u },
   { "cond indirect gpu 0",     true,  true,  0u,  3u, 1u, 2u, COND_NORMAL,   0u },
   { "cond indirect gpu 1",     true,  true,  0u,  3u, 1u, 2u, COND_NORMAL,   1u },
};
#define NCASES (sizeof cases / sizeof cases[0])

struct pc {
   uint32_t off, x, y, z, pred;
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
   if (buf) { *out_size = (size_t)sz; }
   return buf;
}

struct gpu {
   VkDevice              dev;
   VkPhysicalDevice      pd;
   VkQueue               queue;
   uint32_t              qf;
   PFN_vkCmdBeginConditionalRenderingEXT begin_cond;
   PFN_vkCmdEndConditionalRenderingEXT   end_cond;
};

static int
make_buffer(struct gpu *g, VkDeviceSize bytes, VkBufferUsageFlags usage,
            VkBuffer *buf, VkDeviceMemory *mem, uint32_t **map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(g->dev, &bci, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(g->dev, *buf, &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(g->pd, &mp);
   const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t mt = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   }
   if (mt == UINT32_MAX) { fprintf(stderr, "FAILED: no host-visible memory\n"); return 1; }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   CHECK(vkAllocateMemory(g->dev, &mai, NULL, mem));
   CHECK(vkBindBufferMemory(g->dev, *buf, *mem, 0));
   CHECK(vkMapMemory(g->dev, *mem, 0, bytes, 0, (void **)map));
   return 0;
}

static int
make_pipeline(struct gpu *g, const char *path, VkPipelineLayout pl, VkPipeline *pipe)
{
   size_t sz = 0;
   uint32_t *spv = read_spirv(path, &sz);
   if (!spv) { return 1; }
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sz, .pCode = spv,
   };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(g->dev, &smci, NULL, &sm));
   free(spv);
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
      },
      .layout = pl,
   };
   CHECK(vkCreateComputePipelines(g->dev, VK_NULL_HANDLE, 1, &cpci, NULL, pipe));
   vkDestroyShaderModule(g->dev, sm, NULL);
   return 0;
}

int
main(int argc, char **argv)
{
   const char *args_path = (argc > 1) ? argv[1] : "args.comp.spv";
   const char *mark_path = (argc > 2) ? argv[2] : "mark.comp.spv";

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-dispatchindirect",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   struct gpu g = { 0 };
   uint32_t npd = 1;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &g.pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(g.pd, &props);
   printf("device: %s\n", props.deviceName);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nqf, NULL);
   VkQueueFamilyProperties *qfp = calloc(nqf, sizeof(*qfp));
   vkGetPhysicalDeviceQueueFamilyProperties(g.pd, &nqf, qfp);
   g.qf = UINT32_MAX;
   for (uint32_t i = 0; i < nqf; i++) {
      if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qf = i; break; }
   }
   free(qfp);
   if (g.qf == UINT32_MAX) { fprintf(stderr, "FAILED: no compute queue\n"); return 1; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = g.qf, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceConditionalRenderingFeaturesEXT crf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
      .conditionalRendering = VK_TRUE,
   };
   const char *exts[] = { VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &crf,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts,
   };
   CHECK(vkCreateDevice(g.pd, &dci, NULL, &g.dev));
   vkGetDeviceQueue(g.dev, g.qf, 0, &g.queue);
   g.begin_cond = (PFN_vkCmdBeginConditionalRenderingEXT)
      vkGetDeviceProcAddr(g.dev, "vkCmdBeginConditionalRenderingEXT");
   g.end_cond = (PFN_vkCmdEndConditionalRenderingEXT)
      vkGetDeviceProcAddr(g.dev, "vkCmdEndConditionalRenderingEXT");
   if (!g.begin_cond || !g.end_cond) {
      fprintf(stderr, "FAILED: no conditional rendering entry points\n");
      return 1;
   }

   VkBuffer argb, hitb, metab;
   VkDeviceMemory argm, hitm, metam;
   uint32_t *argp, *hitp, *metap;
   if (make_buffer(&g, ARGS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                   VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,
                   &argb, &argm, &argp) ||
       make_buffer(&g, HITS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &hitb, &hitm, &hitp) ||
       make_buffer(&g, META * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &metab, &metam, &metap)) {
      return 1;
   }

   VkDescriptorSetLayoutBinding b[3];
   for (uint32_t i = 0; i < 3; i++) {
      b[i] = (VkDescriptorSetLayoutBinding) {
         .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = b,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(g.dev, &dslci, NULL, &dsl));
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(struct pc),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(g.dev, &plci, NULL, &pl));
   VkPipeline args_pipe, mark_pipe;
   if (make_pipeline(&g, args_path, pl, &args_pipe) ||
       make_pipeline(&g, mark_path, pl, &mark_pipe)) {
      return 1;
   }

   VkDescriptorPoolSize dps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3,
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(g.dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   };
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(g.dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi[3] = {
      { .buffer = argb,  .offset = 0, .range = ARGS * 4 },
      { .buffer = hitb,  .offset = 0, .range = HITS * 4 },
      { .buffer = metab, .offset = 0, .range = META * 4 },
   };
   VkWriteDescriptorSet wds[3];
   for (uint32_t i = 0; i < 3; i++) {
      wds[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i],
      };
   }
   vkUpdateDescriptorSets(g.dev, 3, wds, 0, NULL);

   VkCommandPoolCreateInfo cmpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = g.qf,
   };
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(g.dev, &cmpci, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(g.dev, &cbai, &cb));

   unsigned failed = 0;
   for (unsigned c = 0; c < NCASES; c++) {
      const struct dcase *k = &cases[c];
      /* The record starts out holding counts that would pass a dispatch that
       * ignored the device-written ones. */
      for (uint32_t i = 0; i < ARGS; i++) { argp[i] = 1u; }
      memset(hitp, 0, HITS * 4);
      memset(metap, 0, META * 4);
      if (!k->gpu_args) {
         argp[k->off + 0] = k->x;
         argp[k->off + 1] = k->y;
         argp[k->off + 2] = k->z;
         argp[k->off + 3] = k->pred;
      }

      VkCommandBufferBeginInfo cbbi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      CHECK(vkBeginCommandBuffer(cb, &cbbi));
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
      if (k->gpu_args) {
         const struct pc pc = { k->off, k->x, k->y, k->z, k->pred };
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, args_pipe);
         vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
         vkCmdDispatch(cb, 1, 1, 1);
         VkMemoryBarrier mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                             VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
         };
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                              VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
                              0, 1, &mb, 0, NULL, 0, NULL);
      }
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mark_pipe);
      if (k->cond != COND_NONE) {
         VkConditionalRenderingBeginInfoEXT crbi = {
            .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
            .buffer = argb, .offset = (k->off + 3u) * 4u,
            .flags = k->cond == COND_INVERTED ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0,
         };
         g.begin_cond(cb, &crbi);
      }
      if (k->indirect) {
         vkCmdDispatchIndirect(cb, argb, k->off * 4u);
      } else {
         vkCmdDispatch(cb, k->x, k->y, k->z);
      }
      if (k->cond != COND_NONE) {
         g.end_cond(cb);
      }
      CHECK(vkEndCommandBuffer(cb));
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cb,
      };
      CHECK(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(g.queue));
      CHECK(vkResetCommandBuffer(cb, 0));

      bool runs = k->x && k->y && k->z;
      if (k->cond != COND_NONE) {
         runs = runs && ((k->pred != 0) != (k->cond == COND_INVERTED));
      }
      const uint32_t blocks = runs ? k->x * k->y * k->z : 0;
      unsigned bad = 0;
      for (uint32_t i = 0; i < HITS; i++) {
         const uint32_t want = i < blocks ? 1u : 0u;
         if (hitp[i] != want) {
            if (bad < 4) {
               fprintf(stderr, "  %s: workgroup %u ran %u times, want %u\n",
                       k->name, i, hitp[i], want);
            }
            bad++;
         }
      }
      if (metap[3] != (runs ? 1u : 0u) ||
          (runs && (metap[0] != k->x || metap[1] != k->y || metap[2] != k->z))) {
         fprintf(stderr, "  %s: grid %ux%ux%u over %u launches, want %ux%ux%u over %u\n",
                 k->name, metap[0], metap[1], metap[2], metap[3],
                 k->x, k->y, k->z, runs ? 1u : 0u);
         bad++;
      }
      printf("%-24s %s\n", k->name, bad ? "FAIL" : "ok");
      if (bad) {
         failed++;
      }
   }

   vkDestroyCommandPool(g.dev, cp, NULL);
   vkDestroyDescriptorPool(g.dev, dp, NULL);
   vkDestroyPipeline(g.dev, args_pipe, NULL);
   vkDestroyPipeline(g.dev, mark_pipe, NULL);
   vkDestroyPipelineLayout(g.dev, pl, NULL);
   vkDestroyDescriptorSetLayout(g.dev, dsl, NULL);
   vkDestroyBuffer(g.dev, argb, NULL);
   vkDestroyBuffer(g.dev, hitb, NULL);
   vkDestroyBuffer(g.dev, metab, NULL);
   vkFreeMemory(g.dev, argm, NULL);
   vkFreeMemory(g.dev, hitm, NULL);
   vkFreeMemory(g.dev, metam, NULL);
   vkDestroyDevice(g.dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (dispatchindirect: %u of %u cases)\n", failed, (unsigned)NCASES);
      return 1;
   }
   printf("PASSED (dispatchindirect: %u cases)\n", (unsigned)NCASES);
   return 0;
}
