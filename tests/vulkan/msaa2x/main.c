/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The multisample counts the driver offers must be the ones it can place.
 *
 * The device rasterizer produces a 4-sample coverage mask and nothing else, so
 * 1x and 4x are the only counts whose sample positions it can reproduce.
 * Merging that mask at 2x would take samples 0 and 1 of the 4x pattern and put
 * the coverage in the wrong places -- a render that looks plausible and is
 * wrong, which is the worst failure mode available here.
 *
 * The driver's answer is not to fall back at draw time but to never advertise
 * the count: framebufferColorSampleCounts comes back {1, 4}. An application
 * that honours the limit therefore cannot ask for 2x at all, and the draw-time
 * guard behind it is never reached through a conformant caller.
 *
 * That makes the advertised set itself the contract, and this is what checks
 * it. Widening the set without teaching the merger the matching sample
 * positions is a one-line change with no other symptom -- every existing test
 * would keep passing, because none of them asks for a count nobody offers.
 *
 * The check runs in both directions on purpose. Requiring only that 2x is
 * absent would be satisfied by a driver that advertises nothing; requiring 1x
 * and 4x present as well means the set has to be exactly the served one.
 */

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do {                                              \
   VkResult _r = (x);                                              \
   if (_r != VK_SUCCESS) {                                         \
      fprintf(stderr, "FAILED: %s -> VkResult %d\n", #x, (int)_r); \
      return 1;                                                    \
   }                                                               \
} while (0)

/* The counts the device can position, and the counts it must not offer. 8x and
 * 16x are listed explicitly rather than assumed absent so that adding one is a
 * deliberate edit here as well as in the driver. */
static const struct {
   VkSampleCountFlagBits bit;
   const char           *name;
   bool                  served;
} counts[] = {
   { VK_SAMPLE_COUNT_1_BIT,  "1x",  true  },
   { VK_SAMPLE_COUNT_2_BIT,  "2x",  false },
   { VK_SAMPLE_COUNT_4_BIT,  "4x",  true  },
   { VK_SAMPLE_COUNT_8_BIT,  "8x",  false },
   { VK_SAMPLE_COUNT_16_BIT, "16x", false },
};

int
main(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vortexpipe-msaa2x",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t npd = 1;
   VkPhysicalDevice pd;
   CHECK(vkEnumeratePhysicalDevices(inst, &npd, &pd));
   if (npd == 0) { fprintf(stderr, "FAILED: no physical device\n"); return 1; }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("device: %s\n", props.deviceName);

   const VkSampleCountFlags colour = props.limits.framebufferColorSampleCounts;
   printf("framebufferColorSampleCounts = 0x%x\n", (unsigned)colour);

   unsigned failed = 0;
   for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
      const bool offered = (colour & (VkSampleCountFlags)counts[i].bit) != 0;
      const bool ok = (offered == counts[i].served);
      printf("%-3s: %s  offered=%d expected=%d\n",
             counts[i].name, ok ? "pass" : "FAIL",
             (int)offered, (int)counts[i].served);
      if (!ok) failed++;
   }

   vkDestroyInstance(inst, NULL);

   if (failed) {
      printf("FAILED (%u count(s) misadvertised)\n", failed);
      return 1;
   }
   printf("PASSED (the offered sample counts are the ones the device can place)\n");
   return 0;
}
