# Platform specific configurations for the AMD Alveo V80.
#
# `override` so a command-line `make CONFIGS=...` cannot drop the platform
# settings; without it the AFU is packaged for the wrong memory topology.

override CONFIGS += -DPLATFORM_MEMORY_DATA_WIDTH=512

# The V80 exposes its HBM stacks through the linker's HBM connectivity tags.
# A single wide master fanned across the channels avoids the per-bank base
# address problem: the linker assigns each memory range independently, and
# those bases are not known at synthesis time.
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=34
override CONFIGS += -DPLATFORM_MERGED_MEMORY_INTERFACE

# Kernel clock target (MHz). The linker also accepts a frequency request;
# the runtime can retune within the platform's supported range.
KERNEL_FREQ ?= 200
