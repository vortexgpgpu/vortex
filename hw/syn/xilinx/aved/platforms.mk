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

# Device-memory aperture base. The V80 maps the AFU's m_axi_mem_0 port at
# 0x40_0000_0000 in both build flavours -- simulation puts the BRAM model there
# (vbin.prj/run_pre.tcl) and hardware puts HBM_AXI_00 there (slash's generated
# address map). Vortex's device allocator is based at VX_MEM_USER_BASE_ADDR
# (0x10000) and a 32-bit core cannot emit the aperture base itself, so the AFU
# wrapper rebases every device access -- from the cores and from the CP -- by
# this synthesis-time offset. Without it nothing decodes: writes still collect
# a BRESP from the interconnect while the data goes nowhere.
#
# Sized Verilog literal, NOT decimal. A decimal constant this large is silently
# truncated: "WARNING: [VRFC 10-8884] decimal constant 274877906944 should be
# smaller than 2147483648; using 0 instead".
#
# Tied to MEM_TAG=HBM0; a different tag lands on a different aperture base.
override CONFIGS += -DPLATFORM_MEMORY_OFFSET=40\'h4000000000

# Device-memory connectivity. NOT HBM0: that tag is a single HBM_AXI channel,
# which the compute shell maps as 1 GB at 0x40_0000_0000. Vortex's 32-bit memory
# map puts the stack at the top of a 4 GB space -- vx_start.S sets sp to
# VX_MEM_STACK_BASE_ADDR (0xFFFF0000) and grows down, and that lands just below
# the local-memory window (VX_lsu_slice.sv decodes LMEM *upward* from the same
# base), so it is a global-memory access. Rebased it becomes 0x40_FFFE_Fxxx,
# which a 1 GB aperture cannot answer: the read gets no response, the LSU stalls
# forever, and the AFU never goes idle -- taking the shell's AXI path with it.
#
# The MEM tag routes through the HBM VNOC, which the shell maps at the same
# 0x40_0000_0000 base with 32 GB of range, so it covers the whole 32-bit space
# and PLATFORM_MEMORY_OFFSET above stays correct unchanged.
#
# Set before the Makefile's `MEM_TAG ?= HBM0`, which is why plain `=` suffices.
MEM_TAG = MEM

# Kernel clock target (MHz). The linker also accepts a frequency request;
# the runtime can retune within the platform's supported range.
KERNEL_FREQ ?= 200
