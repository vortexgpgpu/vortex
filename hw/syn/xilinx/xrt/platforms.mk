# Platform specific configurations
# Add your platform specific configurations here

CONFIGS += -DPLATFORM_MEMORY_DATA_WIDTH=512

ifeq ($(DEV_ARCH), zynquplus)
# zynquplus
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=32
else ifeq ($(DEV_ARCH), versal)
# versal
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=32
ifneq ($(findstring xilinx_vck5000,$(XSA)),)
	CONFIGS += -DPLATFORM_MEMORY_OFFSET=40'hC000000000
endif
else
# alveo
# The Command Processor's host-memory master (m_axi_host) reaches host DRAM
# through the platform slave-bridge / Host Memory Access aperture. All Alveo
# XDMA shells expose this as the HOST[0] connectivity tag.
VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_host:HOST[0]
# PLATFORM_MEMORY_TAG/CHANNELS name the device-memory channels the AFU's
# memory masters bind to; the Makefile gives each master its own group.
ifneq ($(findstring xilinx_u55c,$(XSA)),)
  # 16 GB of HBM2 with 32 channels (512 MB per channel)
  PLATFORM_MEMORY_TAG := HBM
  PLATFORM_MEMORY_CHANNELS := 32
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(PLATFORM_MEMORY_CHANNELS) -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=34
else ifneq ($(findstring xilinx_u50,$(XSA)),)
  # 8 GB of HBM2 with 32 channels (256 MB per channel)
  PLATFORM_MEMORY_TAG := HBM
  PLATFORM_MEMORY_CHANNELS := 32
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(PLATFORM_MEMORY_CHANNELS) -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=33
else ifneq ($(findstring xilinx_u280,$(XSA)),)
  # 8 GB of HBM2 with 32 channels (256 MB per channel)
  PLATFORM_MEMORY_TAG := HBM
  PLATFORM_MEMORY_CHANNELS := 32
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(PLATFORM_MEMORY_CHANNELS) -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=33
else ifneq ($(findstring xilinx_u250,$(XSA)),)
  # 16 GB of DDR4 (single channel, bank 0). Multi-bank requires per-bank XRT
  # VA offsets that aren't known at synthesis time.
  PLATFORM_MEMORY_TAG := DDR
  PLATFORM_MEMORY_CHANNELS := 1
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(PLATFORM_MEMORY_CHANNELS) -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=34
  CONFIGS += -DPLATFORM_MEMORY_OFFSET=40\'h4000000000
else ifneq ($(findstring xilinx_u200,$(XSA)),)
  # 64 GB of DDR4 with 4 channels (16 GB per channel)
  PLATFORM_MEMORY_TAG := DDR
  PLATFORM_MEMORY_CHANNELS := 4
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(PLATFORM_MEMORY_CHANNELS) -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=36
else
  CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=32
endif
endif
