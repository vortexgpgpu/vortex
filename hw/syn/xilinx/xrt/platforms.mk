# Platform specific configurations
# Add your platform specific configurations here

CONFIGS += -DPLATFORM_MEMORY_DATA_WIDTH=512

ifeq ($(DEV_ARCH), zynquplus)
# zynquplus
CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=1 -DPLATFORM_MEMORY_ADDR_WIDTH=32
else ifeq ($(DEV_ARCH), versal)
# versal
CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=1 -DPLATFORM_MEMORY_ADDR_WIDTH=32
ifneq ($(findstring xilinx_vck5000,$(XSA)),)
	CONFIGS += -DPLATFORM_MEMORY_OFFSET=40'hC000000000
endif
else
# alveo
ifneq ($(findstring xilinx_u55c,$(XSA)),)
  # 16 GB of HBM2 with 32 channels (512 MB per channel)
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=32 -DPLATFORM_MEMORY_ADDR_WIDTH=34
  CONFIGS += -DPLATFORM_MERGED_MEMORY_INTERFACE
  VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_mem_0:HBM[0:31]
  #VPP_FLAGS += $(foreach i,$(shell seq 0 31), --connectivity.sp vortex_afu_1.m_axi_mem_$(i):HBM[$(i)])
else ifneq ($(findstring xilinx_u50,$(XSA)),)
  # 8 GB of HBM2 with 32 channels (256 MB per channel)
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=32 -DPLATFORM_MEMORY_ADDR_WIDTH=33
  VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_mem_0:HBM[0:31]
else ifneq ($(findstring xilinx_u280,$(XSA)),)
  # 8 GB of HBM2 with 32 channels (256 MB per channel)
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=32 -DPLATFORM_MEMORY_ADDR_WIDTH=33
  VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_mem_0:HBM[0:31]
else ifneq ($(findstring xilinx_u250,$(XSA)),)
  # 64 GB of DDR4 with 4 channels (16 GB per channel)
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=4 -DPLATFORM_MEMORY_ADDR_WIDTH=36
else ifneq ($(findstring xilinx_u200,$(XSA)),)
  # 64 GB of DDR4 with 4 channels (16 GB per channel)
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=4 -DPLATFORM_MEMORY_ADDR_WIDTH=36
else
  CONFIGS += -DPLATFORM_MEMORY_NUM_BANKS=1 -DPLATFORM_MEMORY_ADDR_WIDTH=32
endif
endif
