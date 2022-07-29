PROJECT = VX_fpu_fpga
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

FPU_INCLUDE = $(RTL_DIR)/fpu_unit;$(THIRD_PARTY_DIR)/fpnew/src;$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src
RTL_INCLUDE = $(FPU_INCLUDE);$(RTL_DIR);$(RTL_DIR)/libs;$(RTL_DIR)/interfaces;$(IP_DIR)
