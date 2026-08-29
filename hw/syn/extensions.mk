# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared extension-source wiring for every synthesis backend (xilinx/*, altera/*,
# synopsys, yosys). Single source of truth: appends per-extension RTL packages and
# include paths to RTL_PKGS / RTL_INCLUDE based on the resolved XCONFIGS.
#
# Include AFTER the flow has computed XCONFIGS and set its base RTL_DIR,
# THIRD_PARTY_DIR, RTL_PKGS and RTL_INCLUDE, e.g.:
#     include $(VORTEX_HOME)/hw/syn/extensions.mk
# Use the $(ROOT_DIR) absolute path — the xilinx/dut flow copies each target
# Makefile into a <target>/<PREFIX>/ build subdir where a relative path would break.
#
# $(filter) without a % is a whole-word match, so each _ENABLE guard has to list
# every spelling gen_config.py can emit for "enabled" — which depends on how the
# caller wrote the flag:
#
#   caller passes -DVX_CFG_EXT_TCU_ENABLE=1  ->  emits  _ENABLE=1
#   caller passes -DVX_CFG_EXT_TCU_ENABLE    ->  emits  _ENABLE  and  _ENABLED=1
#   extension left disabled                  ->  emits  _ENABLED=0
#
# Matching only one spelling makes the guard silently never fire for the others,
# dropping the extension's package from RTL_PKGS; synthesis then fails with
# "'VX_<x>_pkg' is not declared". Listing all three enabled forms covers every
# caller, and none of them collides with the _ENABLED=0 disabled form.
#
# The TCU_TYPE_* flags below are always emitted bare and are matched as such.

# Add TCU extension sources
ifneq (,$(filter -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_TCU_ENABLE=1 -DVX_CFG_EXT_TCU_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tcu
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_FPNEW, $(XCONFIGS)))
		RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/fpnew
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_TFR, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
	endif
endif

# Add DXA extension sources
ifneq (,$(filter -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DXA_ENABLE=1 -DVX_CFG_EXT_DXA_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/dxa
endif

# Add RTU extension sources
ifneq (,$(filter -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_EXT_RTU_ENABLE=1 -DVX_CFG_EXT_RTU_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/rtu/VX_rtu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/rtu
endif

# Add RASTER extension sources
ifneq (,$(filter -DVX_CFG_EXT_RASTER_ENABLE -DVX_CFG_EXT_RASTER_ENABLE=1 -DVX_CFG_EXT_RASTER_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/raster/VX_raster_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/raster
endif

# Add TEX extension sources
ifneq (,$(filter -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_TEX_ENABLE=1 -DVX_CFG_EXT_TEX_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tex/VX_tex_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tex
endif

# Add OM extension sources
ifneq (,$(filter -DVX_CFG_EXT_OM_ENABLE -DVX_CFG_EXT_OM_ENABLE=1 -DVX_CFG_EXT_OM_ENABLED=1, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/om/VX_om_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/om
endif
