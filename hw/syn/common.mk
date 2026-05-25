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

# Shared variables for the synthesis backends (yosys, synopsys, xilinx/*,
# altera/*). Tool paths live here rather than in the build-root config.mk
# because synthesis is a single domain — putting these in config.mk would
# leak them into every test/sim build that has no use for them.
#
# Mirror the per-domain pattern used by tests/{opencl,hip,vulkan}/common.mk
# (POCL_PATH / CHIPSTAR_PATH / MESA_PATH there); only the syn backends
# touch these.

# config.mk gives us TOOLDIR; each syn backend Makefile includes us via
# include $(ROOT_DIR)/hw/syn/common.mk

ifndef TOOLDIR
$(error TOOLDIR not set — include $$(ROOT_DIR)/config.mk before hw/syn/common.mk)
endif

# Tool install prefixes (overridable; default under $(TOOLDIR)).
SV2V_PATH  ?= $(TOOLDIR)/sv2v
YOSYS_PATH ?= $(TOOLDIR)/yosys
STA_PATH   ?= $(TOOLDIR)/sta

# Absolute tool binaries. Scripts that previously relied on PATH (e.g.
# hw/scripts/sv2v.sh, hw/syn/yosys/run_synth.sh) take these via env var
# so the build is self-contained and multiple Vortex trees can coexist
# without a sourced toolchain_env.sh polluting the global PATH.
SV2V  ?= $(SV2V_PATH)/bin/sv2v
YOSYS ?= $(YOSYS_PATH)/bin/yosys
STA   ?= $(STA_PATH)/bin/sta

export SV2V_PATH YOSYS_PATH STA_PATH SV2V YOSYS STA
