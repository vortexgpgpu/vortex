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

# Shared variables for synthesis backends (yosys, synopsys, xilinx/*, altera/*).
# Tool paths are kept here rather than in config.mk to avoid leaking them
# into test/sim builds.

# config.mk provides TOOLDIR; syn backend Makefiles include this file.

ifndef TOOLDIR
$(error TOOLDIR not set — include $$(ROOT_DIR)/config.mk before hw/syn/common.mk)
endif

# Tool install prefixes (overridable; default under $(TOOLDIR)).
SV2V_PATH  ?= $(TOOLDIR)/sv2v
YOSYS_PATH ?= $(TOOLDIR)/yosys
STA_PATH   ?= $(TOOLDIR)/sta
VERILATOR_PATH ?= $(TOOLDIR)/verilator

# Absolute tool binaries exported via env var so builds are self-contained
# and multiple trees can coexist without a sourced env polluting PATH.
SV2V  ?= $(SV2V_PATH)/bin/sv2v
YOSYS ?= $(YOSYS_PATH)/bin/yosys
STA   ?= $(STA_PATH)/bin/sta
VERILATOR ?= $(VERILATOR_PATH)/bin/verilator

export SV2V_PATH YOSYS_PATH STA_PATH VERILATOR_PATH SV2V YOSYS STA VERILATOR
