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

# Tool install prefixes (overridable; default under $(TOOLDIR)). CIRCT is the
# SystemVerilog frontend of every flow: its preprocessor (gen_sources.sh -P)
# feeds Vivado, and its elaboration into one netlist (circt_elaborate.sh;
# gen_sources.sh -E) feeds Yosys, Beacon and Foldsim. Before either, the RAM
# fold (gen_sources.sh -F, hw/scripts/ram_fold.py) patches the preprocessed
# sources so that asynchronous RAM reads map to block RAM.
CIRCT_PATH ?= $(TOOLDIR)/circt
YOSYS_PATH ?= $(TOOLDIR)/yosys
STA_PATH   ?= $(TOOLDIR)/sta
VERILATOR_PATH ?= $(TOOLDIR)/verilator

# Absolute tool binaries exported via env var so builds are self-contained
# and multiple trees can coexist without a sourced env polluting PATH.
CIRCT_VERILOG ?= $(CIRCT_PATH)/bin/circt-verilog
CIRCT_OPT     ?= $(CIRCT_PATH)/bin/circt-opt
FIRTOOL       ?= $(CIRCT_PATH)/bin/firtool
FOLDSIM       ?= $(FOLDSIM_PATH)/bin/foldsim
FOLDSIM_OPT   ?= $(FOLDSIM_PATH)/bin/foldsim-opt
FOLDSIM_LIFT_TOOL ?= $(FOLDSIM_PATH)/bin/foldsim-lift
YOSYS ?= $(YOSYS_PATH)/bin/yosys
STA   ?= $(STA_PATH)/bin/sta
VERILATOR ?= $(VERILATOR_PATH)/bin/verilator
# a python3 with pyslang 11 or 12 (pip install "pyslang>=11,<13")
RAM_FOLD_PYTHON ?= python3

export CIRCT_PATH FOLDSIM_PATH FOLDSIM_SHELL FOLDSIM_LIFT_TOOL YOSYS_PATH STA_PATH VERILATOR_PATH CIRCT_VERILOG CIRCT_OPT FIRTOOL FOLDSIM FOLDSIM_OPT YOSYS STA VERILATOR RAM_FOLD_PYTHON
