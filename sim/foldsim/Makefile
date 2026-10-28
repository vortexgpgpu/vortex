include ../common.mk

# The foldsim kernel in Verilator, as a library the runtime's avedsim target
# links: the same simulator hw/syn/foldsim generates for the image, in
# Foldsim's shell and its in-process host (FOLDSIM_SHELL), so a run here is
# the image's run, cycle for cycle, but for device memory's timing. Vortex adds
# only the Xilinx floating-point cores' models its FPU may instantiate.

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/sim/foldsim
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl
WORK_DIR := $(DESTDIR)/foldsim_emul
GEN_SV := $(WORK_DIR)/gen/foldsim_sim.sv
# The kernel's memory ports, as the image's flow derived them.
KERNEL_VH := $(WORK_DIR)/gen/foldsim_kernel.vh

CONFIG_FILE := $(DESTDIR)/foldsim_emul_config.stamp

CXXFLAGS += -std=c++17 -fPIC -O2 -DNDEBUG -I$(FOLDSIM_SHELL)/include
LDFLAGS += -shared -pthread
# The Xilinx floating-point cores' simulation models compute with Vortex's DPI
# floating point.
CXXFLAGS += -I$(SW_COMMON_DIR) -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

VL_FLAGS += -O3 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN --x-initial 0 --x-assign 0
VL_FLAGS += -I$(FOLDSIM_RTL_DIR) -I$(WORK_DIR)/gen -I$(DPI_DIR)
ifdef VCD
VL_FLAGS += --trace --trace-structs
endif
THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(min(32, mp.cpu_count()))')
VL_FLAGS += -j $(THREADS)

TOP := foldsim_afu
# The simulation models of the Xilinx floating-point cores the simulator may
# instantiate (its FPU of type DSP): Verilator cannot run their netlists.
RTL_SRCS := $(SRC_DIR)/xilinx_fp_models.sv $(GEN_SV) $(FOLDSIM_RTL_DIR)/foldsim_token.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_mem_bmap.sv $(FOLDSIM_RTL_DIR)/foldsim_mem_bank.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_mem.sv $(FOLDSIM_RTL_DIR)/foldsim_ctrl.sv $(FOLDSIM_RTL_DIR)/foldsim_afu.sv
SRCS := $(FOLDSIM_SHELL)/emul/foldsim_emul.cpp $(DPI_DIR)/dpi_float.cpp $(DPI_DIR)/dpi_util.cpp \
        $(SW_COMMON_DIR)/softfloat_ext.cpp $(SW_COMMON_DIR)/rvfloats.cpp

PROJECT := libfoldsim_emul.so

.PHONY: all lib force clean

all: lib
lib: $(DESTDIR)/$(PROJECT)

# The simulator, generated for these CONFIGS by the image's own flow.
$(GEN_SV): force
	$(MAKE) -C $(ROOT_DIR)/hw/syn/foldsim generate TARGET=hw FOLDSIM_WORK_DIR=$(WORK_DIR) CONFIGS='$(CONFIGS)' \
		$(if $(PERF),PERF=$(PERF)) $(if $(FOLDSIM_CLASSES),FOLDSIM_CLASSES='$(FOLDSIM_CLASSES)') \
		$(if $(FOLDSIM_LIFT),FOLDSIM_LIFT='$(FOLDSIM_LIFT)') $(if $(CIRCT_BIN),CIRCT_BIN=$(CIRCT_BIN)) \
		$(if $(FOLDSIM_FLAGS),FOLDSIM_FLAGS='$(FOLDSIM_FLAGS)') \
		$(if $(LIFT_PYTHON),LIFT_PYTHON=$(LIFT_PYTHON)) $(if $(FOLDSIM),FOLDSIM=$(FOLDSIM))

$(DESTDIR)/$(PROJECT): $(GEN_SV) $(SRCS) $(SRC_DIR)/xilinx_fp_models.sv $(wildcard $(FOLDSIM_RTL_DIR)/*) \
		$(FOLDSIM_SHELL)/include/foldsim_emul.h $(CONFIG_FILE)
	$(VERILATOR_PATH)/bin/verilator --build --exe $(VL_FLAGS) --cc $(RTL_SRCS) --top-module $(TOP) \
		$(SRCS) -CFLAGS '$(CXXFLAGS) -DFOLDSIM_MEM_PORTS='$$(sed -n 's/^`define FOLDSIM_MEM_PORTS //p' $(KERNEL_VH)) \
		-LDFLAGS '$(LDFLAGS)' --Mdir $@.obj_dir -o $@

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(VL_FLAGS)" "$(CXXFLAGS)" "$(CONFIGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

clean:
	rm -rf $(DESTDIR)/$(PROJECT).obj_dir $(DESTDIR)/$(PROJECT) $(WORK_DIR)
