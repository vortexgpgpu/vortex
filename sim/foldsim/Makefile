include ../common.mk

# The foldsim kernel in Verilator, as a library the runtime's avedsim target
# links: the same simulator hw/syn/foldsim generates for the image, so a run
# here is the image's run, cycle for cycle, but for device memory's timing.

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/sim/foldsim
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim
WORK_DIR := $(DESTDIR)/foldsim_emul
GEN_SV := $(WORK_DIR)/gen/foldsim_sim.sv
# The kernel's memory ports, as the image's flow derived them, and the
# simulator's virtual memories, each on a master of its own after them.
KERNEL_VH := $(WORK_DIR)/gen/foldsim_kernel.vh
SIM_VH := $(WORK_DIR)/gen/foldsim_sim.vh

CONFIG_FILE := $(DESTDIR)/foldsim_emul_config.stamp

CXXFLAGS += -std=c++17 -fPIC -O2 -DNDEBUG -I$(SRC_DIR)
LDFLAGS += -shared -pthread
# The Xilinx floating-point cores' simulation models compute with Vortex's DPI
# floating point.
CXXFLAGS += -I$(SW_COMMON_DIR) -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

VL_FLAGS += -O3 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN --x-initial 0 --x-assign 0
VL_FLAGS += -I$(FOLDSIM_RTL_DIR) -I$(WORK_DIR)/gen -I$(DPI_DIR)
# The kernel expands its virtual-memory controllers from one macro line
# (FOLDSIM_VMEM_REPEAT): past ~32 controllers that line holds more than
# Verilator's default 40000 preprocessor tokens.
VL_FLAGS += --preproc-token-limit 4000000
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
            $(FOLDSIM_RTL_DIR)/foldsim_mem.sv $(FOLDSIM_RTL_DIR)/foldsim_vmem.sv $(FOLDSIM_RTL_DIR)/foldsim_vmem_share.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_vmem_group.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_ctrl.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_afu.sv
# Folded memory ports (FOLDSIM_PORTS in the generated foldsim_sim.vh): the folded
# endpoint around the memory endpoint threaded by portfold_gen.sh, from the
# kernel sources (hw/syn/foldsim's FOLD_SRCS).
FOLD_SRCS = $(if $(shell grep -s '^.define FOLDSIM_PORTS ' $(SIM_VH)),\
                 $(FOLDSIM_RTL_DIR)/foldsim_mem_threaded.sv $(FOLDSIM_RTL_DIR)/foldsim_mem_fold.sv)
SRCS := $(SRC_DIR)/foldsim_emul.cpp $(DPI_DIR)/dpi_float.cpp $(DPI_DIR)/dpi_util.cpp \
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
		$(if $(LIFT_PYTHON),LIFT_PYTHON=$(LIFT_PYTHON)) $(if $(FOLDSIM),FOLDSIM=$(FOLDSIM)) \
		$(if $(FOLDSIM_VMEM_CACHE_ROWS),FOLDSIM_VMEM_CACHE_ROWS=$(FOLDSIM_VMEM_CACHE_ROWS)) \
		$(if $(FOLDSIM_VMEM_MASTERS),FOLDSIM_VMEM_MASTERS=$(FOLDSIM_VMEM_MASTERS))

$(DESTDIR)/$(PROJECT): $(GEN_SV) $(SRCS) $(SRC_DIR)/xilinx_fp_models.sv $(wildcard $(FOLDSIM_RTL_DIR)/*) $(CONFIG_FILE)
	$(VERILATOR_PATH)/bin/verilator --build --exe $(VL_FLAGS) --cc $(RTL_SRCS) $(FOLD_SRCS) --top-module $(TOP) \
		$(SRCS) -CFLAGS '$(CXXFLAGS) -DFOLDSIM_MEM_MASTERS='$$(sed -n 's/^`define FOLDSIM_MEM_MASTERS //p' $(KERNEL_VH)) \
		-CFLAGS -DFOLDSIM_VMEM_COUNT=$$(sed -n -e 's/^`define FOLDSIM_VMEM_GROUPS //p' -e 's/^`define FOLDSIM_VMEM_COUNT //p' $(SIM_VH) | sort -n | head -1 | grep . || echo 0) \
		$$(sed -n -e 's/^`define FOLDSIM_VMEM_MASTERS \(.*\)/-CFLAGS -DFOLDSIM_VMEM_MASTERS=\1/p' \
		          -e 's/^`define FOLDSIM_VMEM_ON_PORTS.*/-CFLAGS -DFOLDSIM_VMEM_MASTERS=0/p' $(KERNEL_VH)) \
		-LDFLAGS '$(LDFLAGS)' --Mdir $@.obj_dir -o $@

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(VL_FLAGS)" "$(CXXFLAGS)" "$(CONFIGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

clean:
	rm -rf $(DESTDIR)/$(PROJECT).obj_dir $(DESTDIR)/$(PROJECT) $(WORK_DIR)
