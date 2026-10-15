include ../common.mk

# The foldsim kernel in Verilator, as a library the runtime's avedsim target
# links: the same simulator hw/syn/foldsim generates for the image, so a run
# here is the image's run, cycle for cycle, but for device memory's timing.

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/sim/foldsim
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim
WORK_DIR := $(DESTDIR)/foldsim_emul
GEN_SV := $(WORK_DIR)/gen/foldsim_sim.sv

CONFIG_FILE := $(DESTDIR)/foldsim_emul_config.stamp

CXXFLAGS += -std=c++17 -fPIC -O2 -DNDEBUG -I$(SRC_DIR)
LDFLAGS += -shared -pthread

VL_FLAGS += -O3 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN --x-initial 0 --x-assign 0
VL_FLAGS += -I$(FOLDSIM_RTL_DIR) -I$(WORK_DIR)/gen
ifdef VCD
VL_FLAGS += --trace --trace-structs
endif
THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(min(32, mp.cpu_count()))')
VL_FLAGS += -j $(THREADS)

TOP := foldsim_afu
RTL_SRCS := $(GEN_SV) $(FOLDSIM_RTL_DIR)/foldsim_token.sv $(FOLDSIM_RTL_DIR)/foldsim_mem.sv \
            $(FOLDSIM_RTL_DIR)/foldsim_ctrl.sv $(FOLDSIM_RTL_DIR)/foldsim_afu.sv
SRCS := $(SRC_DIR)/foldsim_emul.cpp

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

$(DESTDIR)/$(PROJECT): $(GEN_SV) $(SRCS) $(wildcard $(FOLDSIM_RTL_DIR)/*) $(CONFIG_FILE)
	$(VERILATOR_PATH)/bin/verilator --build --exe $(VL_FLAGS) --cc $(RTL_SRCS) --top-module $(TOP) \
		$(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' --Mdir $@.obj_dir -o $@

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(VL_FLAGS)" "$(CXXFLAGS)" "$(CONFIGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

clean:
	rm -rf $(DESTDIR)/$(PROJECT).obj_dir $(DESTDIR)/$(PROJECT) $(WORK_DIR)
