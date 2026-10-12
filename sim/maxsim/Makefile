include ../common.mk

# The maxsim kernel in Verilator, as a library the runtime's avedsim target
# links: the same simulator hw/syn/maxsim generates for the image, so a run
# here is the image's run, cycle for cycle, but for device memory's timing.

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/sim/maxsim
MAXSIM_RTL_DIR := $(RTL_DIR)/afu/maxsim
WORK_DIR := $(DESTDIR)/maxsim_emul
GEN_SV := $(WORK_DIR)/gen/maxsim_sim.sv

CONFIG_FILE := $(DESTDIR)/maxsim_emul_config.stamp

CXXFLAGS += -std=c++17 -fPIC -O2 -DNDEBUG -I$(SRC_DIR)
LDFLAGS += -shared -pthread

VL_FLAGS += -O3 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN --x-initial 0 --x-assign 0
VL_FLAGS += -I$(MAXSIM_RTL_DIR) -I$(WORK_DIR)/gen
THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(min(32, mp.cpu_count()))')
VL_FLAGS += -j $(THREADS)

TOP := maxsim_afu
RTL_SRCS := $(GEN_SV) $(MAXSIM_RTL_DIR)/maxsim_token.sv $(MAXSIM_RTL_DIR)/maxsim_mem.sv \
            $(MAXSIM_RTL_DIR)/maxsim_ctrl.sv $(MAXSIM_RTL_DIR)/maxsim_afu.sv
SRCS := $(SRC_DIR)/maxsim_emul.cpp

PROJECT := libmaxsim_emul.so

.PHONY: all lib force clean

all: lib
lib: $(DESTDIR)/$(PROJECT)

# The simulator, generated for these CONFIGS by the image's own flow.
$(GEN_SV): force
	$(MAKE) -C $(ROOT_DIR)/hw/syn/maxsim generate TARGET=hw MAXSIM_WORK_DIR=$(WORK_DIR) CONFIGS='$(CONFIGS)' \
		$(if $(PERF),PERF=$(PERF)) $(if $(MAXSIM_CLASSES),MAXSIM_CLASSES='$(MAXSIM_CLASSES)') \
		$(if $(MAXSIM_LIFT),MAXSIM_LIFT='$(MAXSIM_LIFT)') $(if $(CIRCT_BIN),CIRCT_BIN=$(CIRCT_BIN)) \
		$(if $(LIFT_PYTHON),LIFT_PYTHON=$(LIFT_PYTHON))

$(DESTDIR)/$(PROJECT): $(GEN_SV) $(SRCS) $(wildcard $(MAXSIM_RTL_DIR)/*) $(CONFIG_FILE)
	$(VERILATOR_PATH)/bin/verilator --build --exe $(VL_FLAGS) --cc $(RTL_SRCS) --top-module $(TOP) \
		$(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' --Mdir $@.obj_dir -o $@

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(VL_FLAGS)" "$(CXXFLAGS)" "$(CONFIGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

clean:
	rm -rf $(DESTDIR)/$(PROJECT).obj_dir $(DESTDIR)/$(PROJECT) $(WORK_DIR)
