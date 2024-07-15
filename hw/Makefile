ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

HW_DIR := $(VORTEX_HOME)/hw
SCRIPT_DIR := $(HW_DIR)/scripts
RTL_DIR := $(HW_DIR)/rtl

all: config

config: VX_config.h VX_types.h

VX_config.h: $(RTL_DIR)/VX_config.vh
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_config.vh -o VX_config.h

VX_types.h: $(RTL_DIR)/VX_types.vh
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_types.vh -o VX_types.h

clean:
	$(MAKE) -C unittest clean
	rm -f VX_config.h VX_types.h

.PHONY: VX_config.h VX_types.h