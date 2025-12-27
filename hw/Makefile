ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

SRC_DIR := $(VORTEX_HOME)/hw
DST_DIR := $(ROOT_DIR)/hw

all: config

config: VX_config.h VX_config.vh VX_types.h VX_types.vh vortex_opae.vh vortex_opae.h

VX_config.h: $(SRC_DIR)/VX_config.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/VX_config.toml --output $(DST_DIR)/VX_config.h --format cpp

VX_config.vh: $(SRC_DIR)/VX_config.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/VX_config.toml --output $(DST_DIR)/VX_config.vh --format verilog

VX_types.h: $(SRC_DIR)/VX_types.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/VX_types.toml --output $(DST_DIR)/VX_types.h --format cpp --resolved

VX_types.vh: $(SRC_DIR)/VX_types.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/VX_types.toml --output $(DST_DIR)/VX_types.vh --format verilog --resolved

vortex_opae.h: $(SRC_DIR)/vortex_opae.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/vortex_opae.toml --output $(DST_DIR)/vortex_opae.h --format cpp

vortex_opae.vh: $(SRC_DIR)/vortex_opae.toml
	$(ROOT_DIR)/ci/gen_config.py --config $(SRC_DIR)/vortex_opae.toml --output $(DST_DIR)/vortex_opae.vh --format verilog

clean:
	rm -f VX_config.h VX_config.vh VX_types.h VX_types.vh vortex_opae.vh vortex_opae.h

.PHONY: config