ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

PREFIX ?= build

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, 3 -- default. NOT supported on this backend.
OPT_LEVEL ?= 3
ifneq ($(OPT_LEVEL),3)
$(error OPT_LEVEL=$(OPT_LEVEL) requested but not supported on the altera/dut backend)
endif
export OPT_LEVEL

BUILD_DIR := $(PREFIX)_$(DEVICE_FAMILY)

SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

IP_CACHE_DIR := $(ROOT_DIR)/hw/syn/altera/ip_cache/$(DEVICE_FAMILY)

# DUTS + the per-DUT definitions. Adding a DUT touches catalog.mk only.
include catalog.mk

.PHONY: ip-gen list $(DUTS)

ip-gen: $(IP_CACHE_DIR)/ip_gen.log
$(IP_CACHE_DIR)/ip_gen.log:
	$(SCRIPT_DIR)/altera_ip_gen.sh $(IP_CACHE_DIR)

# Each DUT builds out-of-tree in its own <BUILD_DIR>_<dut>/ so parallel builds
# cannot collide. build.mk is copied in as the Makefile, exactly as the per-DUT
# Makefiles used to be.
$(DUTS):
	mkdir -p $(BUILD_DIR)_$@
	cp build.mk $(BUILD_DIR)_$@/Makefile
	$(MAKE) -C $(BUILD_DIR)_$@ DUT=$@ clean && $(MAKE) -C $(BUILD_DIR)_$@ DUT=$@ > $(BUILD_DIR)_$@/build.log 2>&1 &

list:
	@echo $(DUTS) | tr ' ' '\n'
