ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

PREFIX ?= build

SAIF_FILE ?=
SAIF_INST ?=

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile (no opt), 3 -- default. Forwarded to each sub-build.
OPT_LEVEL ?= 3
export OPT_LEVEL

# DUTS + the per-DUT definitions. Adding a DUT touches catalog.mk only.
include catalog.mk

# Each DUT builds out-of-tree in its own <PREFIX>_<dut>/ so parallel builds (and
# a hand-run synthesis on the same machine) cannot collide. build.mk is copied
# in as the Makefile, exactly as the per-DUT Makefiles used to be.
.PHONY: $(DUTS)
$(DUTS):
	mkdir -p $(PREFIX)_$@
	cp build.mk $(PREFIX)_$@/Makefile
	$(MAKE) -C $(PREFIX)_$@ DUT=$@ clean && $(MAKE) -C $(PREFIX)_$@ DUT=$@ > $(PREFIX)_$@/build.log 2>&1 &

# Re-run power analysis on an existing build without rebuilding the design.
# Usage: make <target>-power SAIF_FILE=<saif-path> [SAIF_INST=<top-module-instance-path>]
%-power:
	@if [ ! -d $(PREFIX)_$* ]; then \
	  echo "ERROR: $(PREFIX)_$* not found. Run 'make $*' first."; exit 1; \
	fi
	$(MAKE) -C $(PREFIX)_$* DUT=$* power SAIF_FILE=$(SAIF_FILE) SAIF_INST=$(SAIF_INST)

.PHONY: list
list:
	@echo $(DUTS) | tr ' ' '\n'
