ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

PREFIX ?= build

BUILD_DIR := $(PREFIX)

SAIF_FILE ?=
SAIF_INST ?=

.PHONY: unittest scope mem_unit lmem cache fpu tcu dxa tex raster om core issue vortex top

unittest:
	mkdir -p unittest/$(BUILD_DIR)
	cp unittest/Makefile unittest/$(BUILD_DIR)
	$(MAKE) -C unittest/$(BUILD_DIR) clean && $(MAKE) -C unittest/$(BUILD_DIR) > unittest/$(BUILD_DIR)/build.log 2>&1 &

scope:
	mkdir -p scope/$(BUILD_DIR)
	cp scope/Makefile scope/$(BUILD_DIR)
	$(MAKE) -C scope/$(BUILD_DIR) clean && $(MAKE) -C scope/$(BUILD_DIR) > scope/$(BUILD_DIR)/build.log 2>&1 &

mem_unit:
	mkdir -p mem_unit/$(BUILD_DIR)
	cp mem_unit/Makefile mem_unit/$(BUILD_DIR)
	$(MAKE) -C mem_unit/$(BUILD_DIR) clean && $(MAKE) -C mem_unit/$(BUILD_DIR) > mem_unit/$(BUILD_DIR)/build.log 2>&1 &

lmem:
	mkdir -p lmem/$(BUILD_DIR)
	cp lmem/Makefile lmem/$(BUILD_DIR)
	$(MAKE) -C lmem/$(BUILD_DIR) clean && $(MAKE) -C lmem/$(BUILD_DIR) > lmem/$(BUILD_DIR)/build.log 2>&1 &

cache:
	mkdir -p cache/$(BUILD_DIR)
	cp cache/Makefile cache/$(BUILD_DIR)
	$(MAKE) -C cache/$(BUILD_DIR) clean && $(MAKE) -C cache/$(BUILD_DIR) > cache/$(BUILD_DIR)/build.log 2>&1 &

fpu:
	mkdir -p fpu/$(BUILD_DIR)
	cp fpu/Makefile fpu/$(BUILD_DIR)
	$(MAKE) -C fpu/$(BUILD_DIR) clean && $(MAKE) -C fpu/$(BUILD_DIR) > fpu/$(BUILD_DIR)/build.log 2>&1 &

tcu:
	mkdir -p tcu/$(BUILD_DIR)
	cp tcu/Makefile tcu/$(BUILD_DIR)
	$(MAKE) -C tcu/$(BUILD_DIR) clean && $(MAKE) -C tcu/$(BUILD_DIR) > tcu/$(BUILD_DIR)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILD_DIR)
	cp core/Makefile core/$(BUILD_DIR)
	$(MAKE) -C core/$(BUILD_DIR) clean && $(MAKE) -C core/$(BUILD_DIR) > core/$(BUILD_DIR)/build.log 2>&1 &

issue:
	mkdir -p issue/$(BUILD_DIR)
	cp issue/Makefile issue/$(BUILD_DIR)
	$(MAKE) -C issue/$(BUILD_DIR) clean && $(MAKE) -C issue/$(BUILD_DIR) > issue/$(BUILD_DIR)/build.log 2>&1 &

vortex:
	mkdir -p vortex/$(BUILD_DIR)
	cp vortex/Makefile vortex/$(BUILD_DIR)
	$(MAKE) -C vortex/$(BUILD_DIR) clean && $(MAKE) -C vortex/$(BUILD_DIR) > vortex/$(BUILD_DIR)/build.log 2>&1 &

top:
	mkdir -p top/$(BUILD_DIR)
	cp top/Makefile top/$(BUILD_DIR)
	$(MAKE) -C top/$(BUILD_DIR) clean && $(MAKE) -C top/$(BUILD_DIR) > top/$(BUILD_DIR)/build.log 2>&1 &

dxa:
	mkdir -p dxa/$(BUILD_DIR)
	cp dxa/Makefile dxa/$(BUILD_DIR)
	$(MAKE) -C dxa/$(BUILD_DIR) clean && $(MAKE) -C dxa/$(BUILD_DIR) > dxa/$(BUILD_DIR)/build.log 2>&1 &

tex:
	mkdir -p tex/$(BUILD_DIR)
	cp tex/Makefile tex/$(BUILD_DIR)
	$(MAKE) -C tex/$(BUILD_DIR) clean && $(MAKE) -C tex/$(BUILD_DIR) > tex/$(BUILD_DIR)/build.log 2>&1 &

raster:
	mkdir -p raster/$(BUILD_DIR)
	cp raster/Makefile raster/$(BUILD_DIR)
	$(MAKE) -C raster/$(BUILD_DIR) clean && $(MAKE) -C raster/$(BUILD_DIR) > raster/$(BUILD_DIR)/build.log 2>&1 &

om:
	mkdir -p om/$(BUILD_DIR)
	cp om/Makefile om/$(BUILD_DIR)
	$(MAKE) -C om/$(BUILD_DIR) clean && $(MAKE) -C om/$(BUILD_DIR) > om/$(BUILD_DIR)/build.log 2>&1 &

# Re-run power analysis on an existing build without rebuilding the design.
# Usage: make <target>-power SAIF_FILE=<saif-path> [SAIF_INST=<top-module-instance-path>]
%-power:
	@target=$*; \
	if [ ! -d $$target/$(BUILD_DIR) ]; then \
	  echo "ERROR: $$target/$(BUILD_DIR) not found. Run 'make $$target' first."; exit 1; \
	fi
	$(MAKE) -C $*/$(BUILD_DIR) power SAIF_FILE=$(SAIF_FILE) SAIF_INST=$(SAIF_INST)