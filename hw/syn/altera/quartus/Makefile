BUILD_DIR ?= build

BUILD_DIR_X=$(BUILD_DIR)_$(DEVICE_FAMILY)

.PHONY: dogfood unittest pipeline smem cache fpu core vortex vortex-gfx top top-gfx texunit test

dogfood:
	mkdir -p dogfood/$(BUILD_DIR_X)
	cp dogfood/Makefile dogfood/$(BUILD_DIR_X)
	$(MAKE) -C dogfood/$(BUILD_DIR_X) clean && $(MAKE) -C dogfood/$(BUILD_DIR_X) > dogfood/$(BUILD_DIR_X)/build.log 2>&1 &

unittest:
	mkdir -p unittest/$(BUILD_DIR_X)
	cp unittest/Makefile unittest/$(BUILD_DIR_X)
	$(MAKE) -C unittest/$(BUILD_DIR_X) clean && $(MAKE) -C unittest/$(BUILD_DIR_X) > unittest/$(BUILD_DIR_X)/build.log 2>&1 &

pipeline:
	mkdir -p pipeline/$(BUILD_DIR_X)
	cp pipeline/Makefile pipeline/$(BUILD_DIR_X)
	$(MAKE) -C pipeline/$(BUILD_DIR_X) clean && $(MAKE) -C pipeline/$(BUILD_DIR_X) > pipeline/$(BUILD_DIR_X)/build.log 2>&1 &

smem:
	mkdir -p smem/$(BUILD_DIR_X)
	cp smem/Makefile smem/$(BUILD_DIR_X)
	$(MAKE) -C smem/$(BUILD_DIR_X) clean && $(MAKE) -C smem/$(BUILD_DIR_X) > smem/$(BUILD_DIR_X)/build.log 2>&1 &

cache:
	mkdir -p cache/$(BUILD_DIR_X)
	cp cache/Makefile cache/$(BUILD_DIR_X)
	$(MAKE) -C cache/$(BUILD_DIR_X) clean && $(MAKE) -C cache/$(BUILD_DIR_X) > cache/$(BUILD_DIR_X)/build.log 2>&1 &

fpu:
	mkdir -p fpu/$(BUILD_DIR_X)
	cp fpu/Makefile fpu/$(BUILD_DIR_X)
	$(MAKE) -C fpu/$(BUILD_DIR_X) clean && $(MAKE) -C fpu/$(BUILD_DIR_X) > fpu/$(BUILD_DIR_X)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILD_DIR_X)
	cp core/Makefile core/$(BUILD_DIR_X)
	$(MAKE) -C core/$(BUILD_DIR_X) clean && $(MAKE) -C core/$(BUILD_DIR_X) > core/$(BUILD_DIR_X)/build.log 2>&1 &

vortex:
	mkdir -p vortex/$(BUILD_DIR_X)
	cp vortex/Makefile vortex/$(BUILD_DIR_X)
	$(MAKE) -C vortex/$(BUILD_DIR_X) clean && $(MAKE) -C vortex/$(BUILD_DIR_X) > vortex/$(BUILD_DIR_X)/build.log 2>&1 &

vortex-gfx:
	mkdir -p vortex-gfx/$(BUILD_DIR_X)
	cp vortex-gfx/Makefile vortex-gfx/$(BUILD_DIR_X)
	$(MAKE) -C vortex-gfx/$(BUILD_DIR_X) clean && $(MAKE) -C vortex-gfx/$(BUILD_DIR_X) > vortex-gfx/$(BUILD_DIR_X)/build.log 2>&1 &

top:
	mkdir -p top/$(BUILD_DIR_X)
	cp top/Makefile top/$(BUILD_DIR_X)
	$(MAKE) -C top/$(BUILD_DIR_X) clean && $(MAKE) -C top/$(BUILD_DIR_X) > top/$(BUILD_DIR_X)/build.log 2>&1 &

top-gfx:
	mkdir -p top-gfx/$(BUILD_DIR_X)
	cp top-gfx/Makefile top-gfx/$(BUILD_DIR_X)
	$(MAKE) -C top-gfx/$(BUILD_DIR_X) clean && $(MAKE) -C top-gfx/$(BUILD_DIR_X) > top-gfx/$(BUILD_DIR_X)/build.log 2>&1 &

texunit:
	mkdir -p texunit/$(BUILD_DIR_X)
	cp texunit/Makefile texunit/$(BUILD_DIR_X)
	$(MAKE) -C texunit/$(BUILD_DIR_X) clean && $(MAKE) -C texunit/$(BUILD_DIR_X) > texunit/$(BUILD_DIR_X)/build.log 2>&1 &

test:
	mkdir -p test/$(BUILD_DIR_X)
	cp test/Makefile test/$(BUILD_DIR_X)
	$(MAKE) -C test/$(BUILD_DIR_X) clean && $(MAKE) -C test/$(BUILD_DIR_X) > test/$(BUILD_DIR_X)/build.log 2>&1 &