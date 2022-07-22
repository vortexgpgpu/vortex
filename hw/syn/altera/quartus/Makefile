BUILD_DIR ?= build

.PHONY: dogfood unittest pipeline smem cache fpu_core core vortex vortex-gfx top top-gfx texunit test

dogfood:
	mkdir -p dogfood/$(BUILD_DIR)
	cp dogfood/Makefile dogfood/$(BUILD_DIR)
	$(MAKE) -C dogfood/$(BUILD_DIR) clean && $(MAKE) -C dogfood/$(BUILD_DIR) > dogfood/$(BUILD_DIR)/build.log 2>&1 &

unittest:
	mkdir -p unittest/$(BUILD_DIR)
	cp unittest/Makefile unittest/$(BUILD_DIR)
	$(MAKE) -C unittest/$(BUILD_DIR) clean && $(MAKE) -C unittest/$(BUILD_DIR) > unittest/$(BUILD_DIR)/build.log 2>&1 &

pipeline:
	mkdir -p pipeline/$(BUILD_DIR)
	cp pipeline/Makefile pipeline/$(BUILD_DIR)
	$(MAKE) -C pipeline/$(BUILD_DIR) clean && $(MAKE) -C pipeline/$(BUILD_DIR) > pipeline/$(BUILD_DIR)/build.log 2>&1 &

smem:
	mkdir -p smem/$(BUILD_DIR)
	cp smem/Makefile smem/$(BUILD_DIR)
	$(MAKE) -C smem/$(BUILD_DIR) clean && $(MAKE) -C smem/$(BUILD_DIR) > smem/$(BUILD_DIR)/build.log 2>&1 &

cache:
	mkdir -p cache/$(BUILD_DIR)
	cp cache/Makefile cache/$(BUILD_DIR)
	$(MAKE) -C cache/$(BUILD_DIR) clean && $(MAKE) -C cache/$(BUILD_DIR) > cache/$(BUILD_DIR)/build.log 2>&1 &

fpu_core:
	mkdir -p fpu_core/$(BUILD_DIR)
	cp fpu_core/Makefile fpu_core/$(BUILD_DIR)
	$(MAKE) -C fpu_core/$(BUILD_DIR) clean && $(MAKE) -C fpu_core/$(BUILD_DIR) > fpu_core/$(BUILD_DIR)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILD_DIR)
	cp core/Makefile core/$(BUILD_DIR)
	$(MAKE) -C core/$(BUILD_DIR) clean && $(MAKE) -C core/$(BUILD_DIR) > core/$(BUILD_DIR)/build.log 2>&1 &

vortex:
	mkdir -p vortex/$(BUILD_DIR)
	cp vortex/Makefile vortex/$(BUILD_DIR)
	$(MAKE) -C vortex/$(BUILD_DIR) clean && $(MAKE) -C vortex/$(BUILD_DIR) > vortex/$(BUILD_DIR)/build.log 2>&1 &

vortex-gfx:
	mkdir -p vortex-gfx/$(BUILD_DIR)
	cp vortex-gfx/Makefile vortex-gfx/$(BUILD_DIR)
	$(MAKE) -C vortex-gfx/$(BUILD_DIR) clean && $(MAKE) -C vortex-gfx/$(BUILD_DIR) > vortex-gfx/$(BUILD_DIR)/build.log 2>&1 &

top:
	mkdir -p top/$(BUILD_DIR)
	cp top/Makefile top/$(BUILD_DIR)
	$(MAKE) -C top/$(BUILD_DIR) clean && $(MAKE) -C top/$(BUILD_DIR) > top/$(BUILD_DIR)/build.log 2>&1 &

top-gfx:
	mkdir -p top-gfx/$(BUILD_DIR)
	cp top-gfx/Makefile top-gfx/$(BUILD_DIR)
	$(MAKE) -C top-gfx/$(BUILD_DIR) clean && $(MAKE) -C top-gfx/$(BUILD_DIR) > top-gfx/$(BUILD_DIR)/build.log 2>&1 &

texunit:
	mkdir -p texunit/$(BUILD_DIR)
	cp texunit/Makefile texunit/$(BUILD_DIR)
	$(MAKE) -C texunit/$(BUILD_DIR) clean && $(MAKE) -C texunit/$(BUILD_DIR) > texunit/$(BUILD_DIR)/build.log 2>&1 &

test:
	mkdir -p test/$(BUILD_DIR)
	cp test/Makefile test/$(BUILD_DIR)
	$(MAKE) -C test/$(BUILD_DIR) clean && $(MAKE) -C test/$(BUILD_DIR) > test/$(BUILD_DIR)/build.log 2>&1 &