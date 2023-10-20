PREFIX ?= build

BUILD_DIR=$(PREFIX)_$(DEVICE_FAMILY)

IP_CACHE_DIR=../ip_cache/$(DEVICE_FAMILY)

.PHONY: dogfood unittest pipeline smem cache fpu core vortex top test

ip-gen: $(IP_CACHE_DIR)/ip_gen.log
$(IP_CACHE_DIR)/ip_gen.log:
	../ip_gen.sh $(IP_CACHE_DIR)

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

fpu: ip-gen
	mkdir -p fpu/$(BUILD_DIR)
	cp fpu/Makefile fpu/$(BUILD_DIR)
	$(MAKE) -C fpu/$(BUILD_DIR) clean && $(MAKE) -C fpu/$(BUILD_DIR) > fpu/$(BUILD_DIR)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILD_DIR)
	cp core/Makefile core/$(BUILD_DIR)
	$(MAKE) -C core/$(BUILD_DIR) clean && $(MAKE) -C core/$(BUILD_DIR) > core/$(BUILD_DIR)/build.log 2>&1 &

vortex: ip-gen
	mkdir -p vortex/$(BUILD_DIR)
	cp vortex/Makefile vortex/$(BUILD_DIR)
	$(MAKE) -C vortex/$(BUILD_DIR) clean && $(MAKE) -C vortex/$(BUILD_DIR) > vortex/$(BUILD_DIR)/build.log 2>&1 &

top: ip-gen
	mkdir -p top/$(BUILD_DIR)
	cp top/Makefile top/$(BUILD_DIR)
	$(MAKE) -C top/$(BUILD_DIR) clean && $(MAKE) -C top/$(BUILD_DIR) > top/$(BUILD_DIR)/build.log 2>&1 &

test: ip-gen
	mkdir -p test/$(BUILD_DIR)
	cp test/Makefile test/$(BUILD_DIR)
	$(MAKE) -C test/$(BUILD_DIR) clean && $(MAKE) -C test/$(BUILD_DIR) > test/$(BUILD_DIR)/build.log 2>&1 &
