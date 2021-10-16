BUILD_DIR ?= build

.PHONY: dogfood unittest pipeline smem cache fpu_core core vortex top1 top2 top4 top8 top16 top32 top64

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

top1:
	mkdir -p top1/$(BUILD_DIR)
	cp top1/Makefile top1/$(BUILD_DIR)
	$(MAKE) -C top1/$(BUILD_DIR) clean && $(MAKE) -C top1/$(BUILD_DIR) > top1/$(BUILD_DIR)/build.log 2>&1 &

top2:
	mkdir -p top2/$(BUILD_DIR)
	cp top2/Makefile top2/$(BUILD_DIR)
	$(MAKE) -C top2/$(BUILD_DIR) clean && $(MAKE) -C top2/$(BUILD_DIR) > top2/$(BUILD_DIR)/build.log 2>&1 &

top4:
	mkdir -p top4/$(BUILD_DIR)
	cp top4/Makefile top4/$(BUILD_DIR)
	$(MAKE) -C top4/$(BUILD_DIR) clean && $(MAKE) -C top4/$(BUILD_DIR) > top4/$(BUILD_DIR)/build.log 2>&1 &

top8:
	mkdir -p top8/$(BUILD_DIR)
	cp top8/Makefile top8/$(BUILD_DIR)
	$(MAKE) -C top8/$(BUILD_DIR) clean && $(MAKE) -C top8/$(BUILD_DIR) > top8/$(BUILD_DIR)/build.log 2>&1 &

top16:
	mkdir -p top16/$(BUILD_DIR)
	cp top16/Makefile top16/$(BUILD_DIR)
	$(MAKE) -C top16/$(BUILD_DIR) clean && $(MAKE) -C top16/$(BUILD_DIR) > top16/$(BUILD_DIR)/build.log 2>&1 &

top32:
	mkdir -p top32/$(BUILD_DIR)
	cp top32/Makefile top32/$(BUILD_DIR)
	$(MAKE) -C top32/$(BUILD_DIR) clean && $(MAKE) -C top32/$(BUILD_DIR) > top32/$(BUILD_DIR)/build.log 2>&1 &

top64:
	mkdir -p top64/$(BUILD_DIR)
	cp top64/Makefile top64/$(BUILD_DIR)
	$(MAKE) -C top64/$(BUILD_DIR) clean && $(MAKE) -C top64/$(BUILD_DIR) > top64/$(BUILD_DIR)/build.log 2>&1 &