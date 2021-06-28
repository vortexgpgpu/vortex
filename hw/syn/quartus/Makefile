BUILDIR ?= build

.PHONY: unittest pipeline smem cache core vortex top1 top2 top4 top8 top16 top32 top64

unittest:
	mkdir -p unittest/$(BUILDIR)
	cp unittest/Makefile unittest/$(BUILDIR)
	$(MAKE) -C unittest/$(BUILDIR) clean && $(MAKE) -C unittest/$(BUILDIR) > unittest/$(BUILDIR)/build.log 2>&1 &

pipeline:
	mkdir -p pipeline/$(BUILDIR)
	cp pipeline/Makefile pipeline/$(BUILDIR)
	$(MAKE) -C pipeline/$(BUILDIR) clean && $(MAKE) -C pipeline/$(BUILDIR) > pipeline/$(BUILDIR)/build.log 2>&1 &

smem:
	mkdir -p smem/$(BUILDIR)
	cp smem/Makefile smem/$(BUILDIR)
	$(MAKE) -C smem/$(BUILDIR) clean && $(MAKE) -C smem/$(BUILDIR) > smem/$(BUILDIR)/build.log 2>&1 &

cache:
	mkdir -p cache/$(BUILDIR)
	cp cache/Makefile cache/$(BUILDIR)
	$(MAKE) -C cache/$(BUILDIR) clean && $(MAKE) -C cache/$(BUILDIR) > cache/$(BUILDIR)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C core/$(BUILDIR) clean && $(MAKE) -C core/$(BUILDIR) > core/$(BUILDIR)/build.log 2>&1 &

vortex:
	mkdir -p vortex/$(BUILDIR)
	cp vortex/Makefile vortex/$(BUILDIR)
	$(MAKE) -C vortex/$(BUILDIR) clean && $(MAKE) -C vortex/$(BUILDIR) > vortex/$(BUILDIR)/build.log 2>&1 &

top1:
	mkdir -p top1/$(BUILDIR)
	cp top1/Makefile top1/$(BUILDIR)
	$(MAKE) -C top1/$(BUILDIR) clean && $(MAKE) -C top1/$(BUILDIR) > top1/$(BUILDIR)/build.log 2>&1 &

top2:
	mkdir -p top2/$(BUILDIR)
	cp top2/Makefile top2/$(BUILDIR)
	$(MAKE) -C top2/$(BUILDIR) clean && $(MAKE) -C top2/$(BUILDIR) > top2/$(BUILDIR)/build.log 2>&1 &

top4:
	mkdir -p top4/$(BUILDIR)
	cp top4/Makefile top4/$(BUILDIR)
	$(MAKE) -C top4/$(BUILDIR) clean && $(MAKE) -C top4/$(BUILDIR) > top4/$(BUILDIR)/build.log 2>&1 &

top8:
	mkdir -p top8/$(BUILDIR)
	cp top8/Makefile top8/$(BUILDIR)
	$(MAKE) -C top8/$(BUILDIR) clean && $(MAKE) -C top8/$(BUILDIR) > top8/$(BUILDIR)/build.log 2>&1 &

top16:
	mkdir -p top16/$(BUILDIR)
	cp top16/Makefile top16/$(BUILDIR)
	$(MAKE) -C top16/$(BUILDIR) clean && $(MAKE) -C top16/$(BUILDIR) > top16/$(BUILDIR)/build.log 2>&1 &

top32:
	mkdir -p top32/$(BUILDIR)
	cp top32/Makefile top32/$(BUILDIR)
	$(MAKE) -C top32/$(BUILDIR) clean && $(MAKE) -C top32/$(BUILDIR) > top32/$(BUILDIR)/build.log 2>&1 &

top64:
	mkdir -p top64/$(BUILDIR)
	cp top64/Makefile top64/$(BUILDIR)
	$(MAKE) -C top64/$(BUILDIR) clean && $(MAKE) -C top64/$(BUILDIR) > top64/$(BUILDIR)/build.log 2>&1 &