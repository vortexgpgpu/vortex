BUILDIR ?= build

.PHONY: unittest pipeline cache core vortex top1 top2 top4 top8 top16 top32 top64

unittest:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C unittest/$(BUILDIR) clean && $(MAKE) -C unittest/$(BUILDIR) > unittest//$(BUILDIR)build.log 2>&1 &

pipeline:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C pipeline/$(BUILDIR) clean && $(MAKE) -C pipeline/$(BUILDIR) > pipeline/$(BUILDIR)/build.log 2>&1 &

cache:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C cache/$(BUILDIR) clean && $(MAKE) -C cache/$(BUILDIR) > cache/$(BUILDIR)/build.log 2>&1 &

core:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C core/$(BUILDIR) clean && $(MAKE) -C core/$(BUILDIR) > core/$(BUILDIR)/build.log 2>&1 &

vortex:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C vortex/$(BUILDIR) clean && $(MAKE) -C vortex/$(BUILDIR) > vortex/$(BUILDIR)/build.log 2>&1 &

top1:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top1/$(BUILDIR) clean && $(MAKE) -C top1/$(BUILDIR) > top1/$(BUILDIR)/build.log 2>&1 &

top2:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top2/$(BUILDIR) clean && $(MAKE) -C top2/$(BUILDIR) > top2/$(BUILDIR)/build.log 2>&1 &

top4:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top4/$(BUILDIR) clean && $(MAKE) -C top4/$(BUILDIR) > top4/$(BUILDIR)/build.log 2>&1 &

top8:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top8/$(BUILDIR) clean && $(MAKE) -C top8/$(BUILDIR) > top8/$(BUILDIR)/build.log 2>&1 &

top16:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top16/$(BUILDIR) clean && $(MAKE) -C top16/$(BUILDIR) > top16/$(BUILDIR)build.log 2>&1 &

top32:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top32/$(BUILDIR) clean && $(MAKE) -C top32/$(BUILDIR) > top32/$(BUILDIR)/build.log 2>&1 &

top64:
	mkdir -p core/$(BUILDIR)
	cp core/Makefile core/$(BUILDIR)
	$(MAKE) -C top64/$(BUILDIR) clean && $(MAKE) -C top64/$(BUILDIR) > top64/$(BUILDIR)/build.log 2>&1 &