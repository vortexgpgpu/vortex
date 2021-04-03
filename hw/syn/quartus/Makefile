.PHONY: unittest pipeline cache core vortex top1 top2 top4 top8 top16 top32 top64

unittest:
	$(MAKE) -C unittest clean && $(MAKE) -C unittest > unittest/build.log 2>&1 &

pipeline:
	$(MAKE) -C pipeline clean && $(MAKE) -C pipeline > pipeline/build.log 2>&1 &

cache:
	$(MAKE) -C cache clean && $(MAKE) -C cache > cache/build.log 2>&1 &

core:
	$(MAKE) -C core clean && $(MAKE) -C core > core/build.log 2>&1 &

vortex:
	$(MAKE) -C vortex clean && $(MAKE) -C vortex > vortex/build.log 2>&1 &

top1:
	$(MAKE) -C top1 clean && $(MAKE) -C top1 > top1/build.log 2>&1 &

top2:
	$(MAKE) -C top2 clean && $(MAKE) -C top2 > top2/build.log 2>&1 &

top4:
	$(MAKE) -C top4 clean && $(MAKE) -C top4 > top4/build.log 2>&1 &

top8:
	$(MAKE) -C top8 clean && $(MAKE) -C top8 > top8/build.log 2>&1 &

top16:
	$(MAKE) -C top16 clean && $(MAKE) -C top16 > top16/build.log 2>&1 &

top32:
	$(MAKE) -C top32 clean && $(MAKE) -C top32 > top32/build.log 2>&1 &

top64:
	$(MAKE) -C top64 clean && $(MAKE) -C top64 > top64/build.log 2>&1 &