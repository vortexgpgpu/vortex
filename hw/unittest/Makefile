all:
	$(MAKE) -C cache
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_streamer
	$(MAKE) -C cache_top
	$(MAKE) -C core_top
	$(MAKE) -C issue_top

run:
	$(MAKE) -C cache run
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_streamer run
	$(MAKE) -C cache_top run
	$(MAKE) -C core_top run
	$(MAKE) -C issue_top run

clean:
	$(MAKE) -C cache clean
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_streamer clean
	$(MAKE) -C cache_top clean
	$(MAKE) -C core_top clean
	$(MAKE) -C issue_top clean