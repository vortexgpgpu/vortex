all:
	$(MAKE) -C cache
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_streamer
	$(MAKE) -C top_modules

run:
	$(MAKE) -C cache run
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_streamer run
	$(MAKE) -C top_modules run

clean:
	$(MAKE) -C cache clean
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_streamer clean
	$(MAKE) -C top_modules clean
