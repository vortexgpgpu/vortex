all:
	$(MAKE) -C simple
	$(MAKE) -C dev
	$(MAKE) -C hello
	$(MAKE) -C nlTest
	$(MAKE) -C vecadd

run:
	cd simple && $(MAKE) run
	cd dev && $(MAKE) run	
	cd hello && $(MAKE) run
	cd nlTest && $(MAKE) run
	cd vecadd && $(MAKE) run

clean:
	$(MAKE) -C simple clean
	$(MAKE) -C dev clean
	$(MAKE) -C hello clean
	$(MAKE) -C nlTest clean
	$(MAKE) -C vecadd clean

