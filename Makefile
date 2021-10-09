all:
	$(MAKE) -C hw
	$(MAKE) -C sim
	$(MAKE) -C driver
	$(MAKE) -C runtime
	$(MAKE) -C tests
	
clean:
	$(MAKE) -C hw clean
	$(MAKE) -C sim clean
	$(MAKE) -C driver clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean