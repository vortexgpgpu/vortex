
all:
	$(MAKE) -C hw
	$(MAKE) -C driver
	$(MAKE) -C runtime
	$(MAKE) -C simX	
	$(MAKE) -C tests
	
clean:
	$(MAKE) -C hw clean
	$(MAKE) -C driver clean
	$(MAKE) -C simX clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean