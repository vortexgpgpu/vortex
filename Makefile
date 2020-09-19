
all:
	$(MAKE) -C hw
	$(MAKE) -C driver
	$(MAKE) -C runtime
	$(MAKE) -C simX	

clean:
	$(MAKE) -C hw clean
	$(MAKE) -C driver clean
	$(MAKE) -C simX clean
	$(MAKE) -C runtime clean

