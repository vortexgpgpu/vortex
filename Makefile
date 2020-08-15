
all:
	$(MAKE) -C hw
	$(MAKE) -C driver
	$(MAKE) -C simX
	$(MAKE) -C runtime

clean:
	$(MAKE) -C hw clean
	$(MAKE) -C driver clean
	$(MAKE) -C simX clean
	$(MAKE) -C runtime clean

