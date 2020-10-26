
all:
	$(MAKE) -C hw
	$(MAKE) -C driver
	$(MAKE) -C runtime
	$(MAKE) -C simX	
	$(MAKE) -C ben benchmarks/opencl

clean:
	$(MAKE) -C hw clean
	$(MAKE) -C driver clean
	$(MAKE) -C simX clean
	$(MAKE) -C runtime clean
	$(MAKE) -C ben benchmarks/opencl clean

