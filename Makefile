
all:
	$(MAKE) -C hw
	$(MAKE) -C driver
	$(MAKE) -C runtime
	$(MAKE) -C simX	
	$(MAKE) -C benchmarks/opencl

perf-demo:
	$(MAKE) -C hw
	$(MAKE) -C driver rtlsim
	$(MAKE) -C driver/tests/demo/ run-rtlsim
	
clean:
	$(MAKE) -C hw clean
	$(MAKE) -C driver clean
	$(MAKE) -C simX clean
	$(MAKE) -C runtime clean
	$(MAKE) -C benchmarks/opencl clean

