all:
	$(MAKE) -C third_party
	$(MAKE) -C hw
	$(MAKE) -C sim
	$(MAKE) -C kernel
	$(MAKE) -C runtime
	$(MAKE) -C tests

clean:
	$(MAKE) -C hw clean
	$(MAKE) -C sim clean
	$(MAKE) -C kernel clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean

clean-all:
	$(MAKE) -C third_party clean
	$(MAKE) -C hw clean
	$(MAKE) -C sim clean
	$(MAKE) -C kernel clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean-all

crtlsim:
	$(MAKE) -C sim clean

brtlsim:
	$(MAKE) -C sim
