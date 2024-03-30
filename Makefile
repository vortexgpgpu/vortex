include config.mk

all: 
	$(MAKE) -C $(VORTEX_HOME)/third_party
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
	$(MAKE) -C hw clean
	$(MAKE) -C sim clean
	$(MAKE) -C kernel clean
	$(MAKE) -C runtime clean
	$(MAKE) -C tests clean-all
