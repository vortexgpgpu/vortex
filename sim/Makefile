ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: simx rtlsim opaesim xrtsim

simx:
	$(MAKE) -C simx

rtlsim:
	$(MAKE) -C rtlsim

opaesim:
	$(MAKE) -C opaesim

xrtsim:
	$(MAKE) -C xrtsim

clean:
	$(MAKE) -C simx clean
	$(MAKE) -C rtlsim clean
	$(MAKE) -C opaesim clean
	$(MAKE) -C xrtsim clean

.PHONY: all simx rtlsim opaesim xrtsim clean