ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

simx:
	$(MAKE) -C simx

all:
	$(MAKE) -C simx
	$(MAKE) -C rtlsim
	$(MAKE) -C opaesim
	$(MAKE) -C xrtsim

clean:
	$(MAKE) -C simx clean
	$(MAKE) -C rtlsim clean
	$(MAKE) -C opaesim clean
	$(MAKE) -C xrtsim clean