ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C conform
	$(MAKE) -C hello	
	$(MAKE) -C fibonacci	

run-simx:
	$(MAKE) -C conform run-simx
	$(MAKE) -C hello run-simx
	$(MAKE) -C fibonacci run-simx	

run-rtlsim:
	$(MAKE) -C conform run-rtlsim
	$(MAKE) -C hello run-rtlsim
	$(MAKE) -C fibonacci run-rtlsim
	
clean:
	$(MAKE) -C conform clean
	$(MAKE) -C hello clean
	$(MAKE) -C fibonacci clean
