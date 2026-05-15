ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm

run-simx:
	$(MAKE) -C vecadd run-simx
	$(MAKE) -C sgemm run-simx

run-rtlsim:
	$(MAKE) -C vecadd run-rtlsim
	$(MAKE) -C sgemm run-rtlsim

clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean

.PHONY: all run-simx run-rtlsim clean
