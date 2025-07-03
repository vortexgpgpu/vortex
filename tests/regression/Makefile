ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C dropout
	$(MAKE) -C dotproduct
	$(MAKE) -C mstress
	$(MAKE) -C io_addr
	$(MAKE) -C printf
	$(MAKE) -C diverge
	$(MAKE) -C sort
	$(MAKE) -C fence
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C conv3
	$(MAKE) -C relu
	$(MAKE) -C sgemv
	$(MAKE) -C sgemm2
	$(MAKE) -C madmax
	$(MAKE) -C stencil3d

run-simx:
	$(MAKE) -C basic run-simx
	$(MAKE) -C demo run-simx
	$(MAKE) -C dogfood run-simx
	$(MAKE) -C dropout run-simx
	$(MAKE) -C dotproduct run-simx
	$(MAKE) -C mstress run-simx
	$(MAKE) -C io_addr run-simx
	$(MAKE) -C printf run-simx
	$(MAKE) -C diverge run-simx
	$(MAKE) -C sort run-simx
	$(MAKE) -C fence run-simx
	$(MAKE) -C vecadd run-simx
	$(MAKE) -C sgemm run-simx
	$(MAKE) -C conv3 run-simx
	$(MAKE) -C relu run-simx
	$(MAKE) -C sgemv run-simx
	$(MAKE) -C sgemm2 run-simx
	$(MAKE) -C madmax run-simx
	$(MAKE) -C stencil3d run-simx

run-rtlsim:
	$(MAKE) -C basic run-rtlsim
	$(MAKE) -C demo run-rtlsim
	$(MAKE) -C dogfood run-rtlsim
	$(MAKE) -C dropout run-rtlsim
	$(MAKE) -C dotproduct run-rtlsim
	$(MAKE) -C mstress run-rtlsim
	$(MAKE) -C io_addr run-rtlsim
	$(MAKE) -C printf run-rtlsim
	$(MAKE) -C diverge run-rtlsim
	$(MAKE) -C sort run-rtlsim
	$(MAKE) -C fence run-rtlsim
	$(MAKE) -C vecadd run-rtlsim
	$(MAKE) -C sgemm run-rtlsim
	$(MAKE) -C conv3 run-rtlsim
	$(MAKE) -C relu run-rtlsim
	$(MAKE) -C sgemv run-rtlsim
	$(MAKE) -C sgemm2 run-rtlsim
	$(MAKE) -C madmax run-rtlsim
	$(MAKE) -C stencil3d run-rtlsim

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C dropout clean
	$(MAKE) -C dotproduct clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean
	$(MAKE) -C printf clean
	$(MAKE) -C diverge clean
	$(MAKE) -C sort clean
	$(MAKE) -C fence clean
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C conv3 clean
	$(MAKE) -C relu clean
	$(MAKE) -C sgemv clean
	$(MAKE) -C sgemm2 clean
	$(MAKE) -C madmax clean
	$(MAKE) -C stencil3d clean
