ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C mstress
	$(MAKE) -C io_addr
	$(MAKE) -C printf
	$(MAKE) -C diverge
	$(MAKE) -C sort
	$(MAKE) -C fence
	$(MAKE) -C vecaddx
	$(MAKE) -C sgemmx
	$(MAKE) -C conv3x
	$(MAKE) -C sgemm2x
	$(MAKE) -C vote
	$(MAKE) -C shfl
	$(MAKE) -C coop
	$(MAKE) -C cg_mse_cm_hw
	$(MAKE) -C cg_mse_tm_hw

run-simx:
	$(MAKE) -C basic run-simx
	$(MAKE) -C demo run-simx
	$(MAKE) -C dogfood run-simx
	$(MAKE) -C mstress run-simx
	$(MAKE) -C io_addr run-simx
	$(MAKE) -C printf run-simx
	$(MAKE) -C diverge run-simx
	$(MAKE) -C sort run-simx
	$(MAKE) -C fence run-simx
	$(MAKE) -C vecaddx run-simx
	$(MAKE) -C sgemmx run-simx
	$(MAKE) -C conv3x run-simx
	$(MAKE) -C sgemm2x run-simx
	$(MAKE) -C vote run-simx
	$(MAKE) -C shfl run-simx
	$(MAKE) -C coop run-simx
	$(MAKE) -C cg_mse_cm_hw run-simx
	$(MAKE) -C cg_mse_tm_hw run-simx

run-rtlsim:
	$(MAKE) -C basic run-rtlsim
	$(MAKE) -C demo run-rtlsim
	$(MAKE) -C dogfood run-rtlsim
	$(MAKE) -C mstress run-rtlsim
	$(MAKE) -C io_addr run-rtlsim
	$(MAKE) -C printf run-rtlsim
	$(MAKE) -C diverge run-rtlsim
	$(MAKE) -C sort run-rtlsim
	$(MAKE) -C fence run-rtlsim
	$(MAKE) -C vecaddx run-rtlsim
	$(MAKE) -C sgemmx run-rtlsim
	$(MAKE) -C conv3x run-rtlsim
	$(MAKE) -C sgemm2x run-rtlsim
	$(MAKE) -C vote run-rtlsim
	$(MAKE) -C shfl run-rtlsim
	$(MAKE) -C coop run-rtlsim
	$(MAKE) -C cg_mse_cm_hw run-rtlsim
	$(MAKE) -C cg_mse_tm_hw run-rtlsim

run-opae:
	$(MAKE) -C basic run-opae
	$(MAKE) -C demo run-opae
	$(MAKE) -C dogfood run-opae
	$(MAKE) -C mstress run-opae
	$(MAKE) -C io_addr run-opae
	$(MAKE) -C printf run-opae
	$(MAKE) -C diverge run-opae
	$(MAKE) -C sort run-opae
	$(MAKE) -C fence run-opae
	$(MAKE) -C vecaddx run-opae
	$(MAKE) -C sgemmx run-opae
	$(MAKE) -C conv3x run-opae
	$(MAKE) -C sgemm2x run-opae
	$(MAKE) -C vote run-opae
	$(MAKE) -C shfl run-opae
	$(MAKE) -C coop run-opae
	$(MAKE) -C cg_mse_cm_hw run-opae
	$(MAKE) -C cg_mse_tm_hw run-opae

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean
	$(MAKE) -C printf clean
	$(MAKE) -C diverge clean
	$(MAKE) -C sort clean
	$(MAKE) -C fence clean
	$(MAKE) -C vecaddx clean
	$(MAKE) -C sgemmx clean
	$(MAKE) -C conv3x clean
	$(MAKE) -C sgemm2x clean
	$(MAKE) -C vote clean
	$(MAKE) -C shfl clean
	$(MAKE) -C coop clean
	$(MAKE) -C cg_mse_cm_hw clean
	$(MAKE) -C cg_mse_tm_hw clean