all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C mstress
	$(MAKE) -C io_addr
	$(MAKE) -C printf
	$(MAKE) -C diverge
	$(MAKE) -C fence

run-simx:
	$(MAKE) -C basic run-simx
	$(MAKE) -C demo run-simx
	$(MAKE) -C dogfood run-simx
	$(MAKE) -C mstress run-simx
	$(MAKE) -C io_addr run-simx
	$(MAKE) -C printf run-simx
	$(MAKE) -C diverge run-simx
	$(MAKE) -C fence run-simx

run-vlsim:
	$(MAKE) -C basic run-vlsim
	$(MAKE) -C demo run-vlsim
	$(MAKE) -C dogfood run-vlsim
	$(MAKE) -C mstress run-vlsim
	$(MAKE) -C io_addr run-vlsim
	$(MAKE) -C printf run-vlsim
	$(MAKE) -C diverge run-vlsim
	$(MAKE) -C fence run-vlsim

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean
	$(MAKE) -C printf clean
	$(MAKE) -C diverge clean
	$(MAKE) -C fence clean

clean-all:
	$(MAKE) -C basic clean-all
	$(MAKE) -C demo clean-all
	$(MAKE) -C dogfood clean-all
	$(MAKE) -C mstress clean-all
	$(MAKE) -C io_addr clean-all
	$(MAKE) -C printf clean-all
	$(MAKE) -C diverge clean-all
	$(MAKE) -C fence clean-all

