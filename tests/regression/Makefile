all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C mstress
	$(MAKE) -C io_addr
	$(MAKE) -C printf
	$(MAKE) -C diverge

run:
	$(MAKE) -C basic run-vlsim
	$(MAKE) -C demo run-vlsim
	$(MAKE) -C dogfood run-vlsim
	$(MAKE) -C mstress run-vlsim
	$(MAKE) -C io_addr run-vlsim
	$(MAKE) -C printf run-vlsim
	$(MAKE) -C diverge run-vlsim

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean
	$(MAKE) -C printf clean
	$(MAKE) -C diverge clean

clean-all:
	$(MAKE) -C basic clean-all
	$(MAKE) -C demo clean-all
	$(MAKE) -C dogfood clean-all
	$(MAKE) -C mstress clean-all
	$(MAKE) -C io_addr clean-all
	$(MAKE) -C printf clean-all
	$(MAKE) -C diverge clean-all

