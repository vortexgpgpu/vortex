all:
	$(MAKE) -C dogfood
	$(MAKE) -C mstress
	$(MAKE) -C io_addr

run:
	$(MAKE) -C dogfood run-vlsim
	$(MAKE) -C mstress run-vlsim
	$(MAKE) -C io_addr run-vlsim

clean:
	$(MAKE) -C dogfood clean
	$(MAKE) -C mstress clean
	$(MAKE) -C io_addr clean

clean-all:
	$(MAKE) -C dogfood clean-all
	$(MAKE) -C mstress clean-all
	$(MAKE) -C io_addr clean-all

