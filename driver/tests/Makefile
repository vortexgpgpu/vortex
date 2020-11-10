all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood

run:
	$(MAKE) -C basic run-rtlsim
	$(MAKE) -C demo run-rtlsim
	$(MAKE) -C dogfood run-rtlsim

	$(MAKE) -C basic run-vlsim
	$(MAKE) -C demo run-vlsim
	$(MAKE) -C dogfood run-vlsim

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean

