all:
	$(MAKE) -C basic
	$(MAKE) -C demo
	$(MAKE) -C dogfood
	$(MAKE) -C sha256

run:
	$(MAKE) -C basic run-vlsim
	$(MAKE) -C demo run-vlsim
	$(MAKE) -C dogfood run-vlsim
	$(MAKE) -C sha256 run-vlsim

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C demo clean
	$(MAKE) -C dogfood clean
	$(MAKE) -C sha256 clean

clean-all:
	$(MAKE) -C basic clean-all
	$(MAKE) -C demo clean-all
	$(MAKE) -C dogfood clean-all
	$(MAKE) -C sha256 clean-all

