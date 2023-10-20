all: stub rtlsim simx opae

stub:
	$(MAKE) -C stub

simx:
	$(MAKE) -C simx

rtlsim:
	$(MAKE) -C rtlsim

opae:
	$(MAKE) -C opae

xrt:
	$(MAKE) -C xrt

clean:
	$(MAKE) clean -C stub
	$(MAKE) clean -C simx
	$(MAKE) clean -C rtlsim
	$(MAKE) clean -C opae
	$(MAKE) clean -C xrt

.PHONY: all stub simx rtlsim opae xrt clean