all: stub rtlsim simx opae

stub:
	$(MAKE) -C stub

opae:
	$(MAKE) -C opae

rtlsim:
	$(MAKE) -C rtlsim

simx:
	$(MAKE) -C simx

clean:
	$(MAKE) clean -C stub
	$(MAKE) clean -C opae
	$(MAKE) clean -C rtlsim
	$(MAKE) clean -C simx

.PHONY: all stub opae rtlsim simx clean