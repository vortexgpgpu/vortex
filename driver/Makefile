all: stub rtlsim simx opae tests

stub:
	$(MAKE) -C stub

opae:
	$(MAKE) -C opae

rtlsim:
	$(MAKE) -C rtlsim

simx:
	$(MAKE) -C simx

tests:
	$(MAKE) -C tests

clean:
	$(MAKE) clean -C stub
	$(MAKE) clean -C opae
	$(MAKE) clean -C rtlsim
	$(MAKE) clean -C simx
	$(MAKE) clean -C tests

.PHONY: all stub opae rtlsim simx tests clean