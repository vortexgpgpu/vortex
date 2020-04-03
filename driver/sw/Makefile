

all: dummy

dummy:
	$(MAKE) -C dummy

opae:
	$(MAKE) -C opae

rtlsim:
	$(MAKE) -C rtlsim

simx:
	$(MAKE) -C simx

clean:
	$(MAKE) clean -C dummy
	$(MAKE) clean -C opae
	$(MAKE) clean -C rtlsim
	$(MAKE) clean -C simx

.PHONY: all opae rtlsim simx clean