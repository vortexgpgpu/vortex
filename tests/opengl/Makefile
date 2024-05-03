all:
	$(MAKE) -C lib
	$(MAKE) -C triangle
	$(MAKE) -C cube
	$(MAKE) -C texture

run-simx:
	$(MAKE) -C lib run-simx
	$(MAKE) -C triangle run-simx
	$(MAKE) -C cube run-simx
	$(MAKE) -C texture run-simx

run-rtlsim:
	$(MAKE) -C lib run-rtlsim
	$(MAKE) -C triangle run-rtlsim
	$(MAKE) -C cube run-rtlsim
	$(MAKE) -C texture run-rtlsim

run-opae:
	$(MAKE) -C lib run-opae
	$(MAKE) -C triangle run-opae
	$(MAKE) -C cube run-opae
	$(MAKE) -C texture run-opae

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C triangle clean
	$(MAKE) -C cube clean
	$(MAKE) -C texture clean

clean-all:
	$(MAKE) -C lib clean-all
	$(MAKE) -C triangle clean-all
	$(MAKE) -C cube clean-all
	$(MAKE) -C texture clean-all
