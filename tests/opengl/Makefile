all:
	$(MAKE) -C lib
	$(MAKE) -C triangle
	$(MAKE) -C subadd

run-simx:
	$(MAKE) -C lib run-simx
	$(MAKE) -C triangle run-simx
	$(MAKE) -C subadd run-simx

run-rtlsim:
	$(MAKE) -C lib run-rtlsim
	$(MAKE) -C triangle run-rtlsim
	$(MAKE) -C subadd run-rtlsim

run-opae:
	$(MAKE) -C lib run-opae
	$(MAKE) -C triangle run-opae
	$(MAKE) -C subadd run-opae

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C triangle clean
	$(MAKE) -C subadd clean

clean-all:
	$(MAKE) -C lib clean-all
	$(MAKE) -C triangle clean-all
	$(MAKE) -C subadd clean-all
