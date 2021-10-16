all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C saxpy
	$(MAKE) -C sfilter
	$(MAKE) -C nearn
	$(MAKE) -C guassian
	$(MAKE) -C oclprintf
	$(MAKE) -C psort
	
run-simx:
	$(MAKE) -C vecadd run-simx 
	$(MAKE) -C sgemm run-simx
	$(MAKE) -C saxpy run-simx
	$(MAKE) -C sfilter run-simx
	$(MAKE) -C nearn run-simx
	$(MAKE) -C guassian run-simx
	$(MAKE) -C oclprintf run-simx
	$(MAKE) -C psort run-simx

run-rtlsim:
	$(MAKE) -C vecadd run-rtlsim 
	$(MAKE) -C sgemm run-rtlsim
	$(MAKE) -C saxpy run-rtlsim
	$(MAKE) -C sfilter run-rtlsim
	$(MAKE) -C nearn run-rtlsim
	$(MAKE) -C guassian run-rtlsim
	$(MAKE) -C oclprintf run-rtlsim
	$(MAKE) -C psort run-rtlsim

run-vlsim:
	$(MAKE) -C vecadd run-vlsim 
	$(MAKE) -C sgemm run-vlsim
	$(MAKE) -C saxpy run-vlsim
	$(MAKE) -C sfilter run-vlsim
	$(MAKE) -C nearn run-vlsim
	$(MAKE) -C guassian run-vlsim
	$(MAKE) -C oclprintf run-vlsim
	$(MAKE) -C psort run-vlsim

clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C saxpy clean
	$(MAKE) -C sfilter clean
	$(MAKE) -C nearn clean
	$(MAKE) -C guassian clean
	$(MAKE) -C oclprintf clean
	$(MAKE) -C psort clean

clean-all:
	$(MAKE) -C vecadd clean-all
	$(MAKE) -C sgemm clean-all
	$(MAKE) -C saxpy clean-all
	$(MAKE) -C sfilter clean-all
	$(MAKE) -C nearn clean-all
	$(MAKE) -C guassian clean-all
	$(MAKE) -C oclprintf clean-all
	$(MAKE) -C psort clean-all