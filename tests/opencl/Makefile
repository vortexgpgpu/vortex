all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C saxpy
	$(MAKE) -C sfilter
	$(MAKE) -C nearn
	$(MAKE) -C guassian
	$(MAKE) -C oclprintf
	$(MAKE) -C psort
	$(MAKE) -C DotProduct

run-simx:
	$(MAKE) -C vecadd run-simx
	$(MAKE) -C sgemm run-simx
	$(MAKE) -C saxpy run-simx
	$(MAKE) -C sfilter run-simx
#   $(MAKE) -C nearn run-simx
	$(MAKE) -C guassian run-simx
#   $(MAKE) -C oclprintf run-simx
	$(MAKE) -C psort run-simx
	$(MAKE) -C DotProduct run-simx

run-rtlsim:
	$(MAKE) -C vecadd run-rtlsim
	$(MAKE) -C sgemm run-rtlsim
	$(MAKE) -C saxpy run-rtlsim
	$(MAKE) -C sfilter run-rtlsim
#   $(MAKE) -C nearn run-rtlsim
	$(MAKE) -C guassian run-rtlsim
#   $(MAKE) -C oclprintf run-rtlsim
	$(MAKE) -C psort run-rtlsim
	$(MAKE) -C DotProduct run-rtlsim

run-opae:
	$(MAKE) -C vecadd run-opae
	$(MAKE) -C sgemm run-opae
	$(MAKE) -C saxpy run-opae
	$(MAKE) -C sfilter run-opae
#   $(MAKE) -C nearn run-opae
	$(MAKE) -C guassian run-opae
#   $(MAKE) -C oclprintf run-opae
	$(MAKE) -C psort run-opae
	$(MAKE) -C DotProduct run-opae

clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C saxpy clean
	$(MAKE) -C sfilter clean
	$(MAKE) -C nearn clean
	$(MAKE) -C guassian clean
	$(MAKE) -C oclprintf clean
	$(MAKE) -C psort clean
	$(MAKE) -C DotProduct clean

clean-all:
	$(MAKE) -C vecadd clean-all
	$(MAKE) -C sgemm clean-all
	$(MAKE) -C saxpy clean-all
	$(MAKE) -C sfilter clean-all
	$(MAKE) -C nearn clean-all
	$(MAKE) -C guassian clean-all
	$(MAKE) -C oclprintf clean-all
	$(MAKE) -C psort clean-all
	$(MAKE) -C DotProduct clean-all
