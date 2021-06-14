all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C saxpy
	$(MAKE) -C sfilter
	$(MAKE) -C nearn
	$(MAKE) -C guassian
	
run:
	$(MAKE) -C vecadd run-vlsim 
	$(MAKE) -C sgemm run-vlsim
	$(MAKE) -C saxpy run-vlsim
	$(MAKE) -C sfilter run-vlsim
	$(MAKE) -C nearn run-vlsim
	$(MAKE) -C guassian run-vlsim

clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C saxpy clean
	$(MAKE) -C sfilter clean
	$(MAKE) -C nearn clean
	$(MAKE) -C guassian clean

clean-all:
	$(MAKE) -C vecadd clean-all
	$(MAKE) -C sgemm clean-all
	$(MAKE) -C saxpy clean-all
	$(MAKE) -C sfilter clean-all
	$(MAKE) -C nearn clean-all
	$(MAKE) -C guassian clean-all