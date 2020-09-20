all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C saxpy
	$(MAKE) -C sfilter
	$(MAKE) -C nearn
	$(MAKE) -C guassian
	$(MAKE) -C bfs
run:
	$(MAKE) -C vecadd run-rtlsim 
	$(MAKE) -C sgemm run-rtlsim
	$(MAKE) -C saxpy run-rtlsim
	$(MAKE) -C sfilter run-rtlsim
	$(MAKE) -C nearn run-rtlsim
	$(MAKE) -C guassian run-rtlsim
	$(MAKE) -C bfs run-rtlsim
clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C saxpy clean
	$(MAKE) -C sfilter clean
	$(MAKE) -C nearn clean
	$(MAKE) -C guassian clean
	$(MAKE) -C bfs clean