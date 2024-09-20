ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C vecadd
	$(MAKE) -C sgemm
	$(MAKE) -C conv3
	$(MAKE) -C psort
	$(MAKE) -C saxpy
	$(MAKE) -C sfilter
	$(MAKE) -C sgemm2
	$(MAKE) -C sgemm3
	$(MAKE) -C psum
	$(MAKE) -C oclprintf
	$(MAKE) -C dotproduct
	$(MAKE) -C transpose
	$(MAKE) -C spmv
	$(MAKE) -C stencil
	$(MAKE) -C lbm
	$(MAKE) -C nearn
	$(MAKE) -C guassian
	$(MAKE) -C kmeans
	$(MAKE) -C blackscholes
	$(MAKE) -C bfs

run-simx:
	$(MAKE) -C vecadd run-simx
	$(MAKE) -C sgemm run-simx
	$(MAKE) -C conv3 run-simx
	$(MAKE) -C psort run-simx
	$(MAKE) -C saxpy run-simx
	$(MAKE) -C sfilter run-simx
	$(MAKE) -C sgemm2 run-simx
	$(MAKE) -C sgemm3 run-simx
	$(MAKE) -C psum run-simx
	$(MAKE) -C oclprintf run-simx
	$(MAKE) -C dotproduct run-simx
	$(MAKE) -C transpose run-simx
	$(MAKE) -C spmv run-simx
	$(MAKE) -C stencil run-simx
	$(MAKE) -C nearn run-simx
	$(MAKE) -C guassian run-simx
	$(MAKE) -C kmeans run-simx
	$(MAKE) -C blackscholes run-simx
	$(MAKE) -C bfs run-simx

run-rtlsim:
	$(MAKE) -C vecadd run-rtlsim
	$(MAKE) -C sgemm run-rtlsim
	$(MAKE) -C conv3 run-rtlsim
	$(MAKE) -C psort run-rtlsim
	$(MAKE) -C saxpy run-rtlsim
	$(MAKE) -C sfilter run-rtlsim
	$(MAKE) -C sgemm2 run-rtlsim
	$(MAKE) -C sgemm3 run-rtlsim
	$(MAKE) -C psum run-rtlsim
	$(MAKE) -C oclprintf run-rtlsim
	$(MAKE) -C dotproduct run-rtlsim
	$(MAKE) -C transpose run-rtlsim
	$(MAKE) -C spmv run-rtlsim
	$(MAKE) -C stencil run-rtlsim
	$(MAKE) -C nearn run-rtlsim
	$(MAKE) -C guassian run-rtlsim
	$(MAKE) -C kmeans run-rtlsim
	$(MAKE) -C blackscholes run-rtlsim
	$(MAKE) -C bfs run-rtlsim

clean:
	$(MAKE) -C vecadd clean
	$(MAKE) -C sgemm clean
	$(MAKE) -C conv3 clean
	$(MAKE) -C psort clean
	$(MAKE) -C saxpy clean
	$(MAKE) -C sfilter clean
	$(MAKE) -C sgemm2 clean
	$(MAKE) -C sgemm3 clean
	$(MAKE) -C psum clean
	$(MAKE) -C oclprintf clean
	$(MAKE) -C dotproduct clean
	$(MAKE) -C transpose clean
	$(MAKE) -C spmv clean
	$(MAKE) -C stencil clean
	$(MAKE) -C lbm clean
	$(MAKE) -C nearn clean
	$(MAKE) -C guassian clean
	$(MAKE) -C kmeans clean
	$(MAKE) -C blackscholes clean
	$(MAKE) -C bfs clean