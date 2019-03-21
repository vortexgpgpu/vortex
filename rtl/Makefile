


all: RUNFILE


VERILATOR:
	verilator -Wall -cc vortex.v VX_f_d_reg.v VX_fetch.v --exe test_bench.cpp 

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f Vvortex.mk)
	
