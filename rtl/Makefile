


all: RUNFILE


VERILATOR:
	verilator -Wall -cc Vortex.v --exe test_bench.cpp 

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVortex.mk)

clean:
	rm ./obj_dir/*
	
