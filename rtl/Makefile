


all: RUNFILE


VERILATOR:
	verilator -Wall -cc Vortex.v --exe test_bench.cpp -CFLAGS -std=c++11

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVortex.mk)

clean:
	rm ./obj_dir/*
	
