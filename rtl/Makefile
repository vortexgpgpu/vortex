all: RUNFILE


VERILATOR:
	verilator --compiler gcc -Wall -cc Vortex.v -I. -Iinterfaces/ -Ipipe_regs/ --exe test_bench.cpp -CFLAGS -std=c++11 -O3

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVortex.mk)

clean:
	rm ./obj_dir/*