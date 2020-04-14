all: RUNFILE


VERILATOR:
	verilator --compiler gcc --Wno-UNOPTFLAT -Wall --trace -cc VX_d_cache_encapsulate.v -Iinterfaces/ --exe d_cache_test_bench.cpp -CFLAGS -std=c++11

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVX_d_cache_encapsulate.mk)

clean:
	rm ./obj_dir/*
	
