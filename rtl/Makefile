all: RUNFILE


VERILATOR:
	echo "#define VCD_OFF" > tb_debug.h
	verilator --compiler gcc -Wall -cc Vortex.v -I. -Iinterfaces/ -Ipipe_regs/ --exe test_bench.cpp -CFLAGS -std=c++11 -O3

compdebug:
	echo "#define VCD_OUTPUT" > tb_debug.h
	verilator --compiler gcc --prof-cfuncs -DVL_DEBUG=1 --coverage -Wall --trace -cc Vortex.v -I. -Iinterfaces/ -Ipipe_regs/ --exe test_bench.cpp -CFLAGS '-std=c++11 -DVL_DEBUG'

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVortex.mk)

debug: compdebug
	(cd obj_dir && make -j -f VVortex.mk)

clean:
	rm obj_dir/*