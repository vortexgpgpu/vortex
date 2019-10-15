all: RUNFILE

# -LDFLAGS '-lsystemc'
VERILATOR:
	echo "#define VCD_OFF" > tb_debug.h
	verilator --compiler gcc -Wno-fatal -Wno-UNOPTFLAT -Wno-UNDRIVEN -Wno-UNSIGNED -Wno-lint -cc Vortex.v -I. -I../models/memory/cln28hpc/rf2_32x128_wm1/ -I/usr/local/systemc/ -Iinterfaces/ -Ipipe_regs/ --exe test_bench.cpp -CFLAGS '-std=c++11 -O3' -LDFLAGS '-L/usr/local/systemc/'

compdebug:
	echo "#define VCD_OUTPUT" > tb_debug.h
	verilator --compiler gcc -Wno-fatal -Wno-UNOPTFLAT -Wno-UNDRIVEN -Wno-UNSIGNED -Wno-lint --prof-cfuncs -DVL_DEBUG=1 --coverage --trace -cc Vortex.v -I/usr/local/systemc/ -I. -I../models/memory/cln28hpc/rf2_32x128_wm1/ -Iinterfaces/ -Ipipe_regs/ --exe test_bench.cpp -CFLAGS '-std=c++11 -DVL_DEBUG' -LDFLAGS '-L/usr/local/systemc/'

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f VVortex.mk)

debug: compdebug
	(cd obj_dir && make -j -f VVortex.mk)

clean:
	rm obj_dir/*