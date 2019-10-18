all: RUNFILE

INCLUDE=-I. -I../models/memory/cln28hpc/rf2_32x128_wm1/ -I/usr/local/systemc/ -Iinterfaces/ -Ipipe_regs/

FILE=Vortex.v

EXE=--exe test_bench.cpp

COMP=--compiler gcc

WNO=-Wno-fatal -Wno-UNOPTFLAT -Wno-UNDRIVEN -Wno-UNSIGNED -Wno-lint --Wno-PINMISSING

LIB=-LDFLAGS '-L/usr/local/systemc/'

CF=-CFLAGS '-std=c++11 -O3'

DEB=--prof-cfuncs -DVL_DEBUG=1 --coverage --trace 


MAKECPP=(cd obj_dir && make -j -f VVortex.mk)

# -LDFLAGS '-lsystemc'
VERILATOR:
	echo "#define VCD_OFF" > tb_debug.h
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF)

VERILATORnoWarnings:
	echo "#define VCD_OFF" > tb_debug.h
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF)  $(WNO)

compdebug:
	echo "#define VCD_OUTPUT" > tb_debug.h
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '-std=c++11 -DVL_DEBUG' $(WNO)  $(DEB)

RUNFILE: VERILATOR
	$(MAKECPP)

debug: compdebug
	$(MAKECPP)

w: VERILATORnoWarnings
	$(MAKECPP)

clean:
	rm obj_dir/*