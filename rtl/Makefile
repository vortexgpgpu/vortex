all: RUNFILE

# /rf2_256x128_wm1/
BaseMEM=../models/memory/cln28hpm

INCLUDE=-I. -Ishared_memory -Icache -I$(BaseMEM)/rf2_128x128_wm1/ -I$(BaseMEM)/rf2_256x128_wm1/ -I$(BaseMEM)/rf2_256x19_wm0/ -I$(BaseMEM)/rf2_32x128_wm1/ -Iinterfaces/ -Ipipe_regs/ -Icompat/ -Isimulate

FILE=Vortex.v

EXE=--exe ./simulate/test_bench.cpp

COMP=--compiler gcc

WNO=-Wno-UNDRIVEN --Wno-PINMISSING -Wno-STMTDLY -Wno-WIDTH -Wno-UNSIGNED -Wno-UNOPTFLAT
# WNO=

# LIGHTW=
LIGHTW=-Wno-UNOPTFLAT
# LIB=-LDFLAGS '-L/usr/local/systemc/'
LIB=

CF=-CFLAGS '-std=c++11 -O3'

DEB=--trace --prof-cfuncs -DVL_DEBUG=1 


MAKECPP=(cd obj_dir && make -j -f VVortex.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

# -LDFLAGS '-lsystemc'
VERILATOR:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF) $(LIGHTW)

VERILATORnoWarnings:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) $(CF)  $(WNO) $(DEB)

compdebug:
	echo "#define VCD_OUTPUT" > simulate/tb_debug.h
	verilator_bin_dbg $(COMP) -cc $(FILE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '-std=c++11 -DVL_DEBUG' $(WNO)  $(DEB)

RUNFILE: VERILATOR
	$(MAKECPP)

debug: compdebug
	$(MAKECPP)

w: VERILATORnoWarnings
	$(MAKECPP)

clean:
	rm obj_dir/*
