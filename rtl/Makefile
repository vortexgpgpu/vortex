all: RUNFILE


INCLUDE=-I. -Ishared_memory -Icache -IVX_cache -IVX_cache/interfaces -Iinterfaces/ -Ipipe_regs/ -Icompat/ -Isimulate

SINGLE_CORE=Vortex.v
MULTI_CORE=Vortex_SOC.v

EXE=--exe ./simulate/test_bench.cpp
MULTI_EXE=--exe ./simulate/multi_test_bench.cpp

COMP=--compiler gcc --language 1800-2009

WNO=-Wno-UNDRIVEN --Wno-PINMISSING -Wno-STMTDLY -Wno-WIDTH -Wno-UNSIGNED -Wno-UNOPTFLAT -Wno-LITENDIAN
# WNO=

# LIGHTW=
LIGHTW=-Wno-UNOPTFLAT
# LIB=-LDFLAGS '-L/usr/local/systemc/'
LIB=

CF=-CFLAGS '-std=c++11 -fms-extensions'

CFRel=-CFLAGS '-std=c++11 -fms-extensions -O3 -DVL_THREADED'

DEB=--trace --prof-cfuncs -DVL_DEBUG=1 


MAKECPP=(cd obj_dir && make -j -f VVortex.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

MAKECPPRel=(cd obj_dir && make -j -f VVortex.mk)

MAKEMULTICPP=(cd obj_dir && make -j -f VVortex_SOC.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))' )

# -LDFLAGS '-lsystemc'
VERILATOR:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) $(CF) $(LIGHTW)

VERILATORnoWarnings:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) $(CF)  $(WNO) $(DEB)

VERILATORnoWarningsRel:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) $(CFRel)  $(WNO) --threads $(THREADS)


VERILATORMULTInoWarnings:
	echo "#define VCD_OFF" > simulate/tb_debug.h
	verilator $(COMP) -cc $(MULTI_CORE) $(INCLUDE) $(MULTI_EXE) $(LIB) $(CF)  $(WNO) $(DEB)

compdebug:
	echo "#define VCD_OUTPUT" > simulate/tb_debug.h
	verilator_bin_dbg $(COMP) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '-std=c++11 -DVL_DEBUG' $(WNO)  $(DEB)

compdebugmulti:
	echo "#define VCD_OUTPUT" > simulate/tb_debug.h
	verilator_bin_dbg $(COMP) -cc $(MULTI_CORE) $(INCLUDE) $(MULTI_EXE) $(LIB) -CFLAGS '-std=c++11 -DVL_DEBUG' $(WNO)  $(DEB)

RUNFILE: VERILATOR
	$(MAKECPP)

debug: compdebug
	$(MAKECPP)

w: VERILATORnoWarnings
	$(MAKECPP)

wRel: VERILATORnoWarningsRel
	$(MAKECPPRel)

multicore: VERILATORMULTInoWarnings
	$(MAKEMULTICPP)

dmulticore: compdebugmulti
	$(MAKEMULTICPP)

run: w
	(cd obj_dir && ./VVortex)

runRel: wRel
	(cd obj_dir && ./VVortex)

clean:
	rm obj_dir/*
