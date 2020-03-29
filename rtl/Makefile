all: RUNFILE

INCLUDE = -I. -Ishared_memory -Icache -IVX_cache -IVX_cache/interfaces -Iinterfaces/ -Ipipe_regs/ -Icompat/ -Isimulate

SINGLE_CORE = Vortex.v

MULTI_CORE = Vortex_SOC.v

EXE += --exe ./simulate/test_bench.cpp ./simulate/Vortex.cpp

MULTI_EXE += --exe ./simulate/multi_test_bench.cpp ./simulate/Vortex_SOC.cpp

VF += -compiler gcc --language 1800-2009

WNO += -Wno-UNDRIVEN --Wno-PINMISSING -Wno-STMTDLY -Wno-WIDTH -Wno-UNSIGNED -Wno-UNOPTFLAT -Wno-LITENDIAN
# WNO=

# LIGHTW=
LIGHTW += -Wno-UNOPTFLAT

# LIB=-LDFLAGS '-L/usr/local/systemc/'
LIB +=

CF += -std=c++11 -fms-extensions

DEB += --trace -DVL_DEBUG=1 

MAKECPP += (cd obj_dir && make -j -f VVortex.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

MAKECPPRel += (cd obj_dir && make -j -f VVortex.mk)

MAKEMULTICPP += (cd obj_dir && make -j -f VVortex_SOC.mk OPT='-DVL_DEBUG' VL_DEBUG=1 DVL_DEBUG=1)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

.PHONY: build_config
build_config:
	./gen_config.py --rtl_locations

# -LDFLAGS '-lsystemc'
VERILATOR: build_config
	verilator $(VF) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '$(CF)' $(LIGHTW)

VERILATORnoWarnings: build_config
	verilator $(VF) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '$(CF)' $(WNO) $(DEB)

VERILATORnoWarningsRel: build_config
	verilator $(VF) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '$(CF) -O3 -DVL_THREADED' $(WNO) --threads $(THREADS)

VERILATORMULTInoWarnings: build_config
	verilator $(VF) -cc $(MULTI_CORE) $(INCLUDE) $(MULTI_EXE) $(LIB) -CFLAGS '$(CF)' $(WNO) $(DEB)

compdebug: build_config
	verilator_bin_dbg $(VF) -cc $(SINGLE_CORE) $(INCLUDE) $(EXE) $(LIB) -CFLAGS '$(CF) -DVCD_OUTPUT -DVL_DEBUG' $(WNO)  $(DEB)

compdebugmulti: build_config
	verilator_bin_dbg $(VF) -cc $(MULTI_CORE) $(INCLUDE) $(MULTI_EXE) $(LIB) -CFLAGS '$(CF) -DVCD_OUTPUT -DVL_DEBUG' $(WNO)  $(DEB)

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

run-multicore: multicore
	(cd obj_dir && ./VVortex_SOC)

runRel: wRel
	(cd obj_dir && ./VVortex)

clean:
	rm -rf obj_dir