all: RUNFILE

# -LDFLAGS '-lsystemc'
VERILATOR:
	verilator rf2_32x128_wm1_rtl.v -cc --exe testbench.cpp --Wno-UNOPTFLAT --Wno-WIDTH --Wno-STMTDLY --Wno-UNSIGNED -LDFLAGS '-lsystemc'


RUNFILE: VERILATOR
	(cd obj_dir && make -j -f Vrf2_32x128_wm1_rtl.mk)

clean:
	rm obj_dir/*