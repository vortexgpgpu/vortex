# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, 3 -- default. NOT supported on this backend.
OPT_LEVEL ?= 3
ifneq ($(OPT_LEVEL),3)
$(error OPT_LEVEL=$(OPT_LEVEL) requested but not supported on this backend)
endif

all: RUNFILE

# -LDFLAGS '-lsystemc'
VERILATOR:
	verilator rf2_32x128_wm1_rtl.v -cc --exe testbench.cpp --Wno-UNOPTFLAT --Wno-WIDTH --Wno-STMTDLY --Wno-UNSIGNED -LDFLAGS '-lsystemc'


RUNFILE: VERILATOR
	(cd obj_dir && make -j -f Vrf2_32x128_wm1_rtl.mk)

clean:
	rm obj_dir/*