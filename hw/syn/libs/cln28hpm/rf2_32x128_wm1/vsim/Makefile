

ALL:sim

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, 3 -- default. NOT supported on this backend.
OPT_LEVEL ?= 3
ifneq ($(OPT_LEVEL),3)
$(error OPT_LEVEL=$(OPT_LEVEL) requested but not supported on this backend)
endif

#TOOL INPUT
SRC = \
rf2_32x128_wm1_tb.v \
../rf2_32x128_wm1.v 



CMD= \
-do "vcd file rf2_32x128_wm1_tb.vcd; \
	 vcd add -r /rf2_32x128_wm1_tb/*; \
	 run -all; \
	 quit -f"


OPT=-sv -sv12compat

LIB = rf_lib

# LOG=-logfile rf2_32x128_wm1_tb.log
LOG=

comp:
	vlog $(OPT) -work $(LIB) $(SRC)


sim: comp
	vsim rf2_32x128_wm1_tb $(LOG) -c -lib $(LIB) $(CMD)





