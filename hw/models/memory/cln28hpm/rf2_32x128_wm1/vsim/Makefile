

ALL:sim

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





