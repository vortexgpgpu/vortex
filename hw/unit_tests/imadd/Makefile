# TODO: make this a proper makefile
default: verilate build

verilate:
	verilator -cc --trace ../../rtl/VX_imadd.sv -I../../rtl -I../../../.. -I../../../../dpi -I../../../common -I../../rtl/fpu -v ../../rtl/libs/VX_multiplier.sv ../../rtl/libs/VX_shift_register.sv ../../rtl/libs/VX_pipe_register.sv --exe imadd_tb.cpp

build:
	make -C obj_dir -f VVX_imadd.mk

clean:
	rm -rf obj_dir
	rm -f wave.vcd