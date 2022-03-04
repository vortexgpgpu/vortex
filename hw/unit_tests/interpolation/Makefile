# TODO: make this a proper makefile
default: verilate build

verilate:
	verilator -cc --trace ../../rtl/VX_interpolation.sv -I../../rtl -I../../../.. -I../../../../dpi -I../../../common -I../../rtl/fp_cores -v ../../rtl/libs/VX_multiplier.sv ../../rtl/libs/VX_shift_register.sv ../../rtl/libs/VX_pipe_register.sv --exe imadd_tb.cpp

build:
	make -C obj_dir -f VVX_interpolation.mk

clean:
	rm -rf obj_dir
	rm -f wave.vcd