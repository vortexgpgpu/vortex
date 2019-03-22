


all: RUNFILE


VERILATOR:
	verilator -Wall -cc Vortex.v VX_fetch.v VX_f_d_reg.v VX_decode.v VX_register_file.v VX_d_e_reg.v VX_execute.v VX_e_m_reg.v VX_memory.v VX_m_w_reg.v VX_writeback.v VX_csr_handler.v VX_forwarding.v --exe test_bench.cpp 

RUNFILE: VERILATOR
	(cd obj_dir && make -j -f Vvortex.mk)
	
