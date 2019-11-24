# Verilated -*- Makefile -*-
# DESCRIPTION: Verilator output: Make include file with class lists
#
# This file lists generated Verilated files, for including in higher level makefiles.
# See Vcache_simX.mk for the caller.

### Switches...
# Coverage output mode?  0/1 (from --coverage)
VM_COVERAGE = 0
# Tracing output mode?  0/1 (from --trace)
VM_TRACE = 1

### Object file lists...
# Generated module classes, fast-path, compile with highest optimization
VM_CLASSES_FAST += \
	Vcache_simX \
	Vcache_simX_cache_simX \
	Vcache_simX_VX_dmem_controller__V0_VB1000 \
	Vcache_simX_VX_icache_request_inter \
	Vcache_simX_VX_icache_response_inter \
	Vcache_simX_VX_dram_req_rsp_inter__N4_NB4 \
	Vcache_simX_VX_dram_req_rsp_inter__N1_NB4 \
	Vcache_simX_VX_dcache_request_inter \
	Vcache_simX_VX_dcache_response_inter \

# Generated module classes, non-fast-path, compile with low/medium optimization
VM_CLASSES_SLOW += \

# Generated support classes, fast-path, compile with highest optimization
VM_SUPPORT_FAST += \
	Vcache_simX__Trace \

# Generated support classes, non-fast-path, compile with low/medium optimization
VM_SUPPORT_SLOW += \
	Vcache_simX__Syms \
	Vcache_simX__Trace__Slow \

# Global classes, need linked once per executable, fast-path, compile with highest optimization
VM_GLOBAL_FAST += \
	verilated \
	verilated_vcd_c \

# Global classes, need linked once per executable, non-fast-path, compile with low/medium optimization
VM_GLOBAL_SLOW += \


# Verilated -*- Makefile -*-
