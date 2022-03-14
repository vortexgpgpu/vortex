# Verilated -*- Makefile -*-
# DESCRIPTION: Verilator output: Makefile for building Verilated archive or executable
#
# Execute this makefile from the object directory:
#    make -f VVX_raster_req_switch.mk

default: ../VX_raster_req_switch

### Constants...
# Perl executable (from $PERL)
PERL = perl
# Path to Verilator kit (from $VERILATOR_ROOT)
VERILATOR_ROOT = /opt/verilator
# SystemC include directory with systemc.h (from $SYSTEMC_INCLUDE)
SYSTEMC_INCLUDE ?= 
# SystemC library directory with libsystemc.a (from $SYSTEMC_LIBDIR)
SYSTEMC_LIBDIR ?= 

### Switches...
# SystemC output mode?  0/1 (from --sc)
VM_SC = 0
# Legacy or SystemC output mode?  0/1 (from --sc)
VM_SP_OR_SC = $(VM_SC)
# Deprecated
VM_PCLI = 1
# Deprecated: SystemC architecture to find link library path (from $SYSTEMC_ARCH)
VM_SC_TARGET_ARCH = linux

### Vars...
# Design prefix (from --prefix)
VM_PREFIX = VVX_raster_req_switch
# Module prefix (from --prefix)
VM_MODPREFIX = VVX_raster_req_switch
# User CFLAGS (from -CFLAGS on Verilator command line)
VM_USER_CFLAGS = \
	-std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -Wno-maybe-uninitialized -I../../../dpi/.. -I../../../dpi -I../../../dpi/../common -I/nethome/vsaxena36/vortex-dev/hw -g -O0 -DVCD_OUTPUT -DDBG_TRACE_CORE_PIPELINE   -DDBG_TRACE_CORE_ICACHE -DDBG_TRACE_CORE_DCACHE -DDBG_TRACE_CORE_MEM -DDBG_TRACE_CACHE_BANK  -DDBG_TRACE_CACHE_MSHR -DDBG_TRACE_CACHE_TAG -DDBG_TRACE_CACHE_DATA -DDBG_TRACE_AFU -DDBG_TRACE_SCOPE -DDBG_TRACE_TEX -DDBG_TRACE_RASTER -DDBG_TRACE_ROP \

# User LDLIBS (from -LDFLAGS on Verilator command line)
VM_USER_LDLIBS = \

# User .cpp files (from .cpp's on Verilator command line)
VM_USER_CLASSES = \
	util_dpi \
	testbench \

# User .cpp directories (from .cpp's on Verilator command line)
VM_USER_DIR = \
	. \
	../../../dpi \


### Default rules...
# Include list of all generated classes
include VVX_raster_req_switch_classes.mk
# Include global rules
include $(VERILATOR_ROOT)/include/verilated.mk

### Executable rules... (from --exe)
VPATH += $(VM_USER_DIR)

util_dpi.o: ../../../dpi/util_dpi.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<
testbench.o: testbench.cpp
	$(OBJCACHE) $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(OPT_FAST) -c -o $@ $<

### Link rules... (from --exe)
../VX_raster_req_switch: $(VK_USER_OBJS) $(VK_GLOBAL_OBJS) $(VM_PREFIX)__ALL.a
	$(LINK) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) $(LIBS) $(SC_LIBS) -o $@


# Verilated -*- Makefile -*-
