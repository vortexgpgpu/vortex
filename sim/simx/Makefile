XLEN ?= 64
DESTDIR ?= .
RTL_DIR = ../hw/rtl
THIRD_PARTY_DIR = ../../third_party

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a 
LDFLAGS += -L$(THIRD_PARTY_DIR)/cocogfx -lcocogfx 
LDFLAGS += -L$(THIRD_PARTY_DIR)/ramulator -lramulator

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp ../common/graphics.cpp
SRCS += args.cpp processor.cpp cluster.cpp core.cpp warp.cpp decode.cpp execute.cpp exe_unit.cpp cache_sim.cpp mem_sim.cpp shared_mem.cpp dcrs.cpp
SRCS += tex_unit.cpp raster_unit.cpp rop_unit.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = simx

all: $(DESTDIR)/$(PROJECT)
	
$(DESTDIR)/$(PROJECT): $(SRCS) main.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(DESTDIR)/lib$(PROJECT).so: $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(DESTDIR)/$(PROJECT) $(DESTDIR)/lib$(PROJECT).so
