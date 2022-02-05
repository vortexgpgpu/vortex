DESTDIR ?= .
RTL_DIR = ../hw/rtl
THIRD_PARTY_DIR = ../../third_party

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a 
LDFLAGS += -L$(THIRD_PARTY_DIR)/cocogfx -lcocogfx 
LDFLAGS += -L$(THIRD_PARTY_DIR)/ramulator -lramulator

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp
SRCS += args.cpp cache.cpp memsim.cpp warp.cpp core.cpp decode.cpp execute.cpp exeunit.cpp tex_unit.cpp processor.cpp

OBJS := $(patsubst %.cpp, obj_dir/%.o, $(notdir $(SRCS)))
VPATH := $(sort $(dir $(SRCS)))

#$(info OBJS is $(OBJS))
#$(info VPATH is $(VPATH))

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

# XLEN parameterization
ifdef XLEN
	CXXFLAGS += -DXLEN=$(XLEN)
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
	rm -rf obj_dir $(DESTDIR)/$(PROJECT) $(DESTDIR)/lib$(PROJECT).so
