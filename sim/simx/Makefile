XLEN ?= 32
DESTDIR ?= $(CURDIR)
HW_DIR = $(abspath ../../hw)
COMMON_DIR = $(abspath ../common)
THIRD_PARTY_DIR = $(abspath ../../third_party)

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(CURDIR) -I$(COMMON_DIR) -I$(HW_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -L$(THIRD_PARTY_DIR)/ramulator -lramulator

SRCS = $(COMMON_DIR)/util.cpp $(COMMON_DIR)/mem.cpp $(COMMON_DIR)/rvfloats.cpp
SRCS += processor.cpp cluster.cpp socket.cpp core.cpp warp.cpp decode.cpp execute.cpp exe_unit.cpp cache_sim.cpp mem_sim.cpp shared_mem.cpp dcrs.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
	#CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG) -fsanitize=address -fno-omit-frame-pointer
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = simx

all: $(DESTDIR)/$(PROJECT)
	
$(DESTDIR)/$(PROJECT): $(SRCS) main.cpp
	$(CXX) $(CXXFLAGS) -DSTARTUP_ADDR=0x80000000 $^ $(LDFLAGS) -o $@

$(DESTDIR)/lib$(PROJECT).so: $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(DESTDIR)/$(PROJECT) $(DESTDIR)/lib$(PROJECT).so
