PROJECT = libvortex.so
#PROJECT = libvortex.dylib

SIMX_DIR = ../../simX

CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
#CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -DUSE_SIMX -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../include -I../../hw -I$(SIMX_DIR)

CONFIGS ?= -DNUM_CLUSTERS=1 -DNUM_CORES=1

CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread
#LDFLAGS += -dynamiclib -pthread

DBG_FLAGS += -DDEBUG_LEVEL=3

SRCS = vortex.cpp ../common/vx_utils.cpp 
SRCS += $(SIMX_DIR)/util.cpp $(SIMX_DIR)/args.cpp $(SIMX_DIR)/mem.cpp $(SIMX_DIR)/pipeline.cpp $(SIMX_DIR)/warp.cpp $(SIMX_DIR)/core.cpp $(SIMX_DIR)/decode.cpp $(SIMX_DIR)/execute.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += $(DBG_FLAGS)
else    
	CXXFLAGS += -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;	

clean:
	rm -rf $(PROJECT) *.o .depend