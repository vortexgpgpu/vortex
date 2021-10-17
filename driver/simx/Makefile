SIMX_DIR = ../../sim/simX

CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
#CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../include -I../../hw -I$(SIMX_DIR) -I$(SIMX_DIR)/../common
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread
LDFLAGS += $(SIMX_DIR)/libsimX.a

SRCS = vortex.cpp ../common/vx_utils.cpp 

PROJECT = libvortex.so

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(MAKE) -C $(SIMX_DIR) static
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	$(MAKE) -C $(SIMX_DIR) clean-static
	rm -rf $(PROJECT) *.o