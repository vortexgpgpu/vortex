SIMX_DIR = ../../sim/simx

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../include -I../common -I../../hw -I$(SIMX_DIR) -I$(SIMX_DIR)/../common
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread
LDFLAGS += -L. -lsimx

SRCS = vortex.cpp ../common/vx_utils.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = libvortex.so

all: $(PROJECT)

$(PROJECT): $(SRCS)
	DESTDIR=../../driver/simx $(MAKE) -C $(SIMX_DIR) ../../driver/simx/libsimx.so
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	DESTDIR=../../driver/simx $(MAKE) -C $(SIMX_DIR) clean
	rm -rf libsimx.so $(PROJECT) *.o