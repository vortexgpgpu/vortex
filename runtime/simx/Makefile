XLEN ?= 32
DESTDIR ?= $(CURDIR)
SIMX_DIR = ../../sim/simx

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../include -I../common -I../../hw -I$(SIMX_DIR) -I$(SIMX_DIR)/../common
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DDUMP_PERF_STATS
CXXFLAGS += -DXLEN_$(XLEN)

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lsimx

SRCS = vortex.cpp ../common/utils.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = libvortex.so

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/$(PROJECT): $(SRCS)
	DESTDIR=$(DESTDIR) $(MAKE) -C $(SIMX_DIR) $(DESTDIR)/libsimx.so
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(SIMX_DIR) clean
	rm -rf $(DESTDIR)/$(PROJECT) *.o