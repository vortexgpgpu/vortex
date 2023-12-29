XLEN ?= 32
DESTDIR ?= $(CURDIR)
RTLSIM_DIR = ../../sim/rtlsim

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I../include -I../common -I../../hw -I$(RTLSIM_DIR) -I$(RTLSIM_DIR)/../common
CXXFLAGS += -DXLEN_$(XLEN)

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lrtlsim

SRCS = vortex.cpp ../common/utils.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = libvortex.so

all: $(DESTDIR)/$(PROJECT)
	
$(DESTDIR)/$(PROJECT): $(SRCS)
	DESTDIR=$(DESTDIR) $(MAKE) -C $(RTLSIM_DIR) $(DESTDIR)/librtlsim.so
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

clean:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(RTLSIM_DIR) clean
	rm -rf $(DESTDIR)/$(PROJECT) *.o