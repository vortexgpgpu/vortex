RTLSIM_DIR = ../../sim/rtlsim

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I../common -I../../hw -I$(RTLSIM_DIR) -I$(RTLSIM_DIR)/../common

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread
LDFLAGS += -L. -lrtlsim

SRCS = vortex.cpp ../common/vx_utils.cpp

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

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	DESTDIR=../../driver/rtlsim $(MAKE) -C $(RTLSIM_DIR) ../../driver/rtlsim/librtlsim.so
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	DESTDIR=../../driver/rtlsim $(MAKE) -C $(RTLSIM_DIR) clean
	rm -rf $(PROJECT) *.o