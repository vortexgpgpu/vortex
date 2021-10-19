RTLSIM_DIR = ../../sim/rtlsim

CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I../../hw -I$(RTLSIM_DIR) -I$(RTLSIM_DIR)/../common

LDFLAGS += $(RTLSIM_DIR)/librtlsim.a

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread

SRCS = vortex.cpp ../common/vx_utils.cpp

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = libvortex.so

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	$(MAKE) -C $(RTLSIM_DIR) static
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	$(MAKE) -C $(RTLSIM_DIR) clean-static
	rm -rf $(PROJECT) *.o