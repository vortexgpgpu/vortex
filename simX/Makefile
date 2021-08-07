#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../hw
CXXFLAGS += -DDUMP_PERF_STATS

TOP = vx_cache_sim

RTL_DIR = ../hw/rtl

PROJECT = simX

SRCS = util.cpp args.cpp mem.cpp pipeline.cpp warp.cpp core.cpp decode.cpp execute.cpp main.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -DDEBUG_LEVEL=$(DEBUG)
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
