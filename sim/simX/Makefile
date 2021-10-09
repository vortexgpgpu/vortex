#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../hw
CXXFLAGS += -DDUMP_PERF_STATS

TOP = vx_cache_sim

RTL_DIR = ../hw/rtl

PROJECT = simX

SRCS = ../common/util.cpp ../common/mem.cpp 
SRCS += args.cpp pipeline.cpp warp.cpp core.cpp decode.cpp execute.cpp main.cpp

OBJS := $(patsubst %.cpp, obj_dir/%.o, $(notdir $(SRCS)))
VPATH := $(sort $(dir $(SRCS)))

#$(info OBJS is $(OBJS))
#$(info VPATH is $(VPATH))

# Debugigng
ifdef DEBUG
	CXXFLAGS += -DDEBUG_LEVEL=$(DEBUG)
else    
	CXXFLAGS += -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

obj_dir/%.o: %.cpp
	mkdir -p obj_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

static: $(OBJS)
	$(AR) rs lib$(PROJECT).a $(OBJS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-static:
	rm -rf lib$(PROJECT).a obj_dir .depend

clean: clean-static
	rm -rf $(PROJECT)