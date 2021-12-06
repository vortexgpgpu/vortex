RTL_DIR = ../hw/rtl

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../hw
CXXFLAGS += -I../common/softfloat/source/include
CXXFLAGS += $(CONFIGS)

LDFLAGS += ../common/softfloat/build/Linux-x86_64-GCC/softfloat.a

TOP = vx_cache_sim

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp 
SRCS += args.cpp pipeline.cpp warp.cpp core.cpp decode.cpp execute.cpp main.cpp

OBJS := $(patsubst %.cpp, obj_dir/%.o, $(notdir $(SRCS)))
VPATH := $(sort $(dir $(SRCS)))

#$(info OBJS is $(OBJS))
#$(info VPATH is $(VPATH))

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = simX

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

obj_dir/%.o: %.cpp
	mkdir -p obj_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

static: $(OBJS)
	$(AR) rcs lib$(PROJECT).a $(OBJS) ../common/softfloat/build/Linux-x86_64-GCC/*.o

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-static:
	rm -rf lib$(PROJECT).a obj_dir .depend

clean: clean-static
	rm -rf $(PROJECT)