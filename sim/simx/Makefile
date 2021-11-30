RTL_DIR = ../hw/rtl
THIRD_PARTY_DIR = ../../third_party

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/cocogfx/include
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a -L$(THIRD_PARTY_DIR)/cocogfx -lcocogfx

TOP = vx_cache_sim

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp 
SRCS += args.cpp cache.cpp memsim.cpp warp.cpp core.cpp decode.cpp execute.cpp exeunit.cpp tex_unit.cpp processor.cpp main.cpp

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

PROJECT = simx

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

obj_dir/%.o: %.cpp
	mkdir -p obj_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

static: $(OBJS)
	$(AR) rcs lib$(PROJECT).a $(OBJS) $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/*.o

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-static:
	rm -rf lib$(PROJECT).a obj_dir .depend

clean: clean-static
	rm -rf $(PROJECT)