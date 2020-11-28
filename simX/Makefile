CFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
#CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -Wno-aligned-new -Wno-maybe-uninitialized

CFLAGS += -I../../hw

TOP = cache_simX

RTL_DIR = ../hw/old_rtl

SRCS = simX.cpp args.cpp mem.cpp core.cpp instruction.cpp enc.cpp util.cpp

RTL_INCLUDE=-I$(RTL_DIR) -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/shared_memory

VL_FLAGS += -O2 --language 1800-2009 --assert
VL_FLAGS += -Wno-DECLFILENAME
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -Wno-UNOPTFLAT -Wno-WIDTH

# Debugigng
ifdef DEBUG
	VL_FLAGS += -DVCD_OUTPUT --trace --trace-structs $(DBG_FLAGS)
	CFLAGS   += -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CFLAGS   += -DNDEBUG
endif

all: simX

simX:
	verilator --exe --cc $(TOP) --top-module $(TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)'
	make -j -C obj_dir -f V$(TOP).mk

clean:
	rm -rf obj_dir
