VORTEX_DRV_PATH ?= $(realpath ../../../driver)

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(VORTEX_DRV_PATH)/common

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = vx_malloc

SRCS = main.cpp

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run:
	./$(PROJECT)

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif