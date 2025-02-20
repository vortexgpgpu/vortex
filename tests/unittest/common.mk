
ROOT_DIR := $(realpath ../../..)

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_HOME)/sim/common

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run:
	./$(PROJECT)

clean:
	rm -rf $(PROJECT) *.o *.log .depend

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
