
CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_RT_PATH)/common

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
