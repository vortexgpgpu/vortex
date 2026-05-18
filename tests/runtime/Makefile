ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

INC_DIR := $(VORTEX_HOME)/sw/runtime/include
RT_DIR  := $(VORTEX_HOME)/build/sw/runtime

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -O2 -DNDEBUG
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

LDFLAGS += -Wl,-rpath,$(RT_DIR) -L$(RT_DIR) -lvortex -pthread

TESTS := test_basic test_async

.PHONY: all run clean

all: $(TESTS)

test_basic: $(VORTEX_HOME)/tests/runtime/test_basic.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_async: $(VORTEX_HOME)/tests/runtime/test_async.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: $(TESTS)
	@for t in $(TESTS); do \
	  echo "[RUN] $$t"; \
	  ./$$t || exit 1; \
	done

clean:
	rm -f $(TESTS)
