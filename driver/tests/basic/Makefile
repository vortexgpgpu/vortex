
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../../sw/include

LDFLAGS += 

PROJECT = basic

SRCS = basic.cpp

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -L../../sw/dummy -lvortex -o $@

run-fpga: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/opae:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-ase: $(PROJECT)
	ASE_LOG=0 LD_LIBRARY_PATH=../../sw/opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-rtlsim: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-simx: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/simx:$(LD_LIBRARY_PATH) ./$(PROJECT)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif