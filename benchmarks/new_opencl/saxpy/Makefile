RISCV_TOOL_PATH ?= $(wildcard ~/dev/riscv-gnu-toolchain/drops)
POCLCC_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_cc)
POCLRT_PATH ?= $(wildcard ..)
DRIVER_PATH ?= $(wildcard ../../../driver/sw)

CXXFLAGS += -std=c++11 -O0 -g -fpermissive -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(POCLRT_PATH)/include

LDFLAGS += -L$(POCLRT_PATH)/lib -L$(DRIVER_PATH)/simx -lOpenCL -lvortex

PROJECT = saxpy

SRCS = main.cc

all: $(PROJECT)

kernel.pocl: kernel.cl
	POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$(RISCV_TOOL_PATH)/lib:$(POCLCC_PATH)/lib:$(DRIVER_PATH)/simx $(POCLCC_PATH)/bin/poclcc -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-fpga: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCLRT_PATH)/lib:$(DRIVER_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-ase: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCLRT_PATH)/lib:$(DRIVER_PATH)/opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-simx: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCLRT_PATH)/lib:$(DRIVER_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-rtlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCLRT_PATH)/lib:$(DRIVER_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.dump .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif