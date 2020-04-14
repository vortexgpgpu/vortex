LLVM_HOME ?= ~/dev/llvm-project/drops
TOOLCHAIN_PATH ?= ~/dev/riscv-gnu-toolchain/drops
SYSROOT ?= $(TOOLCHAIN_PATH)/riscv32-unknown-elf
POCL_CC_PATH ?= $(realpath ../compiler)
POCL_RT_PATH ?= $(realpath ../runtime)
VORTEX_DRV_PATH ?= $(realpath ../../../driver/sw)
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

CXXFLAGS += -std=c++11 -O0 -g -fpermissive -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_DRV_PATH)/stub -lOpenCL -lvortex

PROJECT = kmeans

SRCS = main.cc read_input.c rmse.c kmeans_clustering.c cluster.c getopt.c

all: $(PROJECT)

kernel.pocl: kernel.cl
	TOOLCHAIN_PATH=$(TOOLCHAIN_PATH) SYSROOT=$(SYSROOT) LLVM_HOME=$(LLVM_HOME) VORTEX_RUNTIME_PATH=$(VORTEX_RT_PATH) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_HOME)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-fpga: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-ase: $(PROJECT) kernel.pocl   
	ASE_LOG=0 LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-simx: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT)

run-rtlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.dump .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
