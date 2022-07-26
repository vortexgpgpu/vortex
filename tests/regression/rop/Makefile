XLEN ?= 32

RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_KN_PATH ?= $(wildcard ../../../kernel)

OPTS ?=

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -std=c++11 -march=rv32imf -mabi=ilp32f -O3 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -DENABLE_SW -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_KN_PATH)/libvortexrt.a

VX_SRCS = kernel.cpp

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_KN_PATH)/../hw -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

LDFLAGS += -L/homes/tinebp/tools/boost/lib -L$(VORTEX_RT_PATH)/stub -lvortex $(VORTEX_KN_PATH)/../third_party/cocogfx/libcocogfx.a -lpng -lz

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = rop

SRCS = main.cpp

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.bin kernel.dump
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-fpga: $(PROJECT) kernel.bin kernel.dump 
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/fpga:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT) kernel.bin kernel.dump
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/asesim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-vlsim: $(PROJECT) kernel.bin kernel.dump
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin kernel.dump
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif