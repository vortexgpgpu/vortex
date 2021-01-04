RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_RT_PATH ?= $(wildcard ../../../runtime)

OPTS ?= -n64

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -march=rv32imf -mabi=ilp32f -O3 -Wstack-usage=1024 -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link.ld -nostartfiles -Wl,--gc-sections,-Map,kernel.map
VX_CFLAGS += -I$(VORTEX_RT_PATH)/include

VX_LDFLAGS += $(VORTEX_RT_PATH)/libvortexrt.a
VX_LDFLAGS += -lm

VX_SRCS = kernel.c

#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../../include -I../../../hw

PROJECT = dogfood

SRCS = dogfood.cpp

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -d -r -t kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -L../../stub -lvortex -o $@

run-fpga: $(PROJECT)
	LD_LIBRARY_PATH=../../opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-ase: $(PROJECT)
	ASE_LOG=0 LD_LIBRARY_PATH=../../opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-vlsim: $(PROJECT)
	ASE_LOG=0 LD_LIBRARY_PATH=../../opae/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT)
	LD_LIBRARY_PATH=../../rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-simx: $(PROJECT)
	LD_LIBRARY_PATH=../../simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif