RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(wildcard ../../../runtime)

OPTS ?= -g1

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -march=rv32imf -mabi=ilp32f -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

VX_SRCS = kernel.c

#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -Wfatal-errors

CXXFLAGS += -DLUPNG_USE_ZLIB

CXXFLAGS += -I$(VORTEX_DRV_PATH)/include

LDFLAGS += -L$(VORTEX_DRV_PATH)/stub -lvortex  -lz

PROJECT = tex

SRCS = main.cpp utils.cpp tga.cpp lupng.c

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-fpga: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/fpga:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/asesim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-vlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif