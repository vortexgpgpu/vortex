RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_RT_PATH ?= $(wildcard ../../../runtime)

# See aes_op_type_t in common.h for the possible values for t
OPTS ?= -n 16 -t 3

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -march=rv32imf -mabi=ilp32f -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections -Wall -Werror -Wextra
VX_CFLAGS += -B$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/bin -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw -DAES_NATIVE
#VX_CFLAGS += -B$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/bin -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw -DAES_HYBRID
#VX_CFLAGS += -B$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/bin -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw
# To use the hybrid version, remove -DAES_NATIVE from above and add
# -DAES_HYBRID. To use the pure software version, include neither

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

VX_SRCS = kernel.c aes256-impl.c

#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../../include

PROJECT = aes256

SRCS = aes256.cpp
PROJ_LIBS = $(shell pkg-config --libs openssl)

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -L../../stub -lvortex $(PROJ_LIBS) -o $@

run-fpga: $(PROJECT)
	LD_LIBRARY_PATH=../../opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT)
	ASE_LOG=0 LD_LIBRARY_PATH=../../opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-vlsim: $(PROJECT)
	LD_LIBRARY_PATH=../../opae/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

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
