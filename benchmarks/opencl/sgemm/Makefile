RISCV_TOOL_PATH = $(wildcard ~/dev/riscv-gnu-toolchain/drops)
POCL_CC_PATH = $(wildcard ~/dev/pocl/drops_riscv_cc)
POCL_INC_PATH = $(wildcard ../include)
POCL_LIB_PATH = $(wildcard ../lib)
VX_RT_PATH = $(wildcard ../../../runtime)
VX_SIMX_PATH = $(wildcard ../../../simX/obj_dir)

CC  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
CXX = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-g++
DMP = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
HEX = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy
GDB = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gdb

VX_SRCS =  $(VX_RT_PATH)/newlib/newlib.c
VX_SRCS += $(VX_RT_PATH)/startup/vx_start.s
VX_SRCS += $(VX_RT_PATH)/intrinsics/vx_intrinsics.s
VX_SRCS += $(VX_RT_PATH)/io/vx_io.s $(VX_RT_PATH)/io/vx_io.c
VX_SRCS += $(VX_RT_PATH)/fileio/fileio.s
VX_SRCS += $(VX_RT_PATH)/tests/tests.c
VX_SRCS += $(VX_RT_PATH)/vx_api/vx_api.c
VX_SRCS += $(VX_STR) $(VX_FIO) $(VX_NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST)

VX_CFLAGS = -nostartfiles -Wl,-Bstatic,-T,$(VX_RT_PATH)/mains/vortex_link.ld

CXXFLAGS =  -g -O0 -march=rv32im -mabi=ilp32 
CXXFLAGS += -ffreestanding # program may not begin at main()
CXXFLAGS += -Wl,--gc-sections # enable garbage collection of unused input sections
CXXFLAGS += -fno-rtti -fno-non-call-exceptions # disable RTTI and exceptions
CXXFLAGS += -I$(POCL_INC_PATH)

VX_LIBS = -Wl,--whole-archive lib$(PROJECT).a -Wl,--no-whole-archive $(POCL_LIB_PATH)/libOpenCL.a
QEMU_LIBS = -Wl,--whole-archive lib$(PROJECT).a -Wl,--no-whole-archive $(POCL_LIB_PATH)/qemu/libOpenCL.a

PROJECT=sgemm

all: $(PROJECT).dump $(PROJECT).hex

lib$(PROJECT).a: kernel.cl
	POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$(RISCV_TOOL_PATH)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -o lib$(PROJECT).a kernel.cl

$(PROJECT).elf: main.cc lib$(PROJECT).a
	$(CXX) $(CXXFLAGS) $(VX_CFLAGS) $(VX_SRCS) main.cc $(VX_LIBS) -o $(PROJECT).elf

$(PROJECT).qemu: main.cc lib$(PROJECT).a
	$(CXX) $(CXXFLAGS) main.cc $(QEMU_LIBS) -o $(PROJECT).qemu

$(PROJECT).hex: $(PROJECT).elf
	$(HEX) -O ihex $(PROJECT).elf $(PROJECT).hex

$(PROJECT).dump: $(PROJECT).elf
	$(DMP) -D $(PROJECT).elf > $(PROJECT).dump

run: $(PROJECT).hex
	POCL_DEBUG=all $(VX_SIMX_PATH)/Vcache_simX -E -a rv32i --core $(PROJECT).hex -s -b 1> emulator.debug

qemu: $(PROJECT).qemu
	POCL_DEBUG=all $(RISCV_TOOL_PATH)/bin/qemu-riscv32 -d in_asm -D debug.log $(PROJECT).qemu

gdb-s: $(PROJECT).qemu
	POCL_DEBUG=all $(RISCV_TOOL_PATH)/bin/qemu-riscv32 -g 1234 -d in_asm -D debug.log $(PROJECT).qemu

gdb-c: $(PROJECT).qemu
	$(GDB) $(PROJECT).qemu

clean:
	rm -rf *.o *.elf *.dump *.hex *.a *.pocl *.qemu