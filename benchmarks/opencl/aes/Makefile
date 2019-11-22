
RISCV_TOOL_PATH=$(wildcard ~/dev/riscv-gnu-toolchain/drops)

POCL_CC_PATH=$(wildcard ~/dev/pocl/drops_riscv_cc)
POCL_RT_PATH=$(wildcard ~/dev/pocl/drops_riscv_rt)

VX_RT_PATH=$(wildcard ../../../runtime)
VX_SIMX_PATH=$(wildcard ../../../simX/obj_dir)

CC=$(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
CXX=$(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-g++
DMP=$(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
HEX=$(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy
NEWLIB_PATH=$(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib

VX_NEWLIB = $(VX_RT_PATH)/newlib/newlib.c
VX_STR  = $(VX_RT_PATH)/startup/vx_start.s
VX_INT  = $(VX_RT_PATH)/intrinsics/vx_intrinsics.s
VX_IO   = $(VX_RT_PATH)/io/vx_io.s $(VX_RT_PATH)/io/vx_io.c
VX_FIO  = $(VX_RT_PATH)/fileio/fileio.s
VX_API  = $(VX_RT_PATH)/vx_api/vx_api.c

VX_SRCS  = $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST)

CXXFLAGS = -g -O0 -nostartfiles -Wl,-Bstatic,-T,$(VX_RT_PATH)/mains/vortex_link.ld -march=rv32im -mabi=ilp32 
CXXFLAGS += -ffreestanding # program may not begin at main()
CXXFLAGS += -fno-rtti -fno-non-call-exceptions # disable RTTI and exceptions
CXXFLAGS += -Wl,--gc-sections # enable garbage collection of unused input sections

LIBS = -lOpenCL

PROJECT=saxpy

all: $(PROJECT).dump $(PROJECT).hex

lib$(PROJECT).a: kernel.cl
	POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$(RISCV_TOOL_PATH)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -o lib$(PROJECT).a kernel.cl

$(PROJECT).elf: main.cc lib$(PROJECT).a
	$(CXX) $(CXXFLAGS) -I$(POCL_RT_PATH)/include -L$(POCL_RT_PATH)/lib/static -L. $(VX_SRCS) main.cc -Wl,--whole-archive -l$(PROJECT) -Wl,--no-whole-archive $(LIBS) -o $(PROJECT).elf

$(PROJECT).hex: $(PROJECT).elf
	$(HEX) -O ihex $(PROJECT).elf $(PROJECT).hex

$(PROJECT).dump: $(PROJECT).elf
	$(DMP) -D $(PROJECT).elf > $(PROJECT).dump

run:
	$(VX_SIMX_PATH)/Vcache_simX -E -a rv32i --core $(PROJECT).hex  -s -b 1> emulator.debug

clean:
	rm -rf *.elf *.dump *.hex *.a *.pocl *.o