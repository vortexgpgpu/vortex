LIB_PATH = ../../../runtime

COMP     = /nethome/ekim79/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc

CC_FLAGS = -ffreestanding -O0 -Wl,--gc-sections -nostartfiles -nostdlib -nostartfiles -nodefaultlibs -Wl,-Bstatic,-T,$(LIB_PATH)/startup/vx_link.ld -march=rv32imv -mabi=ilp32

DMP  = /nethome/ekim79/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objdump
CPY  = /nethome/ekim79/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objcopy

# VX_STR  = ../../startup/vx_start.S

NEWLIB  = $(LIB_PATH)/newlib/newlib.c
VX_STR  = $(LIB_PATH)/startup/vx_start.S
VX_INT  = $(LIB_PATH)/intrinsics/vx_intrinsics.S
VX_IO   = $(LIB_PATH)/io/vx_io.S $(LIB_PATH)/io/vx_io.c
VX_API  = $(LIB_PATH)/vx_api/vx_api.c
VX_FIO  = $(LIB_PATH)/fileio/fileio.S
VX_VEC1  = vx_vec_vvaddint32.s
#VX_VEC2  = vx_vec_saxpy.s       #float --> int
#VX_VEC3  = vx_vec_sgemm.s       #float --> int
#VX_VEC4  = vx_vec_vsadd.s 
#VX_VEC5  = vx_vec_memcpy.s 
LIBS    = /nethome/ekim79/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libc.a /nethome/ekim79/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = vx_vec_vecadd

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF:
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC1) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
#	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC2) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
#	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC3) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
#	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC4) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
#	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC5) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf~                                
