ROOT_DIR := $(realpath ../../..)

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(XLEN),64)
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
VX_CFLAGS += -march=rv64imafd -mabi=ilp64d
STARTUP_ADDR ?= 0x180000000
else
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
STARTUP_ADDR ?= 0x80000000
endif

POCL_CC_PATH ?= $(TOOLDIR)/pocl/compiler
POCL_RT_PATH ?= $(TOOLDIR)/pocl/runtime

LLVM_POCL ?= $(TOOLDIR)/llvm-vortex

LIBC_LIB += -L$(LIBC_VORTEX)/lib -lm -lc -lgcc

VX_CFLAGS  += -O3 -mcmodel=medany --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -Xclang -target-feature -Xclang +vortex
VX_CFLAGS  += -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
#VX_CFLAGS += -mllvm -vortex-branch-divergence=0
#VX_CFLAGS += -mllvm -print-after-all
VX_CFLAGS  += -mllvm -disable-loop-idiom-all	# disable memset/memcpy loop idiom
VX_CFLAGS  += -I$(ROOT_DIR)/hw -I$(VORTEX_KN_PATH)/include -DXLEN_$(XLEN) -DNDEBUG
VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(ROOT_DIR)/kernel/libvortexrt.a $(LIBC_LIB)

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter -Wno-narrowing
CXXFLAGS += -pthread
CXXFLAGS += -I$(POCL_RT_PATH)/include

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif	
endif
endif

OBJS := $(addsuffix .o, $(notdir $(SRCS)))
OBJS_HOST := $(addsuffix .host.o, $(notdir $(SRCS)))

.DEFAULT_GOAL := all
all: $(PROJECT) kernel.pocl
 
kernel.cl: $(SRC_DIR)/kernel.cl
	cp $(SRC_DIR)/kernel.cl $@

kernel.pocl: $(SRC_DIR)/kernel.cl
	LD_LIBRARY_PATH=$(LLVM_POCL)/lib:$(POCL_CC_PATH)/lib:$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all POCL_KERNEL_CACHE=0 POCL_VORTEX_CFLAGS="$(VX_CFLAGS)" POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)" $(POCL_CC_PATH)/bin/poclcc -o kernel.pocl $^
 
%.cc.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cpp.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.c.o: $(SRC_DIR)/%.c
	$(CC) $(CXXFLAGS) -c $< -o $@

%.cc.host.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -DHOSTGPU -c $< -o $@

%.cpp.host.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -DHOSTGPU -c $< -o $@

%.c.host.o: $(SRC_DIR)/%.c
	$(CC) $(CXXFLAGS) -DHOSTGPU -c $< -o $@

ifndef USE_SETUP
setup:
endif

$(PROJECT): setup $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out setup, $^) $(LDFLAGS) -L$(ROOT_DIR)/runtime/stub -lvortex -L$(POCL_RT_PATH)/lib -lOpenCL -o $@

$(PROJECT).host: setup $(OBJS_HOST)	
	$(CXX) $(CXXFLAGS) $(filter-out setup, $^) $(LDFLAGS) -lOpenCL -o $@

run-gpu: $(PROJECT).host kernel.cl
	./$(PROJECT).host $(OPTS)

run-simx: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(ROOT_DIR)/runtime/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(ROOT_DIR)/runtime/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.pocl
	SCOPE_JSON_PATH=$(ROOT_DIR)/runtime/opae/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(ROOT_DIR)/runtime/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.pocl
ifeq ($(TARGET), hw)
	XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(ROOT_DIR)/runtime/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(ROOT_DIR)/runtime/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) $(PROJECT).host *.o *.log .depend

clean-all: clean
	rm -rf *.dump *.pocl

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
