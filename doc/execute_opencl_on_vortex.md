# Execute OpenCL on Vortex backend

## Requirements
- [Vortex](https://github.com/vortexgpgpu/vortex)
- [POCL for Vortex](https://github.com/vortexgpgpu/pocl)
- [riscv-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [llvm-riscv](https://github.com/llvm-mirror/llvm)

For installation, please see [Build Instructions](../README.md) for more details.

**For Ubuntu18.04 users, you can directly download pre-build toolchains with [toolchain_install.sh](https://github.com/vortexgpgpu/vortex/blob/master/ci/toolchain_install.sh) script.**
```bash
# please modify the DESTDIR variable in the script before execution
bash toolchain_install.sh -all
```
Assuming we have installed all dependencies in `/opt` path, we can get the following environment:
```bash
tree -L 2 /opt
'''
/opt/
├── llvm-riscv
│   ├── bin
│   ├── include
│   ├── lib
│   ├── libexec
│   └── share
├── pocl
│   ├── compiler
│   └── runtime
├── riscv-gnu-toolchain
│   ├── bin
│   ├── drops
│   ├── include
│   ├── lib
│   ├── libexec
│   ├── riscv32-unknown-elf
│   ├── share
│   └── var
└── verilator
    ├── bin
    ├── examples
    ├── include
    ├── verilator-config.cmake
    └── verilator-config-version.cmake
'''
```
## Execute OpenCL on Vortex
In this tutorial, we show the example of executing a vecadd programs on SIMX backend. 
To execute a OpenCL program on Vortex, we have the following steps:
- Compile the [OpenCL kernels](https://github.com/vortexgpgpu/vortex/blob/master/tests/opencl/vecadd/kernel.cl) into risc-v binary by POCL compiler.
- Compile the [OpenCL host](https://github.com/vortexgpgpu/vortex/blob/master/tests/opencl/vecadd/main.cc) and link with Vortex driver(```-lvortex```).
- Execute the compiled host programs on a backend.

Thus, we can write a Makefile as following:
```Makefile
LLVM_PREFIX ?= /opt/llvm-riscv
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
POCL_CC_PATH ?= /opt/pocl/compiler
POCL_RT_PATH ?= /opt/pocl/runtime

OPTS ?= -n64

# please edit these two variable to your environment
VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

K_LLCFLAGS += "-O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32 -mattr=+m,+f -mattr=+vortex -float-abi=hard -code-model=small"
K_CFLAGS   += "-v -O3 --sysroot=$(SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -march=rv32imf -mabi=ilp32f -Xclang -target-feature -Xclang +vortex -I$(VORTEX_RT_PATH)/include -fno-rtti -fno-exceptions -ffreestanding -nostartfiles -fdata-sections -ffunction-sections"
K_LDFLAGS  += "-Wl,-Bstatic,-T$(VORTEX_RT_PATH)/linker/vx_link.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a -lm"

CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter

CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_DRV_PATH)/stub -lOpenCL -lvortex

PROJECT = vecadd

SRCS = main.cc

all: $(PROJECT) kernel.pocl

kernel.pocl: kernel.cl
	LLVM_PREFIX=$(LLVM_PREFIX) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_PREFIX)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -LLCFLAGS $(K_LLCFLAGS) -CFLAGS $(K_CFLAGS) -LDFLAGS $(K_LDFLAGS) -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-fpga: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/fpga:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/asesim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-vlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-simx: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend 

clean-all: clean
	rm -rf *.pocl *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
```

First, build the host program.
```bash
make all
```
If we want to execute on SIMX, we can execute the command below.
```bash
make run-simx
```
