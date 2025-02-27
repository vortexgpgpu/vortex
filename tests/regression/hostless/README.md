Example Headless Simulation
===========================

This directory contains an example kernel for a headless simulation of the
Vortex GPGPU, which eliminates involvement of the host-side CPU execution and
enables standalone simulation of the GPU.

## Overview

There are three binaries needed for running a kernel on Vortex:

* **Program binary**: Vortex ISA program binary, situated at `0x80000000` in
  the device memory.
* **Argument binary**: Binary that contains the kernel argument.  This is normally
  uploaded from host to the device memory, using `vx_copy_to_dev()` in the host
  program (`main.cpp`).  It is located at a statically known address defined as
  `KERNEL_ARG_DEV_MEM_ADDR` in [`common.h`](common.h).
* **Input binary**: Binary that contains auxiliary input data to the kernel,
  e.g. vector data of FP32 elements used as input to a vector-add kernel. Its
  memory address is communicated to the kernel via the argument binary,
  which contains a pointer field. See `kernel_arg_t.addr_{a,b,dst}` in
  [`common.h`](common.h).

In a GPU-only standalone simulation, we don't have access to host-device calls
such as `vx_copy_to_dev`.  A workaround of this that we use is to _stitch all
of the binaries above into a single kernel binary,_  so that loading the kernel
binary will put relevant bits to the correct device memory addresses with
minimum hassle.

As a tradeoff, the program and argument binaries need to be placed at fixed
device memory addresses so that the kernel can access them at a memory location
statically known ahead of time.  Also, the kernel binary can become large in
size especially when these memory addresses are far apart from each other.

In order to do this, we modify the [`vx_link32.ld` linker
script](/kernel/linker/vx_link32.ld) to add new sections where the argument and
input binaries will get stitched into the kernel ELF:

```
...
MEMORY {
  DRAM0    (rwx): ORIGIN = 0x80000000, LENGTH = 512M
  DRAMARG  (rwx): ORIGIN = 0x9fff0000, LENGTH = 8K
  DRAM1    (rwx): ORIGIN = 0xa0000000, LENGTH = 16M
  DRAM2    (rwx): ORIGIN = 0xa1000000, LENGTH = 16M
  DRAM3    (rwx): ORIGIN = 0xa2000000, LENGTH = 16M
}
...
  .args : {
    *(.args)
    . += 8K;
  }> DRAMARG
  .operand.a : {
    *(.operand.a)
    . += 32K;
  }> DRAM1
  .operand.b : {
    *(.operand.b)
    . += 32K;
  }> DRAM2
  .operand.c : {
    *(.operand.c)
    . += 32K;
  }> DRAM3
...
```

Then, when compiling the kernel, we use the `objcopy` tool to copy the bits
from separate binary files into the corresponding sections using the
`--update-section` flag.  The makefile rule for `kernel.elf` in
[`common.mk`](https://github.com/hansungk/vortex/blob/95710c22806c728a38fb653182f25da2f05e79d4/tests/regression/common.mk#L89-L101) can be modified to automate this:

```
...
kernel.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
	$(VX_CXX) $(VX_CFLAGS) -o $@ $(VX_SRCS) $(VX_LDFLAGS)
	$(OBJCOPY) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .operand.c=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(OBJCOPY) --update-section .operand.a=input.a.bin $@ || true
	$(OBJCOPY) --update-section .operand.b=input.b.bin $@ || true
	$(OBJCOPY) --update-section .operand.c=input.c.bin $@ || true
	$(OBJCOPY) --update-section .args=args.bin $@ || true
...
```

## Setup

### Merging codebase

This codebase is based on a fork at an [older
commit](https://github.com/vortexgpgpu/vortex/tree/5f2b10b8), so a full merge
might be tricky.
However, all changes required for the standalone simulation are contained in
these two files, which should be straightforward to modify manually:

* [kernel/linker/vx_link32.ld](/kernel/linker/vx_link32.ld): Add `MEMORY` block
  and `.args`, `.operand.a`, `.operand.b`, `.operand.c` section descriptions to
  the linker script
* [tests/regression/common.mk](/tests/regression/common.mk): Modify
  `kernel.elf` target to include the `objcopy` commands

The new makefile target requires binary files `args.bin`, `input.a.bin`,
`input.b.bin`, `input.c.bin` to be present in the current working directory.
We added these files into the tree for convenience, but if you want to generate
your custom binaries, comment out [this code block](https://github.com/hansungk/vortex/blob/95710c22806c728a38fb653182f25da2f05e79d4/tests/regression/example/main.cpp#L210-L261) to automatically generate
these binaries from the host side.  Note that the host program should be run at
least once via `make run-simx` to generate the files.

### Steps

To compile the kernel:
```bash
# set up repository and set environment variables
$ source ./ci/toolchain_env.sh
$ cd tests/regression/example
# compile kernel; this will generate kernel.bin that contains all
# program/argument/input binaries
$ make
```

In `kernel.dump`, you should be able to see the new `.args`, `.operand.a` and
`.operand.b` sections added to the ELF:
```
9fff0000 <.args>:
9fff0000: 10 00        	<unknown>
		...
9fff000a: 00 a0        	<unknown>
9fff000c: 00 00        	<unknown>
9fff000e: 00 00        	<unknown>
9fff0010: 00 00        	<unknown>
9fff0012: 00 a1        	<unknown>
    ...

a0000000 <.operand.a>:
a0000000: 00 00        	<unknown>
a0000002: 00 00        	<unknown>
a0000004: 00 00        	<unknown>
a0000006: 00 40        	<unknown>
    ...
```

Please note that we use `9fff0000` as the argument address, different from
`7fff0000` which is the default. Also note that the starting address of the
`.operand.a` section (`a0000000`) should match with what's encoded in the
`args.bin` binary, in particular the `arg->addr_a` field (`9fff0008` in the
device memory):
```bash
$ hexdump -C args.bin
#         <-dim(32b)>              <-----addr_a(64b)----->
00000000  10 00 00 00 00 00 00 00  00 00 00 a0 00 00 00 00  |................|
#         <-----addr_b(64b)----->  <---addr_dst(64b)----->
00000010  00 00 00 a1 00 00 00 00  00 00 00 c0 00 00 00 00  |................|
00000020
```

To run a GPU-standalone simulation either with simx or rtlsim:
```bash
# in tests/regression/example
$ ../../../sim/simx/simx kernel.bin
$ ../../../sim/rtlsim/rtlsim kernel.bin
```

To run a host-GPU simulation to verify the results; note that the host program
`main.cpp` is modified to *not* upload the args/input binaries (because they're
already contained in `kernel.bin`), and only run verification tests at the end:
```bash
# in tests/regression/example
$ make run-simx
```

### Re-compiling

When you update any of the `args.bin`, `input.a.bin`, `input.b.bin` and
`input.c.bin` binary files, the kernel binary needs to be re-built by way of
re-running the `kernel.bin` make target.  The makefile rule is written in a way
to keep track of changes in these files, but you can force recompile by
doing `touch kernel.cpp && make`.

### Program binary size

As a result of putting the program binary and argument/input binaries at
distinct device memory addresses (`80000000` and `a0000000`, `a1000000`, ...),
`kernel.bin` becomes pretty large in size; >512MB in our case.  If this becomes
a problem, the input binary addresses can be moved to somewhere closer to
`80000000`.  The linker script and the relevent fields in `args.bin` should be
updated for this.

## Contact

Please contact Hansung Kim <hansung_kim@berkeley.edu> for any questions about
this README.
