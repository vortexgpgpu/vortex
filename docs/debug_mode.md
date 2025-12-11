# Vortex Debug Mode with GDB

This guide explains how to debug Vortex programs using GDB and OpenOCD with the RISC-V debug interface.

## Prerequisites

Before running the debugger, ensure you have the following dependencies installed:

- **OpenOCD**: Open On-Chip Debugger for JTAG communication
- **RISC-V GDB**: RISC-V cross-debugger (part of RISC-V toolchain, e.g., `riscv64-unknown-elf-gdb`)
- **Build tools**: Make and C++ compiler (g++)

## Building the Simulator and Kernel Files

### Building the Simulator

The simulator must be built with `XLEN=64` to support 64-bit RISC-V binaries (including double-precision floating point):

```bash
cd /vortex
cd build/sim/simx
make clean
make XLEN=64
```

This builds the simulator with:
- **XLEN=64**: 64-bit integer registers
- **EXT_D enabled**: Double-precision floating point support (FLEN=64)
- **RISC-V Debug Module**: Full debug interface support

### Building the Kernel Library

The kernel library (`libvortex.a`) must be built with the same `XLEN` value as the simulator. For 64-bit support:

```bash
cd /vortex/build/kernel
make clean
make XLEN=64
```

This builds the kernel library that provides system calls, startup code, and runtime support for Vortex programs.

### Building Test Binaries

All test binaries must also be built with `XLEN=64` to match the simulator and kernel library:

```bash
# Build a specific test (e.g., fibonacci)
cd /vortex/build/tests/kernel/fibonacci
make clean
make XLEN=64

# Or build all kernel tests
cd /vortex/build/tests/kernel
for dir in */; do
    cd "$dir"
    make clean
    make XLEN=64
    cd ..
done
```

**Important:** The `XLEN` value must be consistent across:
- Simulator (`build/sim/simx`)
- Kernel library (`build/kernel`)
- All test binaries (`build/tests/kernel/*`)

Mismatched `XLEN` values will cause linker errors or runtime failures.

## Quick Start: Debugging Fibonacci

### Step 1: Start Simulator in Debug Mode

```bash
cd /vortex
./build/sim/simx/simx -d build/tests/kernel/fibonacci/fibonacci.bin
```

For verbose debug logging (optional, shows detailed debug module operations):
```bash
./build/sim/simx/simx -d -V 9824 build/tests/kernel/fibonacci/fibonacci.bin
```

The simulator starts halted, waiting for a debugger connection.

### Step 2: Start OpenOCD

```bash
openocd -f vortex.cfg
```

**Note:** `vortex.cfg` uses port 9824. If using default port 9823, either:
- Start simulator with `-p 9824`, or
- Update `vortex.cfg` to use port 9823

### Step 3: Connect GDB

```bash
riscv64-unknown-elf-gdb build/tests/kernel/fibonacci/fibonacci.elf
```

In GDB:
```
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) set $pc = 0x80000000
(gdb) break main
(gdb) continue
```

## Common GDB Commands

```bash
# Breakpoints
(gdb) break main
(gdb) break fibonacci
(gdb) break main.cpp:16

# Execution control
(gdb) continue          # Continue execution
(gdb) step             # Step into function
(gdb) next             # Step over function
(gdb) stepi            # Step one instruction
(gdb) nexti            # Next instruction

# Inspection
(gdb) print variable
(gdb) info registers
(gdb) x/10i $pc        # Disassemble 10 instructions
(gdb) x/s 0x80005740   # Print string at address
```

## Command-Line Options

```bash
./build/sim/simx/simx [options] <program.bin>

Options:
  -d              Enable debug mode
  -p <port>       Remote bitbang port (default: 9823)
  -V              Enable verbose debug module logging (shows detailed debug operations)
  -c <cores>      Number of cores
  -w <warps>      Number of warps per core
  -t <threads>    Number of threads per warp
```

**Note:** The `-V` flag enables verbose logging from the debug module, which shows detailed information about register accesses, memory operations, and debug commands. This is useful for debugging the debugger itself, but can produce a lot of output. Use it when you need to see what the debug module is doing internally.

## Key Addresses (Fibonacci Binary)

| Address | Function/Data |
|---------|---------------|
| 0x80000000 | `_start` (entry point) |
| 0x80000094 | `fibonacci()` |
| 0x80000114 | `main()` |
| 0x800001ac | `init_regs()` (final PC) |
| 0x80005740 | `"fibonacci(%d) = %d\n"` |
| 0x80005754 | `"Passed!\n"` |
| 0x8000575c | `"Failed! value=%d, expected=%d\n"` |

## Troubleshooting

**OpenOCD can't connect:**
- Verify simulator is running with `-d` flag
- Check port numbers match (default 9823, config uses 9824)
- Check simulator output for "Remote bitbang server ready"

## Example Session

```bash
# Terminal 1 (add -V for verbose debug logging)
./build/sim/simx/simx -d -p 9824 build/tests/kernel/fibonacci/fibonacci.bin

# Terminal 2
openocd -f vortex.cfg

# Terminal 3
riscv64-unknown-elf-gdb
(gdb) target remote localhost:3333
(gdb) stepi
(gdb) b *0x80000094
(gdb) continue
(gdb) i r
(gdb) continue
Continuing.
Program Stopped
```

## Additional Resources

- [RISC-V Debug Specification](https://github.com/riscv/riscv-debug-spec)
- [OpenOCD Documentation](http://openocd.org/doc/html/index.html)
- [GDB User Manual](https://sourceware.org/gdb/current/onlinedocs/gdb/)
