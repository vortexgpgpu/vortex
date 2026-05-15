# Building the Vortex Toolchain from Source

This document covers building each Vortex toolchain component from
source. It is intended for **maintainers and developers** who need
to modify a tool, target a new system, or update the prebuilt
toolchain bundles consumed by [install_vortex.md](install_vortex.md).

End users should normally use the prebuilt toolchain via
`./ci/toolchain_install.sh --all` (see
[install_vortex.md](install_vortex.md)) — that is significantly
faster than building from source.

The components covered here are:

1. [Verilator](#1-verilator) — RTL simulator
2. [RISC-V GNU Toolchain](#2-risc-v-gnu-toolchain) — `riscv32-unknown-elf-*` / `riscv64-unknown-elf-*`
3. [LLVM for Vortex](#3-llvm-for-vortex) — Clang + LLVM with the Vortex ISA extensions
4. [compiler-rt for Vortex](#4-compiler-rt-for-vortex) — baremetal builtins
5. [musl libc for Vortex](#5-musl-libc-for-vortex) — C standard library
6. [POCL for Vortex](#6-pocl-for-vortex) — OpenCL implementation with the Vortex device target
7. [OpenSTA](#7-opensta) — static timing analysis (optional)

---

## Prerequisites

Set the install root for all built artifacts:

```bash
export TOOLDIR=$HOME/tools
mkdir -p "$TOOLDIR"
```

`$TOOLDIR` is the canonical install prefix referenced throughout
the rest of the Vortex build infrastructure (`config.mk`,
`ci/toolchain_install.sh`, etc.). Stay consistent with this single
root.

System packages (Ubuntu 22.04 / 24.04):

```bash
sudo apt-get update
sudo apt-get install \
    build-essential cmake ninja-build ccache autoconf flex bison \
    zlib1g-dev libtinfo-dev libncurses-dev uuid-dev \
    libboost-serialization-dev libpng-dev libhwloc-dev \
    libtcl8.6 tcl8.6-dev
```

---

## 1. Verilator

**Purpose**: cycle-accurate RTL simulation backend used by
`sim/rtlsim` and the unit-test suites.

```bash
git clone --depth=1 --recursive https://github.com/verilator/verilator.git
cd verilator
git checkout stable
autoconf
./configure --prefix=$TOOLDIR/verilator
make -j$(nproc)
make install

# Optional convenience layout used by older Vortex scripts:
cp -r $TOOLDIR/verilator/share/verilator/bin     $TOOLDIR/verilator/
cp -r $TOOLDIR/verilator/share/verilator/install $TOOLDIR/verilator/
```

If your system GCC is too old, point the configure step at a
locally-installed newer GCC, e.g.:

```bash
./configure --prefix=$TOOLDIR/verilator \
    CC=$TOOLDIR/gnu/bin/gcc-11 CXX=$TOOLDIR/gnu/bin/g++-11
```

---

## 2. RISC-V GNU Toolchain

**Purpose**: `gcc`, `binutils`, `gdb`, and a baseline C library for
RISC-V baremetal targets. Vortex ships with both 32-bit and
64-bit toolchains so kernels can be built for either XLEN.

```bash
git clone --depth=1 --recursive https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
mkdir build && cd build
```

If the host has a custom GNU prefix, expose its headers/libs to the
configure step:

```bash
export CPATH=$TOOLDIR/gnu/include
export LIBRARY_PATH=$TOOLDIR/gnu/lib
```

### 32-bit (`riscv32-unknown-elf`)

```bash
../configure \
    --prefix=$TOOLDIR/riscv32-gnu-toolchain \
    --with-cmodel=medany \
    --with-arch=rv32imaf --with-abi=ilp32f \
    --enable-multilib
make -j$(nproc)
```

### 64-bit (`riscv64-unknown-elf`)

```bash
../configure \
    --prefix=$TOOLDIR/riscv64-gnu-toolchain \
    --with-cmodel=medany \
    --with-arch=rv64imafd --with-abi=lp64d \
    --enable-multilib
make -j$(nproc)
```

(Optional) build QEMU alongside:

```bash
make -j$(nproc) build-qemu
```

### Reference

- Multilib config:
  `riscv-gnu-toolchain/gcc/gcc/config/riscv/t-elf-multilib`
- Inspect the produced multilib targets:
  `riscv32-unknown-elf-gcc --print-multi-lib`

---

## 3. LLVM for Vortex

**Purpose**: Clang + LLVM with the Vortex ISA extensions, used as:

- The kernel-side compiler for OpenCL/HIP/SYCL device code
  (RISC-V/Vortex backend).
- The host-side compiler for HIP and chipStar host code (X86_64
  backend).
- The linker for both (`ld.lld`).

> **Targets to build:** for the bespoke HIP path
> ([hip_support_proposal.md](proposals/hip_support_proposal.md)
> Path A) and chipStar
> ([chipstar_on_vortex_proposal.md](proposals/chipstar_on_vortex_proposal.md)),
> the same Clang must serve both host and device compilation. We
> therefore enable **both `RISCV` and `X86`** in
> `LLVM_TARGETS_TO_BUILD`, and **enable `lld`** so the host link
> step has a working `ld.lld`.

```bash
git clone --recursive --branch vortex_2.x https://github.com/vortexgpgpu/llvm.git llvm_vortex
cd llvm_vortex
mkdir build && cd build

export LLVM_PREFIX=$TOOLDIR/llvm-vortex
export RISCV_TOOLCHAIN_PATH=$TOOLDIR/riscv32-gnu-toolchain

cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DDEFAULT_SYSROOT=$RISCV_TOOLCHAIN_PATH/riscv32-unknown-elf \
    -DLLVM_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" \
    ../llvm

make -j$(nproc)
make install
```

### Notes

- `DEFAULT_SYSROOT` and `LLVM_DEFAULT_TARGET_TRIPLE` keep
  `riscv32-unknown-elf` as the default triple so existing Vortex
  build infrastructure (which calls `clang foo.c` without an
  explicit `-target` flag) continues to work. The X86 backend is
  available on demand via `clang --target=x86_64-linux-gnu …`.
- `LLVM_ENABLE_PROJECTS="clang;lld"` adds `lld` so the toolchain
  ships with `ld.lld` (some host-link toolchains require it; the
  Vortex device flow uses it via `-fuse-ld=lld`).
- `BUILD_SHARED_LIBS=ON` produces `libLLVM*.so` rather than static
  archives. Tools built against this LLVM (e.g.
  [SPIRV-LLVM-Translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator))
  need `LD_LIBRARY_PATH=$LLVM_PREFIX/lib` at runtime.

### SPIRV-LLVM-Translator (optional, required for chipStar / SPIR-V code path)

The translator binary `llvm-spirv` is built separately against an
installed LLVM and lives alongside it.

```bash
git clone --branch llvm_release_180 --depth 1 \
    https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
      -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX \
      ..
cmake --build . --target llvm-spirv -j$(nproc)
cmake --install .
```

Match the translator branch (`llvm_release_180` for LLVM 18,
`llvm_release_170` for LLVM 17, etc.) to the LLVM version you
just built.

---

## 4. compiler-rt for Vortex

**Purpose**: baremetal `libclang_rt.builtins-riscv{32,64}.a` used
during kernel link to provide `__divdi3`, soft-float helpers, etc.

Build the compiler-rt subdirectory of the Vortex LLVM checkout
twice — once per XLEN.

```bash
cd llvm_vortex
mkdir build_rt && cd build_rt

# Path to the kernel-side runtime archive used by the link step:
export VORTEX_HOME=<path-to-vortex-source>
export VORTEX_BUILD=<path-to-vortex-build>
export RISCV_GCC_TOOLCHAIN=$TOOLDIR/riscv32-gnu-toolchain   # or riscv64-...
```

### 32-bit

```bash
cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/libcrt32 \
    -DCMAKE_AR="$LLVM_PREFIX/bin/llvm-ar" \
    -DCMAKE_LINKER="$LLVM_PREFIX/bin/llvm-lld" \
    -DCMAKE_NM="$LLVM_PREFIX/bin/llvm-nm" \
    -DCMAKE_RANLIB="$LLVM_PREFIX/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER="$LLVM_PREFIX/bin/clang" \
    -DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf" \
    -DCMAKE_C_FLAGS="--gcc-toolchain=$RISCV_GCC_TOOLCHAIN \
        -march=rv32imaf -mabi=ilp32f \
        -Xclang -target-feature -Xclang +vortex \
        -Xclang -target-feature -Xclang +zicond \
        -mcmodel=medany -fno-rtti -fno-exceptions \
        -fdata-sections -ffunction-sections" \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -nostartfiles \
        -Wl,-Bstatic,--gc-sections,-T,$VORTEX_HOME/sw/kernel/scripts/link32.ld,\
--defsym=STARTUP_ADDR=0x80000000 \
        $VORTEX_BUILD/sw/kernel/libvortex.a" \
    -DCMAKE_SYSROOT="$TOOLDIR/riscv32-gnu-toolchain/riscv32-unknown-elf" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCOMPILER_RT_OS_DIR="baremetal" \
    -DCOMPILER_RT_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BAREMETAL_BUILD=ON \
    -DCOMPILER_RT_INCLUDE_TESTS=OFF \
    ../compiler-rt

make -j$(nproc)
make install
```

### 64-bit

Identical to the 32-bit build except for the substitutions:

| 32-bit value | 64-bit value |
|---|---|
| `-DCMAKE_INSTALL_PREFIX=$TOOLDIR/libcrt32` | `$TOOLDIR/libcrt64` |
| `-DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf"` | `riscv64-unknown-elf` |
| `-march=rv32imaf -mabi=ilp32f` | `-march=rv64imafd -mabi=lp64d` |
| `link32.ld` | `link64.ld` |
| `riscv32-gnu-toolchain` (sysroot path) | `riscv64-gnu-toolchain` |
| `riscv32-unknown-elf` (sysroot subdir, default triple) | `riscv64-unknown-elf` |

The output `libclang_rt.builtins-riscv{32,64}.a` lands at
`$TOOLDIR/libcrt{32,64}/lib/baremetal/`.

---

## 5. musl libc for Vortex

**Purpose**: minimal C standard library cross-compiled for
`riscv{32,64}-unknown-elf-vortex`. Provides `libc.a` and `libm.a`
that kernel code links against.

```bash
git clone --recursive https://git.musl-libc.org/git/musl musl-libc
cd musl-libc
git checkout v1.2.5
mkdir build && cd build
```

### 32-bit

```bash
CC=$LLVM_PREFIX/bin/clang \
CFLAGS="--sysroot=$TOOLDIR/riscv32-gnu-toolchain/riscv32-unknown-elf \
        --gcc-toolchain=$TOOLDIR/riscv32-gnu-toolchain \
        -march=rv32imaf -mabi=ilp32f \
        -Xclang -target-feature -Xclang +vortex \
        -Xclang -target-feature -Xclang +zicond \
        -mcmodel=medany -fno-rtti -fno-exceptions \
        -fdata-sections -ffunction-sections \
        -D__riscv_float_abi_single" \
../configure --prefix=$TOOLDIR/libc32 --disable-shared
make -j$(nproc)
make install
```

### 64-bit

```bash
CC=$LLVM_PREFIX/bin/clang \
CFLAGS="--sysroot=$TOOLDIR/riscv64-gnu-toolchain/riscv64-unknown-elf \
        --gcc-toolchain=$TOOLDIR/riscv64-gnu-toolchain \
        -march=rv64imafd -mabi=lp64d \
        -Xclang -target-feature -Xclang +vortex \
        -Xclang -target-feature -Xclang +zicond \
        -mcmodel=medany -fno-rtti -fno-exceptions \
        -fdata-sections -ffunction-sections" \
../configure --prefix=$TOOLDIR/libc64 --disable-shared
make -j$(nproc)
make install
```

`--disable-shared` keeps musl as `.a` archives only — Vortex
kernels are statically linked.

---

## 6. POCL for Vortex

**Purpose**: OpenCL implementation hosting the Vortex device
target. POCL ingests OpenCL-C (and, with SPIR-V enabled,
SPIR-V via `clCreateProgramWithIL`) and dispatches to the
Vortex runtime.

> **Active redesign:** the `vortex_2.x` branch is the current
> baseline. A v3 redesign on top of `upstream/release_6_0` is
> tracked in
> [pocl_vortex_v3_proposal.md](proposals/pocl_vortex_v3_proposal.md);
> consult that proposal for the most current build flags and
> the integration plan with chipStar and the v3 KMU dispatcher.

### Baseline (vortex_2.x) build

```bash
git clone --branch vortex_2.x --recursive https://github.com/vortexgpgpu/pocl
cd pocl
mkdir build && cd build

export POCL_PATH=$TOOLDIR/pocl
export VORTEX_PREFIX=<path-to-vortex-source>

cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$POCL_PATH \
    -DWITH_LLVM_CONFIG=$LLVM_PREFIX/bin/llvm-config \
    -DVORTEX_PREFIX=$VORTEX_PREFIX \
    -DENABLE_VORTEX=ON \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_TESTS=OFF \
    -DKERNEL_CACHE_DEFAULT=OFF \
    -DENABLE_ICD=OFF \
    ..

make -j$(nproc)
make install
cp -r ../include $POCL_PATH    # ship POCL OpenCL headers alongside
```

### Debug build

```bash
cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPOCL_DEBUG_MESSAGES=ON \
    ... (same flags otherwise) ..
```

### v3 redesign (with SPIR-V + chipStar prerequisites)

The redesign branch lives at `pocl_vortex/vortex_3.x` and is
based on `upstream/release_6_0`. Configure flags differ — see
[pocl_vortex_v3_proposal.md §5 Phase 0](proposals/pocl_vortex_v3_proposal.md)
for the canonical recipe. In short:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_VORTEX=ON \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_SPIRV=ON \
    -DENABLE_LOADABLE_DRIVERS=OFF \
    -DWITH_LLVM_CONFIG=$LLVM_PREFIX/bin/llvm-config \
    -DVORTEX_PREFIX=$VORTEX_PREFIX
```

`ENABLE_SPIRV=ON` requires `llvm-spirv` (see
[§3 SPIRV-LLVM-Translator](#spirv-llvm-translator-optional-required-for-chipstar--spir-v-code-path))
to be installed under `$LLVM_PREFIX`.

---

## 7. OpenSTA

**Purpose**: static timing analysis for FPGA / synthesis flows
(see [synthesis_analysis.md](synthesis_analysis.md)).

### Dependency: CUDD

```bash
wget https://github.com/ivmai/cudd/archive/refs/tags/cudd-3.0.0.tar.gz
tar -xzf cudd-3.0.0.tar.gz
cd cudd-cudd-3.0.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/cudd \
    -DUSE_TCL_READLINE=OFF \
    -DTCL_HEADER=/usr/include/tcl8.6/tcl.h \
    -DTCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libtcl8.6.so
make -j$(nproc)
make install
```

### OpenSTA proper

> **TODO**: the OpenSTA build recipe is incomplete in this
> document. The intended install location is
> `$TOOLDIR/sta`, with `-DCUDD_DIR=$TOOLDIR/cudd` pointing at
> the dependency built above. Contributions welcome.

---

## Verifying an installed toolchain

Once components are installed under `$TOOLDIR`, confirm the
expected layout:

```
$TOOLDIR/
├── verilator/           (verilator + bin + install)
├── riscv32-gnu-toolchain/
├── riscv64-gnu-toolchain/
├── llvm-vortex/         (clang, ld.lld, llvm-spirv)
├── libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a
├── libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a
├── libc32/lib/{libc.a,libm.a}
├── libc64/lib/{libc.a,libm.a}
├── pocl/                (libpocl + headers)
└── sta/                 (OpenSTA, optional)
```

The Vortex build script `ci/toolchain_env.sh` exports this layout
into the shell. Confirm a sane environment with:

```bash
$LLVM_PREFIX/bin/clang --version
$LLVM_PREFIX/bin/ld.lld --version
$LLVM_PREFIX/bin/llvm-spirv --version       # if installed
$TOOLDIR/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-gcc --version
$TOOLDIR/verilator/bin/verilator --version
```
