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
7. [chipStar](#7-chipstar-hip-host-runtime) — HIP host runtime layered on POCL
8. [OpenSTA](#8-opensta) — static timing analysis (optional)
9. [Mesa for Vortex](#9-mesa-for-vortex-vulkan) — Vulkan software stack (lavapipe) the Vortex Vulkan driver builds on (optional)

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

> **Targets to build:** the same Clang must serve both host
> compilation (HIP host code via chipStar, plus any C++ host tools)
> and device compilation (Vortex RISC-V kernels). We therefore
> enable **both `RISCV` and `X86`** in `LLVM_TARGETS_TO_BUILD` and
> **enable `lld`** so the host link step has a working `ld.lld`.

> **v3.0 pin:** branch `vortex_3.x` (LLVM 20.1.8, commit
> `87f0227c`). The Vortex-specific `+xvortex` / `+zicond`
> target-feature flags require this branch — upstream LLVM does
> not recognize them.

```bash
git clone --recursive --branch vortex_3.x https://github.com/vortexgpgpu/llvm.git llvm_vortex
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
    ../llvm

make -j$(nproc)
make install
```

### Notes

- **Do not set** `LLVM_DEFAULT_TARGET_TRIPLE` or `DEFAULT_SYSROOT`
  to a RISC-V value. v3.0 leaves the default at the host
  (`x86_64-unknown-linux-gnu`) so that chipStar's `hipcc` and any
  other host-side tool that invokes `clang++` without an explicit
  `--target` keeps working. The Vortex device build passes
  `--target=riscv$(XLEN)-unknown-elf` explicitly (see
  `tests/{kernel,regression,opencl,hip}/common.mk`).
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
git clone --branch llvm_release_200 --depth 1 \
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

Match the translator branch to the LLVM major version you built
(`llvm_release_200` for LLVM 20, `llvm_release_180` for LLVM 18,
etc.). v3.0 uses `llvm_release_200`.

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
        -Xclang -target-feature -Xclang +xvortex \
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

> **Target-feature naming:** v3.0 uses `+xvortex` for the Vortex
> ISA extension. The pre-v3 name `+vortex` is **not** recognized
> by the `vortex_3.x` Clang and will be silently ignored with a
> "not a recognized feature for this target" warning — codegen
> then falls back to plain RISC-V without the Vortex extensions.
> Every consumer (Makefile, build script, CMake) must pass
> `+xvortex`.

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
        -Xclang -target-feature -Xclang +xvortex \
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
        -Xclang -target-feature -Xclang +xvortex \
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

> **v3.0 baseline:** branch `vortex_3.x`, POCL 7.0 derived from
> `upstream/release_6_0`. Includes the SPIR-V ingestion path
> (`clCreateProgramWithIL`) and the
> `cl_ext_buffer_device_address` extension that chipStar's
> `hipMalloc` relies on. See
> [pocl_vortex_v3_proposal.md](proposals/pocl_vortex_v3_proposal.md)
> for the redesign history.

### Build

POCL consumes Vortex through its install tree (`$VORTEX_PATH`) via
`pkg-config`. Run `make install` inside the Vortex build dir first
(default `$VORTEX_BUILD/install`) so `vortex-runtime.pc` /
`vortex-kernel.pc` exist under `$VORTEX_PATH/lib/pkgconfig/`.

```bash
git clone --branch vortex_3.x --recursive https://github.com/vortexgpgpu/pocl
cd pocl
mkdir build && cd build

export POCL_PATH=$TOOLDIR/pocl
export VORTEX_PATH=<path-to-vortex-install-root>     # e.g. <vortex-build>/install
export PKG_CONFIG_PATH=$VORTEX_PATH/lib/pkgconfig:$PKG_CONFIG_PATH

cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$POCL_PATH \
    -DWITH_LLVM_CONFIG=$LLVM_PREFIX/bin/llvm-config \
    -DVORTEX_PATH_64=$VORTEX_PATH \
    -DENABLE_VORTEX=ON \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_SPIRV=ON \
    -DENABLE_LOADABLE_DRIVERS=OFF \
    -DENABLE_TESTS=OFF \
    -DKERNEL_CACHE_DEFAULT=OFF \
    -DENABLE_ICD=OFF \
    ..

make -j$(nproc)
make install

# REQUIRED: ship host-side OpenCL headers alongside the POCL install.
# `make install` with -DENABLE_ICD=OFF does NOT populate $POCL_PATH/include/CL/,
# so without this step tests/opencl host code fails with
# "fatal error: CL/opencl.h: No such file or directory".
cp -r ../include $POCL_PATH
```

`VORTEX_PATH_64` / `VORTEX_PATH_32` select the per-XLEN Vortex
install trees POCL uses to build kernel-side bitcode. Each must
have been produced by `../configure --xlen={32,64}` + `make
install` in its own Vortex build dir. Omitting either skips that
XLEN; setting both builds matching rv32 + rv64 OpenCL kernel
bitcodes.

`ENABLE_SPIRV=ON` requires `llvm-spirv` to be installed under
`$LLVM_PREFIX` (see
[§3 SPIRV-LLVM-Translator](#spirv-llvm-translator-optional-required-for-chipstar--spir-v-code-path)).

### Debug build

```bash
cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPOCL_DEBUG_MESSAGES=ON \
    ... (same flags otherwise) ..
```

---

## 7. chipStar (HIP host runtime)

**Purpose**: translate HIP host calls + SPIR-V device kernels to
OpenCL, so HIP applications can run on POCL/Vortex. Ships
`hipcc`, `hipconfig`, the `CHIP` host library, and SPIR-V helper
tools.

> **v3.0:** chipStar is built with
> `-DCHIP_TARGET_POINTER_WIDTHS="32;64"` so a single `libCHIP.so`
> and `hipcc` serve both rv32 and rv64 Vortex devices. The build
> produces both `hipspv-spirv32.bc` and `hipspv-spirv64.bc` in
> `$TOOLDIR/chipstar/lib/hip-device-lib/`; the runtime picks the
> right rtdevlib SPIR-V variant per device based on
> `CL_DEVICE_ADDRESS_BITS`, and `hipcc --offload-pointer-width={32,64}`
> selects the offload triple per invocation. See
> [chipstar_opencl_32bit_proposal.md](proposals/chipstar_opencl_32bit_proposal.md)
> for the design rationale.
>
> Vortex carries patches against the `HIPCC` and
> `bitcode/ROCm-Device-Libs` submodules (both CHIP-SPV upstream)
> in [`chipStar/HIPCC-patches/`](https://github.com/vortexgpgpu/chipStar/tree/vortex_3.x/HIPCC-patches)
> and [`chipStar/ROCm-Device-Libs-patches/`](https://github.com/vortexgpgpu/chipStar/tree/vortex_3.x/ROCm-Device-Libs-patches).
> chipStar's top-level `CMakeLists.txt` runs an idempotent
> `apply_submodule_patches()` step right after the
> `git submodule update --init` presence check, so the patches
> land automatically at CMake configure time — same shape as
> [`chipStar/llvm-patches/`](https://github.com/vortexgpgpu/chipStar/tree/vortex_3.x/llvm-patches)
> for Clang patches that haven't been upstreamed yet.

```bash
git clone --branch vortex_3.x --recursive https://github.com/vortexgpgpu/chipStar.git
cd chipStar
mkdir build && cd build

# chipStar's FindLLVM probe invokes `llvm-spirv` to detect the SPIR-V
# translator version; that binary dlopens libLLVMPasses.so from the
# Vortex LLVM build, so $LLVM_PREFIX/lib must be on LD_LIBRARY_PATH
# during the cmake step (it is not at first-time install).
export LD_LIBRARY_PATH=$LLVM_PREFIX/lib:$LD_LIBRARY_PATH

cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$TOOLDIR/chipstar \
    -DLLVM_CONFIG_BIN=$LLVM_PREFIX/bin/llvm-config \
    -DCMAKE_C_COMPILER=$LLVM_PREFIX/bin/clang \
    -DCMAKE_CXX_COMPILER=$LLVM_PREFIX/bin/clang++ \
    -DCHIP_TARGET_POINTER_WIDTHS="32;64" \
    -DCHIP_BUILD_TESTS=OFF \
    -DCHIP_BUILD_DOCS=OFF \
    ..

make -j$(nproc)
make install
```

chipStar does not consume Vortex directly — it links OpenCL via
POCL, so it picks up the Vortex device transitively through
`$TOOLDIR/pocl`. Rebuild chipStar after any POCL re-install to
keep the OpenCL ABI in lockstep.

After install, `$TOOLDIR/chipstar/bin/hipcc` drives a HIP build
exactly as on AMD/NVIDIA hosts, but offloads to the POCL/Vortex
OpenCL device.

### Notes

- `hipcc` writes absolute paths into its launcher scripts based
  on `CMAKE_INSTALL_PREFIX` at install time. If you ever move
  `$TOOLDIR/chipstar/` to a different path, re-install rather
  than symlinking — the bin/ launchers will break.
- chipStar shares the same `clang++` as device-side LLVM (§3),
  which is why §3 enables both `RISCV` and `X86` targets.

---

## 8. OpenSTA

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

## 9. Mesa for Vortex (Vulkan)

**Purpose**: Mesa's `lavapipe` Vulkan driver (with the `llvmpipe`
Gallium driver underneath) — the software Vulkan stack the Vortex
Vulkan driver is built on. The `vortexpipe` Gallium driver
(Vulkan-on-Vortex) is developed inside this Mesa fork; see
[proposals/vulkan_support_proposal.md](proposals/vulkan_support_proposal.md).

> **Prebuilt path (recommended):** Mesa-with-`vortexpipe` is a
> prebuilt toolchain component. `ci/toolchain_install.sh --mesa`
> (folded into `--all`) fetches `$TOOLDIR/mesa-vortex` from the
> `vortex-toolchain-prebuilt` release — no Mesa build needed.
> Tests pick the install up via the `MESA_PATH` make var declared in
> `tests/vulkan/common.mk` (defaults to `$(TOOLDIR)/mesa-vortex`).

> **Build-from-source path:** `ci/mesa_install.sh` performs every
> step below (build deps, meson configure, build, install) and is
> the *producer* of the `mesa-vortex` prebuilt — run once per OS by
> the toolchain maintainer, then packaged by
> `ci/toolchain_prebuilt.sh --mesa`. This section documents the
> manual build for maintainers.

> **v3.0 pin:** branch `vortex_3.x` of
> `github.com/vortexgpgpu/mesa` — a fork of upstream Mesa at tag
> `mesa-25.1.0`, carrying the `vortexpipe` Gallium driver. Built
> with `gallium-drivers=llvmpipe,vortexpipe`.

### Dependencies

Mesa needs a current `meson` (≥ 1.4 — the Ubuntu 22.04 distro
`meson` 0.61 is too old) and extra system packages beyond the
[Prerequisites](#prerequisites) list:

```bash
python3 -m pip install --user --upgrade 'meson>=1.4.0'
export PATH=$HOME/.local/bin:$PATH

sudo apt-get install \
    python3-mako flex bison pkg-config \
    libdrm-dev libexpat1-dev zlib1g-dev libzstd-dev libelf-dev \
    libwayland-dev wayland-protocols \
    libx11-dev libxext-dev libxshmfence-dev libxrandr-dev libxfixes-dev \
    libxcb1-dev libxcb-glx0-dev libxcb-shm0-dev libxcb-dri2-0-dev \
    libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libxcb-xfixes0-dev
```

No separate host LLVM is built — Mesa's `llvmpipe` reuses the
Vortex LLVM from [§3](#3-llvm-for-vortex) (`$LLVM_PREFIX`,
LLVM 20.1.8, built with both `X86` and `RISCV`).

### Build

Mesa consumes Vortex through its install tree (`$VORTEX_PATH`) via
`pkg-config` — same shape as POCL above. Run `make install` inside
the Vortex build dir first so `vortex-runtime.pc` is on
`PKG_CONFIG_PATH`.

```bash
git clone --branch vortex_3.x https://github.com/vortexgpgpu/mesa.git mesa_vortex
cd mesa_vortex

# Put the Vortex LLVM's llvm-config first so meson selects it.
export PATH=$HOME/.local/bin:$LLVM_PREFIX/bin:$PATH
export VORTEX_PATH=<path-to-vortex-install-root>
export PKG_CONFIG_PATH=$VORTEX_PATH/lib/pkgconfig:$PKG_CONFIG_PATH

meson setup build \
    --prefix=$TOOLDIR/mesa-vortex \
    --libdir=lib \
    --buildtype=release \
    -D cpp_rtti=false \
    -D gallium-drivers=llvmpipe,vortexpipe \
    -D vulkan-drivers=swrast \
    -D platforms=x11,wayland \
    -D llvm=enabled \
    -D video-codecs= \
    -D gallium-extra-hud=false \
    -D vortex-path=$VORTEX_PATH \
    -D vortex-tooldir=$TOOLDIR

meson compile -C build
meson install -C build
```

`vortexpipe` resolves `libvortex.so` at meson-configure time, so
build the Vortex runtime stub first: `make -C $VORTEX_HOME/build/sw/runtime/stub`.

### Notes

- **`-D cpp_rtti=false`** — Mesa's C++ RTTI setting must match
  LLVM's; `llvm-vortex` is built without RTTI ([§3](#3-llvm-for-vortex),
  no `LLVM_ENABLE_RTTI`). `ci/mesa_install.sh` auto-detects this
  from `llvm-config --cxxflags`.
- **Driver names:** `gallium-drivers=llvmpipe,vortexpipe` — both
  Gallium drivers; `vulkan-drivers=swrast` is lavapipe. `vortexpipe`
  is selected at run time with `GALLIUM_DRIVER=vortexpipe`; the ICD
  stays `lvp_icd.x86_64.json` (no separate Vortex ICD).
- **zstd:** if `libzstd-dev` is unavailable, build zstd from
  source and add it to `PKG_CONFIG_PATH`. zstd bakes `prefix=`
  into its installed `libzstd.pc` at build time — fix that line
  to the real install path or pkg-config resolves the wrong
  headers.
- **Runtime env:** the Vulkan loader finds the driver via
  `VK_ICD_FILENAMES`; `LD_LIBRARY_PATH` must include both
  `mesa-vortex/lib` and `llvm-vortex/lib` (the ICD links
  `libLLVM` shared):

  ```bash
  export VK_ICD_FILENAMES=$TOOLDIR/mesa-vortex/share/vulkan/icd.d/lvp_icd.x86_64.json
  export LD_LIBRARY_PATH=$TOOLDIR/mesa-vortex/lib:$LLVM_PREFIX/lib:$LD_LIBRARY_PATH
  ```

---

## Verifying an installed toolchain

Once components are installed under `$TOOLDIR`, confirm the
expected layout:

```
$TOOLDIR/
├── verilator/                  (verilator + bin + install)
├── riscv32-gnu-toolchain/
├── riscv64-gnu-toolchain/
├── llvm-vortex/                (clang, ld.lld, llvm-spirv; LLVM 20.1.8)
├── libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a
├── libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a
├── libc32/lib/{libc.a,libm.a}
├── libc64/lib/{libc.a,libm.a}
├── pocl/
│   ├── bin/ etc/ lib/libOpenCL.so* share/pocl/
│   └── include/CL/*.h          (REQUIRED — see §6; absence breaks tests/opencl)
├── chipstar/                   (bin/hipcc, lib/libCHIP.so, include/hip/)
├── sta/                        (OpenSTA, optional)
└── mesa-vortex/                (lavapipe Vulkan ICD; share/vulkan/icd.d/, optional)
```

The Vortex build's `config.mk` + per-domain `common.mk` files (e.g.
`tests/vulkan/common.mk`, `hw/syn/common.mk`) bake the paths into
every recipe — no shell env sourcing required. Confirm a sane
toolchain install with:

```bash
$LLVM_PREFIX/bin/clang --version            # expect "clang version 20.1.8"
$LLVM_PREFIX/bin/ld.lld --version
$LLVM_PREFIX/bin/llvm-spirv --version       # if SPIRV-LLVM-Translator installed
$TOOLDIR/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-gcc --version
$TOOLDIR/verilator/bin/verilator --version
$TOOLDIR/chipstar/bin/hipcc --version       # if chipStar installed
test -f $TOOLDIR/pocl/include/CL/opencl.h && echo "POCL host headers OK"

# if Mesa installed — expect a device line "llvmpipe (LLVM 20.1.8, ...)"
VK_ICD_FILENAMES=$TOOLDIR/mesa-vortex/share/vulkan/icd.d/lvp_icd.x86_64.json \
LD_LIBRARY_PATH=$TOOLDIR/mesa-vortex/lib:$TOOLDIR/llvm-vortex/lib \
    vulkaninfo --summary
```

---

## Packaging and installing prebuilt bundles

The toolchain is distributed as prebuilt `.tar.bz2` bundles in the
[vortex-toolchain-prebuilt](https://github.com/vortexgpgpu/vortex-toolchain-prebuilt)
repository. Two scripts manage them; both are generated by `configure`
from `ci/*.sh.in` templates — with `$TOOLDIR`, `$OSVERSION`, and
`$TOOLCHAIN_REV` substituted in — so the runnable copies live under
`ci/` in the Vortex build directory.

Both accept the same per-component flags:

```
--pocl  --chipstar  --verilator  --riscv32  --riscv64  --llvm --mesa
--libcrt32  --libcrt64  --libc32  --libc64  --sv2v  --yosys  --sta
--all      (every component)
```

### `ci/toolchain_prebuilt.sh` — package a built toolchain

Run this **after** building the components above (or after rebuilding
just one — e.g. POCL). For each selected component it `tar`s the
component directory out of `$TOOLDIR`, bzip2-compresses it, and — for
the large ones (`riscv32`, `riscv64`, `llvm`, `verilator`, `yosys`) —
`split`s the archive into 50 MB parts to stay under GitHub's per-file
limit. The bundles are written **relative to the current directory**,
into `./<component>/$OSVERSION/` (`libc*` / `libcrt*` go straight into
`./<component>/`, with no `$OSVERSION`), so run it from a checkout of
the prebuilt repo:

```bash
git clone https://github.com/vortexgpgpu/vortex-toolchain-prebuilt.git
cd vortex-toolchain-prebuilt

# repackage just POCL and chipStar after rebuilding them:
/path/to/vortex/build/ci/toolchain_prebuilt.sh --pocl --chipstar

# or the whole toolchain:
/path/to/vortex/build/ci/toolchain_prebuilt.sh --all
```

Then `git add` / `git commit` the regenerated bundles in that repo and
push, moving the release tag that `$TOOLCHAIN_REV` pins.

### `ci/toolchain_install.sh` — install the prebuilt toolchain

The end-user path, and the fast alternative to building from source.
For each selected component it `wget`s the bundle (reassembling any
split parts) from the `vortex-toolchain-prebuilt` repo at the
`$TOOLCHAIN_REV` revision, extracts it, and installs it into
`$TOOLDIR`, replacing any existing copy. Run it from the Vortex build
directory:

```bash
# install the full toolchain:
./ci/toolchain_install.sh --all

# or refresh a single component (e.g. after a new POCL release):
./ci/toolchain_install.sh --pocl
```

`$TOOLCHAIN_REV` (fixed at `configure` time) selects which release of
the prebuilt repo to pull; `$OSVERSION` selects the matching per-OS
bundle.
