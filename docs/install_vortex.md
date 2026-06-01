# Installing and Setting Up the Vortex Environment

## Ubuntu 22.04

1. Install the following dependencies:

   ```
   sudo apt-get update
   sudo apt-get install build-essential cmake ccache zlib1g-dev libtinfo-dev libncurses-dev uuid-dev libboost-serialization-dev libpng-dev libhwloc-dev
   ```

   (Optional) for roofline/perf plotting:

   ```
   sudo apt-get install python3-numpy python3-matplotlib
   ```

2. Download the Vortex codebase:

   ```
   git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
   ```

3. Build and install Vortex:

   ```
   $ cd vortex
   $ mkdir -p build
   $ cd build
   $ ../configure --xlen=32 --tooldir=$HOME/tools
   $ ./ci/toolchain_install.sh --all
   $ make -s
   $ make install
   $ export VORTEX_PATH=$(pwd)/install
   $ export PKG_CONFIG_PATH=$VORTEX_PATH/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

   `../configure` writes the full toolchain layout (paths, XCONFIGS,
   tool binaries) into the build dir's Makefiles, so no shell env
   sourcing is required and multiple Vortex trees can coexist on one
   machine without a global `~/.bashrc` clobber.

   `make install` lays out the Vortex SDK under `$VORTEX_PATH` (default
   `<build>/install`, override with `../configure --prefix=...`):

   ```
   $VORTEX_PATH/
   ├── kernel/include/        public device-side headers (vx_*.h)
   ├── kernel/lib<XLEN>/      libvortex2.a + libvortex.a
   ├── runtime/include/       public host-side headers (graphics.h,
   │                           vortex2.h, vortex.h, tensor.h, dxa.h)
   ├── runtime/lib/           libvortex.so + libvortex-{simx,xrt,
   │                           rtlsim,opae}.so
   └── lib/pkgconfig/         vortex-runtime.pc + vortex-kernel.pc
   ```

   Downstream tools (mesa, pocl, chipstar) consume Vortex through
   `$VORTEX_PATH` and `pkg-config` — same shape as the CUDA, ROCm
   and oneAPI SDKs. Source-tree paths (`$VORTEX_HOME`) and build-tree
   paths (`$VORTEX_BUILD_DIR`) are not exposed to those consumers.

## RHEL 8
Note: depending on the system, some of the toolchain may need to be recompiled for non-Ubuntu Linux. The source for the tools can be found [here](https://github.com/vortexgpgpu/).

1. Install the following dependencies:

   ```
   sudo yum install libpng-devel boost boost-devel boost-serialization libuuid-devel opencl-headers hwloc hwloc-devel gmp-devel compat-hwloc1
   ```

   (Optional) for roofline/perf plotting:

   ```
   sudo yum install python3-numpy python3-matplotlib
   ```

2. Upgrade GCC to 11:

   ```
   sudo yum install gcc-toolset-11
   ```

   Multiple gcc versions on Red Hat can be managed with scl

3. Install MPFR 4.2.0:

   Download [the source](https://ftp.gnu.org/gnu/mpfr/) and follow [the installation documentation](https://www.mpfr.org/mpfr-current/mpfr.html#How-to-Install).

4. Download the Vortex codebase:

   ```
   git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
   ```

5. Build and install Vortex

   ```
   $ cd vortex
   $ mkdir -p build
   $ cd build
   $ ../configure --xlen=32 --tooldir=$HOME/tools
   $ ./ci/toolchain_install.sh --all
   $ make -s
   $ make install
   $ export VORTEX_PATH=$(pwd)/install
   $ export PKG_CONFIG_PATH=$VORTEX_PATH/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

   See the Ubuntu section above for the install-tree layout.
