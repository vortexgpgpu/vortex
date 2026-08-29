#!/bin/sh

# Copyright 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install Vortex's system (apt) dependencies.
#
# By default only the baseline packages needed to build and run the Vortex
# core (SimX, RTL sim, the riscv regression suite) are installed. Optional
# feature groups are opt-in via flags so that minimal environments (e.g.
# container base images) don't pull heavy/specialized dependencies they
# don't use:
#
#   install_dependencies.sh              baseline core only
#   install_dependencies.sh --vulkan     + Vulkan loader + glslc  (tests/vulkan)
#   install_dependencies.sh --gem5       + gem5.opt runtime libs  (gem5 tests)
#   install_dependencies.sh --mpi        + OpenMPI                (tests/mpi)
#   install_dependencies.sh --all        baseline + every group
#
# Flags compose: e.g. `--gem5 --mpi` installs both groups. CI workflows
# (ci.yml, apptainer-ci.yml) call this with `--all`.

set -e

usage() {
    echo "Usage: $0 [--vulkan] [--gem5] [--mpi] [--all] [-h|--help]"
    echo "  (no flags)   install baseline packages for the Vortex core only"
    echo "  --vulkan     also install the Vulkan loader and glslc (tests/vulkan)"
    echo "  --gem5       also install gem5.opt's runtime shared libraries"
    echo "  --mpi        also install OpenMPI (tests/mpi)"
    echo "  --all        install baseline + vulkan + gem5 + mpi"
}

enable_vulkan=0
enable_gem5=0
enable_mpi=0

for arg in "$@"; do
    case "$arg" in
        --vulkan) enable_vulkan=1 ;;
        --gem5)   enable_gem5=1 ;;
        --mpi)    enable_mpi=1 ;;
        --all)    enable_vulkan=1; enable_gem5=1; enable_mpi=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Invalid argument: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

# Function to check if GCC version is less than 11
check_gcc_version() {
    local gcc_version
    gcc_version=$(gcc -dumpversion)
    if dpkg --compare-versions "$gcc_version" lt 11; then
        return 0  # GCC version is less than 11
    else
        return 1  # GCC version is 11 or greater
    fi
}

# Update package list
apt-get update -y

###############################################################################
# Baseline — required to build/run the Vortex core and the riscv regressions.
#
# libpng-dev + libboost-serialization-dev are required by third_party/cocogfx
# (src/png.cpp pulls in png.h, src/cgltrace.cpp pulls in boost/serialization).
# opencl-headers ships CL/opencl.h needed by tests/opencl host code; the prebuilt
# POCL tarball doesn't include host-API headers, so this is the system fallback.
# ocl-icd-opencl-dev provides libOpenCL.so symlinks for host-side linking.
###############################################################################
apt-get install -y build-essential valgrind libstdc++6 binutils python3 uuid-dev ccache cmake libffi8 libpng-dev libboost-serialization-dev opencl-headers ocl-icd-opencl-dev

# Check and install GCC 11 if necessary
if check_gcc_version; then
    echo "GCC version is less than 11. Installing GCC 11..."
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get update
    apt-get install -y g++-11 gcc-11
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
else
    echo "GCC version is 11 or greater. No need to install GCC 11."
fi

###############################################################################
# --mpi — OpenMPI provides mpic++ / mpicc and the runtime needed by
# tests/mpi/mpi_* targets (MPI=1). SST builds its own OpenMPI under
# $TOOLDIR/openmpi_install for its internal linkage, but the test recipes
# pick up the system mpic++.
###############################################################################
if [ "$enable_mpi" -eq 1 ]; then
    echo "Installing MPI dependencies..."
    apt-get install -y openmpi-bin libopenmpi-dev
fi

###############################################################################
# --gem5 — gem5.opt runtime shared libraries. gem5.opt is built in the setup
# job (where ci/gem5_install.sh installs the -dev packages: libgoogle-perftools-dev,
# libprotobuf-dev, libhdf5-serial-dev, python3-dev, ...), but the tests job only
# runs this script and then executes the prebuilt gem5.opt. So the *runtime*
# equivalents of everything gem5.opt links must be installed here too, or it
# fails to start with "cannot open shared object file". This list mirrors
# `ldd gem5.opt`'s non-base entries:
#   libtcmalloc-minimal4 -> libtcmalloc_minimal.so.4   (gperftools)
#   libprotobuf23        -> libprotobuf.so.23          (trace protobufs)
#   libhdf5-103-1        -> libhdf5_serial.so.103      (stats HDF5)
#   libhdf5-cpp-103-1    -> libhdf5_serial_cpp.so.103
#   libpython3.10        -> libpython3.10.so.1.0       (embedded interpreter)
# (libaec0/libsz2 arrive transitively via the HDF5 packages.) These are all
# jammy-native, so they ABI-match the gem5.opt built on the same image.
###############################################################################
if [ "$enable_gem5" -eq 1 ]; then
    echo "Installing gem5 runtime dependencies..."
    apt-get install -y libtcmalloc-minimal4 libprotobuf23 libhdf5-103-1 libhdf5-cpp-103-1 libpython3.10
fi

###############################################################################
# --vulkan — the Vulkan loader (libvulkan-dev) plus glslc, the GLSL->SPIR-V
# compiler tests/vulkan uses to build its .spv shaders. glslc ships in
# LunarG's `shaderc` package, which is not in the Ubuntu archive, so add the
# LunarG Vulkan SDK repo — the same source this project's dev machines use.
# Pinned to jammy to match the ubuntu-22.04 CI image; bump alongside the runner OS.
###############################################################################
if [ "$enable_vulkan" -eq 1 ]; then
    echo "Installing Vulkan dependencies..."
    apt-get install -y libvulkan-dev
    # glslc (shaderc) lives only in LunarG's repo. Best-effort so a LunarG
    # outage / signature change can't abort the whole dependency install (set -e).
    if wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc > /dev/null \
       && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list \
       && apt-get update -y; then
        apt-get install -y shaderc || echo "WARNING: shaderc (glslc) install failed" >&2
    else
        echo "WARNING: LunarG repo unavailable; glslc may be missing" >&2
    fi
    # The umbrella header /usr/include/vulkan/vulkan.h is shipped by Ubuntu's
    # libvulkan-dev, but LunarG's libvulkan-dev is loader-only and splits the
    # headers into a separate `vulkan-headers` package (dpkg -S vulkan.h ->
    # vulkan-headers). When this script runs twice per cell (setup-vortex +
    # per-category) the second invocation pulls LunarG's libvulkan-dev with the
    # LunarG repo already configured, so the header vanishes unless
    # vulkan-headers is installed too — the CI v2 vulkan-build regression. Pull
    # it explicitly if the header is absent, then hard-verify.
    if [ ! -f /usr/include/vulkan/vulkan.h ]; then
        apt-get install -y vulkan-headers || true
    fi
    if [ ! -f /usr/include/vulkan/vulkan.h ]; then
        echo "ERROR: vulkan/vulkan.h missing after Vulkan dependency install" >&2
        exit 1
    fi
fi
