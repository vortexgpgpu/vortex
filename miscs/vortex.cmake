# Vortex environment variables
set(TOOLDIR $ENV{HOME}/tools)
set(VORTEX_HOME $ENV{HOME}/dev/vortex)
set(VORTEX_BUILD $ENV{HOME}/dev/vortex/build)
set(STARTUP_ADDR 0x80000000)

# Set the system name to indicate cross-compiling
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

# Specify the binary utilities
set(CMAKE_AR "${TOOLDIR}/llvm-vortex/bin/llvm-ar")
#set(CMAKE_ASM_COMPILER "${TOOLDIR}/llvm-vortex/bin/llvm-as")
set(CMAKE_LINKER "${TOOLDIR}/llvm-vortex/bin/llvm-lld")
set(CMAKE_NM "${TOOLDIR}/llvm-vortex/bin/llvm-nm")
set(CMAKE_RANLIB "${TOOLDIR}/llvm-vortex/bin/llvm-ranlib")

# Specify the compilers
set(CMAKE_C_COMPILER "${TOOLDIR}/llvm-vortex/bin/clang")
set(CMAKE_CXX_COMPILER "${TOOLDIR}/llvm-vortex/bin/clang++")

# Compiler flags for C and C++
set(CMAKE_C_FLAGS "-v --gcc-toolchain=${TOOLDIR}/riscv-gnu-toolchain -march=rv32imaf -mabi=ilp32f -Xclang -target-feature -Xclang +vortex -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections")
set(CMAKE_CXX_FLAGS "-v --gcc-toolchain=${TOOLDIR}/riscv-gnu-toolchain -march=rv32imaf -mabi=ilp32f -Xclang -target-feature -Xclang +vortex -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections")

# Set the sysroot
set(CMAKE_SYSROOT "${TOOLDIR}/riscv32-gnu-toolchain/riscv32-unknown-elf")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld -nostartfiles -Wl,-Bstatic,--gc-sections,-T,${VORTEX_HOME}/kernel/scripts/link32.ld ${VORTEX_BUILD}/kernel/libvortex.a")

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(HAVE_CXX_ATOMICS64_WITHOUT_LIB True)
set(HAVE_CXX_ATOMICS_WITHOUT_LIB True)

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
