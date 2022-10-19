# Create the folder ahead of time so people won't get confused
mkdir -p $HOME/vortex-workspace

export DESTDIR=$HOME/vortex-workspace

# I don't know where the OPAE driver is installed
# But other simulation/driver should work
export OPAE_HOME=/tools/opae/1.4.0

# Need to modify vortex/runtime/makefile
# replace RISCV_TOOLCHAIN_PATH = /opt/riscv-gnu-toolchain with RISCV_TOOLCHAIN_PATH=$DESTDIR/riscv-gnu-toolchain
# in vscode, use search and replace
#   search:         RISCV_TOOLCHAIN_PATH = /opt/(.*)
#   replace with:   RISCV_TOOLCHAIN_PATH=$DESTDIR/$1
export RISCV_TOOLCHAIN_PATH=$DESTDIR/riscv-gnu-toolchain
export LLVM_PREFIX=$DESTDIR/llvm-riscv
export POCL_CC_PATH=$DESTDIR/pocl/compiler
export POCL_RT_PATH=$DESTDIR/pocl/runtime

# Modification to sim/rtlsim/Makefile
# Replace DESTDIR with RTLSIM_DESTDIR

# Modification to sim/vlsim/Makefile
# Replace DESTDIR with VLSIM_DESTDIR

# In their corresponding driver Makefile
# replace DESTDIR with RTLSIM_DESTDIR and VLSIM_DESTDIR

# Add verilator to path
export VERILATOR_ROOT=$DESTDIR/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH
