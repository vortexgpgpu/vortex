Conda enviornment setup:
conda install -c conda-forge gxx=11.2.0
conda install -c conda-forge zlib=1.2.11
conda install -c conda-forge libhwloc=2.7.0
conda install -c conda-forge sysroot_linux-64=2.17  ## By default it installs 2.12 woth gxx, which is insufficient for GLIBC2.14 calls

vortex_3 environment is working currently.
vortex_3 project directory is working currently.

Add the conda lib dir to LD_LIBRARY_PATH, example: /volume1/users/nshah/anaconda3/anaconda3/envs/vortex/lib/
Add the conda include dir to CPATH, /volume1/users/nshah/anaconda3/anaconda3/envs/vortex/include/
This is automated by wih activate.d and deactivate.d. See script ~/conda_env_var.sh

IMPORTANT:
Prebuilt POCL of centos is not compiling, probably due to compiler mismatch, as we are using gxx=11.2.0 instead of the deafult 4.8.5 in centos. 
Hence, use the ubuntu prebuilt for POCL.
Also, using the ubunut prebuilt for verilator, as there were no errors. 
Centos prebuilt of verilator also works, but with the following change:
Copy ./verilator/share/verilator/ to ./verilator/

Add the following to the bashrc before making vortex:
export VORTEX_ROOT="/esat/puck1/users/nshah/Software_tools/no_backup/vortex_3/"
export VORTEX_OPT_UBUNTU="/esat/puck1/users/nshah/Software_tools/no_backup/vortex_opt_ubuntu"
export VORTEX_OPT_CENTOS="/esat/puck1/users/nshah/Software_tools/no_backup/vortex_opt_centos"
export RISCV_TOOLCHAIN_PATH="${VORTEX_OPT_CENTOS}/riscv-gnu-toolchain"
export SYSROOT="${RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf"
export VERILATOR_ROOT="${VORTEX_OPT_UBUNTU}/verilator"
export PATH="${VERILATOR_ROOT}/bin:${PATH}"
export LLVM_PREFIX="${VORTEX_OPT_CENTOS}/llvm-riscv"
export POCL_CC_PATH="${VORTEX_OPT_UBUNTU}/pocl/compiler"
export POCL_RT_PATH="${VORTEX_OPT_UBUNTU}/pocl/runtime"

