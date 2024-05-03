# Installing and Setting Up the Vortex Environment

## Ubuntu 18.04, 20.04

1. Install the following dependencies:

   ```
   sudo apt-get install build-essential zlib1g-dev libtinfo-dev libncurses5 uuid-dev libboost-serialization-dev libpng-dev libhwloc-dev
   ```

2. Upgrade GCC to 11:

   ```
   sudo apt-get install gcc-11 g++-11
   ```
   
   Multiple gcc versions on Ubuntu can be managed with update-alternatives, e.g.:
   
   ```
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
   ```

3. Download the Vortex codebase:

   ```
   git clone --recursive https://github.com/vortexgpgpu/vortex.git
   ```

4. Install Vortex's prebuilt toolchain:

   ```
   $ cd vortex
   
   # By default, the toolchain will install to /opt folder which requires sudo access. Alternatively, you could also install the toolchain to a different location of your choice by setting the TOOLDIR environment variable
    
   $ export TOOLDIR=$HOME/tools    
   $ ./ci/toolchain_install.sh --all
   ```

5. Set up environment:

   ```
   $ source ./ci/toolchain_env.sh
   ```

6. Build Vortex

   ```
   $ make
   ```


## RHEL 8
Note: depending on the system, some of the toolchain may need to be recompiled for non-Ubuntu Linux. The source for the tools can be found [here](https://github.com/vortexgpgpu/).

1. Install the following dependencies:

   ```
   sudo yum install libpng-devel boost boost-devel boost-serialization libuuid-devel opencl-headers hwloc hwloc-devel gmp-devel compat-hwloc1
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
   git clone --recursive https://github.com/vortexgpgpu/vortex.git
   ```

5. Install Vortex's prebuilt toolchain:

   ```
   $ cd vortex
   
   # By default, the toolchain will install to /opt folder which requires sudo access. Alternatively, you could also install the toolchain to a different location of your choice by setting the TOOLDIR environment variable
    
   $ export TOOLDIR=$HOME/tools    
   $ ./ci/toolchain_install.sh --all
   ```

6. Set up environment:

   ```
   $ source ./ci/toolchain_env.sh
   ```

7. Build Vortex

   ```
   $ make
   ```
