# Environment Setup

These instructions apply to the development vortex repo using the updated toolchain. The updated toolchain is considered to be any commit of `master` pulled from July 2, 2023 onwards. The toolchain update in question can be viewed in this [commit](https://github.com/vortexgpgpu/vortex-dev/commit/0048496ba28d7b9a209a0e569d52d60f2b68fc04). Therefore, if you are unsure whether you are using the new toolchain or not, then you should check the `ci` folder for the existence of the `toolchain_prebuilt.sh` script. Furthermore, you should notice that the `toolchain_install.sh` script has the legacy `llvm()` split into `llvm-vortex()` and `llvm-pocl()`.

## Set Up on Your Own System

The toolchain binaries provided with Vortex are built on Ubuntu-based systems. To install Vortex on your own system, [follow these instructions](install_vortex.md).

## Servers for Georgia Tech Students and Collaborators

### Volvo

Volvo is a 64-core server provided by HPArch. You need valid credentials to access it. If you don't already have access, you can get in contact with your mentor to ask about setting your account up.

Setup on Volvo:

1. Connect to Georgia Tech's VPN or ssh into another machine on campus
2. `ssh volvo.cc.gatech.edu`
3. Clone Vortex to your home directory: `git clone --recursive https://github.com/vortexgpgpu/vortex.git`
4. `source /nethome/software/set_vortex_env.sh` to set up the necessary environment variables.
5. `make -s` in the `vortex` root directory
6. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

### Nio

Nio is a 20-core desktop server provided by HPArch. If you have access to Volvo, you also have access to Nio.

Setup on Nio:

1. Connect to Georgia Tech's VPN or ssh into another machine on campus
2. `ssh nio.cc.gatech.edu`
3. Clone Vortex to your home directory: `git clone --recursive https://github.com/vortexgpgpu/vortex.git`
4. `source /opt/set_vortex_env_dev.sh` to set up the necessary environment variables.
5. `make -s` in the `vortex` root directory
6. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

## Docker (Experimental)

Docker allows for isolated pre-built environments to be created, shared and used. The emulation mode required for ARM-based processors will incur a decrease in performance. Currently, the dockerfile is not included with the official vortex repository and is not actively maintained or supported.

### Setup with Docker

1. Clone repo recursively onto your local machine: `git clone --recursive https://github.com/vortexgpgpu/vortex.git`
2. Download the dockerfile from [here](https://github.gatech.edu/gist/usubramanya3/f1bf3e953faa38a6372e1292ffd0b65c) and place it in the root of the repo.
3. Build the Dockerfile into an image: `docker build --platform=linux/amd64 -t vortex -f dockerfile .`
4. Run a container based on the image: `docker run --rm -v ./:/root/vortex/ -it --name vtx-dev --privileged=true --platform=linux/amd64 vortex`
5. Install the toolchain `./ci/toolchain_install.sh --all` (once per container)
6. `make -s` in `vortex` root directory
7. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

You may exit from a container and resume a container you have exited or start a second terminal session `docker exec -it <container-name> bash`
