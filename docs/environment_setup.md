# Environment Setup# Vortex Dev Environment Setup
These instructions apply to the development vortex repo using the *updated toolchain*. The updated toolchain is considered to be any commit of `master` pulled from *July 2, 2023* onwards. The toolchain update in question can be viewed in this [commit](https://github.com/vortexgpgpu/vortex-dev/commit/0048496ba28d7b9a209a0e569d52d60f2b68fc04). Therefore, if you are unsure whether you are using the new toolchain or not, then you should check the `ci` folder for the existence of the `toolchain_prebuilt.sh` script. Furthermore, you should notice that the `toolchain_install.sh` script has the legacy `llvm()` split into `llvm-vortex()` and `llvm-pocl()`.

> Note: As it stands right now, there a few test suites which are not working due to this toolchain migration. We are working to determine an exact list of which ones are working and which ones are not. For now, if the repo builds at a minimum, then you can consider all these steps to have worked successfully.

## Choosing an Development Environment
There are three primary environments you can use. Each has its own pros and cons. Refer to this section to help you determine which environment best suits your needs.
1. Volvo
2. Docker
3. Local

### Volvo
Volvo is a server provided by Georgia Tech. As such, it provides high performance compute, but you need valid credentials to access it. If you don't already have credentials, you can get in contact with your mentor to ask about setting your account up.

Pros:

1. Native x86_64 architecture, AMD EPYC 7702P 64-Core Processor (*fast*)
2. Packages and difficult configurations are already done for you
3. Consistent environment as others, allowing for easier troubleshooting
4. Just need to SSH into Volvo, minimal impact on local computer resources
5. VScode remote development tools are phenomenal over SSH

Cons:
1. Volvo is accessed via gatech vpn, external contributors might encounter issues with it -- especially from other university networks
2. Account creation is not immediate and is subject to processing time
3. Volvo might have outtages (*pretty uncommon*)
5. SSH development requires internet and other remote development tools (*vscode works!*)

### Docker

Docker allows for isolated pre-built environments to be created, shared and used. They are much more resource efficient than a Virtual Machine, and have great tooling and support available. The main motivation for Docker is bringing a consistent development environment to your local computer, across all platforms.

Pros:

1. If you are native to x86_64, the container will also run natively, yielding better performance. However, if you have aarch64 (arm) processor, you can still run the Docker container without configuration changes.
2. Consistent environment as others, allowing for easier troubleshooting
3. Works out of the box, just have a working installation of Docker
4. Vortex uses a build system, so once you build the repo once, only new code changes need to be recompiled
5. Docker offers helpful tools and extensions to monitor the performance of your container

Cons:

1. If you are using an arm processor, the container will be run in emulation mode, so it will inherently run slower, as it needs to translate all the x86_64 instructions. It's still usable on Apple Silicon, however.
2. Limited to your computer's performance, and Vortex is a large repo to build
3. Will utilize a few gigabytes of storage on your computer for saving binaries to run the container


### Local
You can reverse engineer the Dockerfile and scripts above to get a working environment setup locally. This option is for experienced users, who have already considered the pros and cons of Volvo and Docker.

## Setup on Volvo
1. Clone Repo Recursively: `git clone --recursive https://github.com/vortexgpgpu/vortex-dev.git`
2. Source `/opt/set_vortex_env_dev.sh` to initialize pre-installed toolchain
3. `make -s` in `vortex-dev` root directory
4. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

## Setup with Docker
Currently the Dockerfile is not included with the official vortex-dev repository, however you can quickly add it to repo and get started.
1. Clone repo recursively onto your local machine: `git clone --recursive https://github.com/vortexgpgpu/vortex-dev.git`
2. Download a copy of `Dockerfile.dev` and place it in the root of the repo.
3. Build the Dockerfile into an image: `docker build --platform=linux/amd64 -t vortex-dev -f Dockerfile.dev .`
4. Run a container based on the image: `docker run --rm -v ./:/root/vortex-dev/ -it --name vtx-dev --privileged=true --platform=linux/amd64 vortex-dev`
5. Install the toolchain `./ci/toolchain_install.sh --all` (once per container)
6. `make -s` in `vortex-dev` root directory
7. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`


### Additional Docker Commands
- Exit from a container (does not stop or remove it)
- Resume a container you have exited or start a second terminal session `docker exec -it <container-name> bash`

