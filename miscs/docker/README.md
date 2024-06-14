You can install Docker desktop on MAC or PC or Ubuntu.

- PC: https://docs.docker.com/desktop/install/windows-install
- MAC: https://docs.docker.com/desktop/install/mac-install
- Ubuntu: https://docs.docker.com/desktop/install/ubuntu

### 1- Create a Docker image from the Dockerfile
    $ docker build -f Dockerfile.ubuntu -t vortex

### 2- Build the Docker image
    $ docker docker run -it vortex /bin/bash

### 3- Build the project
One you login the Docker terminal, you will be in the build directory.

    $ make -s

### 4- Run a simple test

    $ ./ci/blackbox.sh --cores=2 --app=vecadd