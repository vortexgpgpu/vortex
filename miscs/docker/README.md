You can install Docker desktop on MAC or PC or Ubuntu.

- PC: https://docs.docker.com/desktop/install/windows-install
- MAC: https://docs.docker.com/desktop/install/mac-install
- Ubuntu: https://docs.docker.com/desktop/install/ubuntu

### 1- Build a Docker Image from the Dockerfile
    $ docker build --platform=linux/amd64 -t vortex-packaged -f Dockerfile.prod .

### 2- Construct and run a Container from the Docker Image
    $ docker run -it --name vortex --privileged=true --platform=linux/amd64 vortex-packaged

### 3- Build the Project
One you login the Docker terminal, you will be in the build directory.

    $ make -s

### 4- Run a Simple Test
See `docs/` to learn more!

    $ ./ci/blackbox.sh --cores=2 --app=vecadd

### 5- Exit the Container
    
    $ exit
    $ docker stop vortex

### 6- Restart and Re-Enter the Container
If you ran step `2` and then step `5` then, you have to start and re-enter the container
    
    $ docker start vortex
    $ docker exec -it vortex

---
Note: Apple Silicon macs will run the container in emulation mode, so compiling and running will take a considerable amount of time -- but it still works!