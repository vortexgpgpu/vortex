#!/bin/bash

#!/bin/bash
#SBATCH -Jvortex-test                        # Job name
#SBATCH -N1 --cpus-per-task=4                	 # Number of nodes and CPUs per node required
#SBATCH --mem-per-cpu=4G                         # Memory per core
#SBATCH -t 00:30:00                              # Duration of the job (Ex: 30 mins)
#SBATCH -p rg-nextgen-hpc                        # Partition Name
#SBATCH -o /tools/ci-reports/vortex-test-%j.out   # Combined output and error messages file
#SBATCH -W                                       # Do not exit until the submitted job terminates.

##Add commands here to build and execute
## cd $GITHUB_WORKSPACE
hostname
#This line allows the GH runner to use the module command on the targeted node
source /tools/misc/.read_profile


# Set environment variables with fallback default values.
# The :- means if the variable CONTAINER_IMAGE is not set or is empty, then use the default value vortex.sif
export GIT_REPO_PATH=${GIT_REPO_PATH:-~/USERSCRATCH/vortex}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/projects/tools/x86_64/containers/vortex_micro25.sif}
export DEVNULL_BIND=${DEVNULL_BIND:-~/USERSCRATCH/devnull}
export VORTEX_TOOLCHAIN_PATH=${VORTEX_TOOLCHAIN_PATH:-/projects/tools/x86_64/common-tools/vortex-tools}

# Launch the Apptainer container with the bind mount
apptainer shell --fakeroot --cleanenv --writable-tmpfs \
  --bind /dev/bus/usb,/sys/bus/pci \
  --bind /projects:/projects \
  --bind /lib/firmware:/lib/firmware \
  --bind /opt/xilinx/:/opt/xilinx/ \
  --bind /tools:/tools \
  --bind /netscratch:/netscratch \
  --bind "$VORTEX_TOOLCHAIN_PATH":/home/tools \
  --bind "$GIT_REPO_PATH":/home/vortex \
  "$CONTAINER_IMAGE"
