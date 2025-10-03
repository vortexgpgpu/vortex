## MPI With SIMX


### Usage

```
Apptainer> ./ci/blackbox.sh --cores=2 --app=mpi_vecadd --driver=simx --np=4 --args="-n5000"
CONFIGS=-DNUM_CORES=2
Running: CONFIGS="-DNUM_CORES=2" make -C ./ci/../runtime/simx > /dev/null
Running: OPTS="-n5000" NP=4 make -C "./ci/../tests/regression/mpi_vecadd" run-simx
make: Entering directory '/home/vortex/build/tests/regression/mpi_vecadd'
LD_LIBRARY_PATH=/home/vortex/build/runtime:/opt/boost-1.66/lib:/opt/openssl-1.1/lib::/.singularity.d/libs VORTEX_DRIVER=simx  mpirun  --allow-run-as-root --oversubscribe -np 4 ./mpi_vecadd -n5000
rank = 3, world_size = 4
rank = 0, world_size = 4
rank = 1, world_size = 4
rank = 2, world_size = 4
Rank: 3- Upload kernel binary
Rank: 0- Upload kernel binary
Rank: 1- Upload kernel binary
Rank: 2- Upload kernel binary
PERF: core0: instrs=22440, cycles=59003, IPC=0.380320
PERF: core1: instrs=22440, cycles=58635, IPC=0.382707
PERF: instrs=44880, cycles=59003, IPC=0.760639
PERF: core0: instrs=22440, cycles=59003, IPC=0.380320
PERF: core1: instrs=22440, cycles=58635, IPC=0.382707
PERF: instrs=44880, cycles=59003, IPC=0.760639
PERF: core0: instrs=22440, cycles=59003, IPC=0.380320
PERF: core1: instrs=22440, cycles=58635, IPC=0.382707
PERF: instrs=44880, cycles=59003, IPC=0.760639
PASSED!
PERF: core0: instrs=22440, cycles=59003, IPC=0.380320
PERF: core1: instrs=22440, cycles=58635, IPC=0.382707
PERF: instrs=44880, cycles=59003, IPC=0.760639
make: Leaving directory '/home/vortex/build/tests/regression/mpi_vecadd'
Apptainer> 
```


###  High-Level Summary of main.cpp

#### MPI Setup

Calls MPI_Init, gets the rank (MPI_Comm_rank) and world size (MPI_Comm_size).

Each MPI rank prints its rank and total world_size.

#### Argument Parsing

Reads -n <size> from the command line (number of elements in the vector).

Rank 0 parses this value, then broadcasts it to all ranks with MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD).

This ensures every rank sees the same problem size.

#### Data Partitioning

Total work = size elements.

Each rank computes its chunk:

```
local_size = size / world_size;
start = rank * local_size;
end   = start + local_size;
```


So if size=50 and np=8, each rank gets about 6–7 elements.

#### Kernel Upload + Execution

Each rank loads the Vortex kernel binary (mpi_vecadd) into its own Vortex instance.

That’s why you see “Upload kernel binary” printed for every rank, not just once.

Then each rank launches the kernel for its assigned portion of the data.

#### Performance Reporting

After kernel finishes, each rank prints Vortex perf stats (instrs, cycles, IPC).

These numbers are per rank’s Vortex instance, not shared across ranks.


#### Verification

Each rank validates its results (checks that vector addition is correct).

Finally, the ranks synchronize (MPI_Barrier) and finalize (MPI_Finalize).