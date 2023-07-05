# Testing

## Running a Vortex application

The framework provides a utility script: blakcbox.sh under the /ci/ folder for executing applications in the tests tree.
You can query the commandline options of the tool using:

    $ ./ci/blakcbox.sh --help

To execute sgemm test program on the simx driver and passing "-n10" as argument to sgemm:

    $ ./ci/blakcbox.sh --driver=simx --app=sgemm --args="-n10"

You can execute the same application of a GPU architecture with 2 cores:

    $ ./ci/blakcbox.sh --core=2 --driver=simx --app=sgemm --args="-n10"

When excuting, Blackbox needs to recompile the driver if the desired architecture changes. 
It tracks the latest configuration in a file under the current directory blackbox.<driver>.cache.
To avoid having to rebuild the driver all the time, Blackbox checks if the latest cached configuration matches the current.

## Running Benchmarks

The Vortex test suite is located under the /test/ folder
You can execute the default regression suite by running the following commands at the root folder.

    $ make -C tests/regression run-simx 
    $ make -C tests/regression run-rtlsim

You can execute the default opncl suite by running the following commands at the root folder.

    $ make -C tests/opencl run-simx 
    $ make -C tests/opencl run-rtlsim