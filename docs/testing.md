# Testing

## Running a Vortex application

The framework provides a utility script: blackbox.sh under the /ci/ folder for executing applications in the tests tree. It gets copied into the `build` directory with all the environment variables resolved, so you should run it from the `build` directory as follows:
You can query the commandline options of the tool using:

    $ ./ci/blackbox.sh --help

To execute sgemm test program on the simx driver and passing "-n10" as argument to sgemm:

    $ ./ci/blackbox.sh --driver=simx --app=sgemm --args="-n10"

You can execute the same application of a GPU architecture with 2 cores:

    $ ./ci/blackbox.sh --core=2 --driver=simx --app=sgemm --args="-n10"

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

## Creating Your Own Regression Test

Inside `tests/regression` you will find a series of folders which are named based on what they test.
You can view the tests to see which ones have tests similar to what you are trying to create new tests for.
Once you have found a similar baseline, you can copy the folder and rename it to what you are planning to test.
A regression test typically implements the following files:
- ***kernel.cpp*** contains the GPU kernel code.
- ***main.cpp*** contains the host CPU code.
- ***Makefile*** defines the compiler build commands for the CPU and GPU binaries.

Sync your build folder: `$ ../configure`

Compile your test: `$ make -C tests/regression/<test-name>`

Run your test: `$ ./ci/blackbox.sh --driver=simx --app=<test-name> --debug`

## Adding Your Tests to the CI Pipeline
If you are a contributor, then you will need to add tests that integrate into the continuous integration pipeline. Remember, Pull Requests cannot be merged unless new code has tests and existing tests do not regress. Furthermore, if you are contributing a new feature, it is recommended that you add the ability to enable / disable the new feature that you are adding. See more at [contributing.md](contributing.md) and [continuous_integration.md](continuous_integration.md).