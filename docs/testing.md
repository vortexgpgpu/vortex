# Testing

## Running a Vortex application

The framework provides a utility script: blackbox.sh under the /ci/ folder for executing applications in the tests tree.
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

## Creating Your Own Regression Tests
- Inside `test/` you will find a series of folders which are named based on what they test
- You can view the tests to see which ones have tests similar to what you are trying to create new tests for
- once you have found a similar baseline, you can copy the folder and rename it to what you are planning to test
- `testcases.h` contains each of the test case templates
- `main.cpp` contains the implementation of each of the test cases and builds a test suite of all the tests cases you want

Compile the test case: `make -C tests/regression/<testcase-name>/ clean-all && make -C tests/regression/<testcase-name>/`

Run the test case: `./ci/blackbox.sh --driver=simx --cores=4 --app=<testcase-name> --debug`

## Adding Your Tests to the CI Pipeline
see `continuous_integration.md`