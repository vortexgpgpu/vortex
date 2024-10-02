# Debugging Vortex GPU

## Testing changes to the RTL or simulator GPU driver.

The Blackbox utility script will not pick up your changes if the h/w configuration is the same as during teh last run.
To force the utility to build the driver, you need pass the --rebuild=1 option when running tests.
Using --rebuild=0 will prevent the rebuild even if the h/w configuration is different from last run.

    $ ./ci/blackbox.sh --driver=simx --app=demo --rebuild=1

## SimX Debugging

SimX cycle-approximate simulator allows faster debugging of Vortex kernels' execution.
The recommended method to enable debugging is to pass the `--debug=<level>` flag to `blackbox` tool when running a program.

    // Running demo program on SimX in debug mode
    $ ./ci/blackbox.sh --driver=simx --app=demo --debug=1

A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (decoded instruction, register states, pipeline states, etc..). You can increase the verbosity of the trace by changing the debug level.

    // Using SimX in debug mode with verbose level 3
    $ ./ci/blackbox.sh --driver=simx --app=demo --debug=3

## RTL Debugging

To debug the processor RTL, you need to use VLSIM or RTLSIM driver. VLSIM simulates the full processor including the AFU command processor (using `/rtl/afu/opae/vortex_afu.sv` as top module). RTLSIM simulates the Vortex processor only (using `/rtl/Vortex.v` as top module).

The recommended method to enable debugging is to pass the `--debug` flag to `blackbox` tool when running a program.

    // Running demo program on the opae simulator in debug mode
    $ TARGET=opaesim ./ci/blackbox.sh --driver=opae --app=demo --debug=1

    // Running demo program on rtlsim in debug mode
    $ ./ci/blackbox.sh --driver=rtlsim --app=demo --debug=1

A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (memory, caches, pipeline, stalls, etc..). A waveform trace `trace.vcd` is also generated in the current directory during the program execution. You can visualize the waveform trace using any tool that can open VCD files (Modelsim, Quartus, Vivado, etc..). [GTKwave] (http://gtkwave.sourceforge.net) is a great open-source scope analyzer that also works with VCD files.

## FPGA Debugging

Debugging the FPGA directly may be necessary to investigate runtime bugs that the RTL simulation cannot catch. We have implemented an in-house scope analyzer for Vortex that works when the FPGA is running. To enable the FPGA scope analyzer, the FPGA bitstream should be built using `SCOPE=1` flag

    & cd /hw/syn/opae
    $ CONFIGS="-DSCOPE=1" TARGET=fpga make

When running the program on the FPGA, you need to pass the `--scope` flag to the `blackbox` tool.

    // Running demo program on FPGA with scope enabled
    $ ./ci/blackbox.sh --driver=fpga --app=demo --scope


A waveform trace `trace.vcd` will be generated in the current directory during the program execution. This trace includes a limited set of signals that are defined in `/hw/scripts/scope.json`. You can expand your signals' selection by updating the json file.

## Analyzing Vortex trace log

When debugging Vortex RTL or SimX Simulator, reading the trace run.log file can be overwhelming when the trace gets really large.
We provide a trace sanitizer tool under ./hw/scripts/trace_csv.py that you can use to convert the large trace into a CSV file containing all the instructions that executed with their source and destination operands.

    $ ./ci/blackbox.sh --driver=rtlsim --app=demo --debug=3 --log=run_rtlsim.log
    $ ./ci/trace_csv.py -trtlsim run_rtlsim.log -otrace_rtlsim.csv

    $ ./ci/blackbox.sh --driver=simx --app=demo --debug=3 --log=run_simx.log
    $ ./ci/trace_csv.py -tsimx run_simx.log -otrace_simx.csv

    $ diff trace_rtlsim.csv trace_simx.csv

The first column in the CSV trace is UUID (universal unique identifier) of the instruction and the content is sorted by the UUID.
You can use the UUID to trace the same instruction running on either the RTL hw or SimX simulator.
This can be very effective if you want to use SimX to debugging your RTL hardware by comparing CSV traces.