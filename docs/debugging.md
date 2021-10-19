# Debugging Vortex Hardware

## SimX Debugging

SimX cycle-approximate simulator allows faster debugging of Vortex kernels' execution. 
The recommended method to enable debugging is to pass the `--debug` flag to `blackbox` tool when running a program.

    // Running demo program on SimX in debug mode
    $ ./ci/blackbox.sh --driver=simx --app=demo --debug 

A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (decoded instruction, register states, pipeline states, etc..). You can increase the verbosity level of the trace by changing the `DEBUG_LEVEL` variable to a value [1-5] (default is 3).

    // Using SimX in debug mode with verbose level 4
    $ CONFIGS=-DDEBUG_LEVEL=4 ./ci/blackbox.sh --driver=simx --app=demo --debug

## RTL Debugging

To debug the processor RTL, you need to use VLSIM or RTLSIM driver. VLSIM simulates the full processor including the AFU command processor (using `/rtl/afu/vortex_afu.sv` as top module). RTLSIM simulates the Vortex processor only (using `/rtl/Vortex.v` as top module).

The recommended method to enable debugging is to pass the `--debug` flag to `blackbox` tool when running a program.

    // Running demo program on vlsim in debug mode
    $ ./ci/blackbox.sh --driver=vlsim --app=demo --debug

    // Running demo program on rtlsim in debug mode
    $ ./ci/blackbox.sh --driver=rtlsim --app=demo --debug

A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (memory, caches, pipeline, stalls, etc..). A waveform trace `trace.vcd` is also generated in the current directory during the program execution. You can visualize the waveform trace using any tool that can open VCD files (Modelsim, Quartus, Vivado, etc..). [GTKwave] (http://gtkwave.sourceforge.net) is a great open-source scope analyzer that also works with VCD files.

## FPGA Debugging

Debugging the FPGA directly may be necessary to investigate runtime bugs that the RTL simulation cannot catch. We have implemented an in-house scope analyzer for Vortex that works when the FPGA is running. To enable the FPGA scope analyzer, the FPGA bitstream should be built using `SCOPE=1` flag

    & cd /hw/syn/opae
    $ CONFIGS=-DSCOPE=1 make fpga-4c

When running the program on the FPGA, you need to pass the `--scope` flag to the `blackbox` tool.

    // Running demo program on FPGA with scope enabled
    $ ./ci/blackbox.sh --driver=fpga --app=demo --scope


A waveform trace `trace.vcd` will be generated in the current directory during the program execution. This trace includes a limited set of signals that are defined in `/hw/scripts/scope.json`. You can expand your signals' selection by updating the json file.