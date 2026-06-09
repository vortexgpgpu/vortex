# Debugging Vortex GPU

## Testing changes to the RTL or simulator GPU driver.

The Blackbox utility script will not pick up your changes if the h/w configuration is the same as the last run.
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

A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (memory, caches, pipeline, stalls, etc..). A waveform trace `trace.vcd` is also generated in the current directory during the program execution.
By default all library modules under the /libs/ folder are excluded from the trace to reduce the waveform file size, you can change that behavior by either explicitly commenting out `TRACING_OFF`/`TRACING_ON` inside a lib module source (e.g. VX_stream_buffer.sv) or simply enabling a full trace by defining TRACING_ALL as follows.

    // Debugging the demo program with rtlsim in full tracing mode
    $ CONFIGS="-DTRACING_ALL" ./ci/blackbox.sh --driver=rtlsim --app=demo --debug=1

You can visualize the waveform trace using any tool that can open VCD files (Modelsim, Quartus, Vivado, etc..). [GTKwave] (http://gtkwave.sourceforge.net) is a great open-source scope analyzer that also works with VCD files.

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

## SimX as Oracle for RTL Debug

When RTL debugging stalls — sparse-mode numerical failures, pipeline races, anything where rtlsim is "close but wrong" and the failure mode does not localize to one module — **don't keep poking at the RTL**. Switch to the SimX-as-oracle strategy:

1. **Build / extend the SimX C++ model so it mirrors the *new* RTL architecture.** Match the actual structural pipeline the RTL implements: same FU boundaries, same uop expansion, same SRAM layout, same metadata flow, same client-port shapes for shared resources. SimX semantics should track the RTL, not the legacy reference.

2. **Get SimX to pass the failing test first.** SimX is orders of magnitude faster than rtlsim and trivially debuggable with a normal C++ debugger. A failing SimX is much cheaper to fix than a failing rtlsim.

3. **Add matching trace dumps on both sides.** Use the same CSV format for both SimX and rtlsim — cycle, UUID, instruction, FU dispatch, SRAM write/read addresses + data, scoreboard hazards. The `trace_csv.py` UUID-sorted format above is the starting point; extend it with module-specific columns (e.g. for TCU: `meta_wr_en`, `meta_wr_wid`, `meta_wr_idx`, `meta_wr_data[0..N-1]`) so SimX and rtlsim emit identical fields.

4. **Diff the traces.** `diff trace_simx.csv trace_rtlsim.csv` — the first divergence is the bug. You're no longer guessing from output values ("actual=9.69 vs expected=10.33"); you're reading the exact UUID + cycle where the RTL disagrees with the oracle.

**When to trigger this pattern:**
- Numerical failures persist across multiple speculative RTL edits.
- The bug spans more than one module (so unit tests can't catch it).
- You're tempted to litter `$display` everywhere — that's the cue to commit to the trace-diff loop instead.

**Why not the other way around** (RTL-as-oracle): the only signals rtlsim gives you out-of-the-box are output values and assertion failures. SimX runs in-process, accepts arbitrary instrumentation, and stops on user-set breakpoints. Use it as the leverage point.

## SAIF trace for power analysis

Use the `--saif` flag to capture switching activity during RTL simulation. The trace.saif file will be generated in the current directory. Use `SAIF_FILE` and `SAIF_INST` argument during synthesis build to generate accurate power report.
