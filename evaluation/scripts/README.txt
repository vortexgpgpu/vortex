-build.sh-

Description: Makes the build in the opae directory with the specified core
	     count and optional performance profiling. If a build already 
	     exists, a make clean command is ran before the build. Script waits
             until the inteldev script or quartus program is finished running.

Usage: ./build.sh -c [1|2|4|8|16] [-p [y|n]]

Options:
	-c
	  Core count (1, 2, 4, 8, or 16).

	-p
	  Performance profiling enable (y or n). Changes the source file in the
	  opae directory to include/exclude "+define+PERF_ENABLE".

_______________________________________________________________________________


-build_all_perf.sh-

Description: Runs build.sh with performance profiling enabled for all valid
	     core configurations.

_______________________________________________________________________________


-program_fpga.sh-

Description: Signs and programs the fpga for a specified core count. Prompts
	     for PACSign are all automatically answered 'yes'.

Usage: ./program_fpga.sh -c [1|2|4|8|16]

Options:
        -c
          Core count (1, 2, 4, 8, or 16).

_______________________________________________________________________________


-gather_perf_results.sh-

Description: Creates directory named perf_YYYY_MM_DD and core subfolders in
	     evaluation. Copies relevant build output files to specified core
	     directory. Runs and redirects outputs of sgemm, vecadd, saxpy,
	     sfilter, nearn, and gaussian benchmarks to specified core
	     directory. Build should already be made before running this.

Usage: ./gather_perf_results.sh -c [1|2|4|8|16]

Options:
        -c
          Core count (1, 2, 4, 8, or 16).

_______________________________________________________________________________


-gather_all_perf_results.sh-

Description: Programs fpga and runs gather_perf_results.sh for all valid core
	     configurations. All builds should already be made before running
	     this.
