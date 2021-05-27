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
_______________________________________________________________________________


-program_fpga.sh-

Description: Signs and programs the fpga for a specified core count. Prompts
	     for PACSign are all automatically answered 'yes'.

Usage: ./program_fpga.sh -c [1|2|4|8|16]

Options:
        -c
          Core count (1, 2, 4, 8, or 16).

_______________________________________________________________________________
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

_______________________________________________________________________________
_______________________________________________________________________________


-export_csv.sh-

Description: Creates specified .csv output file from an input directory, file, 
and parameter. The .csv file contains two columns: cores, and the input
parameter. The output file is located within the directory specified with -d.

Usage: ./export_csv.sh -c [cores] -d [directory] -i [input filename] -o
	[output filename] -p '[parameter]'

Example: ./export_csv.sh -c 16 -d perf_2021_03_07 -i sgemm.result -o output.csv
	 -p 'PERF: scoreboard stalls'

Options:
	-c
	  Upper limit of cores to be read in. Core directories should exist in
	  the directory specified by -d e.g. 1c, 2c, 4c for -c 4.

	-d
	  The directory of the form perf_{date} located in the evaluation
	  directory.

	-i
	  The input filename located in each core directory within the
	  directory specified by -d.

	-o
	  The output filename to be created within the directory specified
	  by -d.

	-p
	  The parameter corresponding to the core count in the .csv file. The
	  full name of the parameter from the start of the line should be
	  inputted to avoid the parameter name being matched multiple times.

_______________________________________________________________________________


-export_ipc_csv.sh-

Description: Runs export_csv.sh for the parameter IPC.

Usage: ./export_csv.sh -c [cores] -d [directory] -i [input filename] -o
        [output filename]

Example: ./export_ipc.sh -c 16 -d perf_2021_03_07 -i sgemm.result -o output.csv
