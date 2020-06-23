/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#ifndef SHR_UTILS_H
#define SHR_UTILS_H

// *********************************************************************
// Generic utilities for NVIDIA GPU Computing SDK 
// *********************************************************************

// reminders for output window and build log
#ifdef _WIN32
    #pragma message ("Note: including windows.h")
    #pragma message ("Note: including math.h")
    #pragma message ("Note: including assert.h")
#endif

// OS dependent includes
#ifdef _WIN32
    // Headers needed for Windows
    #include <windows.h>
#else
    // Headers needed for Linux
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/time.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdarg.h>
#endif

// Other headers needed for both Windows and Linux
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Un-comment the following #define to enable profiling code in SDK apps
//#define GPU_PROFILING

// Beginning of GPU Architecture definitions
inline int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}
// end of GPU Architecture definitions


// Defines and enum for use with logging functions
// *********************************************************************
#define DEFAULTLOGFILE "SdkConsoleLog.txt"
#define MASTERLOGFILE "SdkMasterLog.csv"
enum LOGMODES 
{
    LOGCONSOLE = 1, // bit to signal "log to console" 
    LOGFILE    = 2, // bit to signal "log to file" 
    LOGBOTH    = 3, // convenience union of first 2 bits to signal "log to both"
    APPENDMODE = 4, // bit to set "file append" mode instead of "replace mode" on open
    MASTER     = 8, // bit to signal master .csv log output
    ERRORMSG   = 16, // bit to signal "pre-pend Error" 
    CLOSELOG   = 32  // bit to close log file, if open, after any requested file write
};
#define HDASHLINE "-----------------------------------------------------------\n"

// Standardized boolean
enum shrBOOL
{
    shrFALSE = 0,
    shrTRUE = 1
};

// Standardized MAX, MIN and CLAMP
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a

// Error and Exit Handling Macros... 
// *********************************************************************
// Full error handling macro with Cleanup() callback (if supplied)... 
// (Companion Inline Function lower on page)
#define shrCheckErrorEX(a, b, c) __shrCheckErrorEX(a, b, c, __FILE__ , __LINE__) 

// Short version without Cleanup() callback pointer
// Both Input (a) and Reference (b) are specified as args
#define shrCheckError(a, b) shrCheckErrorEX(a, b, 0) 

// Standardized Exit Macro for leaving main()... extended version
// (Companion Inline Function lower on page)
#define shrExitEX(a, b, c) __shrExitEX(a, b, c)

// Standardized Exit Macro for leaving main()... short version
// (Companion Inline Function lower on page)
#define shrEXIT(a, b)        __shrExitEX(a, b, EXIT_SUCCESS)

// Simple argument checker macro
#define ARGCHECK(a) if((a) != shrTRUE)return shrFALSE 

// Define for user-customized error handling
#define STDERROR "file %s, line %i\n\n" , __FILE__ , __LINE__

// Function to deallocate memory allocated within shrUtils
// *********************************************************************
extern "C" void shrFree(void* ptr);

// *********************************************************************
// Helper function to log standardized information to Console, to File or to both
//! Examples: shrLogEx(LOGBOTH, 0, "Function A\n"); 
//!         : shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
//! 
//! Automatically opens file and stores handle if needed and not done yet
//! Closes file and nulls handle on request
//! 
//! @param 0 iLogMode: LOGCONSOLE, LOGFILE, LOGBOTH, APPENDMODE, MASTER, ERRORMSG, CLOSELOG.  
//!          LOGFILE and LOGBOTH may be | 'd  with APPENDMODE to select file append mode instead of overwrite mode 
//!          LOGFILE and LOGBOTH may be | 'd  with CLOSELOG to "write and close" 
//!          First 3 options may be | 'd  with MASTER to enable independent write to master data log file
//!          First 3 options may be | 'd  with ERRORMSG to start line with standard error message
//! @param 2 dValue:    
//!          Positive val = double value for time in secs to be formatted to 6 decimals. 
//!          Negative val is an error code and this give error preformatting.
//! @param 3 cFormatString: String with formatting specifiers like printf or fprintf.  
//!          ALL printf flags, width, precision and type specifiers are supported with this exception: 
//!              Wide char type specifiers intended for wprintf (%S and %C) are NOT supported
//!              Single byte char type specifiers (%s and %c) ARE supported 
//! @param 4... variable args: like printf or fprintf.  Must match format specifer type above.  
//! @return 0 if OK, negative value on error or if error occurs or was passed in. 
// *********************************************************************
extern "C" int shrLogEx(int iLogMode, int iErrNum, const char* cFormatString, ...);

// Short version of shrLogEx defaulting to shrLogEx(LOGBOTH, 0, 
// *********************************************************************
extern "C" int shrLog(const char* cFormatString, ...);

// *********************************************************************
// Delta timer function for up to 3 independent timers using host high performance counters 
// Maintains state for 3 independent counters
//! Example: double dElapsedTime = shrDeltaTime(0);
//! 
//! @param 0 iCounterID: Which timer to check/reset. (0, 1, 2)
//! @return delta time of specified counter since last call in seconds.  Otherwise -9999.0 if error
// *********************************************************************
extern "C" double shrDeltaT(int iCounterID);

// Optional LogFileNameOverride function
// *********************************************************************
extern "C" void shrSetLogFileName (const char* cOverRideName);

// Helper function to init data arrays 
// *********************************************************************
extern "C" void shrFillArray(float* pfData, int iSize);

// Helper function to print data arrays 
// *********************************************************************
extern "C" void shrPrintArray(float* pfData, int iSize);

////////////////////////////////////////////////////////////////////////////
//! Find the path for a filename
//! @return the path if succeeded, otherwise 0
//! @param filename        name of the file
//! @param executablePath  optional absolute path of the executable
////////////////////////////////////////////////////////////////////////////
extern "C" char* shrFindFilePath(const char* filename, const char* executablePath);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing single precision floating point data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFilef( const char* filename, float** data, unsigned int* len, 
              bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing double precision floating point data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFiled( const char* filename, double** data, unsigned int* len, 
              bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing integer data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFilei( const char* filename, int** data, unsigned int* len, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing unsigned integer data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is 
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFileui( const char* filename, unsigned int** data, 
               unsigned int* len, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing char / byte data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is 
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFileb( const char* filename, char** data, unsigned int* len, 
              bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Read file \filename containing unsigned char / byte data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//! @note If a NULL pointer is passed to this function and it is
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrReadFileub( const char* filename, unsigned char** data, 
               unsigned int* len, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing single precision floating point 
//! data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFilef( const char* filename, const float* data, unsigned int len,
               const float epsilon, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing double precision floating point 
//! data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFiled( const char* filename, const float* data, unsigned int len,
               const double epsilon, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing integer data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFilei( const char* filename, const int* data, unsigned int len,
               bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing unsigned integer data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFileui( const char* filename, const unsigned int* data, 
                unsigned int len, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing char / byte data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFileb( const char* filename, const char* data, unsigned int len, 
               bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename containing unsigned char / byte data
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the file to write
//! @param data  pointer to data to write
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrWriteFileub( const char* filename, const unsigned char* data,
                unsigned int len, bool verbose = false);

////////////////////////////////////////////////////////////////////////////
//! Load PPM image file (with unsigned char as data element type), padding 
//! 4th component
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param OutData  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
//! 
//! Note: If *OutData is NULL this function allocates buffer that must be freed by caller
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrLoadPPM4ub(const char* file, unsigned char** OutData, 
                             unsigned int *w, unsigned int *h);

////////////////////////////////////////////////////////////////////////////
//! Save PPM image file (with unsigned char as data element type, padded to 
//! 4 bytes)
//! @return shrTRUE if saving the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrSavePPM4ub( const char* file, unsigned char *data, 
               unsigned int w, unsigned int h);

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned char as data element type)
//! @return shrTRUE if saving the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrSavePGMub( const char* file, unsigned char *data, 
              unsigned int w, unsigned int h); 

////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with unsigned char as data element type)
//! @return shrTRUE if saving the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
//! @note If a NULL pointer is passed to this function and it is initialized 
//!       within shrUtils, then free() has to be used to deallocate the memory
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrLoadPGMub( const char* file, unsigned char** data,
                  unsigned int *w,unsigned int *h);

////////////////////////////////////////////////////////////////////////////
// Command line arguments: General notes
// * All command line arguments begin with '--' followed by the token; 
//   token and value are seperated by '='; example --samples=50
// * Arrays have the form --model=[one.obj,two.obj,three.obj] 
//   (without whitespaces)
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
//! Check if command line argument \a flag-name is given
//! @return shrTRUE if command line argument \a flag_name has been given, 
//!         otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param flag_name  name of command line flag
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCheckCmdLineFlag( const int argc, const char** argv, 
                     const char* flag_name);

////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type int
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrGetCmdLineArgumenti( const int argc, const char** argv, 
                        const char* arg_name, int* val);

////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type unsigned int
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrGetCmdLineArgumentu( const int argc, const char** argv, 
                        const char* arg_name, unsigned int* val);

////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type float
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrGetCmdLineArgumentf( const int argc, const char** argv, 
                        const char* arg_name, float* val);

////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type string
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrGetCmdLineArgumentstr( const int argc, const char** argv, 
                          const char* arg_name, char** val);

////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument list those element are strings
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  command line argument list
//! @param len  length of the list / number of elements
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrGetCmdLineArgumentListstr( const int argc, const char** argv, 
                              const char* arg_name, char** val, 
                              unsigned int* len);

////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparef( const float* reference, const float* data,
             const unsigned int len);

////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparei( const int* reference, const int* data, 
             const unsigned int len ); 

////////////////////////////////////////////////////////////////////////////////
//! Compare two unsigned integer arrays, with epsilon and threshold
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param threshold  tolerance % # of comparison errors (0.15f = 15%)
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCompareuit( const unsigned int* reference, const unsigned int* data,
            const unsigned int len, const float epsilon, const float threshold );

////////////////////////////////////////////////////////////////////////////
//! Compare two unsigned char arrays
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCompareub( const unsigned char* reference, const unsigned char* data,
              const unsigned int len ); 

////////////////////////////////////////////////////////////////////////////////
//! Compare two integers with a tolernance for # of byte errors
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//! @param threshold  tolerance % # of comparison errors (0.15f = 15%)
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCompareubt( const unsigned char* reference, const unsigned char* data,
             const unsigned int len, const float epsilon, const float threshold );

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays witha n epsilon tolerance for equality
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCompareube( const unsigned char* reference, const unsigned char* data,
             const unsigned int len, const float epsilon );

////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparefe( const float* reference, const float* data,
              const unsigned int len, const float epsilon );

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality and a 
//!     threshold for # pixel errors
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparefet( const float* reference, const float* data,
             const unsigned int len, const float epsilon, const float threshold );

////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for 
//! equality
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon );

////////////////////////////////////////////////////////////////////////////////
//! Compare two PPM image files with an epsilon tolerance for equality
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param src_file   filename for the image to be compared
//! @param data       filename for the reference data / gold image
//! @param epsilon    epsilon to use for the comparison
//! @param threshold  threshold of pixels that can still mismatch to pass (i.e. 0.15f = 15% must pass)
//! $param verboseErrors output details of image mismatch to std::err
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparePPM( const char *src_file, const char *ref_file, const float epsilon, const float threshold);

////////////////////////////////////////////////////////////////////////////////
//! Compare two PGM image files with an epsilon tolerance for equality
//! @return shrTRUEif \a reference and \a data are identical, otherwise shrFALSE
//! @param src_file   filename for the image to be compared
//! @param data       filename for the reference data / gold image
//! @param epsilon    epsilon to use for the comparison
//! @param threshold  threshold of pixels that can still mismatch to pass (i.e. 0.15f = 15% must pass)
//! $param verboseErrors output details of image mismatch to std::err
////////////////////////////////////////////////////////////////////////////////
extern "C" shrBOOL shrComparePGM( const char *src_file, const char *ref_file, const float epsilon, const float threshold);

extern "C" unsigned char* shrLoadRawFile(const char* filename, size_t size);

extern "C" size_t shrRoundUp(int group_size, int global_size);

// companion inline function for error checking and exit on error WITH Cleanup Callback (if supplied)
// *********************************************************************
inline void __shrCheckErrorEX(int iSample, int iReference, void (*pCleanup)(int), const char* cFile, const int iLine)
{
    if (iReference != iSample)
    {
        shrLogEx(LOGBOTH | ERRORMSG, iSample, "line %i , in file %s !!!\n\n" , iLine, cFile); 
        if (pCleanup != NULL)
        {
            pCleanup(EXIT_FAILURE);
        }
        else 
        {
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(EXIT_FAILURE);
        }
    }
}

// Standardized Exit
// *********************************************************************
inline void __shrExitEX(int argc, const char** argv, int iExitCode)
{
#ifdef WIN32
    if (!shrCheckCmdLineFlag(argc, argv, "noprompt") && !shrCheckCmdLineFlag(argc, argv, "qatest")) 
#else 
    if (shrCheckCmdLineFlag(argc, argv, "prompt") && !shrCheckCmdLineFlag(argc, argv, "qatest")) 
#endif
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "\nPress <Enter> to Quit...\n");                  
        getchar();                                                           
    }       
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", argv[0]); 
    }
    fflush(stderr);                                                         
    exit(iExitCode);
}

#endif