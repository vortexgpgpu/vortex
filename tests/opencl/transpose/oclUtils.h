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
 
#ifndef OCL_UTILS_H
#define OCL_UTILS_H

// *********************************************************************
// Utilities specific to OpenCL samples in NVIDIA GPU Computing SDK 
// *********************************************************************

// Common headers:  Cross-API utililties and OpenCL header
#include "shrUtils.h"

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

// Includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// For systems with CL_EXT that are not updated with these extensions, we copied these
// extensions from <CL/cl_ext.h>
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
  /* cl_nv_device_attribute_query extension - no extension #define since it has no functions */
  #define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
  #define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
  #define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
  #define CL_DEVICE_WARP_SIZE_NV                      0x4003
  #define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
  #define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
  #define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006
#endif

// reminders for build output window and log
#ifdef _WIN32
    #pragma message ("Note: including shrUtils.h")
    #pragma message ("Note: including opencl.h")
#endif

// SDK Revision #
#define OCL_SDKREVISION "7027912"

// Error and Exit Handling Macros... 
// *********************************************************************
// Full error handling macro with Cleanup() callback (if supplied)... 
// (Companion Inline Function lower on page)
#define oclCheckErrorEX(a, b, c) __oclCheckErrorEX(a, b, c, __FILE__ , __LINE__) 

// Short version without Cleanup() callback pointer
// Both Input (a) and Reference (b) are specified as args
#define oclCheckError(a, b) oclCheckErrorEX(a, b, 0) 

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platform ID
//////////////////////////////////////////////////////////////////////////////
extern "C" cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

//////////////////////////////////////////////////////////////////////////////
//! Print info about the device
//!
//! @param iLogMode       enum LOGBOTH, LOGCONSOLE, LOGFILE
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
extern "C" void oclPrintDevInfo(int iLogMode, cl_device_id device);

//////////////////////////////////////////////////////////////////////////////
//! Get and return device capability
//!
//! @return the 2 digit integer representation of device Cap (major minor). return -1 if NA 
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
extern "C" int oclGetDevCap(cl_device_id device);

//////////////////////////////////////////////////////////////////////////////
//! Print the device name
//!
//! @param iLogMode       enum LOGBOTH, LOGCONSOLE, LOGFILE
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
extern "C" void oclPrintDevName(int iLogMode, cl_device_id device);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the first device from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
extern "C" cl_device_id oclGetFirstDev(cl_context cxGPUContext);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxGPUContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
extern "C" cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int device_idx);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of device with maximal FLOPS from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
extern "C" cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext);

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
extern "C" char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

//////////////////////////////////////////////////////////////////////////////
//! Get the binary (PTX) of the program associated with the device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//! @param binary       returned code
//! @param length       length of returned code
//////////////////////////////////////////////////////////////////////////////
extern "C" void oclGetProgBinary( cl_program cpProgram, cl_device_id cdDevice, char** binary, size_t* length);

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram                   OpenCL program
//! @param cdDevice                    device of interest
//! @param const char*  cPtxFileName   optional PTX file name
//////////////////////////////////////////////////////////////////////////////
extern "C" void oclLogPtx(cl_program cpProgram, cl_device_id cdDevice, const char* cPtxFileName);

//////////////////////////////////////////////////////////////////////////////
//! Get and log the Build Log from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//////////////////////////////////////////////////////////////////////////////
extern "C" void oclLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice);

// Helper function for De-allocating cl objects
// *********************************************************************
extern "C" void oclDeleteMemObjs(cl_mem* cmMemObjs, int iNumObjs);

// Helper function to get OpenCL error string from constant
// *********************************************************************
extern "C" const char* oclErrorString(cl_int error);

// Helper function to get OpenCL image format string (channel order and type) from constant
// *********************************************************************
extern "C" const char* oclImageFormatString(cl_uint uiImageFormat);

// companion inline function for error checking and exit on error WITH Cleanup Callback (if supplied)
// *********************************************************************
inline void __oclCheckErrorEX(cl_int iSample, cl_int iReference, void (*pCleanup)(int), const char* cFile, const int iLine)
{
    // An error condition is defined by the sample/test value not equal to the reference
    if (iReference != iSample)
    {
        // If the sample/test value isn't equal to the ref, it's an error by defnition, so override 0 sample/test value
        iSample = (iSample == 0) ? -9999 : iSample; 

        // Log the error info
        shrLog("\n !!! Error # %i (%s) at line %i , in file %s !!!\n\n", iSample, oclErrorString(iSample), iLine, cFile);

        // Cleanup and exit, or just exit if no cleanup function pointer provided.  Use iSample (error code in this case) as process exit code.
        if (pCleanup != NULL)
        {
            pCleanup(iSample);
        }
        else 
        {
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(iSample);
        }
    }
}

#endif