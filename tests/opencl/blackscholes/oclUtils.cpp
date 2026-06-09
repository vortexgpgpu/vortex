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

// *********************************************************************
// Utilities specific to OpenCL samples in NVIDIA GPU Computing SDK 
// *********************************************************************

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdarg.h>
#include "oclUtils.h"

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        return -1000;
    }
    else 
    {
        if(num_platforms == 0)
        {
            shrLog("No OpenCL platform found!\n\n");
            return -2000;
        }
        else 
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL)
            {
                shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////
//! Print the device name
//!
//! @param iLogMode       enum LOGBOTH, LOGCONSOLE, LOGFILE
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
void oclPrintDevName(int iLogMode, cl_device_id device)
{
    char device_string[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    shrLogEx(iLogMode, 0, "%s", device_string);
}

//////////////////////////////////////////////////////////////////////////////
//! Print info about the device
//!
//! @param iLogMode       enum LOGBOTH, LOGCONSOLE, LOGFILE
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
void oclPrintDevInfo(int iLogMode, cl_device_id device)
{
    char device_string[1024];
    bool nv_device_attibute_query = false;

    // CL_DEVICE_NAME
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_NAME: \t\t\t%s\n", device_string);

    // CL_DEVICE_VENDOR
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

    // CL_DRIVER_VERSION
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    shrLogEx(iLogMode, 0, "  CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

    // CL_DEVICE_VERSION
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_VERSION: \t\t\t%s\n", device_string);

    // CL_DEVICE_OPENCL_C_VERSION (if CL_DEVICE_VERSION version > 1.0)
    if(strncmp("OpenCL 1.0", device_string, 10) != 0) 
    {
        // This code is unused for devices reporting OpenCL 1.0, but a def is needed anyway to allow compilation using v 1.0 headers 
        // This constant isn't #defined in 1.0
        #ifndef CL_DEVICE_OPENCL_C_VERSION
            #define CL_DEVICE_OPENCL_C_VERSION 0x103D   
        #endif

        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_string), &device_string, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_OPENCL_C_VERSION: \t\t%s\n", device_string);
    }

    // CL_DEVICE_TYPE
    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if( type & CL_DEVICE_TYPE_CPU )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
    if( type & CL_DEVICE_TYPE_GPU )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
    if( type & CL_DEVICE_TYPE_ACCELERATOR )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
    if( type & CL_DEVICE_TYPE_DEFAULT )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");
    
    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

	// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    size_t workitem_dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", workitem_dims);

    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);
    
    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t workgroup_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", workgroup_size);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

    // CL_DEVICE_ADDRESS_BITS
    cl_uint addr_bits;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

    // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    cl_ulong max_mem_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

    // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

    // CL_DEVICE_ERROR_CORRECTION_SUPPORT
    cl_bool error_correction_support;
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

    // CL_DEVICE_LOCAL_MEM_TYPE
    cl_device_local_mem_type local_mem_type;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

    // CL_DEVICE_LOCAL_MEM_SIZE
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

    // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

    // CL_DEVICE_QUEUE_PROPERTIES
    cl_command_queue_properties queue_properties;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
    if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");    
    if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
        shrLogEx(iLogMode, 0, "  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

    // CL_DEVICE_IMAGE_SUPPORT
    cl_bool image_support;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

    // CL_DEVICE_MAX_READ_IMAGE_ARGS
    cl_uint max_read_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

    // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
    cl_uint max_write_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);

    // CL_DEVICE_SINGLE_FP_CONFIG
    cl_device_fp_config fp_config;
    clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, NULL);
    shrLogEx(iLogMode, 0, "  CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s%s%s%s%s%s\n",
        fp_config & CL_FP_DENORM ? "denorms " : "",
        fp_config & CL_FP_INF_NAN ? "INF-quietNaNs " : "",
        fp_config & CL_FP_ROUND_TO_NEAREST ? "round-to-nearest " : "",
        fp_config & CL_FP_ROUND_TO_ZERO ? "round-to-zero " : "",
        fp_config & CL_FP_ROUND_TO_INF ? "round-to-inf " : "",
        fp_config & CL_FP_FMA ? "fma " : "");
    
    // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
    size_t szMaxDims[5];
    shrLogEx(iLogMode, 0, "\n  CL_DEVICE_IMAGE <dim>"); 
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
    shrLogEx(iLogMode, 0, "\t\t\t2D_MAX_WIDTH\t %u\n", szMaxDims[0]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
    shrLogEx(iLogMode, 0, "\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", szMaxDims[1]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
    shrLogEx(iLogMode, 0, "\t\t\t\t\t3D_MAX_WIDTH\t %u\n", szMaxDims[2]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
    shrLogEx(iLogMode, 0, "\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", szMaxDims[3]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
    shrLogEx(iLogMode, 0, "\t\t\t\t\t3D_MAX_DEPTH\t %u\n", szMaxDims[4]);

    // CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the string onto separate lines
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_string), &device_string, NULL);
    if (device_string != 0) 
    {
        shrLogEx(iLogMode, 0, "\n  CL_DEVICE_EXTENSIONS:");
        std::string stdDevString;
        stdDevString = std::string(device_string);
        size_t szOldPos = 0;
        size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
        while (szSpacePos != stdDevString.npos)
        {
            if( strcmp("cl_nv_device_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 )
                nv_device_attibute_query = true;

            if (szOldPos > 0)
            {
                shrLogEx(iLogMode, 0, "\t\t");
            }
            shrLogEx(iLogMode, 0, "\t\t\t%s\n", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str());
            
            do {
                szOldPos = szSpacePos + 1;
                szSpacePos = stdDevString.find(' ', szOldPos);
            } while (szSpacePos == szOldPos);
        }
        shrLogEx(iLogMode, 0, "\n");
    }
    else 
    {
        shrLogEx(iLogMode, 0, "  CL_DEVICE_EXTENSIONS: None\n");
    }

    if(nv_device_attibute_query) 
    {
        cl_uint compute_capability_major, compute_capability_minor;
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &compute_capability_major, NULL);
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &compute_capability_minor, NULL);
        shrLogEx(iLogMode, 0, "\n  CL_DEVICE_COMPUTE_CAPABILITY_NV:\t%u.%u\n", compute_capability_major, compute_capability_minor);        

        shrLogEx(iLogMode, 0, "  NUMBER OF MULTIPROCESSORS:\t\t%u\n", compute_units); // this is the same value reported by CL_DEVICE_MAX_COMPUTE_UNITS
        shrLogEx(iLogMode, 0, "  NUMBER OF CUDA CORES:\t\t\t%u\n", ConvertSMVer2Cores(compute_capability_major, compute_capability_minor) * compute_units);

        cl_uint regs_per_block;
        clGetDeviceInfo(device, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(cl_uint), &regs_per_block, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_REGISTERS_PER_BLOCK_NV:\t%u\n", regs_per_block);        

        cl_uint warp_size;
        clGetDeviceInfo(device, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &warp_size, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_WARP_SIZE_NV:\t\t%u\n", warp_size);        

        cl_bool gpu_overlap;
        clGetDeviceInfo(device, CL_DEVICE_GPU_OVERLAP_NV, sizeof(cl_bool), &gpu_overlap, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_GPU_OVERLAP_NV:\t\t%s\n", gpu_overlap == CL_TRUE ? "CL_TRUE" : "CL_FALSE");        

        cl_bool exec_timeout;
        clGetDeviceInfo(device, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &exec_timeout, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:\t%s\n", exec_timeout == CL_TRUE ? "CL_TRUE" : "CL_FALSE");        

        cl_bool integrated_memory;
        clGetDeviceInfo(device, CL_DEVICE_INTEGRATED_MEMORY_NV, sizeof(cl_bool), &integrated_memory, NULL);
        shrLogEx(iLogMode, 0, "  CL_DEVICE_INTEGRATED_MEMORY_NV:\t%s\n", integrated_memory == CL_TRUE ? "CL_TRUE" : "CL_FALSE");        
    }

    // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
    shrLogEx(iLogMode, 0, "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t"); 
    cl_uint vec_width [6];
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
    shrLogEx(iLogMode, 0, "CHAR %u, SHORT %u, INT %u, LONG %u, FLOAT %u, DOUBLE %u\n\n\n", 
           vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4], vec_width[5]); 
}

//////////////////////////////////////////////////////////////////////////////
//! Get and return device capability
//!
//! @return the 2 digit integer representation of device Cap (major minor). return -1 if NA 
//! @param device         OpenCL id of the device
//////////////////////////////////////////////////////////////////////////////
int oclGetDevCap(cl_device_id device)
{
    char cDevString[1024];
    bool bDevAttributeQuery = false;
    int iDevArch = -1;

    // Get device extensions, and if any then search for cl_nv_device_attribute_query
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(cDevString), &cDevString, NULL);
    if (cDevString != 0) 
    {
        std::string stdDevString;
        stdDevString = std::string(cDevString);
        size_t szOldPos = 0;
        size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
        while (szSpacePos != stdDevString.npos)
        {
            if( strcmp("cl_nv_device_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 )
            {
                bDevAttributeQuery = true;
            }
            
            do {
                szOldPos = szSpacePos + 1;
                szSpacePos = stdDevString.find(' ', szOldPos);
            } while (szSpacePos == szOldPos);
        }
    }

    // if search succeeded, get device caps
    if(bDevAttributeQuery) 
    {
        cl_int iComputeCapMajor, iComputeCapMinor;
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), (void*)&iComputeCapMajor, NULL);
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), (void*)&iComputeCapMinor, NULL);
        iDevArch = (10 * iComputeCapMajor) + iComputeCapMinor;
    }

    return iDevArch;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the first device from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetFirstDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of device with maximal FLOPS from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
    int max_flops = 0;
    
    size_t current_device = 0;
    
    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    
    max_flops = compute_units * clock_frequency;
    ++current_device;

    while( current_device < device_count )
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
        
        int flops = compute_units * clock_frequency;
        if( flops > max_flops )
        {
            max_flops        = flops;
            max_flops_device = cdDevices[current_device];
        }
        ++current_device;
    }

    free(cdDevices);

    return max_flops_device;
}

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxGPUContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int nr)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    
    if( szParmDataBytes / sizeof(cl_device_id) <= nr ) {
      return (cl_device_id)-1;
    }
    
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    
    cl_device_id device = cdDevices[nr];
    free(cdDevices);

    return device;
}

//////////////////////////////////////////////////////////////////////////////
//! Get the binary (PTX) of the program associated with the device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//! @param binary       returned code
//! @param length       length of returned code
//////////////////////////////////////////////////////////////////////////////
void oclGetProgBinary( cl_program cpProgram, cl_device_id cdDevice, char** binary, size_t* length)
{
    // Grab the number of devices associated witht the program
    cl_uint num_devices;
    clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // Grab the device ids
    cl_device_id* devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    clGetProgramInfo(cpProgram, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // Grab the sizes of the binaries
    size_t* binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));    
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // Now get the binaries
    char** ptx_code = (char**) malloc(num_devices * sizeof(char*));
    for( unsigned int i=0; i<num_devices; ++i) {
        ptx_code[i]= (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);

    // Find the index of the device of interest
    unsigned int idx = 0;
    while( idx<num_devices && devices[idx] != cdDevice ) ++idx;
    
    // If it is associated prepare the result
    if( idx < num_devices )
    {
        *binary = ptx_code[idx];
        *length = binary_sizes[idx];
    }

    // Cleanup
    free( devices );
    free( binary_sizes );
    for( unsigned int i=0; i<num_devices; ++i) {
        if( i != idx ) free(ptx_code[i]);
    }
    free( ptx_code );
}

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram                   OpenCL program
//! @param cdDevice                    device of interest
//! @param const char*  cPtxFileName   optional PTX file name
//////////////////////////////////////////////////////////////////////////////
void oclLogPtx(cl_program cpProgram, cl_device_id cdDevice, const char* cPtxFileName)
{
    // Grab the number of devices associated with the program
    cl_uint num_devices;
    clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // Grab the device ids
    cl_device_id* devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    clGetProgramInfo(cpProgram, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // Grab the sizes of the binaries
    size_t* binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));    
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // Now get the binaries
    char** ptx_code = (char**)malloc(num_devices * sizeof(char*));
    for( unsigned int i=0; i<num_devices; ++i)
    {
        ptx_code[i] = (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);

    // Find the index of the device of interest
    unsigned int idx = 0;
    while((idx < num_devices) && (devices[idx] != cdDevice)) 
    {
        ++idx;
    }
    
    // If the index is associated, log the result
    if(idx < num_devices)
    {
         
        // if a separate filename is supplied, dump ptx there 
        if (NULL != cPtxFileName)
        {
            shrLog("\nWriting ptx to separate file: %s ...\n\n", cPtxFileName);
            FILE* pFileStream = NULL;
            #ifdef _WIN32
                fopen_s(&pFileStream, cPtxFileName, "wb");
            #else
                pFileStream = fopen(cPtxFileName, "wb");
            #endif

            fwrite(ptx_code[idx], binary_sizes[idx], 1, pFileStream);
            fclose(pFileStream);        
        }
        else // log to logfile and console if no ptx file specified
        {
           shrLog("\n%s\nProgram Binary:\n%s\n%s\n", HDASHLINE, ptx_code[idx], HDASHLINE);
        }
    }

    // Cleanup
    free(devices);
    free(binary_sizes);
    for(unsigned int i = 0; i < num_devices; ++i)
    {
        free(ptx_code[i]);
    }
    free( ptx_code );
}

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//////////////////////////////////////////////////////////////////////////////
void oclLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice)
{
    // write out the build log and ptx, then exit
    char cBuildLog[10240];
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 
                          sizeof(cBuildLog), cBuildLog, NULL );
    shrLog("\n%s\nBuild Log:\n%s\n%s\n", HDASHLINE, cBuildLog, HDASHLINE);
}

// Helper function for De-allocating cl objects
// *********************************************************************
void oclDeleteMemObjs(cl_mem* cmMemObjs, int iNumObjs)
{
    int i;
    for (i = 0; i < iNumObjs; i++)
    {
        if (cmMemObjs[i])clReleaseMemObject(cmMemObjs[i]);
    }
}  

// Helper function to get OpenCL error string from constant
// *********************************************************************
const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

// Helper function to get OpenCL image format string (channel order and type) from constant
// *********************************************************************
const char* oclImageFormatString(cl_uint uiImageFormat)
{
    // cl_channel_order 
    if (uiImageFormat == CL_R)return "CL_R";  
    if (uiImageFormat == CL_A)return "CL_A";  
    if (uiImageFormat == CL_RG)return "CL_RG";  
    if (uiImageFormat == CL_RA)return "CL_RA";  
    if (uiImageFormat == CL_RGB)return "CL_RGB";
    if (uiImageFormat == CL_RGBA)return "CL_RGBA";  
    if (uiImageFormat == CL_BGRA)return "CL_BGRA";  
    if (uiImageFormat == CL_ARGB)return "CL_ARGB";  
    if (uiImageFormat == CL_INTENSITY)return "CL_INTENSITY";  
    if (uiImageFormat == CL_LUMINANCE)return "CL_LUMINANCE";  

    // cl_channel_type 
    if (uiImageFormat == CL_SNORM_INT8)return "CL_SNORM_INT8";
    if (uiImageFormat == CL_SNORM_INT16)return "CL_SNORM_INT16";
    if (uiImageFormat == CL_UNORM_INT8)return "CL_UNORM_INT8";
    if (uiImageFormat == CL_UNORM_INT16)return "CL_UNORM_INT16";
    if (uiImageFormat == CL_UNORM_SHORT_565)return "CL_UNORM_SHORT_565";
    if (uiImageFormat == CL_UNORM_SHORT_555)return "CL_UNORM_SHORT_555";
    if (uiImageFormat == CL_UNORM_INT_101010)return "CL_UNORM_INT_101010";
    if (uiImageFormat == CL_SIGNED_INT8)return "CL_SIGNED_INT8";
    if (uiImageFormat == CL_SIGNED_INT16)return "CL_SIGNED_INT16";
    if (uiImageFormat == CL_SIGNED_INT32)return "CL_SIGNED_INT32";
    if (uiImageFormat == CL_UNSIGNED_INT8)return "CL_UNSIGNED_INT8";
    if (uiImageFormat == CL_UNSIGNED_INT16)return "CL_UNSIGNED_INT16";
    if (uiImageFormat == CL_UNSIGNED_INT32)return "CL_UNSIGNED_INT32";
    if (uiImageFormat == CL_HALF_FLOAT)return "CL_HALF_FLOAT";
    if (uiImageFormat == CL_FLOAT)return "CL_FLOAT";

    // unknown constant
    return "Unknown";
}
