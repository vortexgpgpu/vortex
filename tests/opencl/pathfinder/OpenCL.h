#ifndef __OPENCL_H__
#define __OPENCL_H__

#include <iostream>
#include <stdio.h>
#include <map>
#include <string>
#include <cstring>

// OpenCL header files
#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>
#endif

using namespace std;

class OpenCL
{
public:
	OpenCL(int displayOutput);
	~OpenCL();
	void init();
	void createKernel(string kernelName);
	cl_kernel kernel(string kernelName);
	void gwSize(size_t theSize);
	cl_context ctxt();
	cl_command_queue q();
	void launch(string toLaunch);
	size_t localSize();
	
private:
	int                     VERBOSE;           // Display output text from various functions?
	size_t                  lwsize;            // Local work size.
	size_t                  gwsize;            // Global work size.
	cl_int                  ret;               // Holds the error code returned by cl functions.
	cl_platform_id          platform_id[100];
	cl_device_id            device_id[100];
	map<string, cl_kernel>  kernelArray;
	cl_context              context;
	cl_command_queue        command_queue;
	cl_program              program;
	
	void getDevices();
	void buildKernel();
};

extern int platform_id_inuse;
extern int device_id_inuse;

#endif
