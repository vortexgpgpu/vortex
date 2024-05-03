#include <CL/opencl.h>
#include <stdio.h>

cl_int _err;
// TODO deberia ser independiente de glsc2.c, es decir pasar por parametro de funcion todas las variables que necesitemos

cl_platform_id _getPlatformID() {    
    static cl_platform_id platform_id = NULL;

    if (!platform_id) clGetPlatformIDs(1, &platform_id, NULL);
    
    printf("_getPlatformID() return: %p\n", platform_id);
    return platform_id;
}

cl_device_id _getDeviceID() {
    static cl_device_id device_id = NULL;

    if (!device_id) clGetDeviceIDs(_getPlatformID(), CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    printf("_getDeviceID() return: %p\n", device_id);
    return device_id;
}

cl_context _getContext() {
    static cl_context context = NULL;

    if (!context) {
        cl_device_id device_id = _getDeviceID();
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err);
    }

    printf("_getContext() return: %p\n", context);
    return context;
} 

cl_program createProgramWithBinary(const uint8_t* binary, size_t length) {
    printf("createProgramWithBinary() binary=%p, length=%ld\n", binary, length);

    cl_device_id device_id = _getDeviceID();

    cl_program program = clCreateProgramWithBinary(
        _getContext(), 1, &device_id, &length, &binary, NULL, &_err);
    
    printf("\treturn=%p, error=%d\n", program, _err);
    return program;
}
int buildProgram(void *program) {
    printf("buildProgram() program=%p\n", program);

    cl_device_id device_id = _getDeviceID();

    _err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    printf("\treturn=%d, error=%d\n", _err, _err);
    return _err;
}
/**** BASIC OPERATIONS
 * It works as an interface, OpenGL does not has to know that is implemented with OpenCL,
 * 
*/
#define MEM_READ_ONLY CL_MEM_READ_ONLY
#define MEM_WRITE_ONLY CL_MEM_WRITE_ONLY
#define MEM_READ_WRITE CL_MEM_READ_WRITE
#define MEM_COPY_HOST_PTR CL_MEM_COPY_HOST_PTR 

void* createBuffer(uint64_t flags, size_t size, void* data){

    void *buffer = clCreateBuffer(_getContext(), flags, size, data, &_err);
    
    printf("createBuffer() return=%p, error=%d\n", buffer, _err);

    return buffer;
}

void* createCommandQueue(uint64_t properties) {
    return clCreateCommandQueue(_getContext(), _getDeviceID(), properties, &_err);
}

void* createKernel(void* program, const char* name) {
    printf("createKernel() program=%p, name=%s\n", program, name);
    cl_kernel kernel = clCreateKernel((cl_program) program, name, &_err);

    printf("\treturn=%p, error=%d\n", kernel, _err);
    return kernel;
}

void setKernelArg(void* kernel, unsigned int location, size_t size, const void* value) {
    printf("setKernelArg() location=%d, size=%ld, value=%p\n", location, size, value);
    
    int err = clSetKernelArg((cl_kernel) kernel, location, size, value);

    printf("\terror=%d\n", err);
}

// I decide to make it simple, but maybe it will need to be extendend in future.
void enqueueNDRangeKernel(void* commandQueue, void* kernel, const size_t global_work_size) {
    printf("enqueueND() work=%ld\n", global_work_size);
	int err = clEnqueueNDRangeKernel(
        (cl_command_queue) commandQueue, (cl_kernel) kernel,
        1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    
    printf("\terror=%d\n", err);
}

void enqueueReadBuffer(void* command_queue, void* buffer, size_t bufSize, void* data) {

    clEnqueueReadBuffer(command_queue, (cl_mem) buffer, CL_TRUE, 0, bufSize, data, 0, NULL, NULL);
}

void enqueueWriteBuffer(void* command_queue, void* buffer, size_t size, const void* ptr) {

    clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
}




void finish(void* command_queue) {
    clFinish((cl_command_queue) command_queue);
}

#include "kernel.fill.c"

cl_kernel _getKernelFill() {
    static cl_program program;
    static cl_kernel kernel;

    if (kernel == NULL) {
        program = createProgramWithBinary(GLSC2_kernel_fill_pocl, sizeof(GLSC2_kernel_fill_pocl));
        buildProgram(program);
        kernel = createKernel(program, "gl_fill");
    }

    return kernel;
}

void enqueueFillBuffer(void* command_queue, void* buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size) {
    printf("enqueueFillBuffer() offset=%ld, size=%ld\n", offset, size);
    // just valid for pattern size == 4
    cl_kernel kernel = _getKernelFill();
    uint32_t value = *((uint32_t*) pattern);
    size_t global_work_size = size / 4;
    clSetKernelArg(kernel, 0, 4, &value);
    clSetKernelArg(kernel, 1, sizeof(buffer), &buffer);
    clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    // TODO: It brokes 
    // clEnqueueFillBuffer((cl_command_queue)command_queue, (cl_mem) buffer, pattern, pattern_size, offset, size, 0, NULL, NULL);
}

void* createImage2D(uint64_t flags, const cl_image_format *image_format, size_t image_width, size_t image_height, size_t image_row_pitch, void *host_ptr) {
    
    void *image = clCreateImage2D(_getContext(), flags, image_format, image_width, image_height, image_row_pitch, host_ptr, &_err);
    if (_err) {
        printf("createImage2D() image_format=%p, image_width=%ld, image_height=%ld, image_row_pitch=%ld, host_ptr=%p\n", image_format, image_width, image_height, image_row_pitch, host_ptr);
        printf("\terror=%d\n", _err);
    }

    return image;
}

void* createImage(uint64_t flags, const cl_image_format *image_format, const cl_image_desc* image_desc, void *host_ptr) {
    
    void *image = clCreateImage(_getContext(), flags, image_format, image_desc, host_ptr, &_err);
    if (_err) {
        printf("createImage2D()\n");
        printf("\terror=%d\n", _err);
    }

    return image;
}

int enqueueWriteImage(void* command_queue, void* image, const size_t* origin, const size_t* region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr) {
    int err = clEnqueueWriteImage(command_queue, image, CL_TRUE, origin, region, input_row_pitch, input_slice_pitch, ptr, 0, NULL, NULL );
    
    if (err) {
        printf("enqueueWriteImage()\n");
        printf("\terror=%d\n", err);
    }

    return err;
}

void* createSampler(cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode) {
    cl_sampler sampler = clCreateSampler(_getContext(),normalized_coords, addressing_mode, filter_mode, &_err);

    if (_err) {
        printf("createSampler()\n");
        printf("\terror=%d\n", _err);
    }

    return sampler;
}
