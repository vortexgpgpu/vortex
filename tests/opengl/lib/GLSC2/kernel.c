#include <CL/opencl.h>

cl_int _err;
// TODO deberia ser independiente de glsc2.c, es decir pasar por parametro de funcion todas las variables que necesitemos

cl_platform_id _getPlatformID() {
    static cl_platform_id platform_id = NULL;
    
    if (!platform_id) clGetPlatformIDs(1, &platform_id, NULL);
    return platform_id;
}

cl_device_id _getDeviceID() {
    static cl_device_id device_id = NULL;
    
    if (!device_id) clGetDeviceIDs(_getPlatformID(), CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    return device_id;
}

cl_context _getContext() {
    static cl_context context = NULL;

    if (!context) context = clCreateContext(NULL, 1, _getDeviceID(), NULL, NULL,  &_err);

    return context;
} 

cl_program _createProgram(const char* filename){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    //HOSTCPU
    if (0 != read_kernel_file(filename, &kernel_bin, &kernel_size))
        return -1;
    
    cl_device_id device_id = _getDeviceID();
    cl_program program = clCreateProgramWithBinary(
        _getContext(), 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, NULL, &_err);

    return program;
}

cl_program _getPipelineProgram() {
    static cl_program program = NULL;

    if (!program) program = _createProgram("pipeline.pocl");

    return program;
}

cl_program createProgramWithBinary(binary, length) {
    cl_device_id device_id = _getDeviceID();

    cl_program program = clCreateProgramWithBinary(
        _getContext(), 1, &device_id, &length, (const uint8_t**)&binary, NULL, &_err);
    
    return program;
}

/**** BASIC OPERATIONS
 * It works as an interface, OpenGL does not has to know that is implemented with OpenCL,
 * 
*/
#define MEM_READ_ONLY CL_MEM_READ_ONLY
#define MEM_WRITE_ONLY CL_MEM_WRITE_ONLY
#define MEM_READ_WRITE CL_MEM_READ_WRITE

void* createBuffer(uint64_t flags, size_t size, void* data){
    return clCreateBuffer(_getContext(), flags, size, data, &_err);
}

void* createCommandQueue(uint64_t properties) {
    return clCreateCommandQueue(_getContext(), _getDeviceID(), properties, &_err);
}

void* createKernel(void* program, const char* name) {
    return clCreateKernel((cl_program) program, name, &_err);
}

void setKernelArg(void* kernel, unsigned int location, size_t size, void* value) {
    clSetKernelArg((cl_kernel) kernel, location, size, value);
}

// I decide to make it simple, but maybe it will need to be extendend in future.
void enqueueNDRangeKernel(void* commandQueue, void* kernel, const size_t* global_work_size) {
    clEnqueueNDRangeKernel(
        (cl_command_queue) commandQueue, (cl_kernel) kernel,
        1, NULL, global_work_size, NULL, 0, NULL, NULL);
}

void enqueueReadBuffer(void* command_queue, void* buffer, size_t bufSize, void* data) {

    clEnqueueReadBuffer(command_queue, (cl_mem) buffer, CL_TRUE, 0, bufSize, data, 0, NULL, NULL);
}


void finish(void* command_queue) {
    clFinish((cl_command_queue) command_queue);
}

/**** SHORT CUTS
 * 
*/
void fill(cl_mem buff, size_t size, void* pattern, size_t pattern_size) {
    cl_command_queue commandQueue = clCreateCommandQueue(_getContext(), _getDeviceID(), 0, &_err);
    clEnqueueFillBuffer(commandQueue, buff, pattern, pattern_size, 0, size, 0, NULL, NULL);
}

// formats
#define RGBA8 0x0
#define RGBA4 0x1

void readnPixels(cl_mem buff, int x, int y, int width, int height, unsigned int src_format, unsigned int dst_format, int bufSize, void *data) {

    static cl_program program = NULL;
    if (! program) { program = _createProgram("kernel.readnpixels.pocl"); }
    
    if (src_format == dst_format) {

        cl_command_queue commandQueue = clCreateCommandQueue(_getContext(), _getDeviceID(), 0, &_err);
        clEnqueueReadBuffer(commandQueue, buff, CL_TRUE, 0, bufSize, data, 0, NULL, NULL);

    } else if (src_format == RGBA4 && dst_format == RGBA8) {

        cl_kernel kernel = clCreateKernel(program, "rgba4_rgba8", &_err);
        cl_mem dst_buff = clCreateBuffer(_getContext(), CL_MEM_WRITE_ONLY, bufSize, NULL, &_err);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), buff);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), dst_buff);
        clSetKernelArg(kernel, 2, sizeof(int), &x);
        clSetKernelArg(kernel, 3, sizeof(int), &y);
        clSetKernelArg(kernel, 4, sizeof(int), &width);
        clSetKernelArg(kernel, 5, sizeof(int), &height);

        cl_command_queue commandQueue = clCreateCommandQueue(_getContext(), _getDeviceID(), 0, &_err);
        size_t global_work_size = bufSize/4; // 4 bytes x color
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

        clFinish(commandQueue);  
        clEnqueueReadBuffer(commandQueue, dst_buff, CL_TRUE, 0, bufSize, data, 0, NULL, NULL);

    }

}