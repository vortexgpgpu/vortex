#include <CL/opencl.h>

cl_int _err;

cl_platform_id* _getPlatformID() {
    static cl_platform_id platform_id = NULL;
    
    if (!platform_id) clGetPlatformIDs(1, &platform_id, NULL);
    return &platform_id;
}

cl_device_id* _getDeviceID() {
    static cl_device_id device_id = NULL;
    
    if (!device_id) clGetDeviceIDs(*_getPlatformID(), CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    return &device_id;
}

cl_context _context = clCreateContext(NULL, 1, _getDeviceID(), NULL, NULL,  &_err);

cl_kernel init_vertex_shader_kernel(){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    //HOSTCPU
    if (0 != read_kernel_file(VERTEX_SHADER, &kernel_bin, &kernel_size))
        return -1;
    
    cl_program vertex_shader = clCreateProgramWithSource(
        _context, 1, (const char**)&kernel_bin, &kernel_size, &_err);  
    // Build program
    clBuildProgram(vertex_shader, 1, _getDeviceID(), NULL, NULL, NULL);
  
    // Create kernel
    return clCreateKernel(vertex_shader, "vertex_shader", &_err);
    //Hay que conservar cl_program al salir de la stack? (quien sabe...)
}

cl_kernel init_perspective_division_kernel(){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    //HOSTCPU
    if (0 != read_kernel_file(PERS_DIV, &kernel_bin, &kernel_size))
        return -1;
    
    cl_program perspective_division = clCreateProgramWithSource(
        _context, 1, (const char**)&kernel_bin, &kernel_size, &_err);  
    // Build program
    clBuildProgram(perspective_division, 1, _getDeviceID(), NULL, NULL, NULL);
  
    // Create kernel
    return clCreateKernel(perspective_division, "perspective division", &_err);
    //Hay que conservar cl_program al salir de la stack? (quien sabe...)
}

cl_kernel init_viewport_transformation_kernel(){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    //HOSTCPU
    if (0 != read_kernel_file(VIEWPORT_TRANS, &kernel_bin, &kernel_size))
        return -1;
    
    cl_program viewport_transformation = clCreateProgramWithSource(
        _context, 1, (const char**)&kernel_bin, &kernel_size, &_err);  
    // Build program
    clBuildProgram(viewport_transformation, 1, _getDeviceID(), NULL, NULL, NULL);
  
    // Create kernel
    return clCreateKernel(viewport_transformation, "viewport division", &_err);
    //Hay que conservar cl_program al salir de la stack? (quien sabe...)
}