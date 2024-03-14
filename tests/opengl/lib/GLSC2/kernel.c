#include <CL/opencl.h>

cl_int _err;
extern VIEWPORT_TRANSFORM viewportTransform;
extern PROGRAM_OBJECT _current_program;

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


void vertex_shader(GLint first, GLsizei count, float* clip_coords){
    //VERTEX SHADER
    cl_program vertex_shader = _current_program.binary;
  
    // Create kernel
    //se ve que esto elige un kernel de todos los del programa :D
    cl_kernel vertexShaderKernel = clCreateKernel(vertex_shader, "vertex_shader", &_err);
    //preparar input
    VertexAttrib* pVertexAttr = &vertex_attrib;
    clSetKernelArg(vertexShaderKernel, 0, sizeof(GLint), first);
    VertexAttrib VAO[GL_MAX_VERTEX_ATTRIBS];

    GLint vaos=0;
    for (int i =0; i<VERTEX_ATTR_SIZE; i++)
    {
        if (pVertexAttr->enable){
            vaos++;
            VAO[vaos] = *pVertexAttr;
        }
    pVertexAttr++;
    }
    cl_mem cl_vaos = clCreateBuffer(_context, CL_MEM_READ_ONLY, vaos*sizeof(VertexAttrib), (void*) VAO, &_err);
    clSetKernelArg(vertexShaderKernel, 1, sizeof(cl_mem), (void*) &cl_vaos);
    cl_mem cl_primitives = clCreateBuffer(_context, CL_MEM_WRITE_ONLY, 4*(count-first)*sizeof(float), (void*) VAO, &_err);
    clSetKernelArg(vertexShaderKernel, 2, sizeof(cl_mem), (void*) &cl_primitives);
    if(_binded_buffer){
        clSetKernelArg(vertexShaderKernel, 3, sizeof(cl_mem), (void*) &buffers[_binded_buffer].mem);
    }
    
  cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
  size_t global_work_size[1] = {size};
  clEnqueueNDRangeKernel(commandQueue, vertexShaderKernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
  clFinish(commandQueue);  
  clEnqueueReadBuffer(commandQueue, cl_primitives, CL_TRUE, 0, 4*(count-first)*sizeof(float), clip_coords, 0, NULL, NULL);
}

void perspective_division(unsigned int numVerts, float* clip_coords, float* ndc_coords){
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
    cl_kernel perspective_kernel = clCreateKernel(perspective_division, "perspective division", &_err);

    cl_mem clipCoordsBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, numVerts*4*sizeof(float), NULL, &_err);
    cl_mem ndcCoordsBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numVerts*3*sizeof(float), NULL, &_err);
    
    clSetKernelArg(perspective_kernel, 0, sizeof(cl_mem), (void*)&clipCoordsBuff);
    clSetKernelArg(perspective_kernel, 1, sizeof(cl_mem), (void*)&ndcCoordsBuff);

  cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
  size_t global_work_size[1] = {numVerts};
  clEnqueueNDRangeKernel(commandQueue, perspective_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
  clFinish(commandQueue);  
  clEnqueueReadBuffer(commandQueue, ndcCoordsBuff, CL_TRUE, 0, 3*numVerts*sizeof(float), ndc_coords, 0, NULL, NULL);
}

void viewport_transformation(unsigned int numVerts, float* ndc_coords, float* window_coords){
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
    cl_kernel viewport_kernel = clCreateKernel(viewport_transformation, "viewport division", &_err);
    cl_mem ndcCoordsBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, numVerts*3*sizeof(float), NULL, &_err);
    cl_mem windowCoordsBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numVerts*3*sizeof(float), NULL, &_err);

    clSetKernelArg(viewport_transformation_kernel, 0, sizeof(cl_mem), ndcCoordsBuff);
    clSetKernelArg(viewport_transformation_kernel, 1, sizeof(GLint), viewportTransform.w);
    clSetKernelArg(viewport_transformation_kernel, 2, sizeof(GLint), viewportTransform.h);
    clSetKernelArg(viewport_transformation_kernel, 3, sizeof(GLint), (viewportTransform.x+viewportTransform.w/2));
    clSetKernelArg(viewport_transformation_kernel, 4, sizeof(GLint), (viewportTransform.y+viewportTransform.h/2));
    clSetKernelArg(viewport_transformation_kernel, 5, sizeof(GLfloat), viewportTransform.n);
    clSetKernelArg(viewport_transformation_kernel, 6, sizeof(GLfloat), viewportTransform.f);
    clSetKernelArg(viewport_transformation_kernel, 7, sizeof(cl_mem), windowCoordsBuff);

    cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
    size_t global_work_size[1] = {numVerts};
    clEnqueueNDRangeKernel(commandQueue, viewport_transformation_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(commandQueue);  
    clEnqueueReadBuffer(commandQueue, windowCoordsBuff, CL_TRUE, 0, 3*numVerts*sizeof(float), window_coords, 0, NULL, NULL);
}

void fragment_shader(){
        //VERTEX SHADER
    cl_program fragment_shader = _current_program.binary;
  
    // Create kernel
    cl_kernel fragmentShaderKernel = clCreateKernel(fragment_shader, "fragment_shader", &_err);
    //...
}