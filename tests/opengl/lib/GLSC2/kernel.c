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

cl_program _createProgram(){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    //HOSTCPU
    if (0 != read_kernel_file("pipeline.cl", &kernel_bin, &kernel_size))
        return -1;
    
    cl_program pipeline = clCreateProgramWithSource(
        _context, 1, (const char**)&kernel_bin, &kernel_size, &_err);  
    // Build program
    clBuildProgram(pipeline, 1, _getDeviceID(), NULL, NULL, NULL);
    return pipeline;
}

cl_program pipeline_program = _createProgram();


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

void perspective_division(unsigned int numVerts, float* clip_coords){
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    // Create kernel
    cl_kernel perspective_kernel = clCreateKernel(pipeline_program, "perspective_division", &_err);

    cl_mem clipCoordsBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, numVerts*4*sizeof(float), NULL, &_err);
    
    clSetKernelArg(perspective_kernel, 0, sizeof(cl_mem), (void*)&clipCoordsBuff);

  cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
  size_t global_work_size[1] = {numVerts};
  clEnqueueNDRangeKernel(commandQueue, perspective_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
  clFinish(commandQueue);  
  clEnqueueReadBuffer(commandQueue, clipCoordsBuff, CL_TRUE, 0, numVerts*4*sizeof(float), clip_coords, 0, NULL, NULL);
}

void viewport_transformation(unsigned int numVerts, float* ndc_coords){

    // Create kernel
    cl_kernel viewport_kernel = clCreateKernel(pipeline_program, "viewport_division", &_err);
    cl_mem ndcCoordsBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, numVerts*4*sizeof(float), NULL, &_err);

    clSetKernelArg(viewport_transformation_kernel, 0, sizeof(cl_mem), ndcCoordsBuff);
    clSetKernelArg(viewport_transformation_kernel, 1, sizeof(GLint), viewportTransform.w);
    clSetKernelArg(viewport_transformation_kernel, 2, sizeof(GLint), viewportTransform.h);
    clSetKernelArg(viewport_transformation_kernel, 3, sizeof(GLint), (viewportTransform.x+viewportTransform.w/2));
    clSetKernelArg(viewport_transformation_kernel, 4, sizeof(GLint), (viewportTransform.y+viewportTransform.h/2));
    clSetKernelArg(viewport_transformation_kernel, 5, sizeof(GLfloat), viewportTransform.n);
    clSetKernelArg(viewport_transformation_kernel, 6, sizeof(GLfloat), viewportTransform.f);

    cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
    size_t global_work_size[1] = {numVerts};
    clEnqueueNDRangeKernel(commandQueue, viewport_transformation_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(commandQueue);  
    clEnqueueReadBuffer(commandQueue, ndcCoordsBuff, CL_TRUE, 0, 4*numVerts*sizeof(float), ndcCoordsBuff, 0, NULL, NULL);
}

void rasterization(unsigned int numVerts, float* primitives, float* fragments, unsigned int grid_size){

    // Create kernel
    cl_kernel rasterization_kernel = clCreateKernel(pipeline_program, "rasterization", &_err);
    cl_mem primitivesBuff = clCreateBuffer(_context, CL_MEM_READ_ONLY, numVerts*4*sizeof(float), NULL, &_err);
    cl_mem fragmentsBuff = clCreateBuffer(_context, CL_MEM_WRITE_ONLY, grid_size*4*sizeof(float), NULL, &_err);

    clSetKernelArg(rasterization_kernel, 0, sizeof(GLuint), grid_size);
    clSetKernelArg(rasterization_kernel, 1, sizeof(cl_mem), primitivesBuff);
    clSetKernelArg(rasterization_kernel, 2, sizeof(cl_mem), fragmentsBuff);
    clSetKernelArg(rasterization_kernel, 3, sizeof(GLuint), grid_size);

    cl_command_queue commandQueue = clCreateCommandQueue(_context, _getDeviceID(), 0, &_err);
    size_t global_work_size[1] = {grid_size};
    clEnqueueNDRangeKernel(commandQueue, rasterization_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(commandQueue);  
    clEnqueueReadBuffer(commandQueue, fragmentsBuff, CL_TRUE, 0, 4*grid_size*sizeof(float), fragments, 0, NULL, NULL);
}

void fragment_shader(){
        //VERTEX SHADER
    cl_program fragment_shader = _current_program.binary;
  
    // Create kernel
    cl_kernel fragmentShaderKernel = clCreateKernel(fragment_shader, "fragment_shader", &_err);
    //...
}

void fill(cl_mem buff, size_t size, void* pattern, size_t pattern_size) {
    cl_command_queue commandQueue = clCreateCommandQueue(_context, _getDeviceID(), 0, &_err);
    clEnqueueFillBuffer(commandQueue, buff, pattern, pattern_size, 0, size, 0, NULL, NULL);
}

void read(cl_mem buff, void* ptr, size_t nbytes, size_t offset) {
    cl_command_queue commandQueue = clCreateCommandQueue(_context, _getDeviceID(), 0, &_err);
    clEnqueueReadBuffer(commandQueue, buff, CL_TRUE, offset, nbytes, ptr, 0, NULL, NULL);
}