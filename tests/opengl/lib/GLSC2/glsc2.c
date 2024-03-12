#include <GLSC2/glsc2.h>
#include <CL/opencl.h>


#define VERTEX_SHADER "kernel.vert.cl"
#define PERS_DIV "kernel-persp-div.cl"
// CL 
cl_int _err;

typedef struct vec2f
{
    float x;
    float y;
};

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
    
    cl_program vertex_shader = clCreateProgramWithSource(
        _context, 1, (const char**)&kernel_bin, &kernel_size, &_err);  
    // Build program
  clBuildProgram(vertex_shader, 1, _getDeviceID(), NULL, NULL, NULL);
  
  // Create kernel
  return clCreateKernel(vertex_shader, "perspective division", &_err);
  //Hay que conservar cl_program al salir de la stack? (quien sabe...)
}

void vertex_shader_add_argument_buffer(cl_mem data, unsigned int arg, void* offset, unsigned int size, cl_kernel vertexShaderKernel){
  //position
  clSetKernelArg(vertexShaderKernel, arg, sizeof(cl_mem), (void *)&data);
  clSetKernelArg(vertexShaderKernel, arg+1, sizeof(void*), offset);
  clSetKernelArg(vertexShaderKernel, arg+2, sizeof(unsigned int), size);
}

void vertex_shader_add_argument_host(void* data, unsigned int arg, unsigned int size, cl_kernel vertexShaderKernel){
  //position
  clSetKernelArg(vertexShaderKernel, arg, sizeof(void*), (void *)&data);
  clSetKernelArg(vertexShaderKernel, arg+1, sizeof(void*), (void*)0);
  clSetKernelArg(vertexShaderKernel, arg+2, sizeof(unsigned int), size);
}

void execute_vertex_shader(unsigned int size, unsigned int arg, float* clip_coords, cl_kernel vertexShaderKernel){
  cl_mem primitiveBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4*size*sizeof(float), NULL, &_err);
  clSetKernelArg(vertexShaderKernel, arg, sizeof(cl_mem), clip_coords);

  cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
  size_t global_work_size[1] = {size};
  clEnqueueNDRangeKernel(commandQueue, vertexShaderKernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
  clFinish(commandQueue);  
  clEnqueueReadBuffer(commandQueue, primitiveBuff, CL_TRUE, 0, 4*size*sizeof(float), clip_coords, 0, NULL, NULL);
}

void execute_perspective_division(unsigned int numVerts, float* clip_coords, float* ndc_coords, cl_kernel perspective_kernel){
    cl_mem clipCoordsBuff = clCreateBuffer(context, CL_MEM_READ_ONLY, numVerts*4*sizeof(float), NULL, &_err);
    cl_mem ndcCoordsBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numVerts*3*sizeof(float), NULL, &_err);
    
    clSetKernelArg(perspective_kernel, 0, sizeof(cl_mem), clip_coords);
    clSetKernelArg(perspective_kernel, 1, sizeof(cl_mem), ndc_coords);

  cl_command_queue commandQueue = clCreateCommandQueue(context, _getDeviceID(), 0, &_err);
  size_t global_work_size[1] = {numVerts};
  clEnqueueNDRangeKernel(commandQueue, perspective_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
  clFinish(commandQueue);  
  clEnqueueReadBuffer(commandQueue, ndcCoordsBuff, CL_TRUE, 0, 3*numVerts*sizeof(float), ndc_coords, 0, NULL, NULL);
}

// GL 

struct BUFFER {
    GLboolean used;
    GLenum target;
    cl_mem mem;
};

BUFFER _buffers[255];
GLuint _binded_buffer;

struct VertexAttrib{
    bool enable, normalized;
    GLuint index;
    GLint size;
    GLenum type;
    GLsizei stride;
    const void *pointer;
};

//change to GL_MAX_VERTEX
#define VERTEX_ATTR_SIZE 255

VertexAttrib vertex_attrib[VERTEX_ATTR_SIZE]; // TODO: this has to be the max size GL_MAX_VERTEX_ATTRIBS

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _binded_buffer = buffer;
    _buffers[buffer].target = target;
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    // HINTS for optimize implementation
    if (usage == GL_STATIC_DRAW) {}
    else if (usage == GL_DYNAMIC_DRAW) {}
    else if (usage == GL_STREAM_DRAW) {}

    _buffers[_binded_buffer].target = target;
    _buffers[_binded_buffer].mem = clCreateBuffer(_context, CL_MEM_READ_ONLY, size, data, &_err);
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask);
GL_APICALL GLuint GL_APIENTRY glCreateProgram (void);

inline void vertex_shader(GLint first, GLsizei count, float* clip_coords){
    //VERTEX SHADER
    cl_kernel vertexShaderKernel = init_vertex_shader_kernel();
    //preparar input
    VertexAttrib* pVertexAttr = &vertex_attrib;
    clSetKernelArg(vertexShaderKernel, 0, sizeof(GLint), first);
    unsigned int arg=1;

    //cada uno es un kernel (la idea es hacer count-first kernels)
    //while(first < count) {
        //coger atributos del kernel
    for (int i =0; i<VERTEX_ATTR_SIZE; i++)
    {
        if (pVertexAttr->enable){
            if (_binded_buffer != 0){
                //take from buffer, send to kernel as argument
                vertex_shader_add_argument_buffer(buffers[_binded_buffer].mem, arg, pVertexAttr->pointer, pVertexAttr->size, vertexShaderKernel);
            }else{
                //take from host
                vertex_shader_add_argument_host(pVertexAttr->pointer, arg, pVertexAttr->size, vertexShaderKernel);
            }
            arg+=3;
        }
    pVertexAttr++;
    }
    //}
    execute_vertex_shader(count-first, arg++, clip_coords, vertexShaderKernel);
}

inline void perspective_division(unsigned int numVerts, float* clip_coords, float* ndc_coords){
    cl_kernel perspective_kernel = init_perspective_division_kernel();
    execute_perspective_division(numVerts, clip_coords, ndc_coords, perspective_kernel);
}

inline void gl_pipeline(GLint first, GLsizei count){
    //pipeline
    unsigned int numVerts = count-first;
    //vertex shader
    float clip_coords[4*numVerts];//es ejemplo, no se puede iniciar dinamicamente, pero tampoco se puede usar malloc :D
    vertex_shader(first, count, clip_coords);
    //clip coord
    float ndc_coords[3*numVerts];
    perspective_division(numVerts, clip_coords, ndc_coords);
    //normalized-device-coords

    //rasterization
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (first <0){
        _err= GL_INVALID_VALUE;
        return;
    }
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }
    if (mode==GL_POINTS); // TODO
    else if (mode==GL_LINES); // TODO
    else if (mode==GL_TRIANGLES) {
        gl_pipeline(first, count);
    }
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices){
}
GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    vertex_attrib[index].enable = false;
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    vertex_attrib[index].enable = true;
}

GL_APICALL void GL_APIENTRY glGenBuffers (GLsizei n, GLuint *buffers) {
    GLuint _id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && _id < 256) {
        if (!_buffers[_id].used) {
            _buffers[_id].used = GL_TRUE;
            *buffers = _id;

            buffers += 1; 
            n -= 1;
        }
        _id += 1;
    }
}

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length);

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program);

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    if (size > 4 || size <=0) {
        _err = GL_INVALID_VALUE;
        return;
    }
    
    if (stride < 0) {
        _err = GL_INVALID_VALUE;
        return;
    }

    //check type
    if (type != GL_BYTE || type != GL_UNSIGNED_BYTE || type != GL_SHORT || type != GL_UNSIGNED_SHORT || type != GL_FLOAT){
        _err=GL_INVALID_VALUE;
        return;
    }

    if (normalized == GL_TRUE){
        //normalizar integers
    }

    vertex_attrib[index].size = size;
    vertex_attrib[index].type = type;
    vertex_attrib[index].normalized = normalized;
    vertex_attrib[index].stride = stride;
    vertex_attrib[index].pointer = pointer;
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
