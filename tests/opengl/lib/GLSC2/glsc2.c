#include <GLSC2/glsc2.h>
#include <CL/opencl.h>

// CL 
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

VertexAttrib vertex_attrib[255]; // TODO: this has to be the max size GL_MAX_VERTEX_ATTRIBS

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

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }
    if (mode==GL_POINTS); // TODO
    else if (mode==GL_LINES); // TODO
    else if (mode==GL_TRIANGLES) {
        while(first < count) {


            ++first;
        }
    }
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

    vertex_attrib[index].size = size;
    vertex_attrib[index].type = type;
    vertex_attrib[index].normalized = normalized;
    vertex_attrib[index].stride = stride;
    vertex_attrib[index].pointer = pointer;
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height);
