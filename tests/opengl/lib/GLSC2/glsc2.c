#include <GLSC2/glsc2.h>
#include "kernel.c"

#define VERTEX_SHADER "kernel.vert.cl"
#define PERS_DIV "kernel-persp-div.cl"
#define VIEWPORT_TRANS "kernel-viewport-trans.cl"

#define MAX_PROGRAMS 255
#define _MAX_VERTEX_ATTRIBS 255 // TODO update GL_MAX_VERTEX_ATTRIBS

typedef struct {
    GLint x;
    GLint y;
    GLsizei w;
    GLsizei h;
    GLfloat n=0;
    GLfloat f=0; 
} VIEWPORT_TRANSFORM;

typedef struct {
    GLboolean used;
    GLenum target;
    cl_mem mem;
} BUFFER;

typedef struct {
    GLboolean used;
    GLenum target;
    cl_mem mem;
} FRAMEBUFFER;

typedef struct {
    bool enable, normalized;
    GLuint index;
    GLint size;
    GLenum type;
    GLsizei stride;
    const void *pointer;
} VERTEX_ATTRIB;

VIEWPORT_TRANSFORM viewportTransform;

BUFFER _buffers[255];
GLuint _binded_buffer;

FRAMEBUFFER _framebuffers[255];
GLuint _binded_framebuffer;

VERTEX_ATTRIB vertex_attrib[_MAX_VERTEX_ATTRIBS]; 

//entiendo que
// binary = kernel_bin
// length = kernel_length
typedef struct {
    GLboolean created=0;
    cl_program binary;
    GLsizei length;
}PROGRAM_OBJECT;

PROGRAM_OBJECT programs [MAX_PROGRAMS];
GLboolean _no_program = GL_TRUE;
PROGRAM_OBJECT _current_program;

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _binded_buffer = buffer;
    _buffers[buffer].target = target;
}
GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) {
    if (!_framebuffers[framebuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _binded_framebuffer = framebuffer;
    _framebuffers[framebuffer].target = target;
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

inline void gl_pipeline(GLint first, GLsizei count){
    //pipeline
    unsigned int numVerts = count-first;
    //vertex shader
    float primitives[4*GL_MAX_VERTEX_ATTRIBS];
    if(!_no_program == GL_TRUE)
        vertex_shader(first, count, primitives);
    //clip coord
    perspective_division(numVerts, primitives);
    //normalized-device-coords
    viewport_transformation(numVerts, primitives);
    //rasterization
    float fragments[4*viewportTransform.w*viewportTransform.h];//color
    rasterization(numVerts, primitives, fragments, viewportTransform.w*viewportTransform.h);
    //fragment-shader
    fragment_shader();
    //per-vertex-ops
    per_vertex_operations();
    //entiendo que aqui escribe en frame buff
}

void _glDrawArraysTriangles(GLint first, GLsizei count) {
    gl_pipeline(first, count);
}

GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (first <0){
        _err= GL_INVALID_VALUE;
        return;
    }
    if (mode==GL_POINTS); // TODO
    else if (mode==GL_LINES); // TODO
    else if (mode==GL_TRIANGLES) 
        _glDrawArraysTriangles(first, count);
}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);

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

GL_APICALL void GL_APIENTRY glFinish (void);

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

GL_APICALL void GL_APIENTRY glGenFramebuffers (GLsizei n, GLuint *framebuffers) {
    GLuint _id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && _id < 256) {
        if (!_framebuffers[_id].used) {
            _framebuffers[_id].used = GL_TRUE;
            *framebuffers = _id;

            framebuffers += 1; 
            n -= 1;
        }
        _id += 1;
    }
}

#define CL_PROGRAM 0


GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    for (int i=1; i<MAX_PROGRAMS; i++)
        if (!programs[i].created)
        {
            programs[i].created=GL_TRUE;
            return i;
        }
    return 0;
}

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    if(!programs[program].binary == (void*)0)
        _err = GL_INVALID_OPERATION;
        return;
    if (binaryFormat == CL_PROGRAM){
        programs[program].binary=(*(cl_program*)binary);
        programs[program].length=length;
    }
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    if (program=0){
        _no_program=GL_TRUE;
        return;
    }
    if(programs[program].binary==(void*)0){
        _err = GL_INVALID_OPERATION;
        return;
    }
    _no_program=GL_FALSE;
    _current_program=programs[program];
}

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
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height){
    viewportTransform.x=x;
    viewportTransform.y=y;
    viewportTransform.w=width;
    viewportTransform.h=height;
}

GL_APICALL void GL_APIENTRY glDepthRangef (GLfloat n, GLfloat f){
    viewportTransform.n=n;
    viewportTransform.f=f;
}