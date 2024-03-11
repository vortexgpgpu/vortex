#include <GLSC2/glsc2.h>
#include <CL/opencl.h>

cl_context* _getContext() {
    static cl_context context;

    return &cl_context;
}

cl_int* _getError() {
    static cl_int err;

    return &err;
}


cl_mem BUFFER[255];
unsigned char buffer_tracker=0;

struct {
    bool enable;
    cl_mem mem_ptr;
} VertexAttrib;

VertexAttrib vertex_attrib[255];

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    unsigned int CL_MEM_TYPE;
    if (usage == GL_ARRAY_BUFFER) {
        CL_MEM_TYPE = CL_MEM_READ_ONLY;
    }

    BUFFER[target] = clCreateBuffer(_getContext(), CL_MEM_TYPE, size, data, _getError());
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask);
GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (mode==GL_TRIANGLES) {

    }
}
GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    vertex_attrib[index].enable = false;
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    vertex_attrib[index].enable = true;
}

GL_APICALL void GL_APIENTRY glGenBuffers (GLsizei n, GLuint *buffers) {
    for(GLsizei END=buffers+n; buffers!=END; ++buffers) {
        *buffers = buffer_tracker++;
    }
}


GL_APICALL void GL_APIENTRY glUseProgram (GLuint program);
GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    if (stride < 0) ; // ERROR
    unsigned int CL_MEM_TYPE;
    if (usage == GL_ARRAY_BUFFER) {
        CL_MEM_TYPE = CL_MEM_READ_ONLY;
    }
    if(normalized) { // normalize buffers

    }
    BUFFER[buffer_tracker++] = clCreateBuffer(_getContext(), CL_MEM_TYPE, size, pointer, _getError());
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height) {

}
