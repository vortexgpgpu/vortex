#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA

// Our definitions
#define MAX_BUFFER 256
#define MAX_FRAMEBUFFER 256
#define MAX_RENDERBUFFER 256
#define MAX_PROGRAMS 256
#define MAX_UNIFORM_VECTORS 16
#define MAX_NAME_SIZE 64
#define MAX_INFO_SIZE 256
#define MAX_UNIFORM_SIZE sizeof(float)*4*4 // Limited to a matf4x4

// OpenGL required definitions
#define MAX_VERTEX_ATTRIBS 16
#define MAX_VERTEX_UNIFORM_VECTORS 16 // atMost MAX_UNIFORM_VECTORS
#define MAX_FRAGMENT_UNIFORM_VECTORS 16 // atMost MAX_UNIFORM_VECTORS


/****** DTO objects ******\
 * TODO: externalize to could be imported from kernel cl programs
*/
typedef struct { 
    int type; // byte, ubyte, short, ushort, float 
    int size; 
    void* mem; 
} _attribute_pointer;

typedef struct { int values[4]; } _attribute_int;

typedef struct { float values[4]; } _attribute_float;

typedef union {
    _attribute_int  int4; // 0
    _attribute_float  float4; // 1
    _attribute_pointer pointer; // 2
} _attribute;

typedef struct __attribute__ ((packed)) {
    int type; // 0, 1, 2
    _attribute attribute;
} attribute;

typedef struct {

} FRAGMENT_DATA;

/****** GENERIC objects ******\
 * 
 * 
*/
typedef struct {
    GLint x;
    GLint y;
    GLsizei width;
    GLsizei height;
} BOX;

typedef struct {
    unsigned char info[MAX_INFO_SIZE];
    int length;
} LOG;

BOX _viewport;

/****** PROGRAM objects ******\
 * 
 * 
*/
typedef struct {
    GLint location, size, type;
    unsigned char name[MAX_NAME_SIZE]; 
    uint8_t data[MAX_UNIFORM_SIZE]; // TODO: Use union types
} UNIFORM;

typedef struct {
    GLint location, size, type;
    unsigned char name[MAX_NAME_SIZE]; 
    attribute data;
} ATTRIBUTE;

typedef struct {
    GLboolean used;
    GLuint object_name;
    GLboolean load_status, validation_status;
    LOG log;
    GLuint active_uniforms, active_attributes;
    UNIFORM uniforms[MAX_UNIFORM_VECTORS];
    ATTRIBUTE attributes[16];
    cl_program program;
} PROGRAM;

PROGRAM _programs[MAX_PROGRAMS];
GLuint _current_program; // ZERO is reserved for NULL program

/****** BUFFER objects ******\
 * TODO: Re think this, I think it is actually more tricky than the first though. 
 * Seams that the program object holds also the vertex attributes, and the VAO is on 
 * server side.
 * 
*/

typedef struct {
    GLboolean used;
    GLenum target;
    cl_mem mem;
} BUFFER;

typedef struct {
    GLboolean enable, normalized;
    GLuint index;
    GLint size;
    GLenum type;
    GLsizei stride;
    const void *pointer;
} VERTEX_ATTRIB;

BUFFER _buffers[MAX_BUFFER];
GLuint _buffer_binding;
VERTEX_ATTRIB _vertex_attrib[MAX_VERTEX_ATTRIBS];

/****** FRAMEBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    GLuint color_attachment0, depth_attachment, stencil_attachment;
    GLboolean used;
} FRAMEBUFFER;

FRAMEBUFFER _framebuffers[MAX_FRAMEBUFFER];
GLuint _framebuffer_binding;

/****** RENDERBUFFER objects ******\
 * 
 * 
*/

typedef struct {
    cl_mem mem;
    GLenum internalformat;
    GLsizei width, height;
    GLboolean used;
} RENDERBUFFER;

RENDERBUFFER _renderbuffers[MAX_RENDERBUFFER];
GLuint _renderbuffer_binding;

/****** PER-FRAGMENT objects ******\
 * 
 * 
*/

// Color
typedef struct { GLboolean red, green, blue, alpha } COLOR_MASK;

COLOR_MASK _color_mask = {1, 1, 1, 1};
// Depth
typedef struct { GLfloat n, f } DEPTH_RANGE; // z-near & z-far

GLboolean   _depth_enabled = 0;
GLboolean   _depth_mask = 1;
GLenum      _depth_func = GL_LESS;
DEPTH_RANGE _depth_range = {0.0, 1.0};
// Scissor

GLuint _scissor_enabled = 0;
BOX _scissor_box;
// Stencil
typedef struct { GLboolean front, back } STENCIL_MASK;

GLuint _stencil_enabled = 0;
STENCIL_MASK _stencil_mask = {1, 1};
// TODO blending & dithering

/****** Interface for utils & inline functions ******\
 * Utility or inline function are implemented at the end of the file. 
*/
#define COLOR_ATTACHMENT0 _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0]
#define PROGRAM _programs[_current_program]

void* createVertexKernel(GLenum mode, GLint first, GLsizei count);
void* createPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* createViewportDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* createRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count);
void* createFragmentKernel(GLenum mode, GLint first, GLsizei count);
void* createColorKernel(GLenum mode, GLint first, GLsizei count);

/****** OpenGL Interface Implementations ******\
 * 
 * 
*/

GL_APICALL void GL_APIENTRY glBindBuffer (GLenum target, GLuint buffer) {
    if (!_buffers[buffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    
    _buffer_binding = buffer;
    _buffers[buffer].target = target;
}
GL_APICALL void GL_APIENTRY glBindFramebuffer (GLenum target, GLuint framebuffer) {
    if (!_framebuffers[framebuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_FRAMEBUFFER) {
        _framebuffer_binding = framebuffer;
    }
}

GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    if (target == GL_ARRAY_BUFFER) {
        if (usage == GL_STATIC_DRAW) {
            _buffers[_buffer_binding].mem = clCreateBuffer(_getContext(), CL_MEM_READ_ONLY, size, data, &_err);
        }
        else if (usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_DRAW) {
            _buffers[_buffer_binding].mem = clCreateBuffer(_getContext(), CL_MEM_READ_WRITE, size, data, &_err);
        }
    }
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    if(mask & GL_COLOR_BUFFER_BIT) glClearColor(0.0,0.0,0.0,0.0);
    if(mask & GL_DEPTH_BUFFER_BIT) glClearDepthf(1.0);
    if(mask & GL_STENCIL_BUFFER_BIT) glClearStencil(0);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
    RENDERBUFFER color_attachment0 = _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0];

    if (color_attachment0.internalformat == GL_RGBA4) {
        unsigned short color;
        color |= (unsigned short) (15*red);
        color |= (unsigned short) (15*green) << 4;
        color |= (unsigned short) (15*blue) << 8;
        color |= (unsigned short) (15*alpha) << 12;
        fill(color_attachment0.mem, color_attachment0.width*color_attachment0.height*2, &color, 2);
    }
}
GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    RENDERBUFFER depth_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment];

    if (depth_attachment.internalformat == GL_DEPTH_COMPONENT16) {
        unsigned short value = 65535*d;
        fill(depth_attachment.mem, depth_attachment.width*depth_attachment.height*2, &value, 2);
    }
}
GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    RENDERBUFFER stencil_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment];

    if (stencil_attachment.internalformat == GL_STENCIL_INDEX8) {
        fill(stencil_attachment.mem, stencil_attachment.width*stencil_attachment.height, &s, 1);
    }
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    _color_mask.red = red;
    _color_mask.green = green;
    _color_mask.blue = blue;
    _color_mask.alpha = alpha;
}

GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    GLuint program = 0; // ZERO is reserved
    while(! ++program < MAX_PROGRAMS) {
        if (!_programs[program].used)
        {
            _programs[program].used=GL_TRUE;
            _programs[program].active_attributes=0;
            _programs[program].active_uniforms=0;
            _programs[program].load_status=0;
            _programs[program].validation_status=0;
            return program;
        }
    }
    return 0; // TODO maybe throw some error ??
}

// TODO move this to another file
void _glDrawArraysTriangles(GLint first, GLsizei count) {
    gl_pipeline(first, count);
}

GL_APICALL void GL_APIENTRY glDepthFunc (GLenum func) {
    _depth_func = func;
}

GL_APICALL void GL_APIENTRY glDepthMask (GLboolean flag) {
    _depth_mask = flag;
}

GL_APICALL void GL_APIENTRY glDepthRangef (GLfloat n, GLfloat f) {
    _depth_range.n=n;
    _depth_range.f=f;
}

/**
 * TODO: first is expected to be 0
*/
GL_APICALL void GL_APIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count) {
    if (first <0){
        _err= GL_INVALID_VALUE;
        return;
    }
    
    GLsizei num_vertices = count-first;
    GLsizei num_fragments = COLOR_ATTACHMENT0.width * COLOR_ATTACHMENT0.height;
    GLsizei num_primitives = num_vertices;
    if (mode==GL_LINES) num_primitives /= 2;
    else if (mode==GL_TRIANGLES) num_primitives /= 3;

    // Build memory buffers
    void *gl_Positions = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices);
    void *gl_Primitives = createBuffer(MEM_READ_WRITE, sizeof(float[4])*PROGRAM.active_attributes);
    void *gl_Rasterization = createBuffer(MEM_READ_WRITE, sizeof(float[4])*PROGRAM.active_attributes);
    void *gl_FragCoord = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments);
    void *gl_Discard = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments);
    void *gl_FragColor = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments);

    // Set up kernels
    void* vertex_kernel = createVertexKernel(mode, first, count);
    setKernelArg(vertex_kernel,
        PROGRAM.active_attributes + PROGRAM.active_uniforms,
        sizeof(gl_Positions), gl_Positions
    );
    setKernelArg(vertex_kernel,
        PROGRAM.active_attributes + PROGRAM.active_uniforms + 1,
        sizeof(gl_Primitives), gl_Primitives
    );

    void* perspective_division_kernel = createPerspectiveDivisionKernel(mode, first, count);
    setKernelArg(perspective_division_kernel, 0,
        sizeof(gl_Positions), gl_Positions
    );
    void* viewport_division_kernel = createViewportDivisionKernel(mode, first, count);
    setKernelArg(viewport_division_kernel, 0,
        sizeof(gl_Positions), gl_Positions
    );
    void* rasterization_kernel; 
    if (mode==GL_TRIANGLES) {
        rasterization_kernel = createRasterizationTriangleKernel(mode, first, count);
        setKernelArg(rasterization_kernel, 4,
            sizeof(gl_FragCoord), gl_Positions
        );
        setKernelArg(rasterization_kernel, 5,
            sizeof(gl_Primitives), gl_Primitives
        );
        setKernelArg(rasterization_kernel, 6,
            sizeof(gl_FragCoord), gl_FragCoord
        );
        setKernelArg(rasterization_kernel, 7,
            sizeof(gl_Rasterization), gl_Rasterization
        );
        setKernelArg(rasterization_kernel, 8,
            sizeof(gl_Discard), gl_Discard
        );
    }

    void* fragment_kernel = createFragmentKernel(mode, first, count);
    setKernelArg(fragment_kernel, 
        PROGRAM.active_uniforms,
        sizeof(gl_FragCoord), gl_FragCoord
    );
    setKernelArg(fragment_kernel, 
        PROGRAM.active_uniforms + 1,
        sizeof(gl_Discard), gl_Discard
    );
    setKernelArg(fragment_kernel, 
        PROGRAM.active_uniforms + 2,
        sizeof(gl_FragColor), gl_FragColor
    );
    setKernelArg(fragment_kernel, 
        PROGRAM.active_uniforms + 3,
        sizeof(gl_Rasterization), gl_Rasterization
    );

    void *color_kernel = createColorKernel(mode, first, count);
    setKernelArg(fragment_kernel, 3,
        sizeof(gl_FragCoord), gl_FragCoord
    );
    setKernelArg(fragment_kernel, 4,
        sizeof(gl_Discard), gl_Discard
    );
    setKernelArg(fragment_kernel, 5,
        sizeof(gl_FragColor), gl_FragColor
    );

    // Run Queue
    void *command_queue = createCommandQueue(0);
    // VERTEX

    enqueueNDRangeKernel(command_queue, vertex_kernel, &num_vertices);
    enqueueNDRangeKernel(command_queue, perspective_division_kernel, &num_vertices);
    enqueueNDRangeKernel(command_queue, viewport_division_kernel, &num_vertices);
    // POST-VERTEX
    
    for(uint32_t primitive=0; primitive < num_primitives; ++primitive) {
        // RASTERIZE
        setKernelArg(rasterization_kernel, 0, // TODO: Check in the implementation .cl
            sizeof(primitive), &primitive
        );
        enqueueNDRangeKernel(command_queue, rasterization_kernel, &num_fragments);   
        // FRAGMENT
        enqueueNDRangeKernel(command_queue, fragment_kernel, &num_fragments);   
        // POST-FRAGMENT
        enqueueNDRangeKernel(command_queue, color_kernel, &num_fragments);
    }

}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);

GL_APICALL void GL_APIENTRY glDisable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 0;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 0;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 0;
}

GL_APICALL void GL_APIENTRY glDisableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    _vertex_attrib[index].enable = 0;
}

GL_APICALL void GL_APIENTRY glEnable (GLenum cap) {
    if (cap == GL_SCISSOR_TEST)
        _scissor_enabled = 1;
    else if (cap == GL_DEPTH_TEST)
        _depth_enabled = 1;
    else if (cap == GL_STENCIL_TEST)
        _stencil_enabled = 1;
}

GL_APICALL void GL_APIENTRY glEnableVertexAttribArray (GLuint index) {
    if (index >= GL_MAX_VERTEX_ATTRIBS) {
        _err = GL_INVALID_VALUE;
        return;
    }

    _vertex_attrib[index].enable = 1;
}

GL_APICALL void GL_APIENTRY glFinish (void) {
    // finish(); TODO
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

#define POCL_BINARY 0x0

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    if(!_programs[program].program == (void*)0) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (binaryFormat == POCL_BINARY) {
        _programs[program].program=createProgramWithBinary(binary, length);
    }
}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        if (_framebuffer_binding) {
            RENDERBUFFER color_attachment0 = _renderbuffers[_framebuffers[_framebuffer_binding].color_attachment0];

            unsigned int src_format;
            if (color_attachment0.internalformat == GL_RGBA4) src_format = GL_RGBA4;

            readnPixels(color_attachment0.mem,x,y,width,height, src_format, RGBA8, bufSize, data);
        }
    }
}

GL_APICALL void GL_APIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height) {
    if (width < 0 || height < 0) {
        _err = GL_INVALID_VALUE;
        return;
    }
    _scissor_box.x = x;
    _scissor_box.y = y;
    _scissor_box.width = width;
    _scissor_box.height = height;
}

GL_APICALL void GL_APIENTRY glStencilMask (GLuint mask) {
    _stencil_mask.front = mask;
    _stencil_mask.back = mask;
}
GL_APICALL void GL_APIENTRY glStencilMaskSeparate (GLenum face, GLuint mask) {
    if (GL_FRONT == face) 
        _stencil_mask.front = mask;
    else if (GL_BACK == face) 
        _stencil_mask.back = mask;
    else if (GL_FRONT_AND_BACK == face) {
        _stencil_mask.front = mask;
        _stencil_mask.back = mask;
    }
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    if (program) {
        if (!_programs[program].load_status){
            _err = GL_INVALID_OPERATION;
            return;
        }
        // TODO install program
    }
    _current_program=program;
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

    _vertex_attrib[index].size = size;
    _vertex_attrib[index].type = type;
    _vertex_attrib[index].normalized = normalized;
    _vertex_attrib[index].stride = stride;
    _vertex_attrib[index].pointer = pointer;
}
GL_APICALL void GL_APIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height){
    _viewport.x=x;
    _viewport.y=y;
    _viewport.width=width;
    _viewport.height=height;
}

/**** Utils & inline functions ****\
 *
*/
void* createVertexKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_vs");
    // VAO locations
    GLuint attribute;
    while(attribute < _programs[_current_program].active_attributes) {
        
        if(0); // TODO diff between pointer and not

        setKernelArg(
            kernel, 
            _programs[_current_program].attributes[attribute].location,
            _programs[_current_program].attributes[attribute].size,
            &_programs[_current_program].attributes[attribute].data // TODO: 
        );
        ++attribute;
    }
    // Uniform locations
    GLuint uniform;
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location,
            _programs[_current_program].uniforms[uniform].size, 
            _programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }
    

    return kernel;
}

void* createPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = 0; // TODO

    return kernel;
}

void* createViewportDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = 0; // TODO

    setKernelArg(kernel, 1,
        sizeof(_viewport), &_viewport
    );
    setKernelArg(kernel, 2,
        sizeof(_depth_range), &_depth_range
    );

    return kernel;
}

void* createRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = 0; // TODO

    setKernelArg(kernel, 0,
        sizeof(COLOR_ATTACHMENT0.width), &COLOR_ATTACHMENT0.width
    );
    setKernelArg(kernel, 1,
        sizeof(COLOR_ATTACHMENT0.height), &COLOR_ATTACHMENT0.height
    );
    setKernelArg(kernel, 2,
        sizeof(PROGRAM.active_attributes), &PROGRAM.active_attributes
    );

    return kernel;
}

void* createFragmentKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_fs");
    // Uniform locations
    GLuint uniform;
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location,
            _programs[_current_program].uniforms[uniform].size, 
            _programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }

    return kernel;
}

void* createColorKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = 0; // TODO
    // Uniform locations
    setKernelArg(kernel, 0,
        sizeof(COLOR_ATTACHMENT0.width), &COLOR_ATTACHMENT0.width
    );
    setKernelArg(kernel, 1,
        sizeof(COLOR_ATTACHMENT0.height), &COLOR_ATTACHMENT0.height
    );
    setKernelArg(kernel, 2,
        sizeof(COLOR_ATTACHMENT0.mem), COLOR_ATTACHMENT0.mem
    );

    return kernel;
}