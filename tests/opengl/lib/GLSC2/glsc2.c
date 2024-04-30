#include <GLSC2/glsc2.h>
#include "kernel.c" // TODO: may be interesting to extract it to an interface so could be re implementated with CUDA
#include "binary.c"

#define NOT_IMPLEMENTED              \
    ({                               \
        printf("NOT_IMPLEMENTED");   \
        exit(0);                     \
    })

// Our definitions
#define MAX_BUFFER 256
#define MAX_FRAMEBUFFER 256
#define MAX_RENDERBUFFER 256
#define MAX_TEXTURE 256
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

GLboolean _kernel_load_status;
void *_color_kernel;
void *_rasterization_kernel;
void *_viewport_division_kernel;
void *_perspective_division_kernel;
void *_readnpixels_kernel;

/****** BUFFER objects ******\
 * TODO: Re think this, I think it is actually more tricky than the first though. 
 * Seams that the program object holds also the vertex attributes, and the VAO is on 
 * server side.
 * 
*/

typedef struct {
    GLboolean used;
    GLenum target;
    void* mem;
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

/****** TEXTURE 2D objects ******\
 * 
 * 
*/

typedef struct {
    GLsizei width, height;
    cl_mem mem;
} SAMPLER_2D;

typedef struct {
    cl_mem mem;
    GLenum internalformat;
    GLsizei width, height;
    GLboolean used;
} TEXTURE_2D;

TEXTURE_2D _textures[MAX_RENDERBUFFER];
GLuint _texture_binding;

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

void* getCommandQueue();

void* getPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getViewportDivisionKernel(GLenum mode, GLint first, GLsizei count);
void* getRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count);
void* getColorKernel(GLenum mode, GLint first, GLsizei count);

void* createVertexKernel(GLenum mode, GLint first, GLsizei count);
void* createFragmentKernel(GLenum mode, GLint first, GLsizei count);

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
    printf("glBindFramebuffer(framebuffer: %d)\n", framebuffer);

    if (!_framebuffers[framebuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_FRAMEBUFFER) {
        _framebuffer_binding = framebuffer;
    }
}
GL_APICALL void GL_APIENTRY glBindRenderbuffer (GLenum target, GLuint renderbuffer) {
    if (!_renderbuffers[renderbuffer].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_RENDERBUFFER) {
        _renderbuffer_binding = renderbuffer;
    }
}
GL_APICALL void GL_APIENTRY glBindTexture (GLenum target, GLuint texture) {
    if (!_textures[texture].used) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (target == GL_TEXTURE_2D) {
        _texture_binding = texture;
    }
}

GL_APICALL void GL_APIENTRY glRenderbufferStorage (GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {

    if (internalformat == GL_RGBA4) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*2, NULL);
    } else if (internalformat == GL_DEPTH_COMPONENT16) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*2, NULL);
    } else if (internalformat == GL_STENCIL_INDEX8) {
        _renderbuffers[_renderbuffer_binding].mem = createBuffer(MEM_READ_WRITE, width*height*1, NULL);
    } else {
        printf("NOT IMPLEMENTED\n");
        exit(0);
    }
    _renderbuffers[_renderbuffer_binding].internalformat = internalformat;
    _renderbuffers[_renderbuffer_binding].width = width;
    _renderbuffers[_renderbuffer_binding].height = height;
    
}


GL_APICALL void GL_APIENTRY glBufferData (GLenum target, GLsizeiptr size, const void *data, GLenum usage) {

    if (target == GL_ARRAY_BUFFER) {
        if (usage == GL_STATIC_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_ONLY | MEM_COPY_HOST_PTR, size, data);
        }
        else if (usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_DRAW) {
            _buffers[_buffer_binding].mem = createBuffer(MEM_READ_WRITE | MEM_COPY_HOST_PTR, size, data);
        }
    }
}

GL_APICALL void GL_APIENTRY glClear (GLbitfield mask) {
    if(mask & GL_COLOR_BUFFER_BIT) glClearColor(0.0,0.0,0.0,1.0);
    if(mask & GL_DEPTH_BUFFER_BIT) glClearDepthf(1.0);
    if(mask & GL_STENCIL_BUFFER_BIT) glClearStencil(0);
}

GL_APICALL void GL_APIENTRY glClearColor (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {

    if (COLOR_ATTACHMENT0.internalformat == GL_RGBA4) {
        unsigned int color = 0;
        color |= (unsigned int) (red*15);
        color |= (unsigned int) (green*15) << 4;
        color |= (unsigned int) (blue*15) << 8;
        color |= (unsigned int) (alpha*15) << 12;
        color |= color << 16;
        enqueueFillBuffer(getCommandQueue(), COLOR_ATTACHMENT0.mem, &color, 4, 0, COLOR_ATTACHMENT0.width*COLOR_ATTACHMENT0.height*2);
    
    } else {
        printf("NOT IMPLEMENTED");
        exit(0);
    }
}
// TODO:
GL_APICALL void GL_APIENTRY glClearDepthf (GLfloat d) {
    RENDERBUFFER depth_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].depth_attachment];

    if (depth_attachment.internalformat == GL_DEPTH_COMPONENT16) {
        unsigned short value = 65535*d;
        //fill(depth_attachment.mem, depth_attachment.width*depth_attachment.height*2, &value, 2);
    }
}
// TODO:
GL_APICALL void GL_APIENTRY glClearStencil (GLint s) {
    RENDERBUFFER stencil_attachment = _renderbuffers[_framebuffers[_framebuffer_binding].stencil_attachment];

    if (stencil_attachment.internalformat == GL_STENCIL_INDEX8) {
        //fill(stencil_attachment.mem, stencil_attachment.width*stencil_attachment.height, &s, 1);
    }
}

GL_APICALL void GL_APIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    _color_mask.red = red;
    _color_mask.green = green;
    _color_mask.blue = blue;
    _color_mask.alpha = alpha;
}

GL_APICALL GLuint GL_APIENTRY glCreateProgram (void){
    GLuint program = 1; // ZERO is reserved
    while(program < MAX_PROGRAMS) {
        if (!_programs[program].used) {
            _programs[program].used=GL_TRUE;
            return program;
        } 
        ++program;
    }
    return 0; // TODO maybe throw some error ??
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
    void *gl_Positions = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices, NULL);
    void *gl_Primitives = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_vertices*_programs[_current_program].active_attributes, NULL);
    void *gl_Rasterization = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments*_programs[_current_program].active_attributes, NULL);
    void *gl_FragCoord = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments, NULL);
    void *gl_Discard = createBuffer(MEM_READ_WRITE, sizeof(uint8_t)*num_fragments, NULL);
    void *gl_FragColor = createBuffer(MEM_READ_WRITE, sizeof(float[4])*num_fragments, NULL);

    // Set up kernels
    void* vertex_kernel = createVertexKernel(mode, first, count);
    setKernelArg(vertex_kernel,
        _programs[_current_program].active_attributes + _programs[_current_program].active_uniforms,
        sizeof(gl_Positions), &gl_Positions
    );

    setKernelArg(vertex_kernel,
        _programs[_current_program].active_attributes + _programs[_current_program].active_uniforms + 1,
        sizeof(gl_Primitives), &gl_Primitives
    );

    void* perspective_division_kernel = getPerspectiveDivisionKernel(mode, first, count);
    setKernelArg(perspective_division_kernel, 0,
        sizeof(gl_Positions), &gl_Positions
    );
    void* viewport_division_kernel = getViewportDivisionKernel(mode, first, count);
    setKernelArg(viewport_division_kernel, 0,
        sizeof(gl_Positions), &gl_Positions
    );
    void *rasterization_kernel;
    if (mode==GL_TRIANGLES) {
        rasterization_kernel = getRasterizationTriangleKernel(mode, first, count);
        setKernelArg(rasterization_kernel, 4,
            sizeof(gl_FragCoord), &gl_Positions
        );
        setKernelArg(rasterization_kernel, 5,
            sizeof(gl_Primitives), &gl_Primitives
        );
        setKernelArg(rasterization_kernel, 6,
            sizeof(gl_FragCoord), &gl_FragCoord
        );
        setKernelArg(rasterization_kernel, 7,
            sizeof(gl_Rasterization), &gl_Rasterization
        );
        setKernelArg(rasterization_kernel, 8,
            sizeof(gl_Discard), &gl_Discard
        );
    } else NOT_IMPLEMENTED;

    void* fragment_kernel = createFragmentKernel(mode, first, count);
    int active_textures = _texture_binding != 0;
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*3,
        sizeof(gl_FragCoord), &gl_FragCoord
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*3 + 1,
        sizeof(gl_Rasterization), &gl_Rasterization
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*3 + 2,
        sizeof(gl_Discard), &gl_Discard
    );
    setKernelArg(fragment_kernel, 
        _programs[_current_program].active_uniforms + active_textures*3 + 3,
        sizeof(gl_FragColor), &gl_FragColor
    );

    void *color_kernel = getColorKernel(mode, first, count);
    setKernelArg(color_kernel, 3,
        sizeof(gl_FragCoord), &gl_FragCoord
    );
    setKernelArg(color_kernel, 4,
        sizeof(gl_Discard), &gl_Discard
    );
    setKernelArg(color_kernel, 5,
        sizeof(gl_FragColor), &gl_FragColor
    );

    // Enqueue kernels
    void *command_queue = getCommandQueue();
    // Vertex
    enqueueNDRangeKernel(command_queue, vertex_kernel, num_vertices);
    // Post-Vertex
      float _gl_Positions[num_vertices][4]; 
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }
    
    enqueueNDRangeKernel(command_queue, perspective_division_kernel, num_vertices);
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }
    
    enqueueNDRangeKernel(command_queue, viewport_division_kernel, num_vertices);
    enqueueReadBuffer(command_queue, gl_Positions,sizeof(float[4])*num_vertices,_gl_Positions);
    for (int i = 0; i < num_vertices; i+=1) {
        printf("vertex %d, x=%f, y=%f, z=%f, w=%f\n", i, _gl_Positions[i][0],_gl_Positions[i][1],_gl_Positions[i][2], _gl_Positions[i][3]);
    }

    for(uint32_t primitive=0; primitive < num_primitives; ++primitive) {
        // Rasterization
        setKernelArg(rasterization_kernel, 0, sizeof(primitive), &primitive);
        enqueueNDRangeKernel(command_queue, rasterization_kernel, num_fragments);   
        // Fragment
        enqueueNDRangeKernel(command_queue, fragment_kernel, num_fragments);   
        
	// Post-Fragment
        enqueueNDRangeKernel(command_queue, color_kernel, num_fragments);
    }
    

}

GL_APICALL void GL_APIENTRY glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {}

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
    finish(getCommandQueue());
}

GL_APICALL void GL_APIENTRY glFramebufferRenderbuffer (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    if (attachment == GL_COLOR_ATTACHMENT0)
        _framebuffers[_framebuffer_binding].color_attachment0=renderbuffer;
    else if (attachment == GL_DEPTH_ATTACHMENT)
        _framebuffers[_framebuffer_binding].depth_attachment=renderbuffer;
    else if (attachment == GL_STENCIL_ATTACHMENT)
        _framebuffers[_framebuffer_binding].stencil_attachment=renderbuffer;
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
    GLuint id = 1; // _id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && id < MAX_FRAMEBUFFER) {
        if (!_framebuffers[id].used) {
            _framebuffers[id].used = GL_TRUE;
            
            *framebuffers = id;
            framebuffers += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenRenderbuffers (GLsizei n, GLuint *renderbuffers) {
    GLuint id = 1; // id = 0 is reserved for ARRAY_BUFFER

    while(n > 0 && id < MAX_RENDERBUFFER) {
        if (!_renderbuffers[id].used) {
            _renderbuffers[id].used = GL_TRUE;
            *renderbuffers = id;

            renderbuffers += 1; 
            n -= 1;
        }
        id += 1;
    }
}

GL_APICALL void GL_APIENTRY glGenTextures (GLsizei n, GLuint *textures) {
    GLuint id = 1;

    while(n > 0 && id < MAX_RENDERBUFFER) {
        if (!_textures[id].used) {
            _textures[id].used = GL_TRUE;
            *textures = id;

            textures += 1; 
            n -= 1;
        }
        id += 1;
    }
}

#define POCL_BINARY 0x0

GL_APICALL void GL_APIENTRY glProgramBinary (GLuint program, GLenum binaryFormat, const void *binary, GLsizei length){
    printf("glProgramBinary() program=%d, binaryFormat=%d\n",program,binaryFormat);
    if(!_kernel_load_status) {
        void *gl_program;
        gl_program = createProgramWithBinary(GLSC2_kernel_color_pocl, sizeof(GLSC2_kernel_color_pocl));
        buildProgram(gl_program);
        _color_kernel = createKernel(gl_program, "gl_rgba4");
        gl_program = createProgramWithBinary(GLSC2_kernel_rasterization_triangle_pocl, sizeof(GLSC2_kernel_rasterization_triangle_pocl));
        buildProgram(gl_program);
        _rasterization_kernel = createKernel(gl_program, "gl_rasterization_triangle");
        gl_program = createProgramWithBinary(GLSC2_kernel_viewport_division_pocl, sizeof(GLSC2_kernel_viewport_division_pocl));
        buildProgram(gl_program);
        _viewport_division_kernel = createKernel(gl_program, "gl_viewport_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_perspective_division_pocl, sizeof(GLSC2_kernel_perspective_division_pocl));
        buildProgram(gl_program);
        _perspective_division_kernel = createKernel(gl_program, "gl_perspective_division");
        gl_program = createProgramWithBinary(GLSC2_kernel_readnpixels_pocl, sizeof(GLSC2_kernel_readnpixels_pocl));
        buildProgram(gl_program);
        _readnpixels_kernel = createKernel(gl_program, "gl_rgba4_rgba8");

        _kernel_load_status = 1;
    }
    if(_programs[program].program) {
        _err = GL_INVALID_OPERATION;
        return;
    }
    if (binaryFormat == POCL_BINARY) {
        _programs[program].program=createProgramWithBinary(binary, length);
        buildProgram(_programs[program].program);
        // TODO: Check this logic
        _programs[program].load_status = GL_TRUE;
        _programs[program].validation_status = GL_TRUE;
    }
}

GL_APICALL void GL_APIENTRY glReadnPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data) {
    if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
        if (_framebuffer_binding) {

            unsigned int src_format;
            if (COLOR_ATTACHMENT0.internalformat == GL_RGBA4) src_format = GL_RGBA4;

            void *dst_buff = createBuffer(MEM_WRITE_ONLY, bufSize, NULL);

            setKernelArg(_readnpixels_kernel, 0, sizeof(COLOR_ATTACHMENT0.mem), &COLOR_ATTACHMENT0.mem);
            setKernelArg(_readnpixels_kernel, 1, sizeof(void*), &dst_buff);
            setKernelArg(_readnpixels_kernel, 2, sizeof(int), &x);
            setKernelArg(_readnpixels_kernel, 3, sizeof(int), &y);
            setKernelArg(_readnpixels_kernel, 4, sizeof(int), &width);
            setKernelArg(_readnpixels_kernel, 5, sizeof(int), &height);

            void *command_queue = getCommandQueue();
            size_t global_work_size = bufSize/4; // 4 bytes x color
            enqueueNDRangeKernel(command_queue, _readnpixels_kernel, global_work_size);
            enqueueReadBuffer(command_queue, dst_buff, bufSize, data);
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

GL_APICALL void GL_APIENTRY glTexStorage2D (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
    if ( target == GL_TEXTURE_2D && internalformat == GL_RGBA8) {
        _textures[_texture_binding].width = width;
        _textures[_texture_binding].height = height;
        _textures[_texture_binding].mem = createBuffer(MEM_READ_ONLY, width*height*sizeof(uint32_t), NULL);
    } else NOT_IMPLEMENTED;
}

GL_APICALL void GL_APIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
    if (target == GL_TEXTURE_2D) {
        if (format == GL_RGBA && type == GL_UNSIGNED_BYTE) {
            enqueueWriteBuffer(getCommandQueue(),_textures[_texture_binding].mem,width*height*sizeof(uint8_t[4]),pixels);
        } else NOT_IMPLEMENTED;
    } else NOT_IMPLEMENTED;
}

#define UMAT4 0x0;

GL_APICALL void GL_APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    if (transpose) NOT_IMPLEMENTED;
    if (!_current_program) NOT_IMPLEMENTED;
    if (count > 4) NOT_IMPLEMENTED;
    if (count < 1) NOT_IMPLEMENTED;

    uint32_t uniform_id = _programs[_current_program].active_uniforms;
    _programs[_current_program].uniforms[uniform_id].location = location;
    _programs[_current_program].uniforms[uniform_id].size = sizeof(float[4])*count;
    _programs[_current_program].uniforms[uniform_id].type = UMAT4;

    float *data_ptr = _programs[_current_program].uniforms[uniform_id].data;
    for(uint32_t i=0; i<count; ++i) {
        data_ptr[0] = *(value + 4*i);
        data_ptr[1] = *(value + 4*i + 1);
        data_ptr[2] = *(value + 4*i + 2);
        data_ptr[3] = *(value + 4*i + 3);
        data_ptr +=4;
    }
    _programs[_current_program].active_uniforms += 1;
}

GL_APICALL void GL_APIENTRY glUseProgram (GLuint program){
    printf("glUseProgram() program=%d\n", program);
    if (program) {
        if (!_programs[program].load_status){
            printf("\tERROR load_status=%d\n", _programs[program].load_status);

            _err = GL_INVALID_OPERATION;
            return;
        }
        // TODO install program
    }
    _current_program=program;
}

GL_APICALL void GL_APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer) {
    printf("glVertexAttribPointer() index=%d, size=%d, type=%d, stride=%d, pointer=%x\n", index, size, type, stride, pointer);
    if (index >= MAX_VERTEX_ATTRIBS) {
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

    if (type != GL_BYTE && type != GL_UNSIGNED_BYTE && type != GL_SHORT && type != GL_UNSIGNED_SHORT && type != GL_FLOAT){
        _err=GL_INVALID_VALUE;
        return;
    }
    // TODO: normalized & strid
    if (!_current_program) {
        // TODO:
    } else {
        if (_buffer_binding) {
            _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.mem = _buffers[_buffer_binding].mem;
            _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.type = 0x2;
        }
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.size = size;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].data.attribute.pointer.type = type;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].location = index;
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].size = sizeof(void*);
        _programs[_current_program].attributes[_programs[_current_program].active_attributes].type = type;
        
        _programs[_current_program].active_attributes += 1;
    
        printf("\nactive_attributes=%d\n", _programs[_current_program].active_attributes);
    }
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

void* getCommandQueue() {
    static void* command_queue;
    if (command_queue == NULL) command_queue = createCommandQueue(0);
    return command_queue;
}


void* createVertexKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_vs");
    // VAO locations
    GLuint attribute = 0;
    while(attribute < _programs[_current_program].active_attributes) {
        
        if(_programs[_current_program].attributes[attribute].data.type == 0x2) {
            setKernelArg(
                kernel, 
                _programs[_current_program].attributes[attribute].location,
                _programs[_current_program].attributes[attribute].size,
                &_programs[_current_program].attributes[attribute].data.attribute.pointer.mem
            );
        } else {
            NOT_IMPLEMENTED;
            setKernelArg(
                kernel, 
                _programs[_current_program].attributes[attribute].location,
                _programs[_current_program].attributes[attribute].size,
                _programs[_current_program].attributes[attribute].data.attribute.int4.values // TODO: 
            );
        }
        ++attribute;
    }
    // Uniform locations
    GLuint uniform = 0;
    GLuint active_attributes = _programs[_current_program].active_attributes; 
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location + active_attributes,
            _programs[_current_program].uniforms[uniform].size, 
            &_programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }
    

    return kernel;
}

void* getPerspectiveDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    return _perspective_division_kernel;
}

void* getViewportDivisionKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = _viewport_division_kernel;

    setKernelArg(kernel, 1,
        sizeof(_viewport), &_viewport
    );
    setKernelArg(kernel, 2,
        sizeof(_depth_range), &_depth_range
    );

    return kernel;
}

void* getRasterizationTriangleKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = _rasterization_kernel;

    setKernelArg(kernel, 1,
        sizeof(COLOR_ATTACHMENT0.width), &COLOR_ATTACHMENT0.width
    );
    setKernelArg(kernel, 2,
        sizeof(COLOR_ATTACHMENT0.height), &COLOR_ATTACHMENT0.height
    );
    setKernelArg(kernel, 3,
        sizeof(_programs[_current_program].active_attributes), &_programs[_current_program].active_attributes
    );

    return kernel;
}

void* createFragmentKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = createKernel(_programs[_current_program].program, "gl_main_fs");
    // Uniform locations
    GLuint uniform = 0;
    while(uniform < _programs[_current_program].active_uniforms) {
        setKernelArg(
            kernel, 
            _programs[_current_program].uniforms[uniform].location,
            _programs[_current_program].uniforms[uniform].size, 
            &_programs[_current_program].uniforms[uniform].data
            );
        ++uniform;
    }
    GLuint texture = 0;
    while(texture < 1) {
        SAMPLER_2D sampler;
        sampler.width = _textures[_texture_binding].width;
        sampler.height = _textures[_texture_binding].height;
        sampler.mem = _textures[_texture_binding].mem;
        setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms,
            sizeof(sampler.width), 
            &sampler.width
            );
	setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms+1,
            sizeof(sampler.height), 
            &sampler.height
            );
	setKernelArg(
            kernel, 
            _programs[_current_program].active_uniforms+2,
            sizeof(sampler.mem), 
            &sampler.mem
            );
	
        ++texture;
    }

    return kernel;
}

void* getColorKernel(GLenum mode, GLint first, GLsizei count) {
    void *kernel = _color_kernel;

    setKernelArg(kernel, 0,
        sizeof(COLOR_ATTACHMENT0.width), &COLOR_ATTACHMENT0.width
    );
    setKernelArg(kernel, 1,
        sizeof(COLOR_ATTACHMENT0.height), &COLOR_ATTACHMENT0.height
    );
    setKernelArg(kernel, 2,
        sizeof(COLOR_ATTACHMENT0.mem), &COLOR_ATTACHMENT0.mem
    );

    return kernel;
}
