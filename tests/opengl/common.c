#include <GLSC2/glsc2.h>

typedef struct {
    GLuint framebuffers[1];
    GLuint renderbuffers[3];
    GLsizei activated;
} CONTEXT;

void generateContext(CONTEXT* _context, GLuint width, GLuint height) {
    //Build the main framebuffer
    glGenFramebuffers(1, _context->framebuffers);
    glBindFramebuffer(GL_FRAMEBUFFER, _context->framebuffers[0]);

    //Build the renderbuffers
    glGenRenderbuffers(3, _context->renderbuffers);
    glBindRenderbuffer(GL_RENDERBUFFER, _context->renderbuffers[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, _context->renderbuffers[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, _context->renderbuffers[2]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, width, height);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _context->renderbuffers[0]);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _context->renderbuffers[1]);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _context->renderbuffers[2]);


}

void swapBuffers(CONTEXT* _context) {
    GLsizei to_activate = 1;
    if (_context->activated == 1) to_activate = 0;

    glBindFramebuffer(GL_FRAMEBUFFER ,_context->framebuffers[to_activate]);
    _context->activated = to_activate;
}

void writeBuffer(CONTEXT* _context) {
    //glReadnPixels()
    //fprintf(FILE,char*)
}