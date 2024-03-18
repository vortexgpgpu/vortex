#include <GLSC2/glsc2.h>

typedef struct {
    GLuint framebuffers[2];
    GLsizei activated;
} CONTEXT;

void generateContext(CONTEXT* _context, GLuint w, GLuint h) {
    glGenFramebuffers(2, _context->framebuffers);
    glBindFramebuffer(GL_FRAMEBUFFER ,_context->framebuffers[1]);
    glBufferData(GL_FRAMEBUFFER,w*h*4*sizeof(float), (void *) 0, GL_STATIC_DRAW);

    _context->activated=0;
    glBindFramebuffer(GL_FRAMEBUFFER ,_context->framebuffers[0]);
    glBufferData(GL_FRAMEBUFFER,w*h*4*sizeof(float), (void *) 0, GL_STATIC_DRAW);
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