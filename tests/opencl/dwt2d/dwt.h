
#ifndef _DWT_H
#define _DWT_H

template<typename T> 
int nStage2dDWT(T *in, T *out, T * backup, int pixWidth, int pixHeight, int stages, bool forward);

template<typename T>
int writeNStage2DDWT(T *component_cuda, int width, int height, 
                     int stages, const char * filename, const char * suffix);
template<typename T>
int writeLinear(T *component_cuda, int width, int height, 
                     const char * filename, const char * suffix);

#endif
