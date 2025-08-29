#ifndef _COMMON_H
#define _COMMON_H

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

#define DIVANDRND(a, b) ((((a) % (b)) != 0) ? ((a) / (b) + 1) : ((a) / (b)))
/*
#  define cudaCheckError( msg ) {                                            \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "%s: %i: %s: %s.\n",                                 \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) );         \
        exit(-1);                                                            \
    } }

#  define cudaCheckAsyncError( msg ) {                                       \
    cudaThreadSynchronize();                                                 \
    cudaCheckError( msg );                                                   \
    }
*/

#endif
