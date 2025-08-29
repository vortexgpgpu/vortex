#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <error.h>
/*#include "dwt_cuda/dwt.h"
#include "dwt_cuda/common.h"
#include "dwt.h"
#include "common.h"
*/

#include "common.h"
#include "dwt.h"
#include "common.h"
#include "dwt_cl.h"

// inline void fdwt(float *in, float *out, int width, int height, int levels)
// {
//         dwt_cuda::fdwt97(in, out, width, height, levels);
// }

inline void fdwt(int *in, int *out, int width, int height, int levels)
{
        dwt_cuda::fdwt53(in, out, width, height, levels);
}

// inline void rdwt(float *in, float *out, int width, int height, int levels)
// {
//         dwt_cuda::rdwt97(in, out, width, height, levels);
// }

// inline void rdwt(int *in, int *out, int width, int height, int levels)
// {
//         dwt_cuda::rdwt53(in, out, width, height, levels);
// }

template<typename T>
int nStage2dDWT(T * in, T * out, T * backup, int pixWidth, int pixHeight, int stages, bool forward)
{
    //need add segments
}
template int nStage2dDWT<float>(float*, float*, float*, int, int, int, bool);
template int nStage2dDWT<int>(int*, int*, int*, int, int, int, bool);

void samplesToChar(unsigned char * dst, float * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++) {
        float r = (src[i]+0.5f) * 255;
        if (r > 255) r = 255; 
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}

// void samplesToChar(unsigned char * dst, int * src, int samplesNum)
// {
//     int i;

//     for(i = 0; i < samplesNum; i++) {
//         int r = src[i]+128;
//         if (r > 255) r = 255;
//         if (r < 0)   r = 0; 
//         dst[i] = (unsigned char)r;
//     }
// }

///* Write output linear orderd*/
template<typename T>
int writeLinear(T *component_cuda, int pixWidth, int pixHeight,
                const char * filename, const char * suffix)
{
	//need add segments

}
template int writeLinear<float>(float *component_cuda, int pixWidth, int pixHeight, const char * filename, const char * suffix); 
template int writeLinear<int>(int *component_cuda, int pixWidth, int pixHeight, const char * filename, const char * suffix); 



/* Write output visual ordered */
template<typename T>
int writeNStage2DDWT(T *component_cuda, int pixWidth, int pixHeight, 
                     int stages, const char * filename, const char * suffix) 
{
    //need add segments
}
template int writeNStage2DDWT<float>(float *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 
template int writeNStage2DDWT<int>(int *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 




