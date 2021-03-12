#include <vx_tex.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CORES_MAX 32

int vx_tex(unsigned t, unsigned u, unsigned v, unsigned lod){
    return vx_tex_ld(t,u,v,lod);
}
