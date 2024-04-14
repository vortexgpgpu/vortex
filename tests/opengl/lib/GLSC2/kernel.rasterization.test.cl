__kernel void gl_rasterization_triangle (
    __global float4 *gl_FragCoords,
    __global bool *gl_Discard,
    const int index
)
{
    int gid = get_global_id(0);
    // input values
    __global float4 *fragCoord = gl_FragCoords + gid;

    fragCoord->x = 1.f;
    fragCoord->y = 1.f;
    fragCoord->z = 1.f;
    fragCoord->w = 1.f;

    gl_Discard[gid] = false;
}
