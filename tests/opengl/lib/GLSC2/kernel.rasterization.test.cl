__kernel void gl_rasterization_triangle (
    const int index,
    const int width,
    const int height,
    const int attributes,
    __global const float4 *gl_Positions,
    __global const float4 *gl_Primitives,
    __global float4 *gl_FragCoords,
    __global float4 *gl_Rasterization,
    __global bool *gl_Discard
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
