float cross(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}


float3 calculateBarycentricCoords(float2 point, float2 v0, float2 v1, float2 v2) {

    float3 barycentricCoords;

    float areaABC = cross((float3)(v2 - v0, 0), (float3)(v1 - v0, 0)).z;
    float areaPBC = cross((float3)(v2 - point, 0), (float3)(v1 - point, 0)).z;
    float areaPCA = cross((float3)(v0 - point, 0), (float3)(v2 - point, 0)).z;
    float areaPAB = cross((float3)(v1 - point, 0), (float3)(v0 - point, 0)).z;

    barycentricCoords.x = areaPBC / areaABC;
    barycentricCoords.y = areaPCA / areaABC;
    barycentricCoords.z = areaPAB / areaABC;

    return barycentricCoords;
}

float2 projectToScreenSpace(float4 point3D) {
    //Later on multiplicar por la proyection matrix
    return (float2)(point3D.x, point3D.y);
}

float calculateFragmentDepth(float3 barycentricCoords, float depthV0, float depthV1, float depthV2) {
    return barycentricCoords.x * depthV0 + barycentricCoords.y * depthV1 + barycentricCoords.z * depthV2;
}

bool stencilTestPassed(float3 barycentricCoords, __global int* stencilBuffer, int index) {

    int referenceMask = 0xFFFFFFFF;

    int triangleMask = 0xFF; // Supongamos que solo necesitamos 8 bits de stencil

    int maskedStencilValue = stencilBuffer[index] & referenceMask;

    if ((maskedStencilValue & triangleMask) != 0) {
        return true;
    } else {
        return false;
    }
}

void blendFragmentColor(__global float4* framebufferColor, float4 fragmentColor) {
    // Ejemplo de alpha blending
    float alpha = fragmentColor.w;

    framebufferColor->xyz = (1.0f - alpha) * framebufferColor->xyz + alpha * fragmentColor.xyz;
}

__kernel void fragmentShaderBasic(__global float4* fragColor) {
    
    fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f); // RED
}

__kernel void gl_main_fs(
	__global unsigned int numVertex,
	__global const vec4 *primitives,
	__global vec4 *fragcolor,
	__global unsigned int grid_size,
    __global float *grid,
	__void* vbo,
    int useFragmentShader,
    __global float* depthBuffer,
    __global int* stencilBuffer
) {
	int gid = get_global_id(0);

	int primitive_size = 4; // always attribute 0 is refered to gl_position
	int attribute = 1;
	while(attributes[attribute].active) {
		primitive_size += attributes[attribute].size;
		++attribute;
	}

	rasterization(numVertex, primitives, fragcolor, grid_size, useFragmentShader);

	gl_perspective_division(primitives + gid*primitive_size);

	gl_viewport_division(primitives + gid*primitive_size, viewport, depth_range);
}

void rasterization(__global unsigned int numVertex,
                        __global const float4 *primitives,
	                    __global float4 *fragcolor,
                        __global unsigned int grid_size,
                        __global const float *wc,
	                    __global float *grid,
						__void* vbo,
                        int useFragmentShader,
                        __global float* depthBuffer,
                        __global int* stencilBuffer,
                        __global float4* outputBuffer,
                        )
{
	//get fragment center in wc
	int x = get_global_id(0);
    int y = get_global_id(1);

    //frag coords norm
    float xf = (float)x / width;
    float yf = (float)y / height;

    float index = int index = y * WIDTH + x;

	//iterate over triangles
	float4* it = primitives;
	float4* fragColor = fragcolor[4*get_global_id()];

	for (int i =0; i< numVertex; i++, it++){

        float4 v0 = it++;
        float4 v1 = it++;
        float4 v2 = it++;

        float2 projectedV0 = projectToScreenSpace(v0);
        float2 projectedV1 = projectToScreenSpace(v1);
        float2 projectedV2 = projectToScreenSpace(v2);

		float3 abc = get_baricentric_coords((float2) (xf,yf), projectedV0, projectedV1, projectedV2);

        if (barycentricCoords.x >= 0 && barycentricCoords.y >= 0 && barycentricCoords.z >= 0) {
            
            float fragmentDepth = calculateFragmentDepth(barycentricCoords, v0.z, v1.z, v2.z);
            if (fragmentDepth < depthBuffer[index]) {
                depthBuffer[index] = fragmentDepth;
                outputBuffer[index] = fragmentData[index];

                if (stencilTestPassed(barycentricCoords, stencilBufferm index)) {
                    
                    //blendFragmentColor(&outputBuffer[index], fragColor);

                     if (useFragmentShader == 1) {
                        // Ejecutar el Fragment Shader
                        fragmentShader(&fragColor);
                        return; 
                    }
                    else {
                        t1_color = primitives[2*numVertex+i];
                        t2_color = primitives[2*numVertex+i+1];
                        t3_color = primitives[2*numVertex+i+2];
                    
                        fragColor.x = abc.a*t1_color.x + abc.b*t2_color.x + abc.c*t3_color.x;
                        fragColor.y = abc.a*t1_color.y + abc.b*t2_color.y + abc.c*t3_color.y;
                        fragColor.z = abc.a*t1_color.z + abc.b*t2_color.z + abc.c*t3_color.z;
                        return;
                    }
                }
            }       
        }
	}
}