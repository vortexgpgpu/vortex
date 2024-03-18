struct vec3{
	int x;
	int y;
	int z;
};

float get_area(vec3 p1, vec3 p2, vec3 p3){
	float b = sqrt(pow(v2.x-v1.x,2) + pow(v2.y-v1.y,2));
	//get b/2
	b = dv1v2/2;
	//get h
	vec3 mid_point;
	mid_point.x = v1.x + v2.x-v1.x;
	mid_point.y = v1.y + v2.y-v1.y;
	float h = sqrt(pow(v3.x-mid_point.x,2) + pow(v3.y-mid_point.y,2));

	return b*h/2;
}

//get the area
vec3 get_baricentric_coords(vec3 fp, vec3 v1, vec3 v2, vec3 v3){
	vec3 abc;
	float a_v1v2v3 = get_area(v1,v2,v3);
	
	float a_pv1v2 = get_area(fp,v1,v2);
	float a_pv2v3 = get_area(fp,v2,v3);
	float a_pv1v3 = get_area(fp,v1,v3);
	
	abc.x=a_pv1v2/a_v1v2v3;
	abc.y=a_pv1v2/a_v1v2v3;
	abc.z=a_pv1v2/a_v1v2v3;
}

//per fragment
___kernel void rasterization(__global const float *wc,
	                    __global float *grid,
						__void* vbo)
{
	//get fragment center in wc
	int gid = get_global_id();
	int w = gid%width;
	int h = gid/width;
	//iterate over triangles
	vec3* it = (vec3*)wc;
	vec3 fragColor = grid[gid];

	while ((void*)it != (void*)0){
		vec3 abc = get_baricentric_coords(it++,it++,it++);
		//check inside
		if (abc.a+abc.b+abc.c != 1.f)
			continue;
		//rasterization
		vec3 t1_color = vbo[color];
		vec3 t2_color = vbo[color];
		vec3 t3_color = vbo[color];
		
		fragColor.x = abc.a*t1_color.x + abc.b*t2_color.x + abc.c*t3_color.x;
		fragColor.y = abc.a*t1_color.y + abc.b*t2_color.y + abc.c*t3_color.y;
		fragColor.z = abc.a*t1_color.z + abc.b*t2_color.z + abc.c*t3_color.z;
	}
}