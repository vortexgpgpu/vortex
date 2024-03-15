struct vec4{
	int x;
	int y;
	int z;
    int w;
};

struct vec3{
	int x;
	int y;
	int z;
};

float get_area(vec4 p1, vec4 p2, vec4 p3){
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
vec3 get_baricentric_coords(vec4 fp, vec4 v1, vec4 v2, vec4 v3){
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
___kernel void rasterization(__global unsigned int numVertex,
                        __global const vec4 *primitives,
	                    __global vec4 *fragcolor,
                        __global unsigned int grid_size)
{
	//get fragment center in wc
	int gid = get_global_id();
	//iterate over triangles
	vec4* it = primitives;
	vec4 fragColor = fragcolor[4*gid];

	for (int i =0; i< numVertex; i++, it++){
		vec3 abc = get_baricentric_coords(it++,it++,it++);
		//check inside
		if (abc.a+abc.b+abc.c != 1.f)
			continue;
		//rasterization
		vec4 t1_color = primitives[2*numVertex+i];
		vec4 t2_color = primitives[2*numVertex+i+1];
		vec4 t3_color = primitives[2*numVertex+i+2];
		
		fragColor.x = abc.a*t1_color.x + abc.b*t2_color.x + abc.c*t3_color.x;
		fragColor.y = abc.a*t1_color.y + abc.b*t2_color.y + abc.c*t3_color.y;
		fragColor.z = abc.a*t1_color.z + abc.b*t2_color.z + abc.c*t3_color.z;
	}
}

__kernel void perspective_division(__global const float *cc)
{
  int gid = get_global_id(0);

  float* write_ndc = &cc[4*gid];
  
  *write_ndc++/=w;
  *write_ndc++/=w;
  *write_ndc/=w;
}

__kernel void viewport_division(__global const float *ndc,
                                    __global const int px,
                                    __global const int py,
                                    __global const int cx,
                                    __global const int cy,
                                    __global const float n,
                                    __global const float f)
{
  int gid = get_global_id(0);
  float ndx = ndc[4*gid];
  float ndy = ndc[4*gid+1];
  float ndz = ndc[4*gid+2];


    float* write_wc = &ndc[4*gid];
  //wcx
  *write_wc++=(px/2)*xdx + cx;
  //wcy
  *write_wc++=(h/2)*ndy + cy;
  //wcz
  *write_wc=(f-n)/2 *ndz + (n+f)/2;
}