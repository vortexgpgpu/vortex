
__kernel void perspective_division(__global const float *ndc,
                                    __global const int px,
                                    __global const int py,
                                    __global const int cx,
                                    __global const int cy,
                                    __global const float n,
                                    __global const float f,
	                    __global float *wc)
{
  int gid = get_global_id(0);
  float ndx = cc[3*gid];
  float ndy = cc[3*gid+1];
  float ndz = cc[3*gid+2];


    float* write_wc = &wc[3*gid];
  //wcx
  *write_wc++=(px/2)*xdx + cx;
  //wcy
  *write_wc++=(h/2)*ndy + cy;
  //wcz
  *write_wc=(f-n)/2 *ndz + (n+f)/2;
}