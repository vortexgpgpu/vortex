
__kernel void BFS( __global int* mask,
				   __global int* dist,
				   __global int* mat,
				   __global int* col_ind, 
				   __global int* vec_d, 				   
				   __global int* active_d, 
				   const int iter_d){
	int tid = get_global_id(0);
	int y, rw, point, ci, start;
	int a,b;
	a = tid*4;
	b=(tid+1)*4;
	start = mat[tid];
	int z=tid;
	//printf("iter inside is %d\n", iter_d);
	//int z;
/*for(z=a;z<b;z++)
{*/
	
	rw=mat[(z+1)]-mat[z]; 
	point=mat[z]; 
	//printf("rw is %d", rw);
	for(y=0;y<rw;y++)
	{
		ci=col_ind[(point+y)]; 
		if(mask[z] & vec_d[ci])
		{
		dist[z] = iter_d+1; 
		active_d[z] = 1;
		}
	}

    if(active_d[z])
    { 
        mask[z]=0;            
    }
//}

}
