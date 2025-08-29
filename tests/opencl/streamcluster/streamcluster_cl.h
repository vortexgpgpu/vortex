/***********************************************
	streamcluster_cl.h
	: parallelized code of streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with OpenCL API has been applied by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 15/03/2010
***********************************************/

#define THREADS_PER_BLOCK 256
#define MAXBLOCKS 65536
//#define PROFILE_TMP
#define __CL_ENABLE_EXCEPTIONS
#include "CLHelper.h"
typedef struct {
  float weight;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point_Struct;

/* host memory analogous to device memory */
float *work_mem_h;
float *coord_h;
float *gl_lower;
Point_Struct *p_h;

/* device memory */
cl_mem work_mem_d;
cl_mem coord_d;
cl_mem  center_table_d;
cl_mem  switch_membership_d;
cl_mem p_d;

static int c;			// counters

void quit(char *message){
	printf("%s\n", message);
	exit(1);
}
//free memory
void freeDevMem(){
	try{
	_clFree(work_mem_d);
	_clFree(center_table_d);
	_clFree(switch_membership_d);
	_clFree(p_d);
	_clFree(coord_d);
	/*if(work_mem_h!=NULL)
		free(work_mem_h);*/
	_clFreeHost(1, work_mem_h);
	
	if(coord_h!=NULL)
		free(coord_h);
	if(gl_lower!=NULL)
		free(gl_lower);
	if(p_h!=NULL)
		free(p_h);
	}
	catch(string msg){
		quit(&(msg[0]));
	}	
}

//allocate device memory together
void allocDevMem(int num, int dim, int kmax){
	try{
		work_mem_d = _clMalloc(kmax * num * sizeof(float));
		center_table_d = _clMalloc(num * sizeof(int));
		switch_membership_d = _clMalloc(num * sizeof(char));
		p_d = _clMalloc(num * sizeof(Point));
		coord_d = _clMalloc(num * dim * sizeof(float));
	}
	catch(string msg){
		quit(&(msg[0]));
	}	
}
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, char *switch_membership,
							double *serial, double *cpu_gpu_memcpy, double *memcpy_back, double *gpu_malloc, double *kernel){
	float gl_cost = 0;
	try{
#ifdef PROFILE_TMP
	double t1 = gettime();
#endif
	int K	= *numcenters ;						// number of centers
	int num    =   points->num;				// number of points
	int dim     =   points->dim;				// number of dimension
	kmax++;
	/***** build center index table 1*****/
	int count = 0;
	for( int i=0; i<num; i++){
		if( is_center[i] )
			center_table[i] = count++;
	}
	
#ifdef PROFILE_TMP
	double t2 = gettime();
	*serial += t2 - t1;
#endif

	/***** initial memory allocation and preparation for transfer : execute once -1 *****/
	if( c == 0 ) {
#ifdef PROFILE_TMP
		double t3 = gettime();
#endif
	coord_h = (float*) malloc( num * dim * sizeof(float));								// coordinates (host)
	gl_lower = (float*) malloc( kmax * sizeof(float) );
	work_mem_h = (float*)_clMallocHost(kmax*num*sizeof(float));
	p_h = (Point_Struct*)malloc(num*sizeof(Point_Struct));	//by cambine: not compatibal with original Point
	
	// prepare mapping for point coordinates
	//--cambine: what's the use of point coordinates? for computing distance.
	for(int i=0; i<dim; i++){
		for(int j=0; j<num; j++)
			coord_h[ (num*i)+j ] = points->p[j].coord[i];
	}
#ifdef PROFILE_TMP		
	double t4 = gettime();
	*serial += t4 - t3;
#endif

	allocDevMem(num, dim, kmax);
#ifdef PROFILE_TMP
	double t5 = gettime();
	*gpu_malloc += t5 - t4;
#endif
		
	// copy coordinate to device memory	
	_clMemcpyH2D(coord_d, coord_h, num*dim*sizeof(float));		
#ifdef PROFILE_TMP
	double t6 = gettime();
	*cpu_gpu_memcpy += t6 - t5;
#endif
	}	
#ifdef PROFILE_TMP
	double t100 = gettime();
#endif

	for(int i=0; i<num; i++){
		p_h[i].weight = ((points->p)[i]).weight;
		p_h[i].assign = ((points->p)[i]).assign;
		p_h[i].cost = ((points->p)[i]).cost;	
	}

#ifdef PROFILE_TMP
	double t101 = gettime();
	*serial += t101 - t100;
#endif
#ifdef PROFILE_TMP
	double t7 = gettime();
#endif
	/***** memory transfer from host to device *****/
	/* copy to device memory */
	_clMemcpyH2D(center_table_d,  center_table, num*sizeof(int));
	_clMemcpyH2D(p_d,  p_h,  num * sizeof(Point_Struct));
#ifdef PROFILE_TMP
	double t8 = gettime();
	*cpu_gpu_memcpy += t8 - t7;
#endif

	/***** kernel execution *****/
	/* Determine the number of thread blocks in the x- and y-dimension */
	size_t smSize = dim * sizeof(float);
#ifdef PROFILE_TMP
	double t9 = gettime();
#endif
	_clMemset(switch_membership_d, 0, num*sizeof(char));
	_clMemset(work_mem_d, 0, (K+1)*num*sizeof(float));
	//--cambine: set kernel argument
	int kernel_id = 0;
	int arg_idx = 0;		
	
	kernel_id = 1;
	arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, p_d);
	_clSetArgs(kernel_id, arg_idx++, coord_d);
	_clSetArgs(kernel_id, arg_idx++, work_mem_d);
	_clSetArgs(kernel_id, arg_idx++, center_table_d);
	_clSetArgs(kernel_id, arg_idx++, switch_membership_d);
	_clSetArgs(kernel_id, arg_idx++, 0, smSize);
	_clSetArgs(kernel_id, arg_idx++, &num, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &x, sizeof(long));
	_clSetArgs(kernel_id, arg_idx++, &K, sizeof(int));

	
	_clInvokeKernel(kernel_id, num, THREADS_PER_BLOCK); //--cambine: start kernel here
	_clFinish();
	
	#ifdef PROFILE_TMP
	double t10 = gettime();
	*kernel += t10 - t9;
#endif

	/***** copy back to host for CPU side work *****/
	_clMemcpyD2H(switch_membership, switch_membership_d, num * sizeof(char));	
	_clMemcpyD2H(work_mem_h, work_mem_d, (K+1) *num*sizeof(float));

#ifdef PROFILE_TMP
	double t11 = gettime();
	*memcpy_back += t11 - t10;
#endif
		
	/****** cpu side work *****/
	int numclose = 0;
	gl_cost = z;
	
	/* compute the number of centers to close if we are to open i */
	for(int i=0; i < num; i++){	//--cambine:??
		if( is_center[i] ) {
			float low = z;
			//printf("i=%d  ", i);
		    for( int j = 0; j < num; j++ )
				low += work_mem_h[ j*(K+1) + center_table[i] ];
			//printf("low=%f\n", low);		
		    gl_lower[center_table[i]] = low;
				
		    if ( low > 0 ) {
				numclose++;				
				work_mem_h[i*(K+1)+K] -= low;
		    }
		}
		gl_cost += work_mem_h[i*(K+1)+K];
	}

	/* if opening a center at x saves cost (i.e. cost is negative) do so
		otherwise, do nothing */
	if ( gl_cost < 0 ) {
		for(int i=0; i<num; i++){
		
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
		    if ( (switch_membership[i]=='1') || close_center ) {
				points->p[i].cost = points->p[i].weight * dist(points->p[i], points->p[x], points->dim);
				points->p[i].assign = x;
		    }
	    }
		
		for(int i=0; i<num; i++){
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
				is_center[i] = false;
		}
		
		is_center[x] = true;
		*numcenters = *numcenters +1 - numclose;
	}
	else
		gl_cost = 0;  // the value we'

#ifdef PROFILE_TMP
	double t12 = gettime();
	*serial += t12 - t11;
#endif
	c++;
	}
	catch(string msg){
		printf("--cambine:%s\n", msg.c_str());
		freeDevMem();
		_clRelease();
		exit(-1);		
	}
	catch(...){
		printf("--cambine: unknow reasons in pgain\n");
	}

	/*FILE *fp = fopen("data_opencl.txt", "a");
	fprintf(fp,"%d, %f\n", c, gl_cost);
	fclose(fp);*/
	//printf("%d, %f\n", c, gl_cost);
	//exit(-1);
	return -gl_cost;
}
