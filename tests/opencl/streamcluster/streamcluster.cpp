/***********************************************
	streamcluster.cpp
	: original source code of streamcluster with minor
		modification regarding function calls
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	:revised by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 15/03/2011
***********************************************/

#include "streamcluster.h"
#include "CLHelper.h"
#include "streamcluster_cl.h"

using namespace std;

#define MAXNAMESIZE 1024 	// max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 							// number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 3 						// iterate ITER* k log k times; ITER >= 1

//#define PRINTINFO 			//comment this out to disable output
//#define PROFILE_TMP 					// comment this out to disable instrumentation code
//#define ENABLE_THREADS  // comment this out to disable threads
//#define INSERT_WASTE 		//uncomment this to insert waste computation into dist function

#define CACHE_LINE 512 		// cache line in byte


/* global */
static char *switch_membership;	//whether to switch membership in pgain
static bool *is_center;						//whether a point is a center
static int  *center_table;					//index table of centers
static int nproc; 								//# of threads

/* timing info */
static double serial;
static double cpu_gpu_memcpy;
static double memcpy_back;
static double gpu_malloc;
static double kernel;
static double gpu_free;
static int cnt_speedy;

// instrumentation code
#ifdef PROFILE_TMP
double time_local_search;
double time_speedy;
double time_select_feasible;
double time_gain;
double time_shuffle;
double time_gain_dist;
double time_gain_init;
double time_FL;
#endif 

void inttofile(int data, char *filename){
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "%d ", data);
	fclose(fp);	
}

int isIdentical(float *i, float *j, int D){
// tells whether two points of D dimensions are identical

  int a = 0;
  int equal = 1;

  while (equal && a < D) {
    if (i[a] != j[a]) equal = 0;
    else a++;
  }
  if (equal) return 1;
  else return 0;

}

/* comparator for floating point numbers */
static int floatcomp(const void *i, const void *j)
{
  float a, b;
  a = *(float *)(i);
  b = *(float *)(j);
  if (a > b) return (1);
  if (a < b) return (-1);
  return(0);
}

/* shuffle points into random order */
void shuffle(Points *points)
{
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif
  long i, j;
  Point temp;
  for (i=0;i<points->num-1;i++) {
    j=(lrand48()%(points->num - i)) + i;
    temp = points->p[i];
    points->p[i] = points->p[j];
    points->p[j] = temp;
  }
#ifdef PROFILE_TMP
  double t2 = gettime();
  time_shuffle += t2-t1;
#endif
}

/* shuffle an array of integers */
void intshuffle(int *intarray, int length)
{
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif
  long i, j;
  int temp;
  for (i=0;i<length;i++) {
    j=(lrand48()%(length - i))+i;
    temp = intarray[i];
    intarray[i]=intarray[j];
    intarray[j]=temp;
  }
#ifdef PROFILE_TMP
  double t2 = gettime();
  time_shuffle += t2-t1;
#endif
}

#ifdef INSERT_WASTE
float waste(float s )
{
  for( int i =0 ; i< 4; i++ ) {
    s += pow(s,0.78);
  }
  return s;
}
#endif

/* compute Euclidean distance squared between two points */
float dist(Point p1, Point p2, int dim)
{
  int i;
  float result=0.0;
  for (i=0;i<dim;i++)
    result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);
#ifdef INSERT_WASTE
  float s = waste(result);
  result += s;
  result -= s;
#endif
  return(result);
}

/* run speedy on the points, return total cost of solution */
float pspeedy(Points *points, float z, long *kcenter, int pid, pthread_barrier_t* barrier)
{
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif
cnt_speedy++;
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif
  //my block
  long bsize = points->num/nproc;
  long k1 = bsize * pid;
  long k2 = k1 + bsize;
  if( pid == nproc-1 ) k2 = points->num;
  static float totalcost;
  
  static bool open = false;
  static float* costs; //cost for each thread. 
  static int i;

#ifdef ENABLE_THREADS
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
#endif

#ifdef PRINTINFO
  if( pid == 0 ){
    fprintf(stderr, "Speedy: facility cost %lf\n", z);
  }
#endif

  /* create center at first point, send it to itself */
  for( int k = k1; k < k2; k++ )    {
    float distance = dist(points->p[k],points->p[0],points->dim);
    points->p[k].cost = distance * points->p[k].weight;
    points->p[k].assign=0;
  }

  if( pid==0 )   {
    *kcenter = 1;
    costs = (float*)malloc(sizeof(float)*nproc);
  }
    
  if( pid != 0 ) { // we are not the master threads. we wait until a center is opened.
    while(1) {
#ifdef ENABLE_THREADS
      pthread_mutex_lock(&mutex);
      while(!open) pthread_cond_wait(&cond,&mutex);
      pthread_mutex_unlock(&mutex);
#endif
      if( i >= points->num ) break;
      for( int k = k1; k < k2; k++ )
	{
	  float distance = dist(points->p[i],points->p[k],points->dim);
	  if( distance*points->p[k].weight < points->p[k].cost )
	    {
	      points->p[k].cost = distance * points->p[k].weight;
	      points->p[k].assign=i;
	    }
	}
#ifdef ENABLE_THREADS
      pthread_barrier_wait(barrier);
      pthread_barrier_wait(barrier);
#endif
    } 
  }
  else  { // I am the master thread. I decide whether to open a center and notify others if so. 
    for(i = 1; i < points->num; i++ )  {
      bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z); //--cambine: what standard?
      if( to_open )  {
	(*kcenter)++;
#ifdef ENABLE_THREADS
	pthread_mutex_lock(&mutex);
#endif
	open = true;
#ifdef ENABLE_THREADS
	pthread_mutex_unlock(&mutex);
	pthread_cond_broadcast(&cond);
#endif
	for( int k = k1; k < k2; k++ )  {	//--cambine: for a new open, compute new cost and center.
	  float distance = dist(points->p[i],points->p[k],points->dim);
	  if( distance*points->p[k].weight < points->p[k].cost )  {
	    points->p[k].cost = distance * points->p[k].weight;
	    points->p[k].assign=i;
	  }
	}
#ifdef ENABLE_THREADS
	pthread_barrier_wait(barrier);
#endif
	open = false;
#ifdef ENABLE_THREADS
	pthread_barrier_wait(barrier);
#endif
      }
    }
#ifdef ENABLE_THREADS
    pthread_mutex_lock(&mutex);
#endif
    open = true;
#ifdef ENABLE_THREADS
    pthread_mutex_unlock(&mutex);
    pthread_cond_broadcast(&cond);
#endif
  }
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif
  open = false;
  float mytotal = 0;
  for( int k = k1; k < k2; k++ )  {
    mytotal += points->p[k].cost;
  }
  costs[pid] = mytotal;
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif
  // aggregate costs from each thread
  if( pid == 0 )
    {
      totalcost=z*(*kcenter);
      for( int i = 0; i < nproc; i++ )
	{
	  totalcost += costs[i];
	} 
      free(costs);
    }
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif

#ifdef PRINTINFO
  if( pid == 0 )
    {
      fprintf(stderr, "Speedy opened %d facilities for total cost %lf\n",
	      *kcenter, totalcost);
      fprintf(stderr, "Distance Cost %lf\n", totalcost - z*(*kcenter));
    }
#endif

#ifdef PROFILE_TMP
  double t2 = gettime();
  if( pid== 0 ) {
    time_speedy += t2 -t1;
  }
#endif
  return(totalcost);
}


/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float pFL(Points *points, int *feasible, int numfeasible,
	  float z, long *k, int kmax, float cost, long iter, float e, 
	  int pid, pthread_barrier_t* barrier)
{
#ifdef PROFILE_TMP
	double t1 = gettime();
#endif
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif
  long i;
  long x;
  float change;
  long numberOfPoints;

  change = cost;
  /* continue until we run iter iterations without improvement */
  /* stop instead if improvement is less than e */
  while (change/cost > 1.0*e) {
    change = 0.0;
    numberOfPoints = points->num;
    /* randomize order in which centers are considered */

    if( pid == 0 ) {
      intshuffle(feasible, numfeasible);
    }
#ifdef ENABLE_THREADS
    pthread_barrier_wait(barrier);
#endif
    for (i=0;i<iter;i++) {
	    x = i%numfeasible;
	    //printf("--cambine: feasible x=%ld, z=%f, k=%ld, kmax=%d\n", x, z, *k, kmax);
	    change += pgain(feasible[x], points, z, k, kmax, is_center, center_table, switch_membership,
												&serial, &cpu_gpu_memcpy, &memcpy_back, &gpu_malloc, &kernel);
    }		
    cost -= change;
#ifdef PRINTINFO
    if( pid == 0 ) {
      fprintf(stderr, "%d centers, cost %lf, total distance %lf\n",
	      *k, cost, cost - z*(*k));
    }
#endif
#ifdef ENABLE_THREADS
    pthread_barrier_wait(barrier);
#endif
  }
#ifdef PROFILE_TMP
	double t2 = gettime();
	time_FL += t2 - t1;
#endif
  return(cost);
}

int selectfeasible_fast(Points *points, int **feasible, int kmin, int pid, pthread_barrier_t* barrier)
{
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif

  int numfeasible = points->num;
  if (numfeasible > (ITER*kmin*log((float)kmin)))
    numfeasible = (int)(ITER*kmin*log((float)kmin));
  *feasible = (int *)malloc(numfeasible*sizeof(int));
  
  float* accumweight;
  float totalweight;

  /* 
     Calcuate my block. 
     For now this routine does not seem to be the bottleneck, so it is not parallelized. 
     When necessary, this can be parallelized by setting k1 and k2 to 
     proper values and calling this routine from all threads ( it is called only
     by thread 0 for now ). 
     Note that when parallelized, the randomization might not be the same and it might
     not be difficult to measure the parallel speed-up for the whole program. 
   */
  //  long bsize = numfeasible;
  long k1 = 0;
  long k2 = numfeasible;

  float w;
  int l,r,k;

  /* not many points, all will be feasible */
  if (numfeasible == points->num) {
    for (int i=k1;i<k2;i++)
      (*feasible)[i] = i;
    return numfeasible;
  }

  accumweight= (float*)malloc(sizeof(float)*points->num);
  accumweight[0] = points->p[0].weight;
  totalweight=0;
  for( int i = 1; i < points->num; i++ ) {
    accumweight[i] = accumweight[i-1] + points->p[i].weight;
  }
  totalweight=accumweight[points->num-1];

  for(int i=k1; i<k2; i++ ) {
    w = (lrand48()/(float)INT_MAX)*totalweight;
    //binary search
    l=0;
    r=points->num-1;
    if( accumweight[0] > w )  { 
      (*feasible)[i]=0; 
      continue;
    }
    while( l+1 < r ) {
      k = (l+r)/2;
      if( accumweight[k] > w ) {
	r = k;
      } 
      else {
	l=k;
      }
    }
    (*feasible)[i]=r;
  }
  free(accumweight); 
#ifdef PROFILE_TMP
  double t2 = gettime();
  time_select_feasible += t2-t1;
#endif
  return numfeasible;
}

/* compute approximate kmedian on the points */
float pkmedian(Points *points, long kmin, long kmax, long* kfinal,
	       int pid, pthread_barrier_t* barrier )
{
  int i;
  float cost;
  float lastcost;
  float hiz, loz, z;

  static long k;
  static int *feasible;
  static int numfeasible;
  static float* hizs;

  if( pid==0 ) hizs = (float*)calloc(nproc,sizeof(float));
  hiz = loz = 0.0;
  long numberOfPoints = points->num;
  long ptDimension = points->dim;

  //my block
  long bsize = points->num/nproc;
  long k1 = bsize * pid;
  long k2 = k1 + bsize;
  if( pid == nproc-1 ) k2 = points->num;

#ifdef PRINTINFO
  if( pid == 0 )
    {
      printf("Starting Kmedian procedure\n");
      printf("%i points in %i dimensions\n", numberOfPoints, ptDimension);
    }
#endif

#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif

  float myhiz = 0;
  for (long kk=k1;kk < k2; kk++ ) {
    myhiz += dist(points->p[kk], points->p[0],
		      ptDimension)*points->p[kk].weight;
  }
  hizs[pid] = myhiz;

#ifdef ENABLE_THREADS  
  pthread_barrier_wait(barrier);
#endif

  for( int i = 0; i < nproc; i++ )   {
    hiz += hizs[i];
  }

  loz=0.0; z = (hiz+loz)/2.0;
  /* NEW: Check whether more centers than points! */
  if (points->num <= kmax) {	//--cambine: just ignore for the timebeing
    /* just return all points as facilities */
    for (long kk=k1;kk<k2;kk++) {
      points->p[kk].assign = kk;
      points->p[kk].cost = 0;
    }
    cost = 0;
    if( pid== 0 ) {
      free(hizs); 
      *kfinal = k;
    }
    return cost;
  }

  if( pid == 0 ) shuffle(points);	//--cambine: why need shuffle?
  cost = pspeedy(points, z, &k, pid, barrier);
#ifdef PRINTINFO
  if( pid == 0 )
    printf("thread %d: Finished first call to speedy, cost=%lf, k=%i\n",pid,cost,k);
#endif
  i=0;
  /* give speedy SP chances to get at least kmin/2 facilities */
  while ((k < kmin)&&(i<SP)) {
    cost = pspeedy(points, z, &k, pid, barrier);
    i++;
  }

#ifdef PRINTINFO
  if( pid==0)
    printf("thread %d: second call to speedy, cost=%lf, k=%d\n",pid,cost,k);
#endif 
  /* if still not enough facilities, assume z is too high */
  while (k < kmin) {
#ifdef PRINTINFO
    if( pid == 0 ) {
      printf("%lf %lf\n", loz, hiz);
      printf("Speedy indicates we should try lower z\n");
    }
#endif
    if (i >= SP) {hiz=z; z=(hiz+loz)/2.0; i=0;}
    if( pid == 0 ) shuffle(points);
    cost = pspeedy(points, z, &k, pid, barrier);
    i++;
  }
 
  /* now we begin the binary search for real */
  /* must designate some points as feasible centers */
  /* this creates more consistancy between FL runs */
  /* helps to guarantee correct # of centers at the end */
  
  if( pid == 0 )
    {
      numfeasible = selectfeasible_fast(points,&feasible,kmin,pid,barrier); //--cambine?
      for( int i = 0; i< points->num; i++ ) {
	is_center[points->p[i].assign]= true;
      }
    }
	
#ifdef ENABLE_THREADS
  pthread_barrier_wait(barrier);
#endif

  while(1) {
#ifdef PRINTINFO
    if( pid==0 )
      {
	printf("loz = %lf, hiz = %lf\n", loz, hiz);
	printf("Running Local Search...\n");
      }
#endif
    /* first get a rough estimate on the FL solution */
    //    pthread_barrier_wait(barrier);		
    lastcost = cost;
    cost = pFL(points, feasible, numfeasible,
	       z, &k, kmax, cost, (long)(ITER*kmax*log((float)kmax)), 0.1, pid, barrier);
    /* if number of centers seems good, try a more accurate FL */
    if (((k <= (1.1)*kmax)&&(k >= (0.9)*kmin))||
	((k <= kmax+2)&&(k >= kmin-2))) {
#ifdef PRINTINFO
      if( pid== 0)
	{
	  printf("Trying a more accurate local search...\n");
	}
#endif
      /* may need to run a little longer here before halting without
	 improvement */
	 
      cost = pFL(points, feasible, numfeasible,
		 z, &k, kmax, cost, (long)(ITER*kmax*log((float)kmax)), 0.001, pid, barrier);
    }

    if (k > kmax) {
      /* facilities too cheap */
      /* increase facility cost and up the cost accordingly */
      loz = z; z = (hiz+loz)/2.0;
      cost += (z-loz)*k;
    }
    if (k < kmin) {
      /* facilities too expensive */
      /* decrease facility cost and reduce the cost accordingly */
      hiz = z; z = (hiz+loz)/2.0;
      cost += (z-hiz)*k;
    }

    /* if k is good, return the result */
    /* if we're stuck, just give up and return what we have */
    if (((k <= kmax)&&(k >= kmin))||((loz >= (0.999)*hiz)) )
      { 
	break;
      }
#ifdef ENABLE_THREADS
    pthread_barrier_wait(barrier);
#endif
  }

  //clean up...
  if( pid==0 ) {
    free(feasible); 
    free(hizs);
    *kfinal = k;
  }

  return cost;
}

/* compute the means for the k clusters */
int contcenters(Points *points)
{
  long i, ii;
  float relweight;

  for (i=0;i<points->num;i++) {
    /* compute relative weight of this point to the cluster */
    if (points->p[i].assign != i) {
      relweight=points->p[points->p[i].assign].weight + points->p[i].weight;

      relweight = points->p[i].weight/relweight;
      for (ii=0;ii<points->dim;ii++) {
				points->p[points->p[i].assign].coord[ii]*=1.0-relweight;
				points->p[points->p[i].assign].coord[ii]+=
				points->p[i].coord[ii]*relweight;
      }
      points->p[points->p[i].assign].weight += points->p[i].weight;
    }
  }
  
  return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, long* centerIDs, long offset)
{
  long i;
  long k;

  bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

  /* mark the centers */
  for ( i = 0; i < points->num; i++ ) {
    is_a_median[points->p[i].assign] = 1;
  }

  k=centers->num;

  /* count how many  */
  for ( i = 0; i < points->num; i++ ) {
    if ( is_a_median[i] ) {
      memcpy( centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
      centers->p[k].weight = points->p[i].weight;
      centerIDs[k] = i + offset;
      k++;
    }
  }

  centers->num = k;
  free(is_a_median);
}



void* localSearchSub(void* arg_) {
  pkmedian_arg_t* arg= (pkmedian_arg_t*)arg_;
  pkmedian(arg->points,arg->kmin,arg->kmax,arg->kfinal,arg->pid,arg->barrier);

  return NULL;
}

void localSearch( Points* points, long kmin, long kmax, long* kfinal ) {
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif

    pthread_barrier_t barrier;
#ifdef ENABLE_THREADS
    pthread_barrier_init(&barrier,NULL,nproc);
#endif
    pthread_t* threads = new pthread_t[nproc];
    pkmedian_arg_t* arg = new pkmedian_arg_t[nproc];


    for( int i = 0; i < nproc; i++ ) {
      arg[i].points = points;
      arg[i].kmin = kmin;
      arg[i].kmax = kmax;
      arg[i].pid = i;
      arg[i].kfinal = kfinal;

      arg[i].barrier = &barrier;
#ifdef ENABLE_THREADS
      pthread_create(threads+i,NULL,localSearchSub,(void*)&arg[i]);
#else
      localSearchSub(&arg[0]);
#endif
    }

    for ( int i = 0; i < nproc; i++) {
#ifdef ENABLE_THREADS
      pthread_join(threads[i],NULL);
#endif
    }

    delete[] threads;
    delete[] arg;
#ifdef ENABLE_THREADS
    pthread_barrier_destroy(&barrier);
#endif

#ifdef PROFILE_TMP
  double t2 = gettime();
  time_local_search += t2-t1;
#endif
 
}


void outcenterIDs( Points* centers, long* centerIDs, char* outfile ) {
  FILE* fp = fopen(outfile, "w");
  if( fp==NULL ) {
    fprintf(stderr, "error opening %s\n",outfile);
    exit(1);
  }
  int* is_a_median = (int*)calloc( sizeof(int), centers->num );
  for( int i =0 ; i< centers->num; i++ ) {
    is_a_median[centers->p[i].assign] = 1;
  }

  for( int i = 0; i < centers->num; i++ ) {
    if( is_a_median[i] ) {
      fprintf(fp, "%u\n", centerIDs[i]);
      fprintf(fp, "%lf\n", centers->p[i].weight);
      for( int k = 0; k < centers->dim; k++ ) {
	fprintf(fp, "%lf ", centers->p[i].coord[k]);
      }
      fprintf(fp,"\n\n");
    }
  }
  fclose(fp);
}

void streamCluster( PStream* stream, 
		    long kmin, long kmax, int dim,
		    long chunksize, long centersize, char* outfile )
{

  float* block = (float*)malloc( chunksize*dim*sizeof(float) );
  float* centerBlock = (float*)malloc(centersize*dim*sizeof(float) );
  long* centerIDs = (long*)malloc(centersize*dim*sizeof(long));

  if( block == NULL ) { 
    fprintf(stderr,"not enough memory for a chunk!\n");
    exit(1);
  }

  Points points;
  points.dim = dim;
  points.num = chunksize;
  points.p = (Point *)malloc(chunksize*sizeof(Point));
  for( int i = 0; i < chunksize; i++ ) {
    points.p[i].coord = &block[i*dim];		
  }
	

  Points centers;
  centers.dim = dim;
  centers.p = (Point *)malloc(centersize*sizeof(Point));
  centers.num = 0;

  for( int i = 0; i< centersize; i++ ) {
    centers.p[i].coord = &centerBlock[i*dim];
    centers.p[i].weight = 1.0;
  }

  long IDoffset = 0;
  long kfinal;
  while(1) {

    size_t numRead  = stream->read(block, dim, chunksize ); 
    fprintf(stderr,"read %d points\n",numRead);

    if( stream->ferror() || numRead < (unsigned int)chunksize && !stream->feof() ) {
      fprintf(stderr, "error reading data!\n");
      exit(1);
    }

    points.num = numRead;
    for( int i = 0; i < points.num; i++ ) {
      points.p[i].weight = 1.0;
    }

    switch_membership = (char*)malloc(points.num*sizeof(char));
    is_center = (bool*)calloc(points.num,sizeof(bool));
    center_table = (int*)malloc(points.num*sizeof(int));

    localSearch(&points,kmin, kmax,&kfinal);

    fprintf(stderr,"finish local search\n");
    contcenters(&points);
    if( kfinal + centers.num > centersize ) {
      //here we don't handle the situation where # of centers gets too large. 
      fprintf(stderr,"oops! no more space for centers\n");
      exit(1);
    }

#ifdef PRINTINFO
    printf("finish cont center\n");
#endif

    copycenters(&points, &centers, centerIDs, IDoffset);
    IDoffset += numRead;

#ifdef PRINTINFO
    printf("finish copy centers\n"); 
#endif
    free(is_center);
    free(switch_membership);
    free(center_table);
    if( stream->feof() ) {
      break;
    }
  }

  //finally cluster all temp centers
  switch_membership = (char*)malloc(centers.num*sizeof(char));
  is_center = (bool*)calloc(centers.num,sizeof(bool));
  center_table = (int*)malloc(centers.num*sizeof(int));

  localSearch( &centers, kmin, kmax ,&kfinal );
  contcenters(&centers);
  outcenterIDs( &centers, centerIDs, outfile);
}

int main(int argc, char **argv)
{
  char *outfilename = new char[MAXNAMESIZE];
  char *infilename = new char[MAXNAMESIZE];
  long kmin, kmax, n, chunksize, clustersize;
  int dim;
#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_begin(__parsec_streamcluster);
#endif

  if (argc<11) {
    fprintf(stderr,"usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n",
	    argv[0]);
    fprintf(stderr,"  k1:          Min. number of centers allowed\n");
    fprintf(stderr,"  k2:          Max. number of centers allowed\n");
    fprintf(stderr,"  d:           Dimension of each data point\n");
    fprintf(stderr,"  n:           Number of data points\n");
    fprintf(stderr,"  chunksize:   Number of data points to handle per step\n");
    fprintf(stderr,"  clustersize: Maximum number of intermediate centers\n");
    fprintf(stderr,"  infile:      Input file (if n<=0)\n");
    fprintf(stderr,"  outfile:     Output file\n");
    fprintf(stderr,"  nproc:       Number of threads to use\n");
    fprintf(stderr,"\n");
    fprintf(stderr, "if n > 0, points will be randomly generated instead of reading from infile.\n");
    exit(1);
  }
  kmin = atoi(argv[1]);
  kmax = atoi(argv[2]);
  dim = atoi(argv[3]);
  n = atoi(argv[4]);
  chunksize = atoi(argv[5]);
  clustersize = atoi(argv[6]);
  strcpy(infilename, argv[7]);
  strcpy(outfilename, argv[8]);
  nproc = atoi(argv[9]);
  _clCmdParams(argc, argv);
  try{
	  _clInit(device_type, device_id);
   }
   catch(std::string msg){
   	std::cout<<"exception caught in main function->"<<msg<<std::endl;
   	return -1;
   }
  srand48(SEED);
  PStream* stream;
  if( n > 0 ) {
    stream = new SimStream(n);
  }
  else {
    stream = new FileStream(infilename);
  }
#ifdef PROFILE_TMP
  double t1 = gettime();
#endif
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_begin();
#endif
#ifdef PROFILE_TMP
	serial = 0.0;
	cpu_gpu_memcpy = 0.0;
	gpu_malloc = 0.0;
	gpu_free = 0.0;
	kernel = 0.0;
	time_FL = 0.0;
	cnt_speedy = 0;
#endif
  std::cout<<"before sc"<<std::endl;
  streamCluster(stream, kmin, kmax, dim, chunksize, clustersize, outfilename );
  std::cout<<"after sc"<<std::endl;
#ifdef PROFILE_TMP 
	gpu_free = gettime();
#endif
	freeDevMem();
#ifdef PROFILE_TMP
	gpu_free = gettime() - gpu_free;
#endif
	_clRelease();
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_roi_end();
#endif

#ifdef PROFILE_TMP
  double t2 = gettime();
  printf("time = %lf\n",t2-t1);
#endif

  delete stream;
  
#ifdef PROFILE_TMP
  printf("time pgain = %lf\n", time_gain);
  printf("time pgain_dist = %lf\n", time_gain_dist);
  printf("time pgain_init = %lf\n", time_gain_init);
  printf("time pselect = %lf\n", time_select_feasible);
  printf("time pspeedy = %lf\n", time_speedy);
  printf("time pshuffle = %lf\n", time_shuffle);
  printf("time FL = %lf\n", time_FL);
  printf("time localSearch = %lf\n", time_local_search);
	printf("\n");
	printf("====GPU Timing info====\n");
	printf("time serial = %lf\n", serial);
	printf("time CPU to GPU memory copy = %lf\n", cpu_gpu_memcpy);
	printf("time GPU to CPU memory copy back = %lf\n", memcpy_back);
	printf("time GPU malloc = %lf\n", gpu_malloc);
	printf("time GPU free = %lf\n", gpu_free);
	printf("time kernel = %lf\n", kernel);
	
  FILE *fp = fopen("PD.txt", "w");
  fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", time_FL, cpu_gpu_memcpy, memcpy_back, kernel, gpu_malloc, gpu_free, 0.0);
  fclose(fp);	
 #endif
 _clStatistics(); 
#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_end();
#endif
  
  return 0;
}
