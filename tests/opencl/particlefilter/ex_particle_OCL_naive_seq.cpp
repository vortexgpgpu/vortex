/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <CL/cl.h>
#define PI 3.1415926535897932
#define BLOCK_X 2
#define BLOCK_Y 2

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
 */
long M = INT_MAX;
/**
@var A value for LCG
 */
int A = 1103515245;
/**
@var C value for LCG
 */
int C = 12345;

const int threads_per_block = 128;

#ifdef WIN
#include <windows.h>
#else
#include <pthread.h>
#include <sys/time.h>

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
#endif



#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

// local variables
static cl_context context;
static cl_command_queue cmd_queue;
static cl_device_type device_type;
static cl_device_id * device_list;
static cl_int num_devices;
//GPU copies of arrays
static cl_mem arrayX_GPU;
static cl_mem arrayY_GPU;
static cl_mem xj_GPU;
static cl_mem yj_GPU;
static cl_mem CDF_GPU;
static cl_mem u_GPU;

static cl_kernel kernel_s;
double * likelihood ;
double * arrayX ;
double * arrayY ;
double * xj ;
double * yj ;
double * CDF ;

int * ind ;
double * u;

/***************************************************
  @brief initializes the OpenCL context and detects available platforms
@param use_gpu 
**************************************************/
static int initialize(int use_gpu) {
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs(1,*,0) failed\n");
        return -1;
    }
    cl_context_properties ctxprop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, NULL);
    if (!context) {
        printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU");
        return -1;
    }

    // get the list of GPUs
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (int) (size / sizeof (cl_device_id));

    if (result != CL_SUCCESS || num_devices < 1) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }
    device_list = new cl_device_id[num_devices];
    if (!device_list) {
        printf("ERROR: new cl_device_id[] failed\n");
        return -1;
    }
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
    if (result != CL_SUCCESS) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
    if (!cmd_queue) {
        printf("ERROR: clCreateCommandQueue() failed\n");
        return -1;
    }

    return 0;
}

static int shutdown() {
    // release resources
    if (cmd_queue) clReleaseCommandQueue(cmd_queue);
    if (context) clReleaseContext(context);
    if (device_list) delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;
}

/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) +tv.tv_usec;
}

// Returns the number of seconds elapsed between the two specified times

float elapsed_time(long long start_time, long long end_time) {
    return (float) (end_time - start_time) / (1000 * 1000);
}

/** 
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double roundDouble(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}//eo setIf

/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
double randu(int * seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double) M));
}//eo randu

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
double randn(int * seed, int index) {
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int) (5 * randn(seed, 0));
            }
        }
    }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            double distance = sqrt(pow((double) (x - radius + 1), 2) + pow((double) (y - radius + 1), 2));
            if (distance < radius)
                disk[x * diameter + y] = 1;
        }
    }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++) {
        for (y = startY; y < endY; y++) {
            double distance = sqrt(pow((double) (x - posX), 2) + pow((double) (y - posY), 2));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int * se, int numOnes, double * neighbors, int radius) {
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (se[x * diameter + y]) {
                neighbors[neighY * 2] = (int) (y - center);
                neighbors[neighY * 2 + 1] = (int) (x - center);
                neighY++;
            }
        }
    }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed) {
    int k;
    int max_size = IszX * IszY*Nfr;
    /*get object centers*/
    int x0 = (int) roundDouble(IszY / 2.0);
    int y0 = (int) roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
        xk = abs(x0 + (k - 1));
        yk = abs(y0 - 2 * (k - 1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    int * newMatrix = (int *) malloc(sizeof (int) *IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
        for (y = 0; y < IszY; y++) {
            for (k = 0; k < Nfr; k++) {
                I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
 * Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
 * @param I The 3D matrix
 * @param ind The current ind array
 * @param numOnes The length of ind array
 * @return A double representing the sum
 */
double calcLikelihoodSum(int * I, int * ind, int numOnes) {
    double likelihoodSum = 0.0;
    int y;
    for (y = 0; y < numOnes; y++)
        likelihoodSum += (pow((double) (I[ind[y]] - 100), 2) - pow((double) (I[ind[y]] - 228), 2)) / 50.0;
    return likelihoodSum;
}

/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(double * CDF, int lengthCDF, double value) {
    int index = -1;
    int x;
    for (x = 0; x < lengthCDF; x++) {
        if (CDF[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return lengthCDF - 1;
    }
    return index;
}
/*
*@brief allocates and initializes the data for the computation
 * @param Nparticles 
 * @param countOnes
*
*/
static int allocate(int Nparticles, int countOnes){
	/***** variables ******/

	int sourcesize = 1024 * 1024;
	char * source = (char *) calloc(sourcesize, sizeof (char));
	if (!source) {
		printf("ERROR: calloc(%d) failed\n", sourcesize);
		return -1;
	}

	// read the kernel core source
	char * tempchar = "./particle_naive.cl";
	FILE * fp = fopen(tempchar, "rb");
	if (!fp) {
		printf("ERROR: unable to open '%s'\n", tempchar);
		return -1;
	}
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	// OpenCL initialization
	int use_gpu = 1;
	if (initialize(use_gpu)) return -1;

	// compile kernel
	cl_int err = 0;
	const char * slist[2] = {source, 0};
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateProgramWithSource() => %d\n", err);
		return -1;
	}
	err = clBuildProgram(prog, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	{ // show warnings/errors
		static char log[65536];
		memset(log, 0, sizeof (log));
		cl_device_id device_id[2] = {0};
		err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof (device_id), device_id, NULL);
		if (err != CL_SUCCESS) {
			if (err == CL_INVALID_CONTEXT)
				printf("ERROR: clGetContextInfo() => CL_INVALID_CONTEXT\n");
			if (err == CL_INVALID_VALUE)
				printf("ERROR: clGetContextInfo() => CL_INVALID_VALUE\n");
		}
		err = clGetProgramBuildInfo(prog, device_id[0], CL_PROGRAM_BUILD_LOG, sizeof (log) - 1, log, NULL);
		if (err != CL_SUCCESS) {
			printf("ERROR: clGetProgramBuildInfo() => %d\n", err);
		}
		if (err || strstr(log, "warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}//*/
	if (err != CL_SUCCESS) {
		printf("ERROR: clBuildProgram() => %d\n", err);
		return -1;
	}

	char * particle_kernel = "particle_kernel";

	kernel_s = clCreateKernel(prog, particle_kernel, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateKernel() 0 => %d\n", err);
		return -1;
	}

	clReleaseProgram(prog);


	likelihood = (double *) malloc(sizeof (double) *Nparticles);
	arrayX = (double *) malloc(sizeof (double) *Nparticles);
	arrayY = (double *) malloc(sizeof (double) *Nparticles);
	xj = (double *) malloc(sizeof (double) *Nparticles);
	yj = (double *) malloc(sizeof (double) *Nparticles);
	CDF = (double *) malloc(sizeof (double) *Nparticles);

	ind = (int*) malloc(sizeof (int) *countOnes);
	u = (double *) malloc(sizeof (double) *Nparticles);

	//OpenCL memory allocation
	arrayX_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateBuffer arrayX_GPU (size:%d) => %d\n", Nparticles, err);
		return -1;
	}
	arrayY_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateBuffer arrayY_GPU (size:%d) => %d\n", Nparticles, err);
		return -1;
	}
	xj_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateBuffer xj_GPU (size:%d) => %d\n", Nparticles, err);
		return -1;
	}
	yj_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateBuffer yj_GPU (size:%d) => %d\n", Nparticles, err);
		return -1;
	}
	CDF_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("ERROR: clCreateBuffer CDF_GPU (size:%d) => %d\n", Nparticles, err);
		return -1;
	}
	u_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
	if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer u_GPU (size:%d) => %d\n", Nparticles, err); return -1; }

  

}//eo allocate

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
int particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
	int max_size = IszX * IszY*Nfr;
	long long start = get_time();
	//original particle centroid
	double xe = roundDouble(IszY / 2.0);
	double ye = roundDouble(IszX / 2.0);

	//expected object locations, compared to center
	int radius = 5;
	int diameter = radius * 2 - 1;
	int * disk = (int *) malloc(diameter * diameter * sizeof (int));
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for (x = 0; x < diameter; x++) {
		for (y = 0; y < diameter; y++) {
			if (disk[x * diameter + y] == 1)
				countOnes++;
		}
	}
	double * objxy = (double *) malloc(countOnes * 2 * sizeof (double));
	getneighbors(disk, countOnes, objxy, radius);

	long long get_neighbors = get_time();
	printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
	//initial weights are all equal (1/Nparticles)
	double * weights = (double *) malloc(sizeof (double) *Nparticles);
	for (x = 0; x < Nparticles; x++) {
		weights[x] = 1 / ((double) (Nparticles));
	}
	long long get_weights = get_time();
	printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
	//initial likelihood to 0.0

	//allocate all of the memory for the computation 
	allocate(Nparticles, countOnes);
	
	for (x = 0; x < Nparticles; x++) {
		arrayX[x] = xe;
		arrayY[x] = ye;
	}
	int k;
	int indX, indY;
	for (k = 1; k < Nfr; k++) {
		long long set_arrays = get_time();
		//printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, set_arrays));
		//apply motion model
		//draws sample from motion model (random walk). The only prior information
		//is that the object moves 2x as fast as in the y direction

		for (x = 0; x < Nparticles; x++) {
			arrayX[x] = arrayX[x] + 1.0 + 5.0 * randn(seed, x);
			arrayY[x] = arrayY[x] - 2.0 + 2.0 * randn(seed, x);
		}
		//particle filter likelihood
		long long error = get_time();
		printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
		for (x = 0; x < Nparticles; x++) {

			//compute the likelihood: remember our assumption is that you know
			// foreground and the background image intensity distribution.
			// Notice that we consider here a likelihood ratio, instead of
			// p(z|x). It is possible in this case. why? a hometask for you.            
			//calc ind
			for (y = 0; y < countOnes; y++) {
				indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
				indY = roundDouble(arrayY[x]) + objxy[y * 2];
				ind[y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
				if (ind[y] >= max_size)
					ind[y] = 0;
			}
			likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
			likelihood[x] = likelihood[x] / countOnes;
		}
		long long likelihood_time = get_time();
		printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
		// update & normalize weights
		// using equation (63) of Arulampalam Tutorial          
		for (x = 0; x < Nparticles; x++) {
			weights[x] = weights[x] * exp(likelihood[x]);
		}
		long long exponential = get_time();
		printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
		double sumWeights = 0;
		for (x = 0; x < Nparticles; x++) {
			sumWeights += weights[x];
		}
		long long sum_time = get_time();
		printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
		for (x = 0; x < Nparticles; x++) {
			weights[x] = weights[x] / sumWeights;
		}
		long long normalize = get_time();
		printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
		xe = 0;
		ye = 0;
		// estimate the object location by expected values
		for (x = 0; x < Nparticles; x++) {
			xe += arrayX[x] * weights[x];
			ye += arrayY[x] * weights[x];
		}
		long long move_time = get_time();
		printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
		printf("XE: %lf\n", xe);
		printf("YE: %lf\n", ye);
		double distance = sqrt(pow((double) (xe - (int) roundDouble(IszY / 2.0)), 2) + pow((double) (ye - (int) roundDouble(IszX / 2.0)), 2));
		printf("%lf\n", distance);
		//display(hold off for now)

		//pause(hold off for now)

		//resampling


		CDF[0] = weights[0];
		for (x = 1; x < Nparticles; x++) {
			CDF[x] = weights[x] + CDF[x - 1];
		}
		long long cum_sum = get_time();
		printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
		double u1 = (1 / ((double) (Nparticles))) * randu(seed, 0);
		for (x = 0; x < Nparticles; x++) {
			u[x] = u1 + x / ((double) (Nparticles));
		}
		long long u_time = get_time();
		printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
		long long start_copy = get_time();

		//OpenCL memory allocation
		cl_int err = clEnqueueWriteBuffer(cmd_queue, arrayX_GPU, 1, 0, sizeof (double) *Nparticles, arrayX, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer arrayX_GPU (size:%d) => %d\n", Nparticles, err); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, arrayY_GPU, 1, 0, sizeof (double) *Nparticles, arrayY, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer arrayY_GPU (size:%d) => %d\n", Nparticles, err); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, CDF_GPU, 1, 0, sizeof (double) *Nparticles, CDF, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer CDF_GPU (size:%d) => %d\n", Nparticles, err); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, u_GPU, 1, 0, sizeof (double) *Nparticles, u, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer u_GPU (size:%d) => %d\n", Nparticles, err); return -1; }


		long long end_copy = get_time();
		//Set number of threads
		int num_blocks = ceil((double) Nparticles / (double) threads_per_block);

		clSetKernelArg(kernel_s, 0, sizeof (void *), (void*) &arrayX_GPU);
		clSetKernelArg(kernel_s, 1, sizeof (void *), (void*) &arrayY_GPU);
		clSetKernelArg(kernel_s, 2, sizeof (void *), (void*) &CDF_GPU);
		clSetKernelArg(kernel_s, 3, sizeof (void *), (void*) &u_GPU);
		clSetKernelArg(kernel_s, 4, sizeof (void *), (void*) &xj_GPU);
		clSetKernelArg(kernel_s, 5, sizeof (void *), (void*) &yj_GPU);
		clSetKernelArg(kernel_s, 6, sizeof (cl_int), (void*) &Nparticles);


		//KERNEL FUNCTION CALL
		size_t global_work[3] = {num_blocks*threads_per_block, 1, 1};
		err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);
		clFinish(cmd_queue);
		long long start_copy_back = get_time();
                
                if (err != CL_SUCCESS) {
			printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err);
			return -1;
		}
		//OpenCL memory copying back from GPU to CPU memory
		err = clEnqueueReadBuffer(cmd_queue, yj_GPU, 1, 0, sizeof (double) *Nparticles, yj, 0, 0, 0);
		if (err != CL_SUCCESS) {
			printf("ERROR: Memcopy Out\n");
			return -1;
		}
		err = clEnqueueReadBuffer(cmd_queue, xj_GPU, 1, 0, sizeof (double) *Nparticles, xj, 0, 0, 0);
		if (err != CL_SUCCESS) {
			printf("ERROR: Memcopy Out\n");
			return -1;
		}

		long long end_copy_back = get_time();

		printf("SENDING TO GPU TOOK: %lf\n", elapsed_time(start_copy, end_copy));
        printf("OPEN_CL EXEC TOOK: %lf\n", elapsed_time(end_copy, start_copy_back));
        printf("SENDING BACK FROM GPU TOOK: %lf\n", elapsed_time(start_copy_back, end_copy_back));
        long long xyj_time = get_time();
        printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));
        //reassign arrayX and arrayY
        
        //FIXED: this used to be a memory leak where arrayX was assigned the address of xj
        for (x = 0; x < Nparticles; x++) {
            arrayX[x] = xj[x];
            arrayY[x] = yj[x];
            weights[x] = 1 / ((double) (Nparticles));
        }
        long long reset = get_time();
        printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
    }

    
    //OpenCL freeing of memory

    clReleaseMemObject(u_GPU);
    clReleaseMemObject(CDF_GPU);
    clReleaseMemObject(yj_GPU);
    clReleaseMemObject(xj_GPU);
    clReleaseMemObject(arrayY_GPU);
    clReleaseMemObject(arrayX_GPU);


    //free memory
    free(disk);
    free(objxy);
    free(weights);
    free(likelihood);
    free(xj);
    free(yj);
    free(arrayX);
    free(arrayY);
    free(CDF);
    free(u);
    free(ind);
}

int main(int argc, char * argv[]) {

    char* usage = "naive.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
    //check number of arguments
    if (argc != 9) {
        printf("%s\n", usage);
        return 0;
    }
    //check args deliminators
    if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
        printf("%s\n", usage);
        return 0;
    }

    int IszX, IszY, Nfr, Nparticles;

    //converting a string to a integer
    if (sscanf(argv[2], "%d", &IszX) == EOF) {
        printf("ERROR: dimX input is incorrect");
        return 0;
    }

    if (IszX <= 0) {
        printf("dimX must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[4], "%d", &IszY) == EOF) {
        printf("ERROR: dimY input is incorrect");
        return 0;
    }

    if (IszY <= 0) {
        printf("dimY must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[6], "%d", &Nfr) == EOF) {
        printf("ERROR: Number of frames input is incorrect");
        return 0;
    }

    if (Nfr <= 0) {
        printf("number of frames must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
        printf("ERROR: Number of particles input is incorrect");
        return 0;
    }

    if (Nparticles <= 0) {
        printf("Number of particles must be > 0\n");
        return 0;
    }
    //establish seed
    int * seed = (int *) malloc(sizeof (int) *Nparticles);
    int i;
    for (i = 0; i < Nparticles; i++)
        seed[i] = time(0) * i;
    //malloc matrix
    int * I = (int *) malloc(sizeof (int) *IszX * IszY * Nfr);
    long long start = get_time();
    //call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    long long endVideoSequence = get_time();
    printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
    //call particle filter
    particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
    long long endParticleFilter = get_time();
    printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
    printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

    free(seed);
    free(I);
    return 0;
}
