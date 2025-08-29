#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <CL/cl.h>
//#include "OpenCL_helper_library.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI  3.1415926535897932

int threads_per_block = 512;

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


//#include "oclUtils.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

void ocl_print_double_array(cl_command_queue cmd_q, cl_mem array_GPU, size_t size) {
    //allocate temporary array for printing
    double* mem = (double*) calloc(size, sizeof(double));

    //transfer data from device
    cl_int err = clEnqueueReadBuffer(cmd_q, array_GPU, 1, 0, sizeof (double) *size, mem, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: Memcopy Out\n");
        return;
    }


    printf("PRINTING ARRAY VALUES\n");
    //print values in memory
    for (size_t i = 0; i < size; ++i) {
        printf("[%d]:%0.6f\n", i, mem[i]);
    }
    printf("FINISHED PRINTING ARRAY VALUES\n");

    //clean up memory
    free(mem);
    mem = NULL;
}
// local variables
static cl_context context;
static cl_command_queue cmd_queue;
static cl_device_type device_type;
static cl_device_id * device_list;
static cl_int num_devices;

/*
 * @brief sets up the OpenCL framework by detecting and initializing the available device
 * @param use_gpu flag denoting if the gpu is the desired platform
 */
static int initialize(int use_gpu) {
    cl_int result;
    size_t size;

    // create OpenCL context
    // you have to specify what platform you want to use
    // not uncommon for both NVIDIA and AMD to be installed
    cl_platform_id platform_id[2];

    cl_uint num_avail;
    cl_int err = clGetPlatformIDs(2, platform_id, &num_avail);
    if (err != CL_SUCCESS) {
        if (err == CL_INVALID_VALUE)printf("clGetPlatformIDs() returned invalid_value\n");
        printf("ERROR: clGetPlatformIDs(1,*,0) failed\n");
        return -1;
    }
    printf("number of available platforms:%d.\n",num_avail);
    char info[100];
    clGetPlatformInfo(platform_id[0], CL_PLATFORM_VENDOR, 100, info, NULL);
    printf("clGetPlatformInfo: %s\n", info);

    cl_context_properties ctxprop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id[0], 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, &err);

    if (!context) {
        if (CL_INVALID_PLATFORM == err)
            printf("CL_INVALID_PLATFORM returned by clCreateContextFromType()\n");
        else if (CL_INVALID_VALUE == err)
            printf("CL_INVALID_VALUE returned by clCreateContextFromType()\n");
        else if (CL_INVALID_DEVICE_TYPE == err)
            printf("CL_INVALID_DEVICE_TYPE returned by clCreateContextFromType()\n");
        else if (CL_INVALID_OPERATION == err)
            printf("CL_INVALID_OPERATION returned by clCreateContextFromType()\n");
        else if (CL_DEVICE_NOT_AVAILABLE == err)
            printf("CL_DEVICE_NOT_AVAILABLE returned by clCreateContextFromType()\n");
        else if (CL_DEVICE_NOT_FOUND == err)
            printf("CL_DEVICE_NOT_FOUND returned by clCreateContextFromType()\n");
        else if (CL_OUT_OF_RESOURCES == err)
            printf("CL_OUT_OF_RESOURCES returned by clCreateContextFromType()\n");


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
    size_t max_work_item_sizes[3];
    result = clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                sizeof(max_work_item_sizes), (void*)max_work_item_sizes, NULL);
    if (result != CL_SUCCESS) {
        printf("ERROR: clGetDeviceInfo() failed\n");
        return -1;
    }
  if (max_work_item_sizes[0] < threads_per_block)
    threads_per_block = max_work_item_sizes[0];

   // create command queue for the first device
    cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
    if (!cmd_queue) {
        printf("ERROR: clCreateCommandQueue() failed\n");
        return -1;
    }

    return 0;
}

/*
 @brief cleans up the OpenCL framework
 */
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

double elapsed_time(long long start_time, long long end_time) {
    return (double) (end_time - start_time) / (1000 * 1000);
}

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
}

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
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
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
	    else
                disk[x * diameter + y] = 0;
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
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
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
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
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
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
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
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
    int k;
    int max_size = IszX * IszY * Nfr;
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
    unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
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
int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
    int max_size = IszX * IszY*Nfr;
    //original particle centroid
    double xe = roundDouble(IszY / 2.0);
    double ye = roundDouble(IszX / 2.0);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius * 2 - 1;
    int * disk = (int*) calloc(diameter * diameter, sizeof (int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }
    int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
    getneighbors(disk, countOnes, objxy, radius);
    //initial weights are all equal (1/Nparticles)
    double * weights = (double *) calloc(Nparticles, sizeof(double));
    for (x = 0; x < Nparticles; x++) {
        weights[x] = 1 / ((double) (Nparticles));
    }
    /****************************************************************
     **************   B E G I N   A L L O C A T E *******************
     ****************************************************************/

    /***** kernel variables ******/
    cl_kernel kernel_likelihood;
    cl_kernel kernel_sum;
    cl_kernel kernel_normalize_weights;
    cl_kernel kernel_find_index;

    int sourcesize = 2048 * 2048;
    char * source = (char *) calloc(sourcesize, sizeof (char));
    if (!source) {
        printf("ERROR: calloc(%d) failed\n", sourcesize);
        return -1;
    }

    // read the kernel core source
    char * tempchar = "./particle_double.cl";
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

    err = clBuildProgram(prog, 1, device_list, "-cl-fast-relaxed-math", NULL, NULL);

    if (err != CL_SUCCESS) {
        if (err == CL_INVALID_PROGRAM)
            printf("CL_INVALID_PROGRAM\n");
        else if (err == CL_INVALID_VALUE)
            printf("CL_INVALID_VALUE\n");
        else if (err == CL_INVALID_DEVICE)
            printf("CL_INVALID_DEVICE\n");
        else if (err == CL_INVALID_BINARY)
            printf("CL_INVALID_BINARY\n");
        else if (err == CL_INVALID_BUILD_OPTIONS)
            printf("CL_INVALID_BUILD_OPTIONS\n");
        else if (err == CL_INVALID_OPERATION)
            printf("CL_INVALID_OPERATION\n");
        else if (err == CL_COMPILER_NOT_AVAILABLE)
            printf("CL_COMPILER_NOT_AVAILABLE\n");
        else if (err == CL_BUILD_PROGRAM_FAILURE)
            printf("CL_BUILD_PROGRAM_FAILURE\n");
        else if (err == CL_INVALID_OPERATION)
            printf("CL_INVALID_OPERATION\n");
        else if (err == CL_OUT_OF_RESOURCES)
            printf("CL_OUT_OF_RESOURCES\n");
        else if (err == CL_OUT_OF_HOST_MEMORY)
            printf("CL_OUT_OF_HOST_MEMORY\n");

        printf("ERROR: clBuildProgram() => %d\n", err);

        static char log[65536];
        memset(log, 0, sizeof (log));

        err = clGetProgramBuildInfo(prog, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof (log) - 1, log, NULL);
        if (err != CL_SUCCESS) {
            printf("ERROR: clGetProgramBuildInfo() => %d\n", err);
        }
        if (strstr(log, "warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);


    }
    // { // show warnings/errors
    //     static char log[65536];
    //     memset(log, 0, sizeof (log));
    //     cl_device_id device_id[2] = {0};
    //     err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof (device_id), device_id, NULL);
    //     if (err != CL_SUCCESS) {
    //         if (err == CL_INVALID_CONTEXT)
    //             printf("ERROR: clGetContextInfo() => CL_INVALID_CONTEXT\n");
    //         if (err == CL_INVALID_VALUE)
    //             printf("ERROR: clGetContextInfo() => CL_INVALID_VALUE\n");
    //     }
    // }//*/

    char * s_likelihood_kernel = "likelihood_kernel";
    char * s_sum_kernel = "sum_kernel";
    char * s_normalize_weights_kernel = "normalize_weights_kernel";
    char * s_find_index_kernel = "find_index_kernel";

    kernel_likelihood = clCreateKernel(prog, s_likelihood_kernel, &err);
    if (err != CL_SUCCESS) {
        if (err == CL_INVALID_PROGRAM)
            printf("ERROR: clCreateKernel(likelihood_kernel) 0 => INVALID PROGRAM %d\n", err);
        if (err == CL_INVALID_PROGRAM_EXECUTABLE)
            printf("ERROR: clCreateKernel(likelihood_kernel) 0 => INVALID PROGRAM EXECUTABLE %d\n", err);
        if (err == CL_INVALID_KERNEL_NAME)
            printf("ERROR: clCreateKernel(likelihood_kernel) 0 => INVALID KERNEL NAME %d\n", err);
        if (err == CL_INVALID_KERNEL_DEFINITION)
            printf("ERROR: clCreateKernel(likelihood_kernel) 0 => INVALID KERNEL DEFINITION %d\n", err);
        if (err == CL_INVALID_VALUE)
            printf("ERROR: clCreateKernel(likelihood_kernel) 0 => INVALID CL_INVALID_VALUE %d\n", err);
        printf("ERROR: clCreateKernel(likelihood_kernel) failed.\n");
        return -1;
    }
    kernel_sum = clCreateKernel(prog, s_sum_kernel, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel(sum_kernel) 0 => %d\n", err);
        return -1;
    }
    kernel_normalize_weights = clCreateKernel(prog, s_normalize_weights_kernel, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel(normalize_weights_kernel) 0 => %d\n", err);
        return -1;
    }
    kernel_find_index = clCreateKernel(prog, s_find_index_kernel, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel(find_index_kernel) 0 => %d\n", err);
        return -1;
    }


    //initial likelihood to 0.0
    double * likelihood = (double *) calloc(Nparticles + 1, sizeof (double));
    double * arrayX = (double *) calloc(Nparticles, sizeof (double));
    double * arrayY = (double *) calloc(Nparticles, sizeof (double));
    double * xj = (double *) calloc(Nparticles, sizeof (double));
    double * yj = (double *) calloc(Nparticles, sizeof (double));
    double * CDF = (double *) calloc(Nparticles, sizeof(double));

    //GPU copies of arrays
    cl_mem arrayX_GPU;
    cl_mem arrayY_GPU;
    cl_mem xj_GPU;
    cl_mem yj_GPU;
    cl_mem CDF_GPU;
    cl_mem likelihood_GPU;
    cl_mem I_GPU;
    cl_mem weights_GPU;
    cl_mem objxy_GPU;

    int * ind = (int*) calloc(countOnes, sizeof(int));
    cl_mem ind_GPU;
    double * u = (double *) calloc(Nparticles, sizeof(double));
    cl_mem u_GPU;
    cl_mem seed_GPU;
    cl_mem partial_sums;


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
    CDF_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) * Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer CDF_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    u_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer u_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    likelihood_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer likelihood_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    weights_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (double) *Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer weights_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    I_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (unsigned char) *IszX * IszY * Nfr, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer I_GPU (size:%d) => %d\n", IszX * IszY * Nfr, err);
        return -1;
    }
    objxy_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*sizeof (int) *countOnes, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer objxy_GPU (size:%d) => %d\n", countOnes, err);
        return -1;
    }
    ind_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (int) *countOnes * Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer ind_GPU (size:%d) => %d\n", countOnes * Nparticles, err);
        return -1;
    }
    seed_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (int) *Nparticles, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer seed_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    partial_sums = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof (double) * Nparticles + 1, likelihood, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer partial_sums (size:%d) => %d\n", Nparticles, err);
        return -1;
    }

	//Donnie - this loop is different because in this kernel, arrayX and arrayY
    //  are set equal to xj before every iteration, so effectively, arrayX and
    //  arrayY will be set to xe and ye before the first iteration.
    for (x = 0; x < Nparticles; x++) {

        xj[x] = xe;
        yj[x] = ye;
    }

    int k;
    //double * Ik = (double *)calloc(IszX*IszY, sizeof(double));
    int indX, indY;
    //start send
    long long send_start = get_time();

    //OpenCL memory copy
    err = clEnqueueWriteBuffer(cmd_queue, I_GPU, 1, 0, sizeof (unsigned char) *IszX * IszY*Nfr, I, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer I_GPU (size:%d) => %d\n", IszX * IszY*Nfr, err);
        return -1;
    }
    err = clEnqueueWriteBuffer(cmd_queue, objxy_GPU, 1, 0, 2*sizeof (int) *countOnes, objxy, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer objxy_GPU (size:%d) => %d\n", countOnes, err);
        return -1; }
    err = clEnqueueWriteBuffer(cmd_queue, weights_GPU, 1, 0, sizeof (double) *Nparticles, weights, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer weights_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    err = clEnqueueWriteBuffer(cmd_queue, xj_GPU, 1, 0, sizeof (double) *Nparticles, xj, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer arrayX_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    err = clEnqueueWriteBuffer(cmd_queue, yj_GPU, 1, 0, sizeof (double) *Nparticles, yj, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer arrayY_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    err = clEnqueueWriteBuffer(cmd_queue, seed_GPU, 1, 0, sizeof (int) *Nparticles, seed, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer seed_GPU (size:%d) => %d\n", Nparticles, err);
        return -1;
    }
    /**********************************************************************
     *********** E N D    A L L O C A T E ********************************
     *********************************************************************/

    long long send_end = get_time();
    printf("TIME TO SEND TO GPU: %f\n", elapsed_time(send_start, send_end));
    int num_blocks = ceil((double) Nparticles / (double) threads_per_block);
    printf("threads_per_block=%d \n",threads_per_block);
    size_t local_work[3] = {threads_per_block, 1, 1};
    size_t global_work[3] = {num_blocks*threads_per_block, 1, 1};

    for (k = 1; k < Nfr; k++) {
        /****************** L I K E L I H O O D ************************************/
        clSetKernelArg(kernel_likelihood, 0, sizeof (void *), (void*) &arrayX_GPU);
        clSetKernelArg(kernel_likelihood, 1, sizeof (void *), (void*) &arrayY_GPU);
        clSetKernelArg(kernel_likelihood, 2, sizeof (void *), (void*) &xj_GPU);
        clSetKernelArg(kernel_likelihood, 3, sizeof (void *), (void*) &yj_GPU);
        clSetKernelArg(kernel_likelihood, 4, sizeof (void *), (void*) &CDF_GPU);
        clSetKernelArg(kernel_likelihood, 5, sizeof (void *), (void*) &ind_GPU);
        clSetKernelArg(kernel_likelihood, 6, sizeof (void *), (void*) &objxy_GPU);
        clSetKernelArg(kernel_likelihood, 7, sizeof (void *), (void*) &likelihood_GPU);
        clSetKernelArg(kernel_likelihood, 8, sizeof (void *), (void*) &I_GPU);
        clSetKernelArg(kernel_likelihood, 9, sizeof (void *), (void*) &u_GPU);
        clSetKernelArg(kernel_likelihood, 10, sizeof (void *), (void*) &weights_GPU);
        clSetKernelArg(kernel_likelihood, 11, sizeof (cl_int), (void*) &Nparticles);
        clSetKernelArg(kernel_likelihood, 12, sizeof (cl_int), (void*) &countOnes);
        clSetKernelArg(kernel_likelihood, 13, sizeof (cl_int), (void*) &max_size);
        clSetKernelArg(kernel_likelihood, 14, sizeof (cl_int), (void*) &k);
        clSetKernelArg(kernel_likelihood, 15, sizeof (cl_int), (void*) &IszY);
        clSetKernelArg(kernel_likelihood, 16, sizeof (cl_int), (void*) &Nfr);
        clSetKernelArg(kernel_likelihood, 17, sizeof (void *), (void*) &seed_GPU);
        clSetKernelArg(kernel_likelihood, 18, sizeof (void *), (void*) &partial_sums);
        clSetKernelArg(kernel_likelihood, 19, threads_per_block * sizeof (double), NULL);

        //KERNEL FUNCTION CALL
        err = clEnqueueNDRangeKernel(cmd_queue, kernel_likelihood, 1, NULL, global_work, local_work, 0, 0, 0);
        clFinish(cmd_queue);
        if (err != CL_SUCCESS) {
            printf("ERROR: clEnqueueNDRangeKernel(kernel_likelihood)=>%d failed\n", err);
	    //check_error(err, __FILE__, __LINE__);
            return -1;
        }
        /****************** E N D    L I K E L I H O O D **********************/
        /*************************** S U M ************************************/
        clSetKernelArg(kernel_sum, 0, sizeof (void *), (void*) &partial_sums);
        clSetKernelArg(kernel_sum, 1, sizeof (cl_int), (void*) &Nparticles);

        //KERNEL FUNCTION CALL
        err = clEnqueueNDRangeKernel(cmd_queue, kernel_sum, 1, NULL, global_work, local_work, 0, 0, 0);
        clFinish(cmd_queue);
        if (err != CL_SUCCESS) {
            printf("ERROR: clEnqueueNDRangeKernel(kernel_sum)=>%d failed\n", err);
	    //check_error(err, __FILE__, __LINE__);
            return -1;
        }/*************************** E N D   S U M ****************************/



        /**************** N O R M A L I Z E     W E I G H T S *****************/
        clSetKernelArg(kernel_normalize_weights, 0, sizeof (void *), (void*) &weights_GPU);
        clSetKernelArg(kernel_normalize_weights, 1, sizeof (cl_int), (void*) &Nparticles);
        clSetKernelArg(kernel_normalize_weights, 2, sizeof (void *), (void*) &partial_sums); //*/
        clSetKernelArg(kernel_normalize_weights, 3, sizeof (void *), (void*) &CDF_GPU);
        clSetKernelArg(kernel_normalize_weights, 4, sizeof (void *), (void*) &u_GPU);
        clSetKernelArg(kernel_normalize_weights, 5, sizeof (void *), (void*) &seed_GPU);

        //KERNEL FUNCTION CALL
        err = clEnqueueNDRangeKernel(cmd_queue, kernel_normalize_weights, 1, NULL, global_work, local_work, 0, 0, 0);
        clFinish(cmd_queue);
        if (err != CL_SUCCESS) {
            printf("ERROR: clEnqueueNDRangeKernel(normalize_weights)=>%d failed\n", err);
	    //check_error(err, __FILE__, __LINE__);
            return -1;
        }
        /************* E N D    N O R M A L I Z E     W E I G H T S ***********/

	  //	ocl_print_double_array(cmd_queue, partial_sums, 40);
        // /********* I N T E R M E D I A T E     R E S U L T S ***************/
        // //OpenCL memory copying back from GPU to CPU memory
         err = clEnqueueReadBuffer(cmd_queue, arrayX_GPU, 1, 0, sizeof (double) *Nparticles, arrayX, 0, 0, 0);
         err = clEnqueueReadBuffer(cmd_queue, arrayY_GPU, 1, 0, sizeof (double) *Nparticles, arrayY, 0, 0, 0);
         err = clEnqueueReadBuffer(cmd_queue, weights_GPU, 1, 0, sizeof (double) *Nparticles, weights, 0, 0, 0);

         xe = 0;
         ye = 0;
         double total=0.0;
         // estimate the object location by expected values
	 for (x = 0; x < Nparticles; x++) {
            // if( 0.0000000 < arrayX[x]*weights[x]) printf("arrayX[%d]:%f, arrayY[%d]:%f, weights[%d]:%0.10f\n",x,arrayX[x], x, arrayY[x], x, weights[x]);
	//	printf("arrayX[%d]:%f | arrayY[%d]:%f | weights[%d]:%f\n",
 	//		x, arrayX[x], x, arrayY[x], x, weights[x]); 
             xe += arrayX[x] * weights[x];
             ye += arrayY[x] * weights[x];
             total+= weights[x];
         }
         printf("total weight: %lf\n", total);
         printf("XE: %lf\n", xe);
         printf("YE: %lf\n", ye);
         double distance = sqrt(pow((double) (xe - (int) roundDouble(IszY / 2.0)), 2) + pow((double) (ye - (int) roundDouble(IszX / 2.0)), 2));
         printf("%lf\n", distance);
        // /********* E N D    I N T E R M E D I A T E     R E S U L T S ***************/

        /******************** F I N D    I N D E X ****************************/
        //Set number of threads

        clSetKernelArg(kernel_find_index, 0, sizeof (void *), (void*) &arrayX_GPU);
        clSetKernelArg(kernel_find_index, 1, sizeof (void *), (void*) &arrayY_GPU);
        clSetKernelArg(kernel_find_index, 2, sizeof (void *), (void*) &CDF_GPU);
        clSetKernelArg(kernel_find_index, 3, sizeof (void *), (void*) &u_GPU);
        clSetKernelArg(kernel_find_index, 4, sizeof (void *), (void*) &xj_GPU);
        clSetKernelArg(kernel_find_index, 5, sizeof (void *), (void*) &yj_GPU);
        clSetKernelArg(kernel_find_index, 6, sizeof (void *), (void*) &weights_GPU);
        clSetKernelArg(kernel_find_index, 7, sizeof (cl_int), (void*) &Nparticles);
        //KERNEL FUNCTION CALL
        err = clEnqueueNDRangeKernel(cmd_queue, kernel_find_index, 1, NULL, global_work, local_work, 0, 0, 0);
        clFinish(cmd_queue);
        if (err != CL_SUCCESS) {
            printf("ERROR: clEnqueueNDRangeKernel(find_index)=>%d failed\n", err);
	    //check_error(err, __FILE__, __LINE__);
            return -1;
        }
        /******************* E N D    F I N D    I N D E X ********************/

    }//end loop

    //block till kernels are finished
    //clFinish(cmd_queue);
    long long back_time = get_time();

    //OpenCL freeing of memory
    clReleaseProgram(prog);
    clReleaseMemObject(u_GPU);
    clReleaseMemObject(CDF_GPU);
    clReleaseMemObject(yj_GPU);
    clReleaseMemObject(xj_GPU);
    clReleaseMemObject(likelihood_GPU);
    clReleaseMemObject(I_GPU);
    clReleaseMemObject(objxy_GPU);
    clReleaseMemObject(ind_GPU);
    clReleaseMemObject(seed_GPU);
    clReleaseMemObject(partial_sums);

    long long free_time = get_time();

    //OpenCL memory copying back from GPU to CPU memory
    err = clEnqueueReadBuffer(cmd_queue, arrayX_GPU, 1, 0, sizeof (double) *Nparticles, arrayX, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: Memcopy Out\n");
        return -1;
    }
    long long arrayX_time = get_time();
    err = clEnqueueReadBuffer(cmd_queue, arrayY_GPU, 1, 0, sizeof (double) *Nparticles, arrayY, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: Memcopy Out\n");
        return -1;
    }
    long long arrayY_time = get_time();
    err = clEnqueueReadBuffer(cmd_queue, weights_GPU, 1, 0, sizeof (double) *Nparticles, weights, 0, 0, 0);
    if (err != CL_SUCCESS) {
        printf("ERROR: Memcopy Out\n");
        return -1;
    }
    long long back_end_time = get_time();

    printf("GPU Execution: %lf\n", elapsed_time(send_end, back_time));
    printf("FREE TIME: %lf\n", elapsed_time(back_time, free_time));
    printf("SEND TO SEND BACK: %lf\n", elapsed_time(back_time, back_end_time));
    printf("SEND ARRAY X BACK: %lf\n", elapsed_time(free_time, arrayX_time));
    printf("SEND ARRAY Y BACK: %lf\n", elapsed_time(arrayX_time, arrayY_time));
    printf("SEND WEIGHTS BACK: %lf\n", elapsed_time(arrayY_time, back_end_time));

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
        xe += arrayX[x] * weights[x];
        ye += arrayY[x] * weights[x];
    }
    double distance = sqrt(pow((double) (xe - (int) roundDouble(IszY / 2.0)), 2) + pow((double) (ye - (int) roundDouble(IszX / 2.0)), 2));

    //Output results
    FILE *fid;
    fid=fopen("output.txt", "w+");
    if( fid == NULL ){
      printf( "The file was not opened for writing\n" );
      return -1;
    }
    fprintf(fid, "XE: %lf\n", xe);
    fprintf(fid, "YE: %lf\n", ye);
    fprintf(fid, "distance: %lf\n", distance);
    fclose(fid);


    //OpenCL freeing of memory
    clReleaseMemObject(weights_GPU);
    clReleaseMemObject(arrayY_GPU);
    clReleaseMemObject(arrayX_GPU);

    //free regular memory
    free(likelihood);
    free(arrayX);
    free(arrayY);
    free(xj);
    free(yj);
    free(CDF);
    free(ind);
    free(u);
}

int main(int argc, char * argv[]) {

    char* usage = "double.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
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
    int * seed = (int *) calloc(Nparticles, sizeof(int));
    int i;
    for (i = 0; i < Nparticles; i++)
      seed[i] = i+1;
      //        seed[i] = time(0) * i;
    //calloc matrix
    unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
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
