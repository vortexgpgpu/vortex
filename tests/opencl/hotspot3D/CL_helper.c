#include "CL_helper.h"

#define STR_SIZE 256

long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

char *err_code (cl_int err_in)
{
    switch (err_in) {

        case CL_SUCCESS :
            return (char*)" CL_SUCCESS ";
        case CL_DEVICE_NOT_FOUND :
            return (char*)" CL_DEVICE_NOT_FOUND ";
        case CL_DEVICE_NOT_AVAILABLE :
            return (char*)" CL_DEVICE_NOT_AVAILABLE ";
        case CL_COMPILER_NOT_AVAILABLE :
            return (char*)" CL_COMPILER_NOT_AVAILABLE ";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE :
            return (char*)" CL_MEM_OBJECT_ALLOCATION_FAILURE ";
        case CL_OUT_OF_RESOURCES :
            return (char*)" CL_OUT_OF_RESOURCES ";
        case CL_OUT_OF_HOST_MEMORY :
            return (char*)" CL_OUT_OF_HOST_MEMORY ";
        case CL_PROFILING_INFO_NOT_AVAILABLE :
            return (char*)" CL_PROFILING_INFO_NOT_AVAILABLE ";
        case CL_MEM_COPY_OVERLAP :
            return (char*)" CL_MEM_COPY_OVERLAP ";
        case CL_IMAGE_FORMAT_MISMATCH :
            return (char*)" CL_IMAGE_FORMAT_MISMATCH ";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED :
            return (char*)" CL_IMAGE_FORMAT_NOT_SUPPORTED ";
        case CL_BUILD_PROGRAM_FAILURE :
            return (char*)" CL_BUILD_PROGRAM_FAILURE ";
        case CL_MAP_FAILURE :
            return (char*)" CL_MAP_FAILURE ";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET :
            return (char*)" CL_MISALIGNED_SUB_BUFFER_OFFSET ";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
            return (char*)" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ";
        case CL_INVALID_VALUE :
            return (char*)" CL_INVALID_VALUE ";
        case CL_INVALID_DEVICE_TYPE :
            return (char*)" CL_INVALID_DEVICE_TYPE ";
        case CL_INVALID_PLATFORM :
            return (char*)" CL_INVALID_PLATFORM ";
        case CL_INVALID_DEVICE :
            return (char*)" CL_INVALID_DEVICE ";
        case CL_INVALID_CONTEXT :
            return (char*)" CL_INVALID_CONTEXT ";
        case CL_INVALID_QUEUE_PROPERTIES :
            return (char*)" CL_INVALID_QUEUE_PROPERTIES ";
        case CL_INVALID_COMMAND_QUEUE :
            return (char*)" CL_INVALID_COMMAND_QUEUE ";
        case CL_INVALID_HOST_PTR :
            return (char*)" CL_INVALID_HOST_PTR ";
        case CL_INVALID_MEM_OBJECT :
            return (char*)" CL_INVALID_MEM_OBJECT ";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
            return (char*)" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
        case CL_INVALID_IMAGE_SIZE :
            return (char*)" CL_INVALID_IMAGE_SIZE ";
        case CL_INVALID_SAMPLER :
            return (char*)" CL_INVALID_SAMPLER ";
        case CL_INVALID_BINARY :
            return (char*)" CL_INVALID_BINARY ";
        case CL_INVALID_BUILD_OPTIONS :
            return (char*)" CL_INVALID_BUILD_OPTIONS ";
        case CL_INVALID_PROGRAM :
            return (char*)" CL_INVALID_PROGRAM ";
        case CL_INVALID_PROGRAM_EXECUTABLE :
            return (char*)" CL_INVALID_PROGRAM_EXECUTABLE ";
        case CL_INVALID_KERNEL_NAME :
            return (char*)" CL_INVALID_KERNEL_NAME ";
        case CL_INVALID_KERNEL_DEFINITION :
            return (char*)" CL_INVALID_KERNEL_DEFINITION ";
        case CL_INVALID_KERNEL :
            return (char*)" CL_INVALID_KERNEL ";
        case CL_INVALID_ARG_INDEX :
            return (char*)" CL_INVALID_ARG_INDEX ";
        case CL_INVALID_ARG_VALUE :
            return (char*)" CL_INVALID_ARG_VALUE ";
        case CL_INVALID_ARG_SIZE :
            return (char*)" CL_INVALID_ARG_SIZE ";
        case CL_INVALID_KERNEL_ARGS :
            return (char*)" CL_INVALID_KERNEL_ARGS ";
        case CL_INVALID_WORK_DIMENSION :
            return (char*)" CL_INVALID_WORK_DIMENSION ";
        case CL_INVALID_WORK_GROUP_SIZE :
            return (char*)" CL_INVALID_WORK_GROUP_SIZE ";
        case CL_INVALID_WORK_ITEM_SIZE :
            return (char*)" CL_INVALID_WORK_ITEM_SIZE ";
        case CL_INVALID_GLOBAL_OFFSET :
            return (char*)" CL_INVALID_GLOBAL_OFFSET ";
        case CL_INVALID_EVENT_WAIT_LIST :
            return (char*)" CL_INVALID_EVENT_WAIT_LIST ";
        case CL_INVALID_EVENT :
            return (char*)" CL_INVALID_EVENT ";
        case CL_INVALID_OPERATION :
            return (char*)" CL_INVALID_OPERATION ";
        case CL_INVALID_GL_OBJECT :
            return (char*)" CL_INVALID_GL_OBJECT ";
        case CL_INVALID_BUFFER_SIZE :
            return (char*)" CL_INVALID_BUFFER_SIZE ";
        case CL_INVALID_MIP_LEVEL :
            return (char*)" CL_INVALID_MIP_LEVEL ";
        case CL_INVALID_GLOBAL_WORK_SIZE :
            return (char*)" CL_INVALID_GLOBAL_WORK_SIZE ";
        case CL_INVALID_PROPERTY :
            return (char*)" CL_INVALID_PROPERTY ";
        default:
            return (char*)"UNKNOWN ERROR";

    }
}

int output_device_info(cl_device_id device_id)
{
    int err;                            
    cl_device_type device_type;         
    cl_uint comp_units;                 
    cl_char vendor_name[1024] = {0};    
    cl_char device_name[1024] = {0};    
#ifdef VERBOSE
    cl_uint          max_work_itm_dims;
    size_t           max_wrkgrp_size;
    size_t          *max_loc_size;
#endif


    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device name!\n");
        return EXIT_FAILURE;
    }
    printf(" \n Device is  %s \n",device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device vendor name!\n");
        return EXIT_FAILURE;
    }
    printf(" %s \n",vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }

#ifdef VERBOSE
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                               &max_work_itm_dims, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!\n",
                                                                            err_code(err));
        return EXIT_FAILURE;
    }
    
    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if(max_loc_size == NULL){
       printf(" malloc failed\n");
       return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t), 
                               max_loc_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!\n",err_code(err));
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                               &max_wrkgrp_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)!\n",err_code(err));
        return EXIT_FAILURE;
    }
   printf("work group, work item information");
   printf("\n max loc dim ");
   for(int i=0; i< max_work_itm_dims; i++)
     printf(" %d ",(int)(*(max_loc_size+i)));
   printf("\n");
   printf(" Max work group size = %d\n",(int)max_wrkgrp_size);
#endif

    return CL_SUCCESS;

}


void fatal(char *s)
{
  fprintf(stderr, "Error: %s\n", s);
}

char *load_kernel_source(const char *filename) {
    // Open the source file
    FILE *file = fopen(filename, "r");
    if (file == NULL) fatal("Error opening kernel source file\n");

    // Determine the size of the file
    if (fseek(file, 0, SEEK_END)) fatal("Error reading kernel source file\n");
    size_t size = ftell(file);

    // Allocate space for the source code (plus one for null-terminator)
    char *source = (char *) malloc(size + 1);

    // Read the source code into the string
    fseek(file, 0, SEEK_SET);
    // printf("Number of elements: %lu\nSize = %lu", fread(source, 1, size, file), size);
    // exit(1);
    if (fread(source, 1, size, file) != size) fatal("Error reading kernel source file\n");

    // Null-terminate the string
    source[size] = '\0';

    // Return the pointer to the string
    return source;
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

    int i,j,k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
      fatal( "The file was not opened" );


    for (i=0; i <= grid_rows-1; i++) 
      for (j=0; j <= grid_cols-1; j++)
        for (k=0; k <= layers-1; k++)
          {
            if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
            if (feof(fp))
              fatal("not enough lines in file");
            //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
            if ((sscanf(str, "%f", &val) != 1))
              fatal("invalid file format");
            vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
          }

    fclose(fp);	

}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

    int i,j,k, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
      printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
      for (j=0; j < grid_cols; j++)
        for (k=0; k < layers; k++)

          {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
            fputs(str,fp);
            index++;
          }

    fclose(fp);	
}

void computeTempCPU(float *pIn, float* tIn, float *tOut, 
        int nx, int ny, int nz, float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, float amb_temp, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
        for(z = 0; z < nz; z++)
            for(y = 0; y < ny; y++)
                for(x = 0; x < nx; x++)
                {
                    c = x + y * nx + z * nx * ny;

                    w = (x == 0) ? c : c - 1;
                    e = (x == nx - 1) ? c : c + 1;
                    n = (y == 0) ? c : c - nx;
                    s = (y == ny - 1) ? c : c + nx;
                    b = (z == 0) ? c : c - nx * ny;
                    t = (z == nz - 1) ? c : c + nx * ny;


                    tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
                }
        float *temp = tIn;
        tIn = tOut;
        tOut = temp; 
        i++;
    }
    while(i < numiter);

}

float accuracy(float *arr1, float *arr2, int len)
{
    float err = 0.0; 
    int i;
    for(i = 0; i < len; i++)
    {
        err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
    }

    return (float)sqrt(err/len);
}
 


