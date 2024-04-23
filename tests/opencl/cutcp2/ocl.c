#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include "ocl.h"

char* readFile(const char* fileName)
{
        FILE* fp;
        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error 1!\n");
                exit(1);
        }

        fseek(fp,0,SEEK_END);
        long size = ftell(fp);
        rewind(fp);

        char* buffer = (char*)malloc(sizeof(char)*(size+1));
        if(buffer  == NULL)
        {
                printf("Error 2!\n");
                fclose(fp);
                exit(1);
        }

        size_t res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error 3!\n");
                fclose(fp);
                exit(1);
        }

	buffer[size] = 0;
        fclose(fp);
        return buffer;
}

void clMemSet(cl_command_queue clCommandQueue, cl_mem buf, int val, size_t size)
{
	cl_int clStatus;
	char* temp = (char*)malloc(size);
	memset(temp,val,size);
	clStatus = clEnqueueWriteBuffer(clCommandQueue,buf,CL_TRUE,0,size,temp,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	free(temp);
}
