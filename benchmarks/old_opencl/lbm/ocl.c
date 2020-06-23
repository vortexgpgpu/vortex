#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "ocl.h"

char* readFile(char* fileName)
{
	FILE* fp;
	fp = fopen(fileName,"r");

	if(fp == NULL)
	{
		printf("Error 1!\n");
		return NULL;
	}

	fseek(fp,0,SEEK_END);
	long size = ftell(fp);
	rewind(fp);

	char* buffer = (char*)malloc(sizeof(char)*(size+1));
	if(buffer == NULL)
	{
		printf("Error 2!\n");
		fclose(fp);
		return NULL;
	}

	size_t res = fread(buffer,1,size,fp);
	if(res != size)
	{
		printf("Error 3!\n");
		fclose(fp);
		return NULL;
	}

	buffer[size] = 0;
	fclose(fp);
	return buffer;
}
