ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 

LIBS = -lOpenCL 
OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
OPENCL_INC = $(OPENCL_DIR)/include/ 
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

3D: 3D.c CL_helper.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -o $@

clean:
	rm -f 3D $(OUTPUT) 
