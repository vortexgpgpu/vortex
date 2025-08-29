include ../../common/make.config

#C compiler
CC = g++

CC_FLAGS = -g -O3 -Wall

EXE = nw

$(EXE): nw.c
	$(CC) ${KERNEL_DIM} $(CC_FLAGS) -o $(EXE) nw.c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -f $(EXE)
