include ../../common/make.config

#Can be changed by `make TYPE=CPU`
TYPE = GPU

#Library
ifeq ($(TYPE),GPU)
	OPENCL_INC = $(NV_OPENCL_INC)
	OPENCL_LIB = $(NV_OPENCL_LIB)
else
	OPENCL_INC = $(INTEL_OPENCL_INC)
	OPENCL_LIB = $(INTEL_OPENCL_LIB)
endif

# C compiler
CC = g++
CC_FLAGS = -g -O2

backprop.out: backprop.o imagenet.o facetrain.o
	$(CC) $(CC_FLAGS) backprop_ocl.cpp ../util/timing.c backprop.o imagenet.o facetrain.o -o backprop.out -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -I../util -DTIMING -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c
	
backprop.o: backprop.c
	$(CC) $(CC_FLAGS) -I$(OPENCL_INC) backprop.c -c -Wno-unused-result
	
facetrain.o: facetrain.c
	$(CC) $(CC_FLAGS) -I$(OPENCL_INC) facetrain.c -c

imagenet.o: imagenet.c
	$(CC) $(CC_FLAGS) -I$(OPENCL_INC) imagenet.c -c

run: backprop.out
	./backprop.out -n 4194304 -p 0 -d 0

clean:
	rm -f *.o *~ *.out *.linkinfo
