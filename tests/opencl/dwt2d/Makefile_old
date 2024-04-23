#all:
#	g++ -o components.o -I/usr/local/cuda-5.5/include -c components.cpp
#	g++ -o prog  -I/usr/local/cuda-5.5/include main.cpp components.o -lOpenCL
#
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

ifdef OUTPUT
	override OUTPUT = -DOUTPUT
endif

OUTPUT = -DOUTPUT

C_C = g++
OCL_LIB = -lOpenCL -L$(OPENCL_LIB)
#OCL_INC = -I/usr/local/cuda-5.5/include
OCL_INC = -I$(OPENCL_INC)

default:	dwt2d

components:
	$(C_C) -o components.o $(OCL_INC) -c components.cpp 

dwt2d:
	$(C_C) -o dwt2d.out $(OCL_INC) \
		-I../util -DTIMING \
		main.cpp ../util/timing.c \
		$(components) \
		$(OUTPUT) \
		$(OCL_LIB)

clean:	
	rm -rf *.o dwt2d.out
	rm *.bmp.dwt.*

