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

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp
OCL_LIB =	-L$(OPENCL_LIB) \
			-lOpenCL
OCL_INC = -I$(OPENCL_INC)

# link objects together
a.out:		main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o \
			./timing.o
	$(C_C)	main.o \
			./kernel/kernel_gpu_opencl_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/opencl/opencl.o \
			./timing.o \
			-lm \
			$(OCL_LIB) \
			-o lavaMD.out

# compile function files into objects
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu_opencl_wrapper.h \
			./kernel/kernel_gpu_opencl_wrapper.c \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/opencl/opencl.h \
			./util/opencl/opencl.c
	$(C_C)	$(OUTPUT) $(KERNEL_DIM) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
										./kernel/kernel_gpu_opencl_wrapper.c
	$(C_C)	$(KERNEL_DIM)				./kernel/kernel_gpu_opencl_wrapper.c \
										$(OCL_INC) \
										-I../util -DTIMING \
										-c \
										-o ./kernel/kernel_gpu_opencl_wrapper.o \
										-O3

./timing.o: ../util/timing.c
	$(C_C)	$(OUTPUT) $(KERNEL_DIM) -I../util ../util/timing.c \
			-c \
			-o timing.o \
			-O3

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

./util/opencl/opencl.o:	./util/opencl/opencl.h \
						./util/opencl/opencl.c
	$(C_C)				./util/opencl/opencl.c \
						$(OCL_INC) \
						-c \
						-o ./util/opencl/opencl.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o lavaMD.out \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		./util/opencl/*.o \
