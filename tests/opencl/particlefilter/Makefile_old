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
CC_FLAGS = -O2 
INCLUDES = -I$(OPENCL_INC) -I../util

ALL: OCL_particlefilter_naive OCL_particlefilter_double OCL_particlefilter_single

OCL_particlefilter_naive: ex_particle_OCL_naive_seq.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_naive_seq.o \
		$(INCLUDES) -L$(OPENCL_LIB) -lOpenCL \
		-o OCL_particlefilter_naive.out

OCL_particlefilter_double: ex_particle_OCL_double_seq.o timing.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_double_seq.o timing.o \
		$(INCLUDES)  -L$(OPENCL_LIB) -lOpenCL \
		-o OCL_particlefilter_double.out

OCL_particlefilter_single: ex_particle_OCL_single_seq.o timing.o
	$(CC) $(CC_FLAGS) ex_particle_OCL_single_seq.o timing.o \
		$(INCLUDES)  -L$(OPENCL_LIB) -lOpenCL \
		-o OCL_particlefilter_single.out


%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

timing.o: ../util/timing.c
	$(CC) $(CC_FLAGS) $(INCLUDES) ../util/timing.c -c

ex_particle_OCL_naive_seq.o: ex_particle_OCL_naive_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_naive_seq.cpp -c -Wno-unused-result

ex_particle_OCL_double_seq.o: ex_particle_OCL_double_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_double_seq.cpp -DTIMING -c -Wno-unused-result

ex_particle_OCL_single_seq.o: ex_particle_OCL_single_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_single_seq.cpp -DTIMING -c -Wno-unused-result

clean:
	rm -f *.o *~ OCL_particlefilter_*.out *.linkinfo
