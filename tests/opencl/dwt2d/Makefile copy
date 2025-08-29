#all:
#	g++ -o components.o -I/usr/local/cuda-5.5/include -c components.cpp
#	g++ -o prog  -I/usr/local/cuda-5.5/include main.cpp components.o -lOpenCL
#
include ../../common/make.config


ifdef OUTPUT
	override OUTPUT = -DOUTPUT
endif


OUTPUT = -DOUTPUT

C_C = g++
OCL_LIB = -lOpenCL
OCL_INC = -I/usr/local/cuda-5.5/include 

default:	dwt2d

components:
	$(C_C) -o components.o $(OCL_INC) -c components.cpp 

dwt2d:
	$(C_C) -o dwt2d $(OCL_INC) \
		main.cpp \
		$(components) \
		$(OUTPUT) \
		$(OCL_LIB)

clean:	
	rm -rf *.o dwt2d
	rm *.bmp.dwt.*

