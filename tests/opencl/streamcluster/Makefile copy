include ../../common/make.config

#C compiler
CC = g++


FLAGS = -O3
SRC = streamcluster.cpp
EXE = streamcluster

release:$(SRC)
	$(CC) -lOpenCL  $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)	$(FLAGS)

errmsg:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D ERRMSG $(FLAGS)
	
ptx:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D PTX_MSG $(FLAGS)
	
res:$(SRC)
	$(CC) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D RES_MSG $(FLAGS)
	
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
