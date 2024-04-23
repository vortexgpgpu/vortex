all: OpenCL/leukocyte 

OpenCL/leukocyte:
	cd OpenCL; make
	mv OpenCL/leukocyte.out .

clean:
	cd OpenCL; make clean
	rm -f meschach_lib/meschach.a
	rm -f leukocyte.out
