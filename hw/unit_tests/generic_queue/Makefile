all: testbench.iv

testbench.iv: testbench.v
	iverilog testbench.v -o testbench.iv -I ../../rtl/

run: testbench.iv
	! vvp testbench.iv | grep 'ERROR' || false

clean:
	rm testbench.iv
	
