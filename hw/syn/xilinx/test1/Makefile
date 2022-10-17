
VIVADO = $(XILINX_VIVADO)/bin/vivado

all: create_project

create_project: project_1
project_1: project_1.tcl
	$(VIVADO) -mode batch -source project_1.tcl

run:
	$(VIVADO) project_1/project_1.xpr &

clean:
	rm -rf project_1