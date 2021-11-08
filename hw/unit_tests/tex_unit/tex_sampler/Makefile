TOP = VX_tex_sampler

PARAMS ?=

INCLUDE = -I../../../rtl/ -I../../../rtl/libs -I../../../rtl/tex_unit

SRCS = main.cpp

all: build

CF += -std=c++11 -fms-extensions -I../..
VF += $(PARAMS)

VF += --language 1800-2009 --assert -Wall --trace
VF += -Wno-DECLFILENAME
VF += --x-initial unique
VF += -exe $(SRCS) $(INCLUDE)
VF += $(PARAMS)

gen:
	verilator $(VF) -cc $(TOP).v -CFLAGS '$(CF)' --exe $(SRCS) 
	
build: gen
	(cd obj_dir && make -j -f V$(TOP).mk)

run: build
	(cd obj_dir && ./V$(TOP))

clean:
	rm -rf obj_dir
