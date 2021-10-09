RTL_DIR=./rtl
SCRIPT_DIR=./scripts

all: VX_config.h

VX_config.h: $(RTL_DIR)/VX_config.vh	 
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_config.vh -o VX_config.h

clean:
	rm -f VX_config.h

.PHONY: VX_config.h