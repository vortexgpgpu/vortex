RTL_DIR=./rtl
SCRIPT_DIR=./scripts

all: config

config: VX_config.h VX_types.h

VX_config.h: $(RTL_DIR)/VX_config.vh	 
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_config.vh -o VX_config.h

VX_types.h: $(RTL_DIR)/VX_types.vh	 
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_types.vh -o VX_types.h

clean:
	rm -f VX_config.h VX_types.h

.PHONY: VX_config.h VX_types.h