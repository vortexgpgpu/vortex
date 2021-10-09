RTL_DIR=./rtl
SCRIPT_DIR=./scripts
DESTDIR ?= .

all: VX_config.h vortex_afu.h

VX_config.h: $(RTL_DIR)/VX_config.vh	 
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_config.vh -o $(DESTDIR)/VX_config.h

vortex_afu.h : $(RTL_DIR)/afu/vortex_afu.vh
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/afu/vortex_afu.vh -o $(DESTDIR)/vortex_afu.h

clean:
	rm -f $(DESTDIR)/VX_config.h  $(DESTDIR)/vortex_afu.h

.PHONY: VX_config.h vortex_afu.h