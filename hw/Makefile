.PHONY: build_config

build_config: ./rtl/VX_config.vh	 
	./scripts/gen_config.py -i ./rtl/VX_config.vh -o ./VX_config.h
	$(MAKE) -C simulate	

clean:
	rm -f ./VX_config.h
	$(MAKE) -C simulate clean