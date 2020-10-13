.PHONY: build_config

build_config:
	./scripts/gen_config.py --outv ./rtl/VX_user_config.vh --outc ./VX_config.h
	$(MAKE) -C simulate	

clean:
	rm -f ./rtl/VX_user_config.vh ./VX_config.h
	$(MAKE) -C simulate clean